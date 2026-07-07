from pathlib import Path

import edgeparse
from qdrant_edge import (
    Distance,EdgeConfig,EdgeShard,EdgeVectorParams,
    Point,Query,QueryRequest,UpdateOperation,
)
from qdrant_edge import TurboQuantBitSize, TurboQuantQuantizationConfig
from sentence_transformers import SentenceTransformer

MODEL_DIM = 1024
EMBED_MODEL_NAME = "models/qwen3_embed"
DOCUMENT_PATH = "budget_2026.pdf"
SHARD_PATH = "quantized_shard"

class DocumentIngestion:
    def __init__(self, document_path: str, model_name: str,chunk_size: int = 1024, overlap: int = 0) -> None:
        self.document_path = Path(document_path)
        self.shard_path = Path(SHARD_PATH)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.model = SentenceTransformer(model_name)
        self.shard = self._load_or_create_shard()

    def _load_or_create_shard(self) -> EdgeShard:
        config = EdgeConfig(
            vectors=EdgeVectorParams(size=MODEL_DIM, distance=Distance.Cosine),
            quantization_config=TurboQuantQuantizationConfig(
                always_ram=True,
                bits=TurboQuantBitSize.Bits4,
            ),
        )

        self.shard_path.mkdir(parents=True, exist_ok=True)

        if any(self.shard_path.iterdir()):
            #print(self.shard_path)
            return EdgeShard.load(str(self.shard_path))
        
        return EdgeShard.create(str(self.shard_path), config)

    def chunk_text(self, text: str) -> list[str]:
        return [
            chunk
            for start in range(0, len(text), self.chunk_size - self.overlap)
            if (chunk := text[start : start + self.chunk_size].strip())
        ]

    def index(self) -> int:
        markdown = edgeparse.convert(str(self.document_path), format="markdown")
        chunks = self.chunk_text(markdown)
        embeddings = self.model.encode_document(chunks)

        self.shard.update(
            UpdateOperation.upsert_points(
                [
                    Point(
                        index,
                        embeddings[index - 1].tolist(),
                        {
                            "text": chunk,
                            "source": self.document_path.name,
                            "chunk_index": index - 1,
                        },
                    )
                    for index, chunk in enumerate(chunks, start=1)
                ]
            )
        )
        self.shard.optimize()
        info = self.shard.info()
        print(f"Points: {info.points_count}")
        print(f"Indexed vectors: {info.indexed_vectors_count}")
        return len(chunks)

    def inference(self, user_query: str, limit: int = 2) -> str:
        query_vector = self.model.encode_query(user_query).tolist()
        results = self.shard.query(
            QueryRequest(
                query=Query.Nearest(query_vector),
                limit=limit, with_payload=True,
            )
        )
        context = ""
        for point in results:
            context+= point.payload['text']

        return context

def main() -> None:
    ingestion = DocumentIngestion(document_path=DOCUMENT_PATH, model_name=EMBED_MODEL_NAME)
    indexed_count = ingestion.index()
    print(f"Indexed {indexed_count} chunks")

    user_query = "what is the budget alloted for Biopharma Shakti project?"
    context = ingestion.inference(user_query)
    print(context)

if __name__ == "__main__":
    main()