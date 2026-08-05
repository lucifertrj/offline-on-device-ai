from pathlib import Path
import edgeparse
from qdrant_edge import *
from sentence_transformers import SentenceTransformer

MODEL_DIM = 1024
EMBED_MODEL_NAME = "models/qwen3_embed"
DOCUMENT_PATH = "budget_2026.pdf"
SHARD_PATH = "hybrid_shard"

class HybridDocumentIngestion:
    def __init__(
        self,
        document_path: str,
        model_name: str,
        chunk_size: int = 756,
        overlap: int = 0,
    ) -> None:
        self.document_path = Path(document_path)
        self.shard_path = Path(SHARD_PATH)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.model = SentenceTransformer(model_name)
        self.bm25 = Bm25(Bm25Config(language="english"))
        self.shard = self._load_or_create_shard()

    def _load_or_create_shard(self) -> EdgeShard:
        config = EdgeConfig(
            vectors={"dense": EdgeVectorParams(size=MODEL_DIM, distance=Distance.Cosine)},
            sparse_vectors={"sparse": EdgeSparseVectorParams(modifier=Modifier.Idf)},
        )

        self.shard_path.mkdir(parents=True, exist_ok=True)

        if any(self.shard_path.iterdir()):
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
                        {
                            "dense": embeddings[index - 1].tolist(),
                            "sparse": self.bm25.embed_document(chunk),
                        },
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
        return len(chunks)

    def inference(self, user_query: str, limit: int = 2) -> str:
        dense_query = self.model.encode_query(user_query).tolist()
        sparse_query = self.bm25.embed_query(user_query)

        results = self.shard.query(
            QueryRequest(
                prefetches=[
                    Prefetch(
                        query=Query.Nearest(dense_query, using="dense"),
                        limit=limit,
                    ),
                    Prefetch(
                        query=Query.Nearest(sparse_query, using="sparse"),
                        limit=limit,
                    ),
                ],
                query=Fusion.Rrf(k=60),
                limit=limit,
                with_payload=True,
            )
        )
        return "".join(point.payload["text"] for point in results)


def main() -> None:
    ingestion = HybridDocumentIngestion(
        document_path=DOCUMENT_PATH,
        model_name=EMBED_MODEL_NAME,
    )
    indexed_count = ingestion.index()
    print(f"Indexed {indexed_count} chunks")

    user_query = "what is the budget alloted for Biopharma Shakti project?"
    context = ingestion.inference(user_query)
    print(context)

if __name__ == "__main__":
    main()