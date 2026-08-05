from pathlib import Path

import edgeparse
from qdrant_edge import (
    Distance,EdgeConfig,EdgeShard,EdgeVectorParams,
    Point,Query,QueryRequest,UpdateOperation,
)
from sentence_transformers import SentenceTransformer

from utils import Report

MODEL_DIM = 1024
EMBED_MODEL_NAME = "models/qwen3_embed"
DOCUMENT_PATH = "data/budget_2026.pdf"
SHARD_PATH = "db"

class DocumentIngestion:
    def __init__(
        self,
        document_path: str,
        model_name: str,
        chunk_size: int = 1024,
        overlap: int = 0,
        report: Report | None = None,
    ) -> None:
        self.document_path = Path(document_path)
        self.shard_path = Path(SHARD_PATH)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.report = report or Report("retrieval_report")

        with self.report.stage("load_embed_model"):
            self.model = SentenceTransformer(model_name)

        with self.report.stage("load_or_create_shard"):
            self.shard = self._load_or_create_shard()

    def _load_or_create_shard(self) -> EdgeShard:
        config = EdgeConfig(
            vectors=EdgeVectorParams(size=MODEL_DIM, distance=Distance.Cosine)
        )

        self.shard_path.mkdir(parents=True, exist_ok=True)

        if any(self.shard_path.iterdir()):
            print(f"Already exists {self.shard_path}")
            return EdgeShard.load(str(self.shard_path))
        return EdgeShard.create(str(self.shard_path), config)

    def chunk_text(self, text: str) -> list[str]:
        return [
            chunk
            for start in range(0, len(text), self.chunk_size - self.overlap)
            if (chunk := text[start : start + self.chunk_size].strip())
        ]

    def index(self) -> int:
        with self.report.stage("parse_document"):
            markdown = edgeparse.convert(str(self.document_path), format="markdown") # load the data
            # markdown - raw text
            chunks = self.chunk_text(markdown)
            # convert that into chunks

        with self.report.stage("embed_chunks"):
            embeddings = self.model.encode_document(chunks)
            # encode => text to vectors (dense) using Qwen3 embedding model

        with self.report.stage("upsert_and_optimize"):
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
        return len(chunks)
    
    # https://github.com/qdrant/qdrant/blob/master/lib/edge/python/examples/mmr-query.py
    def inference(self, user_query: str, limit: int = 2) -> str:
        with self.report.stage(f"query: {user_query[:40]}"):
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
    report = Report("retrieval_report")
    ingestion = DocumentIngestion(
        document_path=DOCUMENT_PATH, model_name=EMBED_MODEL_NAME, report=report
    )
    indexed_count = ingestion.index()
    print(f"Indexed {indexed_count} chunks")

    user_query = "what is the budget alloted for Biopharma Shakti project?"
    context = ingestion.inference(user_query)
    print(context)

    report.print_table()
    report.save()

if __name__ == "__main__":
    main()