from pathlib import Path

import edgeparse
from qdrant_edge import (
    Distance,EdgeConfig,EdgeShard,EdgeVectorParams,
    Point,Query,QueryRequest,UpdateOperation,
)
from litert_lm import Backend, EmbeddingEngine

from quiet_native import silence_stderr
from utils import Report

silence_stderr()  # mute the native LiteRT-LM INFO/WARN logs on stderr

MODEL_DIM = 768  # EmbeddingGemma-2 output dim (Qwen3 was 1024)
EMBED_MODEL_PATH = "models/embeddinggemma-2-740m.litertlm"
DOCUMENT_PATH = "data/budget_2026.pdf"
SHARD_PATH = "db_gemma"
# LiteRT-LM writes compiled kernel/weight caches; keep them out of models/.
CACHE_DIR = ".cache"
# EmbeddingGemma-2 is prompt-conditioned: documents and queries MUST be wrapped
# in these task templates, or retrieval quality collapses.
# https://ai.google.dev/gemma/docs/embeddinggemma/model_card
DOC_PROMPT = "title: none | text: {text}"
QUERY_PROMPT = "task: search result | query: {text}"

class DocumentIngestion:
    def __init__(
        self,
        document_path: str,
        model_path: str,
        chunk_size: int = 1024,
        overlap: int = 0,
        report: Report | None = None,
    ) -> None:
        self.document_path = Path(document_path)
        self.shard_path = Path(SHARD_PATH)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.report = report or Report("embeddinggemma2_litert")

        Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
        with self.report.stage("load_embed_model"):
            self.engine = EmbeddingEngine(
                model_path, backend=Backend.GPU(), cache_dir=CACHE_DIR
            )

        with self.report.stage("load_or_create_shard"):
            self.shard = self._load_or_create_shard()

    def close(self) -> None:
        self.engine.close()

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
            documents = [DOC_PROMPT.format(text=chunk) for chunk in chunks]
            responses = self.engine.compute_embedding_batch(documents)
            # text to vectors (dense) using EmbeddingGemma-2 via LiteRT-LM;
            # .embedding is already a plain list[float] (no .tolist() needed)
            embeddings = [response.embedding for response in responses]

        with self.report.stage("upsert_and_optimize"):
            self.shard.update(
                UpdateOperation.upsert_points(
                    [
                        Point(
                            index,
                            embeddings[index - 1],
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
            query_vector = self.engine.compute_embedding(
                QUERY_PROMPT.format(text=user_query)
            ).embedding
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
    report = Report("embeddinggemma2_litert")
    ingestion = DocumentIngestion(
        document_path=DOCUMENT_PATH, model_path=EMBED_MODEL_PATH, report=report
    )
    indexed_count = ingestion.index()
    print(f"Indexed {indexed_count} chunks")

    user_query = "what is the budget alloted for Biopharma Shakti project?"
    context = ingestion.inference(user_query)
    print(context)

    ingestion.close()
    report.print_table()
    report.save()

if __name__ == "__main__":
    main()
