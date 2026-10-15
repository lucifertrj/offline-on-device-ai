import os
import litert_lm
from litert_lm import Backend, EmbeddingEngine
from qdrant_edge import EdgeShard, Query, QueryRequest
from quiet_native import silence_stderr

from utils import Report

os.environ["HF_HUB_OFFLINE"] = "1"
silence_stderr()

EMBED_MODEL_NAME = "models/embeddinggemma-2-740m.litertlm"
SHARD_PATH = "db_gemma"
GEMMA4_LITELLM = "models/gemma4/gemma-4-E2B-it.litertlm"
CACHE_DIR = ".cache"

class GemmaInference:
    def __init__(
        self,
        embed_model_name: str = EMBED_MODEL_NAME,
        shard_path: str = SHARD_PATH,
        gemma_model_path: str = GEMMA4_LITELLM,
        report: Report | None = None,
    ) -> None:
        self.report = report or Report("qdrant_edge_litert_app")

        os.makedirs(CACHE_DIR, exist_ok=True)
        with self.report.stage("load_embed_model"):
            self.model = EmbeddingEngine(
                embed_model_name, backend=Backend.GPU(), cache_dir=CACHE_DIR
            )

        with self.report.stage("load_shard"):
            self.shard = EdgeShard.load(shard_path)

        self.gemma_model_path = gemma_model_path
        with self.report.stage("load_litert_engine"):
            self.engine = litert_lm.Engine(
                self.gemma_model_path,
                backend=litert_lm.Backend.GPU(),
            )

    def close(self) -> None:
        self.engine.close()
        self.model.close()

    def retrieve_context(self, user_query: str, limit: int = 2) -> str:
        query_vector = self.model.compute_embedding(user_query).embedding
        results = self.shard.query(
            QueryRequest(
                query=Query.Nearest(query_vector),
                limit=limit,
                with_payload=True,
            )
        )
        return "".join(point.payload["text"] for point in results)

    def _system_messages(self) -> list[litert_lm.Message]:
        return [
            litert_lm.Message.system(
                "You are a Indian Budget assistant. "
                "Answer the user question using only the provided context. "
                "If the answer is not in the context, say: I don't know"
                "Do not guess or add extra explanation."
                "Answer in 1-2 short sentences. Do not elaborate unless told by user"
            )
        ]

    def answer_stream(self, user_query: str, limit: int = 3, context: str | None = None):
        if context is None:
            context = self.retrieve_context(user_query, limit=limit)
        with self.engine.create_conversation(
            messages=self._system_messages(),
            sampler_config=litert_lm.SamplerConfig(top_k=1),
        ) as conversation:
            for chunk in conversation.send_message_async(
                f"Context:\n{context}\n\nQuestion:\n{user_query}"
            ):
                for item in chunk.get("content", []):
                    if item.get("type") == "text" and item.get("text"):
                        yield item["text"]

    def answer(self, user_query: str, limit: int = 3) -> str:
        return "".join(self.answer_stream(user_query, limit=limit))

QUERIES = [
    "what is the budget alloted for Biopharma Shakti project?",
    "What reforms were announced for direct taxes?",
    "What changes were made to indirect taxes in Budget 2026-2027?",
    "what is the focus on Purvodaya for north-east?"
]

def main() -> None:
    report = Report("qdrant_edge_litert_app")
    inference = GemmaInference(report=report)

    for i, user_query in enumerate(QUERIES, start=1):
        print(f"\n[{i}] {user_query}")
        with report.stage(f"q{i}_retrieve_context"):
            context = inference.retrieve_context(user_query)
        with report.stage(f"q{i}_generate_answer"):
            answer = "".join(inference.answer_stream(user_query, context=context))
        print(answer)

    inference.close()
    report.print_table()
    report.save()

if __name__ == "__main__":
    main()
