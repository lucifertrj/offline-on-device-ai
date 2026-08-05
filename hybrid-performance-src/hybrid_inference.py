import os
import litert_lm
from qdrant_edge import Bm25, Bm25Config, EdgeShard, Fusion, Prefetch, Query, QueryRequest
from sentence_transformers import SentenceTransformer
from quiet_native import silence_stderr

os.environ["HF_HUB_OFFLINE"] = "1"
silence_stderr()

EMBED_MODEL_NAME = "models/qwen3_embed"
SHARD_PATH = "hybrid_shard"
GEMMA4_LITELLM = "models/gemma4/gemma-4-E2B-it.litertlm"

class GemmaInference:
    def __init__(
        self,
        embed_model_name: str = EMBED_MODEL_NAME,
        shard_path: str = SHARD_PATH,
        gemma_model_path: str = GEMMA4_LITELLM,
    ) -> None:
        self.model = SentenceTransformer(embed_model_name)
        self.shard = EdgeShard.load(shard_path)
        self.bm25 = Bm25(Bm25Config(language="english"))
        self.gemma_model_path = gemma_model_path
        self.engine = litert_lm.Engine(
            self.gemma_model_path,
            backend=litert_lm.Backend.GPU(),
        )

    def close(self) -> None:
        self.engine.close()

    def retrieve_context(self, user_query: str, limit: int = 2) -> str:
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

    def _system_messages(self) -> list[litert_lm.Message]:
        return [
            litert_lm.Message.system(
                "You are a Indian Budget assistant. "
                "Answer the user question using only the provided context. "
                "If the answer is not in the context, say: I don't know based on the provided context. "
                "Do not guess or add extra explanation."
                "Be smart and robust to understand the user query intent and context. "
                "Answer in 1-2 short sentences. Do not elaborate."
            )
        ]

    def answer_stream(self, user_query: str, limit: int = 3):
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

def main() -> None:
    inference = GemmaInference()
    user_query = "what is the budget alloted for Biopharma Shakti project?"
    try:
        for token in inference.answer_stream(user_query):
            print(token, end="", flush=True)
        print()
    finally:
        inference.close()

if __name__ == "__main__":
    main()
