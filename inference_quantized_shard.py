import os
import litert_lm
from qdrant_edge import EdgeShard, Query, QueryRequest
from sentence_transformers import SentenceTransformer

os.environ["HF_HUB_OFFLINE"] = "1"
from quiet_native import silence_stderr
silence_stderr()

EMBED_MODEL_NAME = "models/qwen3_embed"
SHARD_PATH = "quantized_shard"
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
        self.gemma_model_path = gemma_model_path

    def retrieve_context(self, user_query: str, limit: int = 2) -> str:
        query_vector = self.model.encode_query(user_query).tolist()
        results = self.shard.query(
            QueryRequest(
                query=Query.Nearest(query_vector),
                limit=limit,
                with_payload=True,
            )
        )
        return "".join(point.payload["text"] for point in results)

    def answer(self, user_query: str, limit: int = 3) -> str:
        context = self.retrieve_context(user_query, limit=limit)
        messages = [
            litert_lm.Message.system(
                "You are a Indian Budget assistant. "
                "Answer the user question using only the provided context. "
                "If the answer is not in the context, say: I don't know based on the provided context. "
                "Do not guess or add extra explanation."
                "Be smart and robust to understand the user query intent and context"
            )
        ]

        with litert_lm.Engine(
            self.gemma_model_path,
            backend=litert_lm.Backend.CPU(),
        ) as engine:
            with engine.create_conversation(messages=messages) as conversation:
                response = conversation.send_message(
                    f"Context:\n{context}\n\nQuestion:\n{user_query}"
                )
        return response["content"][0]["text"]

def main() -> None:
    inference = GemmaInference()
    user_query = "what is the budget alloted for Biopharma Shakti project?"
    print(inference.answer(user_query))

if __name__ == "__main__":
    main()