import os
import streamlit as st
import litert_lm
from qdrant_edge import EdgeShard, Query, QueryRequest
from sentence_transformers import SentenceTransformer
from quiet_native import silence_stderr

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
silence_stderr()

EMBED_MODEL_NAME = "models/qwen3_embed"
MODEL_PATH = "models/gemma4/gemma-4-E2B-it.litertlm"
SHARD_PATH = "qdrant_edge_eg"

@st.cache_resource
def load_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL_NAME)

@st.cache_resource
def load_shard() -> EdgeShard:
    return EdgeShard.load(SHARD_PATH)

@st.cache_resource
def load_engine() -> litert_lm.Engine:
    return litert_lm.Engine(MODEL_PATH, backend=litert_lm.Backend.GPU())

def retrieve_context(user_query: str, limit: int = 2) -> str:
    model = load_embedder()
    shard = load_shard()
    query_vector = model.encode_query(user_query).tolist()
    results = shard.query(
        QueryRequest(
            query=Query.Nearest(query_vector),
            limit=limit,
            with_payload=True,
        )
    )
    return "".join(point.payload["text"] for point in results)

def reply_stream(user_query: str):
    context = retrieve_context(user_query)
    messages = [
        litert_lm.Message.system(
            "You are a Indian Budget assistant. "
            "Answer the user question using only the provided context. "
            "If the answer is not in the context, say: I don't know based on the provided context. "
            "Do not guess or add extra explanation."
            "Answer in 1-2 short sentences. Do not elaborate unless told by user"
        )
    ]
    prompt = f"Context:\n{context}\n\nQuestion:\n{user_query}"

    engine = load_engine()
    with engine.create_conversation(
        messages=messages,
        sampler_config=litert_lm.SamplerConfig(top_k=1),
    ) as conversation:
        for chunk in conversation.send_message_async(prompt):
            for item in chunk.get("content", []):
                if item.get("type") == "text" and item.get("text"):
                    yield item["text"]

st.title("On-device Offline Chat")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input("Ask something"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        answer = st.write_stream(reply_stream(prompt))

    st.session_state.messages.append({"role": "assistant", "content": answer})

# what is the focus on Purvodaya for north-east?