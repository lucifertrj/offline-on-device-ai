from huggingface_hub import hf_hub_download, snapshot_download

def download_embedding_model() -> None:
    print("Downloading Qwen embedding model...")
    EMBEDDING_REPO_ID = "Qwen/Qwen3-Embedding-0.6B"
    EMBEDDING_LOCAL_DIR = "models/qwen3_embed"

    snapshot_download(
        repo_id = EMBEDDING_REPO_ID,
        local_dir = EMBEDDING_LOCAL_DIR,
        local_dir_use_symlinks=False,
    )

    print(f"Qwen model saved at: {EMBEDDING_LOCAL_DIR}")

def download_gemma_litert_model() -> None:
    print("Downloading Gemma LiteRT model...")
    GEMMA_REPO_ID = "litert-community/gemma-4-E2B-it-litert-lm"
    GEMMA_FILENAME = "gemma-4-E2B-it.litertlm"
    GEMMA_LOCAL_DIR = "models/gemma4"

    model_path = hf_hub_download(
        repo_id=GEMMA_REPO_ID,
        filename=GEMMA_FILENAME,
        local_dir=GEMMA_LOCAL_DIR,
    )
    print(f"Gemma LiteRT model saved at: {model_path}")

def main() -> None:
    download_embedding_model()
    download_gemma_litert_model()

if __name__ == "__main__":
    main()