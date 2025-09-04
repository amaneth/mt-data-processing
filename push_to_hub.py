from datasets import load_from_disk, DatasetDict
from huggingface_hub import HfApi
import os

def push_dataset_to_hub(local_dir: str, repo_id: str, private: bool = True, token: str = None):

    # Load dataset from disk
    dataset = load_from_disk(local_dir)

    # If it's a Dataset, wrap in DatasetDict for consistency
    if not isinstance(dataset, DatasetDict):
        dataset = DatasetDict({"train": dataset})

    # Create repo if it doesn’t exist
    api = HfApi()
    try:
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, token=token)
        print(f"✅ Created dataset repo: {repo_id}")
    except Exception as e:
        print(f"ℹ️ Repo might already exist: {e}")

    # Push dataset to Hub
    dataset.push_to_hub(repo_id, private=private, token=token)
    print(f"🚀 Successfully pushed dataset from {local_dir} to {repo_id}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Push a local dataset to Hugging Face Hub.")
    parser.add_argument("--local_dir", type=str, required=True, help="Path to dataset saved with save_to_disk.")
    parser.add_argument("--repo_id", type=str, required=True, help="Repo id (e.g. username/dataset_name).")
    parser.add_argument("--private", action="store_true", help="Upload dataset as private.")
    parser.add_argument("--token", type=str, default=None, help="Optional Hugging Face token.")

    args = parser.parse_args()

    push_dataset_to_hub(args.local_dir, args.repo_id, args.private, args.token)