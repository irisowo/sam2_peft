import os
import argparse

from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download

# Configuration
REPO_ID = "irisqiu/sam2-peft-anomalypcb"
BASE_DIR = os.path.abspath("sam2_logs")
ENV_PATH = os.path.abspath(".env")


def setup_hf_login():
    if "HF_TOKEN" in os.environ:
        del os.environ["HF_TOKEN"]
        print("🧹 Removing previous HF_TOKEN")

    load_dotenv(ENV_PATH, override=True)
    token = os.environ.get("HF_TOKEN")
    token = token.strip()

    if not token:
        raise ValueError(f"❌ Unfoud {ENV_PATH}")

    return HfApi(token=token), token


def upload():
    api, _ = setup_hf_login()

    print(f"Ensuring repository {REPO_ID} exists...")
    api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)

    model_names = ["DORA_r16(0207051004)", "DORA_r16(0207051004)_stage2"]
    for model_name in model_names:
        model_dir = os.path.join(BASE_DIR, model_name)

        if not os.path.isdir(model_dir):
            print(f"⚠️ Directory not found: {model_dir}")
            continue

        local_file_path = os.path.join(model_dir, "checkpoints",
                                       "checkpoint_100.pt")

        if os.path.isfile(local_file_path):
            print(f"⏳ Uploading {model_name} (~1GB)...")
            repo_file_path = f"{model_name}/checkpoints/checkpoint_100.pt"

            try:
                api.upload_file(
                    path_or_fileobj=local_file_path,
                    path_in_repo=repo_file_path,
                    repo_id=REPO_ID,
                    repo_type="model",
                )
                print(f"✅ Successfully uploaded: {model_name}")
            except Exception as e:
                print(f"❌ Failed to upload {model_name}: {e}")
        else:
            print(
                f"⚠️ Checkpoint not found for {model_name} at {local_file_path}"
            )


def download(model_name, epoch, local_dir):
    _, token = setup_hf_login()
    os.makedirs(local_dir, exist_ok=True)

    repo_file_path = f"{model_name}/checkpoints/checkpoint_{epoch}.pt"

    print(f"⏳ Downloading '{model_name}' from '{REPO_ID}'...")

    try:
        downloaded_path = hf_hub_download(repo_id=REPO_ID,
                                          filename=repo_file_path,
                                          local_dir=local_dir,
                                          token=token)
        print(f"✅ Successfully downloaded to: {downloaded_path}")
    except Exception as e:
        print(f"❌ Failed to download {model_name}. Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="action",
                                       required=True,
                                       help="Choose 'upload' or 'download'")
    upload_parser = subparsers.add_parser("upload")
    download_parser = subparsers.add_parser("download")
    download_parser.add_argument("--model",
                                 type=str,
                                 default="DORA_r16(0207051004)_stage2")
    download_parser.add_argument("--epoch", type=int, default=100)
    download_parser.add_argument("--save_dir", type=str, default="sam2_logs")

    args = parser.parse_args()

    if args.action == "upload":
        upload()
    elif args.action == "download":
        download(args.model, args.epoch, args.save_dir)
