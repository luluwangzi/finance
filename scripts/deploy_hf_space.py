import os
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_folder


def main() -> int:
    token = os.environ.get("HF_TOKEN")
    username = os.environ.get("HF_USERNAME")
    space_name = os.environ.get("SPACE_NAME", "mdd-csp-app")

    if not token or not username:
        print("Missing env vars. Please export HF_TOKEN and HF_USERNAME.", file=sys.stderr)
        return 2

    repo_id = f"{username}/{space_name}"
    api = HfApi(token=token)

    print(f"Ensuring Space exists: {repo_id} (SDK=streamlit)")
    create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="streamlit",
        exist_ok=True,
        token=token,
        private=False,
    )

    root = Path("/workspace")
    include = [
        "app.py",
        "requirements.txt",
        "README.md",
        ".streamlit",
    ]

    # Upload selected files/folders
    print("Uploading project files to the Space…")
    for item in include:
        src = root / item
        if not src.exists():
            print(f"Skip missing: {src}")
            continue
        upload_folder(
            repo_id=repo_id,
            repo_type="space",
            folder_path=str(src),
            path_in_repo=item,
            token=token,
            commit_message=f"Deploy {item}",
        )

    url = f"https://huggingface.co/spaces/{repo_id}"
    print(f"Space pushed. Visit: {url}")

    # Optional: brief poll to surface build status
    print("Waiting for Space to start (up to ~90s)…")
    for _ in range(18):
        try:
            info = api.space_info(repo_id)
            stage = getattr(info, "runtime", None)
            status = getattr(stage, "stage", None) if stage else None
            hardware = getattr(stage, "hardware", None) if stage else None
            print(f"Status: {status} / Hardware: {hardware}")
            if status == "RUNNING":
                print("Space is running!")
                break
        except Exception as e:
            print(f"status check error: {e}")
        time.sleep(1)

    print(f"If the UI isn't live yet, it will be shortly: {url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

