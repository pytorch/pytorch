import os
from tools.linter.install.download_bin import download, PYTORCH_ROOT, HASH_PATH

PLATFORM_TO_URL = {
    "Linux": "https://oss-clang-format.s3.us-east-2.amazonaws.com/linux64/clang-tidy",
    "Darwin": "https://oss-clang-format.s3.us-east-2.amazonaws.com/macos/clang-tidy",
}

PLATFORM_TO_HASH = {
    "Linux": os.path.join(HASH_PATH, "clang-tidy-linux64"),
    "Darwin": os.path.join(HASH_PATH, "clang-tidy-macos"),
}

OUTPUT_DIR = os.path.join(PYTORCH_ROOT, ".clang-tidy-bin")
INSTALLATION_PATH = os.path.join(OUTPUT_DIR, "clang-tidy")

if __name__ == "__main__":
    ok = download("clang-tidy", OUTPUT_DIR, PLATFORM_TO_URL, PLATFORM_TO_HASH)
    if not ok:
        print("Installation failed!")
        exit(1)
