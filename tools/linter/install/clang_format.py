import os
from tools.linter.install.download_bin import download, PYTORCH_ROOT, HASH_PATH

PLATFORM_TO_URL = {
    "Darwin": "https://oss-clang-format.s3.us-east-2.amazonaws.com/mac/clang-format-mojave",
    "Linux": "https://oss-clang-format.s3.us-east-2.amazonaws.com/linux64/clang-format-linux64",
}

PLATFORM_TO_HASH = {
    "Darwin": os.path.join(HASH_PATH, "clang-format-mojave"),
    "Linux": os.path.join(HASH_PATH, "clang-format-linux64"),
}

OUTPUT_DIR = os.path.join(PYTORCH_ROOT, ".clang-format-bin")
INSTALLATION_PATH = os.path.join(OUTPUT_DIR, "clang-format")

if __name__ == "__main__":
    ok = download("clang-format", OUTPUT_DIR, PLATFORM_TO_URL, PLATFORM_TO_HASH)
    if not ok:
        print("Installation failed!")
        exit(1)
