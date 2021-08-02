import os
from install.download_bin import download, PYTORCH_ROOT  # type: ignore[import]

# This dictionary maps each platform to the S3 object URL for its clang-format binary.
PLATFORM_TO_CF_URL = {
    "Darwin": "https://oss-clang-format.s3.us-east-2.amazonaws.com/mac/clang-format-mojave",
    "Linux": "https://oss-clang-format.s3.us-east-2.amazonaws.com/linux64/clang-format-linux64",
}

# This dictionary maps each platform to a relative path to a file containing its reference hash.
PLATFORM_TO_HASH = {
    "Darwin": os.path.join("tools", "clang_format_hash", "mac", "clang-format-mojave"),
    "Linux": os.path.join("tools", "clang_format_hash", "linux64", "clang-format-linux64"),
}

CLANG_FORMAT_DIR = os.path.join(PYTORCH_ROOT, ".clang-format-bin")
CLANG_FORMAT_PATH = os.path.join(CLANG_FORMAT_DIR, "clang-format")

def get_and_check_clang_format(verbose: bool = False) -> bool:
    return bool(download("clang-format", CLANG_FORMAT_DIR, PLATFORM_TO_CF_URL, PLATFORM_TO_HASH))
