import os
import subprocess
from pathlib import Path


repo_root = Path(__file__).absolute().parent.parent
third_party_path = os.path.join(repo_root, "third_party")


def read_nccl_pin() -> str:
    nccl_file = "nccl-cu12.txt"
    if os.getenv("DESIRED_CUDA", "").startswith("11") or os.getenv(
        "CUDA_VERSION", ""
    ).startswith("11"):
        nccl_file = "nccl-cu11.txt"
    nccl_pin_path = os.path.join(
        repo_root, ".ci", "docker", "ci_commit_pins", nccl_file
    )
    with open(nccl_pin_path) as f:
        return f.read().strip()


def checkout_nccl() -> None:
    release_tag = read_nccl_pin()
    print(f"-- Checkout nccl release tag: {release_tag}")
    nccl_basedir = os.path.join(third_party_path, "nccl")
    if not os.path.exists(nccl_basedir):
        subprocess.check_call(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                release_tag,
                "https://github.com/NVIDIA/nccl.git",
                "nccl",
            ],
            cwd=third_party_path,
        )


if __name__ == "__main__":
    checkout_nccl()
