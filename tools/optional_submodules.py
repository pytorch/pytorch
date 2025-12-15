from pathlib import Path
from subprocess import check_call


repo_root = Path(__file__).absolute().parent.parent
third_party_path = repo_root / "third_party"


def _read_file(path: Path) -> str:
    with path.open(encoding="utf-8") as f:
        return f.read().strip()


def _checkout_by_tag(repo: str, tag: str) -> None:
    check_call(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            tag,
            repo,
        ],
        cwd=third_party_path,
    )


def read_nccl_pin() -> str:
    # Default NCCL version
    nccl_file = "nccl.txt"

    # If NCCL version diverges for different CUDA versions, uncomment the
    # following block and add the appropriate file (using CUDA 11 as an example)

    # cuda_version = os.getenv("DESIRED_CUDA", os.getenv("CUDA_VERSION", ""))
    # if cuda_version.startswith("11"):
    #     nccl_file = "nccl-cu11.txt"

    nccl_pin_path = repo_root / ".ci" / "docker" / "ci_commit_pins" / nccl_file
    return _read_file(nccl_pin_path)


def checkout_nccl() -> None:
    release_tag = read_nccl_pin()
    print(f"-- Checkout nccl release tag: {release_tag}")
    nccl_basedir = third_party_path / "nccl"
    if not nccl_basedir.exists():
        _checkout_by_tag("https://github.com/NVIDIA/nccl", release_tag)


def checkout_eigen() -> None:
    eigen_tag = _read_file(third_party_path / "eigen_pin.txt")
    print(f"-- Checkout Eigen release tag: {eigen_tag}")
    eigen_basedir = third_party_path / "eigen"
    if not eigen_basedir.exists():
        _checkout_by_tag("https://gitlab.com/libeigen/eigen", eigen_tag)


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        # If no arguments are given checkout all optional dependency
        checkout_nccl()
        checkout_eigen()
    else:
        # Otherwise just call top-level function of choice
        globals()[sys.argv[1]]()
