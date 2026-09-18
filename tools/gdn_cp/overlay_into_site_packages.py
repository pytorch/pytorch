"""Copy GDN-CP Python files from this pytorch tree into an installed torch.

Used on the GPU host after `pip install torch==2.13.0` in an isolated venv.
Does not touch train/.venv.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


REL_PATHS = [
    "torch/nn/attention/gated_delta.py",
    "torch/nn/attention/__init__.py",
    "torch/nn/modules/gated_delta.py",
    "torch/nn/modules/__init__.py",
    "torch/nn/functional.py",
    "torch/distributed/tensor/experimental/_context_parallel/_gated_delta.py",
    "torch/distributed/tensor/experimental/_context_parallel/_attention.py",
    "torch/distributed/tensor/experimental/_context_parallel/__init__.py",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="pytorch source root (this clone)",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=None,
        help="installed torch package dir (default: import torch)",
    )
    args = parser.parse_args()
    src_root = args.src.resolve()
    if args.dst is None:
        import torch

        dst_root = Path(torch.__file__).resolve().parent
    else:
        dst_root = args.dst.resolve()
        if dst_root.name == "torch":
            pass
        else:
            dst_root = dst_root / "torch"

    print(f"overlay {src_root} -> {dst_root}", flush=True)
    for rel in REL_PATHS:
        src = src_root / rel
        # rel is torch/... so strip the torch/ prefix when dst is the package dir
        dst = dst_root / Path(*Path(rel).parts[1:])
        if not src.is_file():
            raise SystemExit(f"missing source file: {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"  {rel} -> {dst}", flush=True)
    print("overlay done", flush=True)


if __name__ == "__main__":
    sys.exit(main() or 0)
