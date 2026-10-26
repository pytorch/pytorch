"""Install test dependencies in a cached CPU image layer."""

from __future__ import annotations

import importlib.metadata
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path


DEPENDENCIES = (
    "fsspec>=0.8.5",
    "onnxruntime==1.28.0",
    "transformers==4.36.2",
)


def install_dependencies(vision_commit: str) -> None:
    torch_version = importlib.metadata.version("torch")
    cache = Path(tempfile.gettempdir()) / "ramjet-dependencies-ccache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.update(
        CCACHE_DIR=str(cache),
        CCACHE_TEMPDIR=str(cache),
        MAX_JOBS="16",
        FORCE_CUDA="1",
        WITH_CUDA="1",
    )
    pip = [sys.executable, "-m", "pip", "--disable-pip-version-check"]
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="ramjet-dependencies-") as temporary:
        directory = Path(temporary)
        constraints = directory / "constraints.txt"
        constraints.write_text(f"torch==={torch_version}\n")
        subprocess.run(
            [*pip, "install", "--constraint", str(constraints), *DEPENDENCIES],
            check=True,
        )
        print(f"Ramjet building torchvision from {vision_commit} on CPU", flush=True)
        subprocess.run(
            [
                *pip,
                "wheel",
                "--no-build-isolation",
                "--no-deps",
                "--wheel-dir",
                str(directory),
                f"git+https://github.com/pytorch/vision.git@{vision_commit}",
            ],
            check=True,
        )
        wheels = list(directory.glob("torchvision-*.whl"))
        if len(wheels) != 1:
            raise RuntimeError(f"Expected one torchvision wheel, found {wheels}")
        subprocess.run(
            [*pip, "install", "--no-index", "--no-deps", str(wheels[0])],
            check=True,
        )
    if importlib.metadata.version("torch") != torch_version:
        raise RuntimeError("Installing test dependencies replaced the PyTorch wheel")
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import torch, torchvision; "
            "print(torchvision.ops.nms(torch.zeros((1, 4)), torch.ones(1), 0.5))",
        ],
        check=True,
    )
    packages = ("torch", "torchvision", "fsspec", "onnxruntime", "transformers")
    versions = ", ".join(
        f"{name}=={importlib.metadata.version(name)}" for name in packages
    )
    print(f"Ramjet test dependencies: {versions}", flush=True)
    print(
        f"Ramjet installed test dependencies in {time.monotonic() - started:.2f} seconds",
        flush=True,
    )


def prepare_dependencies(image, root: Path, snapshots: dict):
    commit = root.joinpath(".github/ci_commit_pins/vision.txt").read_text().strip()
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError(f"Invalid torchvision commit pin: {commit!r}")
    bootstrap, _ = snapshots["bootstrap"]
    module = "tools/ramjet_dependencies.py"
    return image.add_local_file(
        bootstrap / module, f"/opt/ramjet/{module}", copy=True
    ).run_function(
        install_dependencies,
        cpu=16,
        memory=65536,
        gpu=None,
        timeout=1800,
        include_source=False,
        kwargs={"vision_commit": commit},
    )
