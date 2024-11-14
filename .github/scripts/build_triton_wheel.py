#!/usr/bin/env python3

import os
import shutil
import sys
from pathlib import Path
from subprocess import check_call
from tempfile import TemporaryDirectory
from typing import Optional


SCRIPT_DIR = Path(__file__).parent
REPO_DIR = SCRIPT_DIR.parent.parent


def read_triton_pin(device: str = "cuda") -> str:
    triton_file = "triton.txt"
    if device == "xpu":
        triton_file = "triton-xpu.txt"
    with open(REPO_DIR / ".ci" / "docker" / "ci_commit_pins" / triton_file) as f:
        return f.read().strip()


def read_triton_version() -> str:
    with open(REPO_DIR / ".ci" / "docker" / "triton_version.txt") as f:
        return f.read().strip()


def check_and_replace(inp: str, src: str, dst: str) -> str:
    """Checks that `src` can be found in `input` and replaces it with `dst`"""
    if src not in inp:
        raise RuntimeError(f"Can't find ${src} in the input")
    return inp.replace(src, dst)


def patch_init_py(
    path: Path, *, version: str, expected_version: Optional[str] = None
) -> None:
    if not expected_version:
        expected_version = read_triton_version()
    with open(path) as f:
        orig = f.read()
    # Replace version
    orig = check_and_replace(
        orig, f"__version__ = '{expected_version}'", f'__version__ = "{version}"'
    )
    with open(path, "w") as f:
        f.write(orig)


def build_triton(
    *,
    version: str,
    commit_hash: str,
    build_conda: bool = False,
    device: str = "cuda",
    py_version: Optional[str] = None,
    release: bool = False,
) -> Path:
    env = os.environ.copy()
    if "MAX_JOBS" not in env:
        max_jobs = os.cpu_count() or 1
        env["MAX_JOBS"] = str(max_jobs)

    with TemporaryDirectory() as tmpdir:
        triton_basedir = Path(tmpdir) / "triton"
        triton_pythondir = triton_basedir / "python"
        triton_repo = "https://github.com/openai/triton"
        if device == "rocm":
            triton_pkg_name = "pytorch-triton-rocm"
        elif device == "xpu":
            triton_pkg_name = "pytorch-triton-xpu"
            triton_repo = "https://github.com/intel/intel-xpu-backend-for-triton"
        else:
            triton_pkg_name = "pytorch-triton"
        check_call(["git", "clone", triton_repo, "triton"], cwd=tmpdir)
        if release:
            ver, rev, patch = version.split(".")
            check_call(
                ["git", "checkout", f"release/{ver}.{rev}.x"], cwd=triton_basedir
            )
        else:
            check_call(["git", "checkout", commit_hash], cwd=triton_basedir)

        if build_conda:
            with open(triton_basedir / "meta.yaml", "w") as meta:
                print(
                    f"package:\n  name: torchtriton\n  version: {version}\n",
                    file=meta,
                )
                print("source:\n  path: .\n", file=meta)
                print(
                    "build:\n  string: py{{py}}\n  number: 1\n  script: cd python; "
                    "python setup.py install --record=record.txt\n",
                    " script_env:\n   - MAX_JOBS\n",
                    file=meta,
                )
                print(
                    "requirements:\n  host:\n    - python\n    - setuptools\n  run:\n    - python\n"
                    "    - filelock\n    - pytorch\n",
                    file=meta,
                )
                print(
                    "about:\n  home: https://github.com/openai/triton\n  license: MIT\n  summary:"
                    " 'A language and compiler for custom Deep Learning operation'",
                    file=meta,
                )

            patch_init_py(
                triton_pythondir / "triton" / "__init__.py",
                version=f"{version}",
            )
            if py_version is None:
                py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
            check_call(
                [
                    "conda",
                    "build",
                    "--python",
                    py_version,
                    "-c",
                    "pytorch-nightly",
                    "--output-folder",
                    tmpdir,
                    ".",
                ],
                cwd=triton_basedir,
                env=env,
            )
            conda_path = next(iter(Path(tmpdir).glob("linux-64/torchtriton*.bz2")))
            shutil.copy(conda_path, Path.cwd())
            return Path.cwd() / conda_path.name

        # change built wheel name and version
        env["TRITON_WHEEL_NAME"] = triton_pkg_name
        env["TRITON_BUILD_WITH_CLANG_LLD"] = "1"
        patch_init_py(
            triton_pythondir / "triton" / "__init__.py",
            version=f"{version}",
            expected_version=None,
        )

        if device == "rocm":
            check_call(
                [f"{SCRIPT_DIR}/amd/package_triton_wheel.sh"],
                cwd=triton_basedir,
                shell=True,
            )
            print("ROCm libraries setup for triton installation...")

        check_call(
            [sys.executable, "setup.py", "bdist_wheel"], cwd=triton_pythondir, env=env
        )

        whl_path = next(iter((triton_pythondir / "dist").glob("*.whl")))
        shutil.copy(whl_path, Path.cwd())

        if device == "rocm":
            check_call(
                [f"{SCRIPT_DIR}/amd/patch_triton_wheel.sh", Path.cwd()],
                cwd=triton_basedir,
            )

        return Path.cwd() / whl_path.name


def main() -> None:
    from argparse import ArgumentParser

    parser = ArgumentParser("Build Triton binaries")
    parser.add_argument("--release", action="store_true")
    parser.add_argument("--build-conda", action="store_true")
    parser.add_argument(
        "--device", type=str, default="cuda", choices=["cuda", "rocm", "xpu"]
    )
    parser.add_argument("--py-version", type=str)
    parser.add_argument("--commit-hash", type=str)
    parser.add_argument("--triton-version", type=str, default=read_triton_version())
    args = parser.parse_args()

    build_triton(
        device=args.device,
        commit_hash=args.commit_hash
        if args.commit_hash
        else read_triton_pin(args.device),
        version=args.triton_version,
        build_conda=args.build_conda,
        py_version=args.py_version,
        release=args.release,
    )


if __name__ == "__main__":
    main()
