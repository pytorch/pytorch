#!/usr/bin/env python3
from subprocess import check_call
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional
import sys
import shutil
SCRIPT_DIR = Path(__file__).parent

def read_triton_pin() -> str:
    with open(SCRIPT_DIR.parent / "ci_commit_pins" / "triton.txt") as f:
        return f.read().strip()


def check_and_replace(inp: str, src: str, dst: str) -> str:
    """ Checks that `src` can be found in `input` and replaces it with `dst` """
    if src not in inp:
        raise RuntimeError(f"Can't find ${src} in the input")
    return inp.replace(src, dst)


def patch_setup_py(path: Path, *, version: str = "2.0.0", name: str = "triton") -> None:
    with open(path) as f:
        orig = f.read()
    # Replace name
    orig = check_and_replace(orig, "name=\"triton\",", f"name=\"{name}\",")
    # Replace version
    orig = check_and_replace(orig, "version=\"2.0.0\",", f"version=\"{version}\",")
    with open(path, "w") as f:
        f.write(orig)


def build_triton(commit_hash: str, build_conda: bool = False, py_version : Optional[str] = None) -> Path:
    with TemporaryDirectory() as tmpdir:
        triton_basedir = Path(tmpdir) / "triton"
        triton_pythondir = triton_basedir / "python"
        check_call(["git", "clone", "https://github.com/openai/triton"], cwd=tmpdir)
        check_call(["git", "checkout", commit_hash], cwd=triton_basedir)
        if build_conda:
            with open(triton_basedir / "meta.yaml", "w") as meta:
                print("package:\n  name: torchtriton\n  version: 2.0.0\n", file=meta)
                print("source:\n  path: .\n", file=meta)
                print("build:\n  string: py{{py}}\n  number: 1\n  script: cd python; "
                      "python setup.py install --single-version-externally-managed --record=record.txt\n", file=meta)
                print("requirements:\n  host:\n    - python\n    - setuptools\n  run:\n    - python\n"
                      "    - filelock\n    - pytorch\n", file=meta)
                print("about:\n  home: https://github.com/openai/triton\n  license: MIT\n  summary:"
                      " 'A language and compiler for custom Deep Learning operation'", file=meta)

            if py_version is None:
                py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
            check_call(["conda", "build", "--python", py_version,
                        "-c", "pytorch-nightly", "--output-folder", tmpdir, "."], cwd=triton_basedir)
            conda_path = list(Path(tmpdir).glob("linux-64/torchtriton*.bz2"))[0]
            shutil.copy(conda_path, Path.cwd())
            return Path.cwd() / conda_path.name

        patch_setup_py(triton_pythondir / "setup.py", name="pytorch-triton", version=f"2.0.0+{commit_hash[:10]}")
        check_call([sys.executable, "setup.py", "bdist_wheel"], cwd=triton_pythondir)
        whl_path = list((triton_pythondir / "dist").glob("*.whl"))[0]
        shutil.copy(whl_path, Path.cwd())
        return Path.cwd() / whl_path.name


def main() -> None:
    from argparse import ArgumentParser
    parser = ArgumentParser("Build Triton binaries")
    parser.add_argument("--build-conda", action="store_true")
    parser.add_argument("--py-version", type=str)
    args = parser.parse_args()
    pin = read_triton_pin()
    build_triton(pin, build_conda=args.build_conda, py_version=args.py_version)


if __name__ == "__main__":
    main()
