#!/usr/bin/env python3
from subprocess import check_call
from pathlib import Path
from tempfile import TemporaryDirectory
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


def build_triton(commit_hash: str) -> Path:
    with TemporaryDirectory() as tmpdir:
        triton_basedir = Path(tmpdir) / "triton"
        triton_pythondir = triton_basedir / "python"
        check_call(["git", "clone", "https://github.com/openai/triton"], cwd=tmpdir)
        check_call(["git", "checkout", commit_hash], cwd=triton_basedir)
        patch_setup_py(triton_pythondir / "setup.py", name="torchtriton", version=f"2.0.0+{commit_hash[:10]}")
        check_call([sys.executable, "setup.py", "bdist_wheel"], cwd=triton_pythondir)
        whl_path = list((triton_pythondir / "dist").glob("*.whl"))[0]
        shutil.copy(whl_path, Path.cwd())
        return Path.cwd() / whl_path.name


def main() -> None:
    pin = read_triton_pin()
    build_triton(pin)


if __name__ == "__main__":
    main()
