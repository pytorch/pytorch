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


def patch_setup_py(path: Path, *, version: str = "2.0.0", name: str = "triton") -> None:
    with open(path) as f:
        orig = f.read()
    # Replace name
    orig = orig.replace("name=\"triton\",", f"name=\"{name}\",")
    # Replace version
    orig = orig.replace("version=\"2.0.0\",", f"version=\"{version}\",")
    with open(path, "w") as f:
        f.write(orig)


def build_triton(commit_hash: str) -> Path:
    with TemporaryDirectory() as tmpdir:
        triton_basedir = Path(tmpdir) / "triton"
        triton_pythondir = triton_basedir / "python"
        check_call(["git", "clone", "https://github.com/openai/triton"], cwd=tmpdir)
        check_call(["git", "checkout", commit_hash], cwd=triton_basedir)
        patch_setup_py(triton_pythondir / "setup.py", name="pytorch-triton", version=f"2.0.0+{commit_hash[:10]}")
        check_call([sys.executable, "setup.py", "bdist_wheel"], cwd=triton_pythondir)
        whl_path = list((triton_pythondir / "dist").glob("*.whl"))[0]
        shutil.copy(whl_path, Path.cwd())
        return Path.cwd() / whl_path.name


def main() -> None:
    pin = read_triton_pin()
    path = build_triton(pin)
    print(path)


if __name__ == "__main__":
    main()

