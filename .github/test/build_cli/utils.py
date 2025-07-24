import subprocess
import os
import shutil
from pathlib import Path
from typing import Optional
import shlex
import time


def run(
    cmd: str,
    cwd: Optional[str] = None,
    env: Optional[dict] = None,
    logging: bool = False,
):
    if logging:
        print(f">>> {cmd}")
    subprocess.run(shlex.split(cmd), check=True, cwd=cwd, env=env)


def get_post_build_pinned_commit(name: str) -> str:
    path = Path(".github/ci_commit_pins") / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Pin file not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def get_env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def force_create_dir(path: str):
    """
    Forcefully create a fresh directory.
    If the directory exists, it will be removed first.
    """
    remove_dir(path)
    ensure_dir_exists(path)

def ensure_dir_exists(path: str):
    """
    Ensure the directory exists. Create it if necessary.
    """
    abs_path = get_abs_path(path)
    if not os.path.exists(abs_path):
        print(f"[INFO] Creating directory: {abs_path}")
        os.makedirs(abs_path, exist_ok=True)
    else:
        print(f"[INFO] Directory already exists: {abs_path}")

def remove_dir(path: str):
    """
    Remove a directory if it exists.
    """
    abs_path = get_abs_path(path)
    if os.path.exists(abs_path):
        print(f"[INFO] Removing directory: {abs_path}")
        shutil.rmtree(abs_path)
    else:
        print(f"[INFO] Directory not found (skipped): {abs_path}")

def get_abs_path(path:str):
    return os.path.abspath(path)

class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        print(f"Took {self.end - self.start:.3f} seconds")

def get_existing_abs_path(path: str) -> str:
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"Path does not exist: {abs_path}")
    return abs_path
