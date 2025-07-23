
import subprocess
import os
import shutil
from contextlib import contextmanager
from pathlib import Path
import subprocess
from typing import Optional
from time import perf_counter
import shlex

def run(cmd: str, cwd: Optional[str] = None, env: Optional[dict] = None, logging: bool=False):
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


def create_directory(foldername: str):
    shared_wheels_path = os.path.abspath(foldername)

    delete_directory(shared_wheels_path)

    print(f"[INFO] Creating fresh directory: {shared_wheels_path}")
    os.makedirs(shared_wheels_path, exist_ok=True)

def delete_directory(name:str):
    f = os.path.abspath(name)
    if os.path.exists(f):
        print(f"[INFO] Removing existing directory: {f}")
        shutil.rmtree(f)
    else:
        print(f"[INFO] folder {name} does not exists in {f}, skipping")


class Timer:
    def __enter__(self):
        self.start = perf_counter()
        return self

    def __exit__(self, *args):
        self.end = perf_counter()
        self.interval = self.end - self.start
        print(f"⏱️ Took {self.interval:.3f} seconds")
