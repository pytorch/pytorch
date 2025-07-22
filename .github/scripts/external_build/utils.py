
import subprocess
import os

from contextlib import contextmanager
from pathlib import Path
import subprocess
from typing import Optional
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
