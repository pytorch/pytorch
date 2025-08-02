import shlex
import shutil
import subprocess
import sys
from dataclasses import fields
from pathlib import Path
from textwrap import indent
from typing import Optional
import os
import yaml


def run_shell(
    cmd: str,
    logging: bool = True,
    cwd: Optional[str] = None,
    env: Optional[dict] = None,
):
    if logging:
        print(f"[shell]{cmd}", flush=True)
    subprocess.run(
        cmd,
        shell=True,
        executable="/bin/bash",
        stdout=sys.stdout,
        stderr=sys.stderr,
        check=True,
        env=env,
        cwd=cwd,
    )


# eliainwy
def get_post_build_pinned_commit(name: str, prefix=".github/ci_commit_pins") -> str:
    path = Path(prefix) / f"{name}.txt"
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
    if not os.path.exists(path):
        print(f"[INFO] Creating directory: {path}", flush=True)
        os.makedirs(path, exist_ok=True)
    else:
        print(f"[INFO] Directory already exists: {path}", flush=True)


def remove_dir(path: str):
    """
    Remove a directory if it exists.
    """
    if os.path.exists(path):
        print(f"[INFO] Removing directory: {path}", flush=True)
        shutil.rmtree(path)
    else:
        print(f"[INFO] Directory not found (skipped): {path}", flush=True)


def get_abs_path(path: str):
    return os.path.abspath(path)


def generate_dataclass_help(cls) -> str:
    """Auto-generate help text for dataclass default values."""
    lines = []
    for field in fields(cls):
        default = field.default
        if default is not None and default != "":
            lines.append(f"{field.name:<22} = {repr(default)}")
        else:
            lines.append(f"{field.name:<22} = ''")
    return indent("\n".join(lines), "    ")


def get_existing_abs_path(path: str) -> str:
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path does not exist: {path}")
    return path


def clone_vllm(commit: str):
    """
    cloning vllm and checkout pinned commit
    """
    print(f"clonening vllm....", flush=True)
    cwd = "vllm"
    # delete the directory if it exists
    remove_dir(cwd)
    # Clone the repo & checkout commit
    run_shell("git clone https://github.com/vllm-project/vllm.git")
    run_shell(f"git checkout {commit}", cwd=cwd)
    run_shell("git submodule update --init --recursive", cwd=cwd)


def read_yaml_file(file_path: str) -> dict:
    p = get_abs_path(file_path)
    if not os.path.exists(p):
        raise FileNotFoundError(f"YAML file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def pip_install(package: str):
    cmd = f"python3 -m pip install {package}"
    subprocess.run(shlex.split(cmd), check=True)


def uv_pip_install(package: str):
    cmd = f"python3 -m  uv pip install --system {package}"
    subprocess.run(shlex.split(cmd), check=True)
