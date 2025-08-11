"""
File And Path Utility helpers for CLI tasks.
"""

import logging
import os
import shutil
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


def force_create_dir(path: str):
    """
    Ensures that the given directory path is freshly created.

    If the directory already exists, it will be removed along with all its contents.
    Then a new, empty directory will be created at the same path.
    """
    remove_dir(path)
    ensure_dir_exists(path)


def ensure_dir_exists(path: str):
    """
    Ensure the directory exists. Create it if it doesn't exist.
    """
    if not os.path.exists(path):
        logger.info("Creating directory '%s' ....", path)
        os.makedirs(path, exist_ok=True)
    else:
        logger.info("Directory already exists'%s' ", path)


def remove_dir(path: str):
    """
    Remove a directory if it exists.
    """
    if os.path.exists(path):
        logger.info("Removing directory '%s'...", path)
        shutil.rmtree(path)
    else:
        logger.info("skip remove operation, Directory not found: %s", path)


def get_abs_path(path: str):
    """
    Get the absolute path of the given path.
    """
    if not path:
        return ""
    return os.path.abspath(path)


def copy(src: Any, dst: Any, overwrite=True):
    """
    Copy a file or directory from src to dst.
    Creates parent directories for dst if needed.
    If src is a directory and dst exists:
        - overwrite=True will merge/overwrite
        - overwrite=False will raise FileExistsError
    """
    src_path = Path(src).resolve()
    dst_path = Path(dst).resolve()
    if not src_path.exists():
        raise FileNotFoundError(f"Source path does not exist: {src_path}")

    dst_path.parent.mkdir(parents=True, exist_ok=True)

    if src_path.is_file():
        logger.info("try to copy file %s to dst %s", src, dst)
        shutil.copy2(src_path, dst_path)

    elif src_path.is_dir():
        logger.info("try to copy folder %s to dst %s", src, dst)
        shutil.copytree(src_path, dst_path, dirs_exist_ok=overwrite)
    else:
        raise ValueError(f"Unsupported file type: {src_path}")


def get_existing_abs_path(path: str) -> str:
    """
    Get and validate the absolute path of the given path.
    Raises an exception if the path does not exist.
    """

    path = get_abs_path(path)
    if is_path_exist(path):
        raise FileNotFoundError(f"Path does not exist '{path}'")
    return path


def is_path_exist(path: str) -> bool:
    """
    Check if a path exists.
    """
    if not path:
        return False
    return os.path.exists(path)
