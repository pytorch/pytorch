"""
File And Path Utility helpers for CLI tasks.
"""

import logging
import shutil
from pathlib import Path
from typing import Any, Optional, Union


logger = logging.getLogger(__name__)


def get_path(path: Optional[Union[str, Path]], full_path: bool = False) -> Path:
    """
    Get a Path object from a string or Path object.
    If full_path is True, returns the absolute path with currently working directory.
    """
    if path is None:
        raise ValueError("Path cannot be None")
    if isinstance(path, str):
        p = Path(path)
    elif isinstance(path, Path):
        p = path
    else:
        raise TypeError(f"Invalid path type: {type(path)}")

    if full_path:
        return p.resolve()
    return p


def force_create_dir(path: Path):
    """
    Ensures that the given directory path is freshly created.

    If the directory already exists, it will be removed along with all its contents.
    Then a new, empty directory will be created at the same path.
    """
    remove_dir(path)
    ensure_dir_exists(path)


def ensure_dir_exists(path: Optional[Path]):
    """
    Ensure the directory exists. Create it if it doesn't exist.
    if path is None, throw an exception.
    """
    if not path:
        raise ValueError("Path cannot be None or empty")
    p = get_path(path)
    if not p.exists():
        logger.info("creating directory '%s' ....", path)
        p.mkdir(parents=True)
    else:
        logger.info("Directory already exists'%s' ", path)


def remove_dir(path: Optional[Union[Path, str]]):
    """
    Remove a directory if it exists.
    """
    if not path:
        logger.info("skip remove operation, the path is empty or None: %s", path)
        return
    path = Path(path)
    if path.exists():
        logger.info("Removing directory '%s'...", path)
        shutil.rmtree(path)
    else:
        logger.info("skip remove operation, Directory not found: %s", path)


def copy(src: Any, dst: Any, overwrite=True, full_path=True):
    """
    Copy a file or directory from src to dst.
    Creates parent directories for dst if needed.
    If src is a directory and dst exists:
        - overwrite=True will merge/overwrite
        - overwrite=False will raise FileExistsError
    """
    src_path = get_path(src, full_path=full_path)
    dst_path = get_path(dst, full_path=full_path)

    if not src_path.exists():
        raise FileNotFoundError(f"Source path does not exist: {src}")

    dst_path.parent.mkdir(parents=True, exist_ok=True)

    if src_path.is_file():
        logger.info("try to copy file %s to dst %s", src_path, dst_path)
        shutil.copy2(src_path, dst_path)
    elif src_path.is_dir():
        logger.info("try to copy folder %s to dst %s", src_path, dst_path)
        shutil.copytree(src_path, dst_path, dirs_exist_ok=overwrite)
    else:
        raise TypeError(f"Unsupported source type: {type(src_path)}")
    logger.info("Done. Copied file %s to dst %s", src_path, dst_path)


def is_path_exist(path: Union[str, Path]) -> bool:
    """
    Check if a path exists.
    """
    if not path:
        return False
    path = get_path(path)
    return path.exists()
