"""Path utility helpers for CLI tasks."""

import logging
import shutil
from pathlib import Path
from typing import Union


logger = logging.getLogger(__name__)


def get_path(path: Union[str, Path], resolve: bool = False) -> Path:
    """Convert to Path object, optionally resolving to absolute path."""
    if not path:
        raise ValueError("Path cannot be None or empty")
    result = Path(path)
    return result.resolve() if resolve else result


def ensure_dir_exists(path: Union[str, Path]) -> Path:
    """Create directory if it doesn't exist."""
    path_obj = get_path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def remove_dir(path: Union[str, Path, None]) -> None:
    """Remove directory if it exists."""
    if not path:
        return
    path_obj = get_path(path)
    if path_obj.exists():
        shutil.rmtree(path_obj)


def force_create_dir(path: Union[str, Path]) -> Path:
    """Remove directory if exists, then create fresh empty directory."""
    remove_dir(path)
    return ensure_dir_exists(path)


def copy(src: Union[str, Path], dst: Union[str, Path]) -> None:
    """Copy file or directory from src to dst."""
    src_path = get_path(src, resolve=True)
    dst_path = get_path(dst, resolve=True)

    if not src_path.exists():
        raise FileNotFoundError(f"Source does not exist: {src_path}")

    dst_path.parent.mkdir(parents=True, exist_ok=True)

    if src_path.is_file():
        shutil.copy2(src_path, dst_path)
    elif src_path.is_dir():
        shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
    else:
        raise ValueError(f"Unsupported path type: {src_path}")


def is_path_exist(path: Union[str, Path, None]) -> bool:
    """Check if path exists."""
    return bool(path and get_path(path).exists())
