"""
File And Path Utility helpers for CLI tasks.
"""

import logging
from pathlib import Path


logger = logging.getLogger(__name__)


def get_abs_path(path: str) -> Path:
    return Path(path).resolve()


def get_existing_abs_path(path: str) -> str:
    p = get_abs_path(path)
    if not p.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    return path
