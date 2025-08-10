"""
File And Path Utility helpers for CLI tasks.
"""

import logging
import os


logger = logging.getLogger(__name__)


def get_abs_path(path: str):
    return os.path.abspath(path)


def get_existing_abs_path(path: str) -> str:
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path does not exist: {path}")
    return path
