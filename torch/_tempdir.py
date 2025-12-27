"""
Centralized temp directory utilities for PyTorch.

Provides cross-platform temp directory resolution to support Windows,
restricted environments, and custom storage configurations where /tmp
is unavailable or undesirable.
"""
import os
import tempfile
from typing import Optional

# Reason: Separate env var from system TMPDIR allows PyTorch-specific
# override without affecting other applications in the same process
_PYTORCH_TMPDIR_ENV = "PYTORCH_TMPDIR"


def get_pytorch_tmpdir() -> str:
    """
    Get the base temp directory for PyTorch operations.

    The directory specified via the ``PYTORCH_TMPDIR`` environment variable
    must already exist, be a directory, and be writable by the current
    process. If it is not set, this falls back to ``tempfile.gettempdir()``.

    Returns:
        str: Path to temp directory.

    Raises:
        RuntimeError: If ``PYTORCH_TMPDIR`` is set to a path that does not
            exist, is not a directory, or is not writable/executable.
    """
    # Reason: Fallback to tempfile.gettempdir() ensures cross-platform
    # compatibility (Windows, Linux, macOS) without hardcoding paths
    env_value = os.environ.get(_PYTORCH_TMPDIR_ENV)
    if not env_value:
        return tempfile.gettempdir()

    # Reason: Validate custom temp directory upfront to provide clear error
    # messages instead of opaque downstream I/O failures
    if not os.path.exists(env_value):
        raise RuntimeError(
            f"{_PYTORCH_TMPDIR_ENV} is set to '{env_value}', "
            "but this path does not exist."
        )
    if not os.path.isdir(env_value):
        raise RuntimeError(
            f"{_PYTORCH_TMPDIR_ENV} is set to '{env_value}', "
            "but this path is not a directory."
        )
    The directory specified via the ``PYTORCH_TMPDIR`` environment variable
    must already exist, be a directory, and be writable by the current
    process. If it is not set, this falls back to ``tempfile.gettempdir()``.

    Returns:
        str: Path to temp directory.

    Raises:
        RuntimeError: If ``PYTORCH_TMPDIR`` is set to a path that does not
            exist, is not a directory, or is not writable/executable.
    """
    # Reason: Fallback to tempfile.gettempdir() ensures cross-platform
    # compatibility (Windows, Linux, macOS) without hardcoding paths
    env_value = os.environ.get(_PYTORCH_TMPDIR_ENV)
    if not env_value:
        return tempfile.gettempdir()

    # Validate that the custom temp directory exists, is a directory, and is
    # usable (writable and traversable) by the current process. This avoids
    # opaque downstream I/O errors when the path is misconfigured.
    if not os.path.exists(env_value):
        raise RuntimeError(
            f"{_PYTORCH_TMPDIR_ENV} is set to '{env_value}', but this path does not exist."
        )
    if not os.path.isdir(env_value):
        raise RuntimeError(
            f"{_PYTORCH_TMPDIR_ENV} is set to '{env_value}', but this path is not a directory."
        )
    if not os.access(env_value, os.W_OK | os.X_OK):
        raise RuntimeError(
            f"{_PYTORCH_TMPDIR_ENV} is set to '{env_value}', but this directory is not writable/executable."
        )

    return env_value
def get_temp_path(
    subdirectory: Optional[str] = None, filename: Optional[str] = None
) -> str:
    """
    Get a temp path with optional subdirectory and filename.

    Args:
        subdirectory: Optional subdirectory under temp root.
        filename: Optional filename to append.

    Returns:
        str: Full path to temp location. The directory component of the
            path (temp root plus optional subdirectory) is created if it
            does not already exist.
    """
    base = get_pytorch_tmpdir()
    if subdirectory:
        base = os.path.normpath(os.path.join(base, subdirectory))

    # Reason: Create directory automatically so callers can safely create
    # files at the returned path without additional os.makedirs() calls
    os.makedirs(base, exist_ok=True)

    if filename:
        return os.path.join(base, filename)
    return base
