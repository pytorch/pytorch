# Copyright (c) Meta Platforms, Inc. and affiliates.
# patternlint-disable fbcode-nonempty-init-py

import os

from packaging import version


# Minimum PyTorch version required for torch.compile support
# .0.dev0 enables nightly builds and prod builds since "1.dev0" < "1"
MIN_PYTORCH_VERSION_FOR_COMPILE = "2.12.0.dev0"


def is_torch_compile_supported(
    min_version: str = MIN_PYTORCH_VERSION_FOR_COMPILE,
    _current_version: str | None = None,
) -> bool:
    if _current_version is None:
        import torch

        _current_version = torch.__version__
    return os.environ.get(
        "TORCHCOMMS_COMPILE_IGNORE_PYTORCH_VERSION_REQUIREMENT"
    ) == "1" or version.parse(_current_version) >= version.parse(min_version)


def is_torch_compile_supported_and_enabled(
    min_version: str = MIN_PYTORCH_VERSION_FOR_COMPILE,
    _current_version: str | None = None,
) -> bool:
    return (
        is_torch_compile_supported(min_version, _current_version)
        and os.environ.get("TORCHCOMMS_PATCH_FOR_COMPILE") == "1"
    )


__all__ = [
    "is_torch_compile_supported",
    "is_torch_compile_supported_and_enabled",
]
