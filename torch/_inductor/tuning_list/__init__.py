"""Caller-supplied Triton configs for Inductor's codegen kernels; see README.md."""

from .table import apply, InvalidTuningList, lookup


__all__ = ["InvalidTuningList", "apply", "lookup"]
