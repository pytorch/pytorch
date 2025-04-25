import contextlib
import threading
from collections.abc import Generator
from typing import Any

import torch


_TLS = threading.local()


def _freezing_active() -> bool:
    return getattr(_TLS, "freezing_active", False)


@contextlib.contextmanager
def enter_freezing() -> Generator[Any, None, None]:
    """
    Context manager to designate when freezing is active.
    """
    prev = _freezing_active()
    _TLS.freezing_active = True
    try:
        yield
    finally:
        _TLS.freezing_active = prev


def record_has_frozen_params(gm: torch.fx.GraphModule) -> None:
    """
    Mark the gm as having frozen params.
    """
    gm._has_frozen_params = True  # type: ignore[assignment]


def has_frozen_params(gm: torch.fx.GraphModule) -> bool:
    """
    Return True if the gm has frozen parameters.
    """
    return getattr(gm, "_has_frozen_params", False)


def maybe_set_is_frozen_param(t: torch.Tensor) -> None:
    """
    Mark the provided tensor as a frozen param if freezing is active.
    """
    if _freezing_active():
        t._is_frozen_param = True  # type: ignore[attr-defined]


def is_frozen_param(t: torch.Tensor) -> bool:
    """
    Return True if the tensor is a frozen param.
    """
    return getattr(t, "_is_frozen_param", False)
