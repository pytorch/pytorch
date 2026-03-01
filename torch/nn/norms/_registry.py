# mypy: allow-untyped-defs
"""Registry for norm implementations.

Follows the implementation of FA4 registration for SDPA
"""

import logging
from collections.abc import Callable
from typing import Protocol


logger = logging.getLogger(__name__)


class NormHandle(Protocol):
    def remove(self) -> None: ...


_RegisterFn = Callable[..., NormHandle | None]

_NORM_IMPLS: dict[str, _RegisterFn] = {}

_NORM_ACTIVE: tuple[str, NormHandle] | None = None


def register_norm_impl(
    impl: str,
    *,
    register_fn: _RegisterFn,
) -> None:
    """
    Register the callable that activates a norm impl.

    .. note::
        This function is intended for norm backend providers to register their
        implementations. End users should use :func:`activate_norm_impl`
        to activate a registered implementation.

    Args:
        impl: Implementation identifier (e.g., ``"cutedsl_rmsnorm"``).
        register_fn: Callable that performs the actual dispatcher registration.
            This function will be invoked by :func:`activate_norm_impl`
            and should register custom kernels with the PyTorch dispatcher.
            It may optionally return a handle implementing
            :class:`NormHandle` to keep any necessary state alive.
    """
    global _NORM_IMPLS
    _NORM_IMPLS[impl] = register_fn


def activate_norm_impl(
    impl: str,
) -> None:
    """
    Activate into the dispatcher a previously registered norm impl.

    .. note::
        Backend providers should NOT automatically activate their implementation
        on import. Users should explicitly opt-in by calling this function.

    Args:
        impl: Implementation identifier to activate. See
            :func:`~torch.nn.norms.list_norm_impls` for available
            implementations.
    """
    global _NORM_ACTIVE, _NORM_IMPLS

    restore_norm_impl(_raise_warn=False)

    register_fn = _NORM_IMPLS.get(impl)
    if register_fn is None:
        raise ValueError(
            f"Unknown norm impl '{impl}'. "
            f"Available implementations: {list_norm_impls()}"
        )

    handle = register_fn()
    if handle is not None:
        _NORM_ACTIVE = (impl, handle)


def list_norm_impls() -> list[str]:
    """Return the names of all available norm implementations."""
    return sorted(_NORM_IMPLS.keys())


def current_norm_impl() -> str | None:
    """
    Return the currently activated norm impl name, if any.

    ``None`` indicates that no custom impl has been activated.
    """
    return _NORM_ACTIVE[0] if _NORM_ACTIVE is not None else _NORM_ACTIVE


def restore_norm_impl(_raise_warn: bool = True) -> None:
    """
    Restore the default norm implementation.
    """
    global _NORM_ACTIVE

    handle = None
    if _NORM_ACTIVE is not None:
        handle = _NORM_ACTIVE[1]

    if handle is not None:
        handle.remove()
    elif _raise_warn:
        logger.warning(
            "Trying to restore default norm impl when no custom impl was activated"
        )

    _NORM_ACTIVE = None
