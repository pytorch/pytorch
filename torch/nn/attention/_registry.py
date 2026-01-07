# mypy: allow-untyped-defs
"""Registry for flash attention implementations.

This module contains the registration system for flash attention implementations.
It has no torch dependencies to avoid circular imports during initialization.
"""

import logging
from collections.abc import Callable
from typing import Literal, Protocol


logger = logging.getLogger(__name__)


class FlashAttentionHandle(Protocol):
    def remove(self) -> None: ...


_RegisterFn = Callable[..., FlashAttentionHandle | None]
_FlashAttentionImpl = Literal["FA4"]

_FLASH_ATTENTION_IMPLS: dict[str, _RegisterFn] = {}

_FLASH_ATTENTION_ACTIVE: tuple[str, FlashAttentionHandle] | None = None


def register_flash_attention_impl(
    impl: str | _FlashAttentionImpl,
    *,
    register_fn: _RegisterFn,
) -> None:
    """
    Register the callable that activates a flash attention impl.

    .. note::
        This function is intended for SDPA backend providers to register their
        implementations. End users should use :func:`activate_flash_attention_impl`
        to activate a registered implementation.

    Args:
        impl: Implementation identifier (e.g., ``"FA4"``).
        register_fn: Callable that performs the actual dispatcher registration.
            This function will be invoked by :func:`activate_flash_attention_impl`
            and should register custom kernels with the PyTorch dispatcher.
            It may optionally return a handle implementing
            :class:`FlashAttentionHandle` to keep any necessary state alive.

    Example:
        >>> def my_impl_register(module_path: str = "my_flash_impl"):
        ...     # Register custom kernels with torch dispatcher
        ...     pass  # doctest: +SKIP
        >>> register_flash_attention_impl(
        ...     "MyImpl", register_fn=my_impl_register
        ... )  # doctest: +SKIP
    """
    global _FLASH_ATTENTION_IMPLS
    _FLASH_ATTENTION_IMPLS[impl] = register_fn


def activate_flash_attention_impl(
    impl: str | _FlashAttentionImpl,
) -> None:
    """
    Activate into the dispatcher a previously registered flash attention impl.

    .. note::
        Backend providers should NOT automatically activate their implementation
        on import. Users should explicitly opt-in by calling this function or via
        environment variables to ensure multiple provider libraries can coexist.

    Args:
        impl: Implementation identifier to activate. See
            :func:`~torch.nn.attention.list_flash_attention_impls` for available
            implementations.
            If the backend's :func:`register_flash_attention_impl` callable
            returns a :class:`FlashAttentionHandle`, the registry keeps that
            handle alive for the lifetime of the process (until explicit
            uninstall support exists).

    Example:
        >>> activate_flash_attention_impl("FA4")  # doctest: +SKIP
    """
    global _FLASH_ATTENTION_ACTIVE, _FLASH_ATTENTION_IMPLS

    restore_flash_attention_impl(
        _raise_warn=False
    )  # first restore any prev overrides (if any) to default

    register_fn = _FLASH_ATTENTION_IMPLS.get(impl)
    if register_fn is None:
        raise ValueError(
            f"Unknown flash attention impl '{impl}'. "
            f"Available implementations: {list_flash_attention_impls()}"
        )

    handle = register_fn()
    if handle is not None:
        _FLASH_ATTENTION_ACTIVE = (impl, handle)


def list_flash_attention_impls() -> list[str]:
    """Return the names of all available flash attention implementations."""
    return sorted(_FLASH_ATTENTION_IMPLS.keys())


def current_flash_attention_impl() -> str | None:
    """
    Return the currently activated flash attention impl name, if any.

    ``None`` indicates that no custom impl has been activated.
    """
    return (
        _FLASH_ATTENTION_ACTIVE[0]
        if _FLASH_ATTENTION_ACTIVE is not None
        else _FLASH_ATTENTION_ACTIVE
    )


def restore_flash_attention_impl(_raise_warn: bool = True) -> None:
    """
    Restore the default FA2 implementation
    """
    global _FLASH_ATTENTION_ACTIVE

    handle = None
    if _FLASH_ATTENTION_ACTIVE is not None:
        handle = _FLASH_ATTENTION_ACTIVE[1]

    if handle is not None:
        handle.remove()
    elif _raise_warn:
        logger.warning(
            "Trying to restore default FA2 impl when no custom impl was activated"
        )

    _FLASH_ATTENTION_ACTIVE = None  # default
