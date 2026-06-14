"""Forward-pass NaN/Inf guard for :class:`~torch.nn.Module`.

Registers a forward hook on every submodule of a given root and reports the
first submodule whose output contains a NaN or Inf, naming the submodule, the
output path inside its return structure, and tensor stats. Complements
:class:`torch.autograd.detect_anomaly`, which targets the backward pass and
runs through the autograd engine; this guard is forward-only and does not
touch autograd.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Literal

import torch
from torch import nn
from torch.utils import _pytree as pytree
from torch.utils.hooks import RemovableHandle


__all__ = ["NaNGuard", "NaNGuardError", "nan_guard"]


class NaNGuardError(RuntimeError):
    """Raised by :class:`NaNGuard` when a forward output contains NaN or Inf."""


def _is_tensor(x: Any) -> bool:
    return isinstance(x, torch.Tensor)


def _scan_tensor(t: torch.Tensor, check_inf: bool) -> tuple[int, int]:
    if t.numel() == 0:
        return 0, 0
    if not (t.is_floating_point() or t.is_complex()):
        return 0, 0
    nan_count = int(torch.isnan(t).sum().item())
    inf_count = int(torch.isinf(t).sum().item()) if check_inf else 0
    return nan_count, inf_count


_Offender = tuple[str, torch.Tensor, int, int]


def _scan_tree(tree: Any, check_inf: bool) -> list[_Offender]:
    offenders: list[_Offender] = []
    for kp, leaf in pytree.tree_leaves_with_path(tree, is_leaf=_is_tensor):
        if not _is_tensor(leaf):
            continue
        n, i = _scan_tensor(leaf, check_inf)
        if n + i > 0:
            offenders.append((pytree.keystr(kp), leaf, n, i))
    return offenders


def _format_message(
    module_name: str, module: nn.Module, offenders: list[_Offender]
) -> str:
    name_disp = module_name if module_name else "<root>"
    lines = [
        f"NaNGuard: detected NaN/Inf in output of submodule "
        f"{name_disp!r} ({type(module).__name__}):"
    ]
    for path, t, n, i in offenders:
        path_disp = path if path else "<output>"
        lines.append(
            f"  {path_disp}: shape={tuple(t.shape)} dtype={t.dtype} "
            f"device={t.device} nan={n} inf={i}"
        )
    return "\n".join(lines)


class NaNGuard:
    """Context manager that watches an :class:`~torch.nn.Module`'s forward pass
    for NaN or Inf values in submodule outputs.

    On entering, registers a forward hook on every submodule of ``module``
    (including the root). Each hook scans the submodule's output via
    :func:`torch.utils._pytree.tree_leaves_with_path`, applying
    :func:`torch.isnan` (and :func:`torch.isinf` if ``check_inf`` is set) to
    every floating-point or complex tensor leaf. When an offending tensor is
    found, the guard raises :class:`NaNGuardError` (or emits a warning) with
    the submodule's :meth:`~torch.nn.Module.named_modules` name, its class,
    the pytree-style path to the offending leaf within the output structure,
    and the leaf's shape, dtype, device, NaN count, and Inf count.

    All hooks are removed on exit, including when the guarded forward raises.

    Args:
        module: Root module to watch.
        check_inf: If ``True`` (default), Inf values are reported alongside NaN.
        check_inputs: If ``True``, the guard also scans submodule inputs and
            suppresses the report when inputs already contained NaN/Inf,
            isolating the *first producer* rather than firing on every
            downstream propagator. Default ``False``.
        on_detect: ``"raise"`` (default) raises :class:`NaNGuardError`;
            ``"warn"`` emits a :class:`UserWarning` and lets execution continue.

    Example::

        model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
        with torch.utils.nan_guard.NaNGuard(model):
            out = model(x)  # raises NaNGuardError naming the offending submodule

    The guard adds the cost of a reduction over each submodule output, which on
    accelerator devices forces a host sync. Intended for debugging, not steady-
    state training.
    """

    def __init__(
        self,
        module: nn.Module,
        *,
        check_inf: bool = True,
        check_inputs: bool = False,
        on_detect: Literal["raise", "warn"] = "raise",
    ) -> None:
        if on_detect not in ("raise", "warn"):
            raise ValueError(
                f"on_detect must be 'raise' or 'warn', got {on_detect!r}"
            )
        self.module = module
        self.check_inf = check_inf
        self.check_inputs = check_inputs
        self.on_detect: Literal["raise", "warn"] = on_detect
        self._handles: list[RemovableHandle] = []

    def _make_hook(self, name: str) -> Callable[..., None]:
        check_inf = self.check_inf
        check_inputs = self.check_inputs

        def hook(
            module: nn.Module, args: tuple, kwargs: dict, output: Any
        ) -> None:
            offenders = _scan_tree(output, check_inf)
            if not offenders:
                return
            if check_inputs and _scan_tree((args, kwargs), check_inf):
                return
            self._fire(name, module, offenders)

        return hook

    def _fire(
        self, name: str, module: nn.Module, offenders: list[_Offender]
    ) -> None:
        msg = _format_message(name, module, offenders)
        if self.on_detect == "warn":
            warnings.warn(msg, stacklevel=2)
        else:
            raise NaNGuardError(msg)

    def __enter__(self) -> "NaNGuard":
        for name, sub in self.module.named_modules():
            self._handles.append(
                sub.register_forward_hook(self._make_hook(name), with_kwargs=True)
            )
        return self

    def __exit__(self, *exc: object) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


def nan_guard(module: nn.Module, **kwargs: Any) -> NaNGuard:
    """Convenience constructor for :class:`NaNGuard`.

    Equivalent to ``NaNGuard(module, **kwargs)``.
    """
    return NaNGuard(module, **kwargs)
