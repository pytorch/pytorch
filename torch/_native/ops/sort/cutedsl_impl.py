"""CuTeDSL override of ``aten::sort.values_stable`` for narrow out= indices.

Fires only for the out-variant with an indices tensor of dtype uint8, int8,
uint16, int16, uint32 or int32 (``torch.sort(x, out=(values, indices))`` and
``torch.argsort(x, out=indices)``, which reaches it through ``argsort_out``).
Functional sort and int64 out= keep the stock path. Every stock CUDA sort leaf
(see ``policy.py``) has a kernel that mirrors it and is bit-exact with it.
"""

import functools
from collections.abc import Callable

import torch
from torch._prims_common.wrappers import _maybe_resize_out

from ... import cutedsl_utils as cu
from .policy import INDEX_MAX, MAX_NSORT, MAX_ROWS, stock_leaf


_KEY_DTYPES = frozenset(
    (
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.float16,
        torch.bfloat16,
        torch.int32,
        torch.float32,
        torch.int64,
        torch.float64,
    )
)
_INDEX_NAMES = {
    torch.uint8: "uint8",
    torch.int8: "int8",
    torch.uint16: "uint16",
    torch.int16: "int16",
    torch.uint32: "uint32",
    torch.int32: "int32",
}


@functools.cache
def _arch(device: int) -> int:
    major, minor = torch.cuda.get_device_capability(device)
    return 10 * major + minor


def _launcher(
    self: torch.Tensor, nsort: int, stable: bool, descending: bool
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], None]:
    """Kernel for the stock leaf this call takes. Rows longer than 4096 go to
    global_sort, which batches rows and picks the leaf per batch like stock."""
    arch = _arch(self.device.index or 0)
    if stock_leaf(nsort, self.numel() // nsort) == "inplace":
        from .cutedsl_inplace import inplace_sort

        return functools.partial(
            inplace_sort, descending=descending, stable=stable, arch=arch
        )
    from .cutedsl_global import global_sort

    return functools.partial(global_sort, descending=descending, arch=arch)


def _out_cond(
    self: torch.Tensor,
    *,
    stable: bool | None = None,
    dim: int = -1,
    descending: bool = False,
    values: torch.Tensor,
    indices: torch.Tensor,
) -> bool:
    if indices.dtype not in _INDEX_NAMES or self.dtype not in _KEY_DTYPES:
        return False
    if not self.is_cuda or self.dim() == 0 or self.numel() == 0:
        return False
    if values.dtype != self.dtype or values.device != self.device:
        return False
    if indices.device != self.device or _arch(self.device.index or 0) < 90:
        return False
    # Some leaves read x while writing values, and overlapping outs race; stock
    # copies first (or raises on a partial overlap), so leave any aliasing to it.
    is_alias_of = torch._C._is_alias_of  # pyrefly: ignore[missing-attribute]
    if any(
        is_alias_of(a, b)
        for a, b in ((values, self), (indices, self), (values, indices))
    ):
        return False
    nsort = self.size(dim)
    if nsort - 1 > INDEX_MAX[_INDEX_NAMES[indices.dtype]]:
        return False  # meta raises the user-facing error
    if nsort > MAX_NSORT:
        return False
    return self.numel() // nsort <= MAX_ROWS


def _out_impl(
    self: torch.Tensor,
    *,
    stable: bool | None = None,
    dim: int = -1,
    descending: bool = False,
    values: torch.Tensor,
    indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    dim = dim % self.dim()
    nsort = self.size(dim)
    kernel = _launcher(self, nsort, bool(stable), descending)
    # The router replaces the generated out= wrapper, so resize as it would,
    # with self's dense strides like the meta function.
    if values.shape != self.shape or indices.shape != self.shape:
        strides = torch.empty_like(self, device="meta").stride()
        for out in (values, indices):
            if out.shape != self.shape:
                _maybe_resize_out(out, self.shape)
                out.as_strided_(self.shape, strides)

    # Sort a dim-last row-major view; copy in and out only when the caller's
    # layout differs, as sort_cuda_kernel does.
    keys = self.movedim(dim, -1).contiguous()
    out_values = values.movedim(dim, -1)
    out_indices = indices.movedim(dim, -1)
    values_2d = out_values if out_values.is_contiguous() else torch.empty_like(keys)
    indices_2d = (
        out_indices
        if out_indices.is_contiguous()
        else torch.empty(keys.shape, dtype=indices.dtype, device=keys.device)
    )
    rows = keys.view(-1, nsort)

    def launch() -> None:
        kernel(rows, values_2d.view(-1, nsort), indices_2d.view(-1, nsort))

    # The Python dispatch path has no automatic device guard (see topk).
    device = self.get_device()
    if device == torch.cuda.current_device():
        launch()
    else:
        with torch.cuda.device(device):
            launch()
    if values_2d is not out_values:
        out_values.copy_(values_2d)
    if indices_2d is not out_indices:
        out_indices.copy_(indices_2d)
    return values, indices


def register_to_dispatch() -> None:
    cu.register_op_override(
        "aten",
        "sort.values_stable",
        "CUDA",
        cond=_out_cond,
        impl=_out_impl,
        allow_multiple_override=True,
    )
