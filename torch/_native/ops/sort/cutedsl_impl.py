"""CuTeDSL overrides of ``aten::sort.stable`` and ``aten::sort.values_stable``.

The functional overload (``torch.sort(x)``, ``torch.argsort(x)``) returns int64
indices; the out-variant also takes an indices tensor of dtype uint8, int8,
uint16, int16, uint32 or int32 (``torch.sort(x, out=(values, indices))`` and
``torch.argsort(x, out=indices)``, which reaches it through ``argsort_out``).
Every stock CUDA sort leaf (see ``policy.py``) has a kernel that mirrors it and
is bit-exact with it; bool keys and rows of up to 2048 take faster kernels (see
``policy.route``).
"""

import functools
from collections.abc import Callable

import torch
from torch._prims_common.wrappers import _maybe_resize_out

from ... import cutedsl_utils as cu
from .policy import (
    INDEX_MAX,
    jit_pays_off,
    MAX_NSORT,
    MAX_ROWS,
    route,
    stock_leaf,
)


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
    torch.int64: "int64",
}


@functools.cache
def _arch(device: int) -> int:
    major, minor = torch.cuda.get_device_capability(device)
    return 10 * major + minor


def _launcher(
    self: torch.Tensor, nsort: int, stable: bool, descending: bool
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], None]:
    """Launcher for policy.route's kernel. The register kernel's unstable order
    is its stable one, not stock's; global_sort batches rows and picks the stock
    leaf per batch."""
    arch = _arch(self.device.index or 0)
    kernel = route(nsort, self.numel() // nsort, stable, self.dtype == torch.bool)
    if kernel == "partition":
        from .cutedsl_partition import partition_sort

        return functools.partial(partition_sort, descending=descending, arch=arch)
    if kernel == "register":
        from .cutedsl_register import register_sort

        return functools.partial(register_sort, descending=descending, arch=arch)
    if kernel == "inplace":
        from .cutedsl_inplace import inplace_sort

        return functools.partial(
            inplace_sort, descending=descending, stable=stable, arch=arch
        )
    from .cutedsl_global import global_sort

    return functools.partial(global_sort, descending=descending, arch=arch)


def _eligible(self: torch.Tensor, dim: int, index_dtype: torch.dtype) -> bool:
    if index_dtype not in _INDEX_NAMES or self.dtype not in _KEY_DTYPES:
        return False
    if not self.is_cuda or self.dim() == 0 or self.numel() == 0:
        return False
    if _arch(self.device.index or 0) < 90:
        return False
    nsort = self.size(dim)
    if nsort - 1 > INDEX_MAX[_INDEX_NAMES[index_dtype]]:
        return False  # meta raises the user-facing error
    if nsort > MAX_NSORT:
        return False
    nrows = self.numel() // nsort
    if nrows > MAX_ROWS:
        return False
    # TODO: a stock fallback for short int64 calls, whose stock path is one
    # host-cheap launch; until then every int64 call takes the JIT route.
    if index_dtype == torch.int64:
        return True
    return jit_pays_off(self.numel(), nsort, nrows, self.dtype == torch.bool)


def _out_cond(
    self: torch.Tensor,
    *,
    stable: bool | None = None,
    dim: int = -1,
    descending: bool = False,
    values: torch.Tensor,
    indices: torch.Tensor,
) -> bool:
    if values.dtype != self.dtype or values.device != self.device:
        return False
    if indices.device != self.device:
        return False
    # Some leaves read x while writing values, and overlapping outs race; stock
    # copies first (or raises on a partial overlap), so leave any aliasing to it.
    is_alias_of = torch._C._is_alias_of  # pyrefly: ignore[missing-attribute]
    if any(
        is_alias_of(a, b)
        for a, b in ((values, self), (indices, self), (values, indices))
    ):
        return False
    return _eligible(self, dim, indices.dtype)


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
    # The segmented leaf carries the index through every radix pass, so an int64
    # holder receives a narrow index widened once the sort completes; the other
    # leaves write int64 directly.
    index_dtype = indices.dtype
    segmented = stock_leaf(nsort, keys.numel() // nsort) == "segmented"
    if index_dtype == torch.int64 and segmented and self.dtype != torch.bool:
        index_dtype = torch.uint16 if nsort - 1 <= INDEX_MAX["uint16"] else torch.int32
    indices_2d = (
        out_indices
        if out_indices.is_contiguous() and index_dtype == indices.dtype
        else torch.empty(keys.shape, dtype=index_dtype, device=keys.device)
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


def _cond(
    self: torch.Tensor,
    *,
    stable: bool | None = None,
    dim: int = -1,
    descending: bool = False,
) -> bool:
    return _eligible(self, dim, torch.int64)


def _impl(
    self: torch.Tensor,
    *,
    stable: bool | None = None,
    dim: int = -1,
    descending: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    # empty_like keeps self's strides when dense and infers dense ones
    # otherwise, as the sort meta function does.
    values = torch.empty_like(self)
    indices = torch.empty_like(self, dtype=torch.int64)
    return _out_impl(
        self,
        stable=stable,
        dim=dim,
        descending=descending,
        values=values,
        indices=indices,
    )


def register_to_dispatch() -> None:
    for op_symbol, cond, impl in (
        ("sort.stable", _cond, _impl),
        ("sort.values_stable", _out_cond, _out_impl),
    ):
        cu.register_op_override(
            "aten",
            op_symbol,
            "CUDA",
            cond=cond,
            impl=impl,
            allow_multiple_override=True,
        )
