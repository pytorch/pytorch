# mypy: allow-untyped-defs
import warnings
from collections import defaultdict
from collections.abc import Iterator
from typing import Any, Callable, Optional, TypeVar, Union
from typing_extensions import ParamSpec

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from .counter_utils import (
    convert_num_with_suffix,
    convert_to_percent_str,
    get_suffix_str,
    shape_wrapper,
)
from .module_tracker import ModuleTracker


__all__ = ["MemoryCounterMode", "register_memory_formula"]

_T = TypeVar("_T")
_P = ParamSpec("_P")

aten = torch.ops.aten

memory_registry: dict[Any, Any] = {}


def register_memory_formula(
    targets, get_raw=False
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:
    def register_fun(memory_formula: Callable[_P, _T]) -> Callable[_P, _T]:
        if not get_raw:
            memory_formula = shape_wrapper(memory_formula)

        def register(target):
            if not isinstance(target, torch._ops.OpOverloadPacket):
                raise ValueError(
                    f"register_memory_formula(targets): expected each target to be "
                    f"OpOverloadPacket (i.e. torch.ops.mylib.foo), got "
                    f"{target} which is of type {type(target)}"
                )
            if target in memory_registry:
                raise RuntimeError(f"duplicate registrations for {target}")
            memory_registry[target] = memory_formula

        # To handle allowing multiple aten_ops at once
        torch.utils._pytree.tree_map_(register, targets)

        return memory_formula

    return register_fun


@register_memory_formula(aten.mm)
def mm_memory(a_shape, b_shape, *args, out_shape=None, **kwargs) -> int:
    """Count memory accesses for matmul."""
    # TODO
    return 0


@register_memory_formula(aten.addmm)
def addmm_memory(self_shape, a_shape, b_shape, out_shape=None, **kwargs) -> int:
    """Count memory accesses for addmm."""
    # TODO
    return 0


@register_memory_formula(aten.bmm)
def bmm_memory(a_shape, b_shape, out_shape=None, **kwargs) -> int:
    """Count memory accesses for the bmm operation."""
    # TODO
    return 0


@register_memory_formula(aten.baddbmm)
def baddbmm_memory(self_shape, a_shape, b_shape, out_shape=None, **kwargs) -> int:
    """Count memory accesses for the baddbmm operation."""
    # TODO
    return 0


@register_memory_formula(aten._scaled_mm)
def _scaled_mm_memory(
    a_shape,
    b_shape,
    scale_a_shape,
    scale_b_shape,
    bias_shape=None,
    scale_result_shape=None,
    out_dtype=None,
    use_fast_accum=False,
    out_shape=None,
    **kwargs,
) -> int:
    """Count memory accesses for _scaled_mm."""
    # TODO
    return 0


def conv_memory_count(
    x_shape: list[int],
    w_shape: list[int],
    out_shape: list[int],
    transposed: bool = False,
) -> int:
    """Count memory accesses for convolution.

    Args:
        x_shape (list(int)): The input shape before convolution.
        w_shape (list(int)): The filter shape.
        out_shape (list(int)): The output shape after convolution.
        transposed (bool): is the convolution transposed
    Returns:
        int: the number of memory accesses
    """
    # TODO
    return 0


@register_memory_formula([aten.convolution, aten._convolution])
def conv_memory(
    x_shape,
    w_shape,
    _bias,
    _stride,
    _padding,
    _dilation,
    transposed,
    *args,
    out_shape=None,
    **kwargs,
) -> int:
    """Count memory accesses for convolution."""
    # TODO
    return 0


@register_memory_formula(aten.convolution_backward)
def conv_backward_memory(
    grad_out_shape,
    x_shape,
    w_shape,
    _bias,
    _stride,
    _padding,
    _dilation,
    transposed,
    _output_padding,
    _groups,
    output_mask,
    out_shape,
) -> int:
    """Count memory accesses for convolution backward."""
    # TODO
    return 0


def sdpa_memory_count(query_shape, key_shape, value_shape):
    """
    Count memory accesses for self-attention.
    """
    # TODO
    return 0


@register_memory_formula(
    [
        aten._scaled_dot_product_efficient_attention,
        aten._scaled_dot_product_flash_attention,
        aten._scaled_dot_product_cudnn_attention,
    ]
)
def sdpa_memory(
    query_shape, key_shape, value_shape, *args, out_shape=None, **kwargs
) -> int:
    """Count memory accesses for self-attention."""
    # TODO
    return 0


def _offsets_to_lengths(offsets, max_len):
    """
    If the offsets tensor is fake, then we don't know the actual lengths.
    In that case, we can just assume the worst case; each batch has max length.
    """
    from torch._subclasses.fake_tensor import FakeTensor
    from torch._subclasses.functional_tensor import FunctionalTensor

    if (
        not isinstance(offsets, (FakeTensor, FunctionalTensor))
        and offsets.device.type != "meta"
    ):
        return offsets.diff().tolist()
    return [max_len] * (offsets.size(0) - 1)


def _unpack_flash_attention_nested_shapes(
    *,
    query,
    key,
    value,
    grad_out=None,
    cum_seq_q,
    cum_seq_k,
    max_q,
    max_k,
) -> Iterator[
    tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], Optional[tuple[int, ...]]]
]:
    """
    Given inputs to a flash_attention_(forward|backward) kernel, this will handle behavior for
    NestedTensor inputs by effectively unbinding the NestedTensor and yielding the shapes for
    each batch element.

    In the case that this isn't a NestedTensor kernel, then it just yields the original shapes.
    """
    if cum_seq_q is not None:
        # This means we should be dealing with a Nested Jagged Tensor query.
        # The inputs will have shape                  (sum(sequence len), heads, dimension)
        # In comparison, non-Nested inputs have shape (batch, heads, sequence len, dimension)
        # To deal with this, we convert to a shape of (batch, heads, max_seq_len, dimension)
        # So the memory calculation in this case is an overestimate of the actual memory.
        assert len(key.shape) == 3
        assert len(value.shape) == 3
        assert grad_out is None or grad_out.shape == query.shape
        _, h_q, d_q = query.shape
        _, h_k, d_k = key.shape
        _, h_v, d_v = value.shape
        assert cum_seq_q is not None
        assert cum_seq_k is not None
        assert cum_seq_q.shape == cum_seq_k.shape
        seq_q_lengths = _offsets_to_lengths(cum_seq_q, max_q)
        seq_k_lengths = _offsets_to_lengths(cum_seq_k, max_k)
        for seq_q_len, seq_k_len in zip(seq_q_lengths, seq_k_lengths):
            new_query_shape = (1, h_q, seq_q_len, d_q)
            new_key_shape = (1, h_k, seq_k_len, d_k)
            new_value_shape = (1, h_v, seq_k_len, d_v)
            new_grad_out_shape = new_query_shape if grad_out is not None else None
            yield new_query_shape, new_key_shape, new_value_shape, new_grad_out_shape
        return

    yield query.shape, key.shape, value.shape, grad_out.shape if grad_out is not None else None


def _unpack_efficient_attention_nested_shapes(
    *,
    query,
    key,
    value,
    grad_out=None,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
) -> Iterator[
    tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], Optional[tuple[int, ...]]]
]:
    """
    Given inputs to a efficient_attention_(forward|backward) kernel, this will handle behavior for
    NestedTensor inputs by effectively unbinding the NestedTensor and yielding the shapes for
    each batch element.

    In the case that this isn't a NestedTensor kernel, then it just yields the original shapes.
    """
    if cu_seqlens_q is not None:
        # Unlike flash_attention_forward, we get a 4D tensor instead of a 3D tensor for efficient attention.
        #
        # This means we should be dealing with a Nested Jagged Tensor query.
        # The inputs will have shape                  (sum(sequence len), heads, dimension)
        # In comparison, non-Nested inputs have shape (batch, heads, sequence len, dimension)
        # To deal with this, we convert to a shape of (batch, heads, max_seq_len, dimension)
        # So the memory calculation in this case is an overestimate of the actual memory.
        assert len(key.shape) == 4
        assert len(value.shape) == 4
        assert grad_out is None or grad_out.shape == query.shape
        _, _, h_q, d_q = query.shape
        _, _, h_k, d_k = key.shape
        _, _, h_v, d_v = value.shape
        assert cu_seqlens_q is not None
        assert cu_seqlens_k is not None
        assert cu_seqlens_q.shape == cu_seqlens_k.shape
        seqlens_q = _offsets_to_lengths(cu_seqlens_q, max_seqlen_q)
        seqlens_k = _offsets_to_lengths(cu_seqlens_k, max_seqlen_k)
        for len_q, len_k in zip(seqlens_q, seqlens_k):
            new_query_shape = (1, h_q, len_q, d_q)
            new_key_shape = (1, h_k, len_k, d_k)
            new_value_shape = (1, h_v, len_k, d_v)
            new_grad_out_shape = new_query_shape if grad_out is not None else None
            yield new_query_shape, new_key_shape, new_value_shape, new_grad_out_shape
        return

    yield query.shape, key.shape, value.shape, grad_out.shape if grad_out is not None else None


@register_memory_formula(aten._flash_attention_forward, get_raw=True)
def _flash_attention_forward_memory(
    query,
    key,
    value,
    cum_seq_q,
    cum_seq_k,
    max_q,
    max_k,
    *args,
    out_shape=None,
    **kwargs,
) -> int:
    """Count memory accesses for self-attention."""
    # TODO
    sizes = _unpack_flash_attention_nested_shapes(
        query=query,
        key=key,
        value=value,
        cum_seq_q=cum_seq_q,
        cum_seq_k=cum_seq_k,
        max_q=max_q,
        max_k=max_k,
    )
    return sum(
        sdpa_memory_count(query_shape, key_shape, value_shape)
        for query_shape, key_shape, value_shape, _ in sizes
    )


@register_memory_formula(aten._efficient_attention_forward, get_raw=True)
def _efficient_attention_forward_memory(
    query,
    key,
    value,
    bias,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    *args,
    **kwargs,
) -> int:
    """Count memory accesses for self-attention."""
    # TODO
    sizes = _unpack_efficient_attention_nested_shapes(
        query=query,
        key=key,
        value=value,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
    )
    return sum(
        sdpa_memory_count(query_shape, key_shape, value_shape)
        for query_shape, key_shape, value_shape, _ in sizes
    )


def sdpa_backward_memory_count(grad_out_shape, query_shape, key_shape, value_shape):
    """Count memory accesses for self-attention backward."""
    # TODO
    return 0


@register_memory_formula(
    [
        aten._scaled_dot_product_efficient_attention_backward,
        aten._scaled_dot_product_flash_attention_backward,
        aten._scaled_dot_product_cudnn_attention_backward,
    ]
)
def sdpa_backward_memory(
    grad_out_shape, query_shape, key_shape, value_shape, *args, out_shape=None, **kwargs
) -> int:
    """Count memory accesses for self-attention backward."""
    # TODO
    return 0


@register_memory_formula(aten._flash_attention_backward, get_raw=True)
def _flash_attention_backward_memory(
    grad_out,
    query,
    key,
    value,
    out,  # named _out_shape to avoid kwarg collision with out_shape created in wrapper
    logsumexp,
    cum_seq_q,
    cum_seq_k,
    max_q,
    max_k,
    *args,
    **kwargs,
) -> int:
    """Count memory accesses for flash attention backward."""
    # TODO
    shapes = _unpack_flash_attention_nested_shapes(
        query=query,
        key=key,
        value=value,
        grad_out=grad_out,
        cum_seq_q=cum_seq_q,
        cum_seq_k=cum_seq_k,
        max_q=max_q,
        max_k=max_k,
    )
    return sum(
        sdpa_backward_memory_count(grad_out_shape, query_shape, key_shape, value_shape)
        for query_shape, key_shape, value_shape, grad_out_shape in shapes
    )


@register_memory_formula(aten._efficient_attention_backward, get_raw=True)
def _efficient_attention_backward_memory(
    grad_out,
    query,
    key,
    value,
    bias,
    out,  # named _out to avoid kwarg collision with out created in wrapper
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    *args,
    **kwargs,
) -> int:
    """Count memory accesses for efficient attention backward."""
    # TODO
    shapes = _unpack_efficient_attention_nested_shapes(
        query=query,
        key=key,
        value=value,
        grad_out=grad_out,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
    )
    return sum(
        sdpa_backward_memory_count(grad_out_shape, query_shape, key_shape, value_shape)
        for query_shape, key_shape, value_shape, grad_out_shape in shapes
    )


memory_registry = {
    aten.mm: mm_memory,
    aten.addmm: addmm_memory,
    aten.bmm: bmm_memory,
    aten.baddbmm: baddbmm_memory,
    aten._scaled_mm: _scaled_mm_memory,
    aten.convolution: conv_memory,
    aten._convolution: conv_memory,
    aten.convolution_backward: conv_backward_memory,
    aten._scaled_dot_product_efficient_attention: sdpa_memory,
    aten._scaled_dot_product_flash_attention: sdpa_memory,
    aten._scaled_dot_product_cudnn_attention: sdpa_memory,
    aten._scaled_dot_product_efficient_attention_backward: sdpa_backward_memory,
    aten._scaled_dot_product_flash_attention_backward: sdpa_backward_memory,
    aten._scaled_dot_product_cudnn_attention_backward: sdpa_backward_memory,
    aten._flash_attention_forward: _flash_attention_forward_memory,
    aten._efficient_attention_forward: _efficient_attention_forward_memory,
    aten._flash_attention_backward: _flash_attention_backward_memory,
    aten._efficient_attention_backward: _efficient_attention_backward_memory,
}


class MemoryCounterMode:
    """
    ``MemoryCounterMode`` is a context manager that counts the number of memory accesses within its context.

    It does this using a ``TorchDispatchMode``.

    It also supports hierarchical output by passing a module (or list of
    modules) to MemoryCounterMode on construction. If you do not need hierarchical
    output, you do not need to use it with a module.

    Example usage

    .. code-block:: python

        mod = ...
        with MemoryCounterMode(mod) as memory_counter:
            mod.sum().backward()

    """

    def __init__(
        self,
        mods: Optional[Union[torch.nn.Module, list[torch.nn.Module]]] = None,
        depth: int = 2,
        display: bool = True,
        custom_mapping: Optional[dict[Any, Any]] = None,
    ):
        super().__init__()
        self.memory_counts: dict[str, dict[Any, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self.depth = depth
        self.display = display
        self.mode: Optional[_MemoryCounterMode] = None
        if custom_mapping is None:
            custom_mapping = {}
        if mods is not None:
            warnings.warn(
                "mods argument is not needed anymore, you can stop passing it",
                stacklevel=2,
            )
        self.memory_registry = {
            **memory_registry,
            **{
                k: v if getattr(v, "_get_raw", False) else shape_wrapper(v)
                for k, v in custom_mapping.items()
            },
        }
        self.mod_tracker = ModuleTracker()

    def get_total_memory(self) -> int:
        return sum(self.memory_counts["Global"].values())

    def get_memory_counts(self) -> dict[str, dict[Any, int]]:
        """Return the memory access counts as a dictionary of dictionaries.

        The outer
        dictionary is keyed by module name, and the inner dictionary is keyed by
        operation name.

        Returns:
            Dict[str, Dict[Any, int]]: The memory access counts as a dictionary.
        """
        return {k: dict(v) for k, v in self.memory_counts.items()}

    def get_table(self, depth=None):
        if depth is None:
            depth = self.depth
        if depth is None:
            depth = 999999

        import tabulate

        tabulate.PRESERVE_WHITESPACE = True
        header = ["Module", "Memory Accesses", "% Total"]
        values = []
        global_memory = self.get_total_memory()
        global_suffix = get_suffix_str(global_memory)
        is_global_subsumed = False

        def process_mod(mod_name, depth):
            nonlocal is_global_subsumed

            total_memory = sum(self.memory_counts[mod_name].values())

            is_global_subsumed |= total_memory >= global_memory

            padding = " " * depth
            values = []
            values.append(
                [
                    padding + mod_name,
                    convert_num_with_suffix(total_memory, global_suffix),
                    convert_to_percent_str(total_memory, global_memory),
                ]
            )
            for k, v in self.memory_counts[mod_name].items():
                values.append(
                    [
                        padding + " - " + str(k),
                        convert_num_with_suffix(v, global_suffix),
                        convert_to_percent_str(v, global_memory),
                    ]
                )
            return values

        for mod in sorted(self.memory_counts.keys()):
            if mod == "Global":
                continue
            mod_depth = mod.count(".") + 1
            if mod_depth > depth:
                continue

            cur_values = process_mod(mod, mod_depth - 1)
            values.extend(cur_values)

        # We do a bit of messing around here to only output the "Global" value
        # if there are any memory accesses in there that aren't already fully contained by
        # a module.
        if "Global" in self.memory_counts and not is_global_subsumed:
            for value in values:
                value[0] = " " + value[0]

            values = process_mod("Global", 0) + values

        if len(values) == 0:
            values = [["Global", "0", "0%"]]

        return tabulate.tabulate(
            values, headers=header, colalign=("left", "right", "right")
        )

    # NB: This context manager is NOT reentrant
    def __enter__(self):
        self.memory_counts.clear()
        self.mod_tracker.__enter__()
        self.mode = _MemoryCounterMode(self)
        self.mode.__enter__()
        return self

    def __exit__(self, *args):
        assert self.mode is not None
        b = self.mode.__exit__(*args)
        self.mode = None  # break cycles
        self.mod_tracker.__exit__()
        if self.display:
            print(self.get_table(self.depth))
        return b

    def _count_memory(self, func_packet, out, args, kwargs):
        if func_packet in self.memory_registry:
            memory_count_func = self.memory_registry[func_packet]
            memory_count = memory_count_func(*args, **kwargs, out_val=out)  # type: ignore[operator]
            for par in set(self.mod_tracker.parents):
                self.memory_counts[par][func_packet] += memory_count

        return out


class _MemoryCounterMode(TorchDispatchMode):
    def __init__(self, counter: MemoryCounterMode):
        self.counter = counter

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs if kwargs else {}

        # Skip ops from non-standard dispatch_sizes_strides_policy such as NJT
        if func in {
            torch.ops.aten.is_contiguous.default,
            torch.ops.aten.is_contiguous.memory_format,
            torch.ops.aten.is_strides_like_format.default,
            torch.ops.aten.is_non_overlapping_and_dense.default,
            torch.ops.aten.size.default,
            torch.ops.aten.sym_size.default,
            torch.ops.aten.stride.default,
            torch.ops.aten.sym_stride.default,
            torch.ops.aten.storage_offset.default,
            torch.ops.aten.sym_storage_offset.default,
            torch.ops.aten.numel.default,
            torch.ops.aten.sym_numel.default,
            torch.ops.aten.dim.default,
            torch.ops.prim.layout.default,
        }:
            return NotImplemented

        # If we don't have func in memory_registry, see if it can decompose
        if (
            func not in self.counter.memory_registry
            and func is not torch.ops.prim.device.default
        ):
            with self:
                r = func.decompose(*args, **kwargs)
                if r is not NotImplemented:
                    return r

        # no further decomposition; execute & count memory accesses
        out = func(*args, **kwargs)
        return self.counter._count_memory(func._overloadpacket, out, args, kwargs)
