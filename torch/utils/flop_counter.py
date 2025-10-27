# mypy: allow-untyped-defs
import torch
from torch.utils._pytree import tree_map, tree_flatten, tree_unflatten
from .module_tracker import ModuleTracker
from typing import Any, Optional, Union, TypeVar
from collections.abc import Callable
from collections.abc import Iterator
from typing_extensions import ParamSpec
from collections import defaultdict
from torch.utils._python_dispatch import TorchDispatchMode
from math import prod
from functools import wraps
import warnings

__all__ = ["FlopCounterMode", "register_flop_formula"]

_T = TypeVar("_T")
_P = ParamSpec("_P")

aten = torch.ops.aten

def get_shape(i):
    if isinstance(i, torch.Tensor):
        return i.shape
    return i

flop_registry: dict[Any, Any] = {}

def shape_wrapper(f):
    @wraps(f)
    def nf(*args, out_val=None, **kwargs):
        args, kwargs, out_shape = tree_map(get_shape, (args, kwargs, out_val))
        return f(*args, out_shape=out_shape, **kwargs)
    return nf

def register_flop_formula(targets, get_raw=False) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:
    def register_fun(flop_formula: Callable[_P, _T]) -> Callable[_P, _T]:
        if not get_raw:
            flop_formula = shape_wrapper(flop_formula)

        def register(target):
            if not isinstance(target, torch._ops.OpOverloadPacket):
                raise ValueError(
                    f"register_flop_formula(targets): expected each target to be "
                    f"OpOverloadPacket (i.e. torch.ops.mylib.foo), got "
                    f"{target} which is of type {type(target)}")
            if target in flop_registry:
                raise RuntimeError(f"duplicate registrations for {target}")
            flop_registry[target] = flop_formula

        # To handle allowing multiple aten_ops at once
        torch.utils._pytree.tree_map_(register, targets)

        return flop_formula

    return register_fun

@register_flop_formula(aten.mm)
def mm_flop(a_shape, b_shape, *args, out_shape=None, **kwargs) -> int:
    """Count flops for matmul."""
    # Inputs should be a list of length 2.
    # Inputs contains the shapes of two matrices.
    m, k = a_shape
    k2, n = b_shape
    assert k == k2
    # NB(chilli): Should be 2 * k - 1 technically for FLOPs.
    return m * n * 2 * k

@register_flop_formula(aten.addmm)
def addmm_flop(self_shape, a_shape, b_shape, out_shape=None, **kwargs) -> int:
    """Count flops for addmm."""
    return mm_flop(a_shape, b_shape)

@register_flop_formula(aten.bmm)
def bmm_flop(a_shape, b_shape, out_shape=None, **kwargs) -> int:
    """Count flops for the bmm operation."""
    # Inputs should be a list of length 2.
    # Inputs contains the shapes of two tensor.
    b, m, k = a_shape
    b2, k2, n = b_shape
    assert b == b2
    assert k == k2
    # NB(chilli): Should be 2 * k - 1 technically for FLOPs.
    flop = b * m * n * 2 * k
    return flop

@register_flop_formula(aten.baddbmm)
def baddbmm_flop(self_shape, a_shape, b_shape, out_shape=None, **kwargs) -> int:
    """Count flops for the baddbmm operation."""
    # Inputs should be a list of length 3.
    # Inputs contains the shapes of three tensors.
    return bmm_flop(a_shape, b_shape)

@register_flop_formula(aten._scaled_mm)
def _scaled_mm_flop(
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
    """Count flops for _scaled_mm."""
    return mm_flop(a_shape, b_shape)


def conv_flop_count(
    x_shape: list[int],
    w_shape: list[int],
    out_shape: list[int],
    transposed: bool = False,
) -> int:
    """Count flops for convolution.

    Note only multiplication is
    counted. Computation for bias are ignored.
    Flops for a transposed convolution are calculated as
    flops = (x_shape[2:] * prod(w_shape) * batch_size).
    Args:
        x_shape (list(int)): The input shape before convolution.
        w_shape (list(int)): The filter shape.
        out_shape (list(int)): The output shape after convolution.
        transposed (bool): is the convolution transposed
    Returns:
        int: the number of flops
    """
    batch_size = x_shape[0]
    conv_shape = (x_shape if transposed else out_shape)[2:]
    c_out, c_in, *filter_size = w_shape

    """
    General idea here is that for a regular conv, for each point in the output
    spatial dimension we convolve the filter with something (hence
    `prod(conv_shape) * prod(filter_size)` ops). Then, this gets multiplied by
    1. batch_size, 2. the cross product of input and weight channels.

    For the transpose, it's not each point in the *output* spatial dimension but
    each point in the *input* spatial dimension.
    """
    # NB(chilli): I don't think this properly accounts for padding :think:
    # NB(chilli): Should be 2 * c_in - 1 technically for FLOPs.
    flop = prod(conv_shape) * prod(filter_size) * batch_size * c_out * c_in * 2
    return flop

@register_flop_formula([aten.convolution, aten._convolution, aten.cudnn_convolution, aten._slow_conv2d_forward])
def conv_flop(x_shape, w_shape, _bias, _stride, _padding, _dilation, transposed, *args, out_shape=None, **kwargs) -> int:
    """Count flops for convolution."""
    # pyrefly: ignore  # bad-argument-type
    return conv_flop_count(x_shape, w_shape, out_shape, transposed=transposed)


@register_flop_formula(aten.convolution_backward)
def conv_backward_flop(
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
        out_shape) -> int:

    def t(shape):
        return [shape[1], shape[0]] + list(shape[2:])
    flop_count = 0

    """
    Let's say we have a regular 1D conv
    {A, B, C} [inp]
    {i, j} [weight]
    => (conv)
    {Ai + Bj, Bi + Cj} [out]

    And as a reminder, the transposed conv of the above is
    => {Ai, Aj + Bi, Bj + Ci, Cj} [transposed conv out]

    For the backwards of conv, we now have
    {D, E} [grad_out]
    {A, B, C} [inp]
    {i, j} [weight]

    # grad_inp as conv_transpose(grad_out, weight)
    Let's first compute grad_inp. To do so, we can simply look at all the
    multiplications that each element of inp is involved in. For example, A is
    only involved in the first element of the output (and thus only depends upon
    D in grad_out), and C is only involved in the last element of the output
    (and thus only depends upon E in grad_out)

    {Di, Dj + Ei, Ej} [grad_inp]

    Note that this corresponds to the below conv_transpose. This gives us the
    output_mask[0] branch, which is grad_inp.

    {D, E} [inp (grad_out)]
    {i, j} [weight]
    => (conv_transpose)
    {Di, Dj + Ei, Ej} [out (grad_inp)]

    I leave the fact that grad_inp for a transposed conv is just conv(grad_out,
    weight) as an exercise for the reader.

    # grad_weight as conv(inp, grad_out)
    To compute grad_weight, we again look at the terms in the output, which as
    a reminder is:
    => {Ai + Bj, Bi + Cj} [out]
    => {D, E} [grad_out]
    If we manually compute the gradient for the weights, we see it's
    {AD + BE, BD + CE} [grad_weight]

    This corresponds to the below conv
    {A, B, C} [inp]
    {D, E} [weight (grad_out)]
    => (conv)
    {AD + BE, BD + CE} [out (grad_weight)]

    # grad_weight of transposed conv as conv(grad_out, inp)
    As a reminder, the terms of the output of a transposed conv are:
    => {Ai, Aj + Bi, Bj + Ci, Cj} [transposed conv out]
    => {D, E, F, G} [grad_out]

    Manually computing the gradient for the weights, we see it's
    {AD + BE + CF, AE + BF + CG} [grad_weight]

    This corresponds to the below conv
    {D, E, F, G} [inp (grad_out)]
    {A, B, C} [weight (inp)]
    => (conv)
    {AD + BE + CF, AE + BF + CG} [out (grad_weight)]

    For the full backwards formula, there are also some details involving
    transpose of the batch/channel dimensions and groups, but I skip those for
    the sake of brevity (and they're pretty similar to matmul backwards)

    Check [conv backwards decomposition as conv forwards]
    """
    # grad_inp as conv_transpose(grad_out, weight)
    if output_mask[0]:
        grad_input_shape = get_shape(out_shape[0])
        flop_count += conv_flop_count(grad_out_shape, w_shape, grad_input_shape, not transposed)

    if output_mask[1]:
        grad_weight_shape = get_shape(out_shape[1])
        if transposed:
            # grad_weight of transposed conv as conv(grad_out, inp)
            flop_count += conv_flop_count(t(grad_out_shape), t(x_shape), t(grad_weight_shape), transposed=False)
        else:
            # grad_weight as conv(inp, grad_out)
            flop_count += conv_flop_count(t(x_shape), t(grad_out_shape), t(grad_weight_shape), transposed=False)

    return flop_count

def sdpa_flop_count(query_shape, key_shape, value_shape):
    """
    Count flops for self-attention.

    NB: We can assume that value_shape == key_shape
    """
    b, h, s_q, d_q = query_shape
    _b2, _h2, s_k, _d2 = key_shape
    _b3, _h3, _s3, d_v = value_shape
    assert b == _b2 == _b3 and h == _h2 == _h3 and d_q == _d2 and s_k == _s3 and d_q == _d2
    total_flops = 0
    # q: [b, h, s_q, d_q] @ k: [b, h, d_q, s_k] -> scores: [b, h, s_q, s_k]
    total_flops += bmm_flop((b * h, s_q, d_q), (b * h, d_q, s_k))
    # scores: [b, h, s_q, s_k] @ v: [b, h, s_k, d_v] -> out: [b, h, s_q, d_v]
    total_flops += bmm_flop((b * h, s_q, s_k), (b * h, s_k, d_v))
    return total_flops


@register_flop_formula([aten._scaled_dot_product_efficient_attention,
                        aten._scaled_dot_product_flash_attention,
                        aten._scaled_dot_product_cudnn_attention])
def sdpa_flop(query_shape, key_shape, value_shape, *args, out_shape=None, **kwargs) -> int:
    """Count flops for self-attention."""
    # NB: We aren't accounting for causal attention here
    return sdpa_flop_count(query_shape, key_shape, value_shape)


def _offsets_to_lengths(offsets, max_len):
    """
    If the offsets tensor is fake, then we don't know the actual lengths.
    In that case, we can just assume the worst case; each batch has max length.
    """
    from torch._subclasses.fake_tensor import FakeTensor
    from torch._subclasses.functional_tensor import FunctionalTensor
    if not isinstance(offsets, (FakeTensor, FunctionalTensor)) and offsets.device.type != "meta":
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
) -> Iterator[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], Optional[tuple[int, ...]]]]:
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
        # So the flops calculation in this case is an overestimate of the actual flops.
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
        for (seq_q_len, seq_k_len) in zip(seq_q_lengths, seq_k_lengths):
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
) -> Iterator[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], Optional[tuple[int, ...]]]]:
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
        # So the flops calculation in this case is an overestimate of the actual flops.
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


@register_flop_formula(aten._flash_attention_forward, get_raw=True)
def _flash_attention_forward_flop(
    query,
    key,
    value,
    cum_seq_q,
    cum_seq_k,
    max_q,
    max_k,
    *args,
    out_shape=None,
    **kwargs
) -> int:
    """Count flops for self-attention."""
    # NB: We aren't accounting for causal attention here
    # in case this is a nested tensor, we unpack the individual batch elements
    # and then sum the flops per batch element
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
        sdpa_flop_count(query_shape, key_shape, value_shape)
        for query_shape, key_shape, value_shape, _ in sizes
    )


@register_flop_formula(aten._efficient_attention_forward, get_raw=True)
def _efficient_attention_forward_flop(
    query,
    key,
    value,
    bias,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    *args,
    **kwargs
) -> int:
    """Count flops for self-attention."""
    # NB: We aren't accounting for causal attention here
    # in case this is a nested tensor, we unpack the individual batch elements
    # and then sum the flops per batch element
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
        sdpa_flop_count(query_shape, key_shape, value_shape)
        for query_shape, key_shape, value_shape, _ in sizes
    )


def sdpa_backward_flop_count(grad_out_shape, query_shape, key_shape, value_shape):
    total_flops = 0
    b, h, s_q, d_q = query_shape
    _b2, _h2, s_k, _d2 = key_shape
    _b3, _h3, _s3, d_v = value_shape
    _b4, _h4, _s4, _d4 = grad_out_shape
    assert b == _b2 == _b3 == _b4 and h == _h2 == _h3 == _h4 and d_q == _d2
    assert d_v == _d4 and s_k == _s3 and s_q == _s4
    total_flops = 0
    # Step 1: We recompute the scores matrix.
    # q: [b, h, s_q, d_q] @ k: [b, h, d_q, s_k] -> scores: [b, h, s_q, s_k]
    total_flops += bmm_flop((b * h, s_q, d_q), (b * h, d_q, s_k))

    # Step 2: We propagate the gradients through the score @ v operation.
    # gradOut: [b, h, s_q, d_v] @ v: [b, h, d_v, s_k] -> gradScores: [b, h, s_q, s_k]
    total_flops += bmm_flop((b * h, s_q, d_v), (b * h, d_v, s_k))
    # scores: [b, h, s_k, s_q] @ gradOut: [b, h, s_q, d_v] -> gradV: [b, h, s_k, d_v]
    total_flops += bmm_flop((b * h, s_k, s_q), (b * h, s_q, d_v))

    # Step 3: We propagate th gradients through the k @ v operation
    # gradScores: [b, h, s_q, s_k] @ k: [b, h, s_k, d_q] -> gradQ: [b, h, s_q, d_q]
    total_flops += bmm_flop((b * h, s_q, s_k), (b * h, s_k, d_q))
    # q: [b, h, d_q, s_q] @ gradScores: [b, h, s_q, s_k] -> gradK: [b, h, d_q, s_k]
    total_flops += bmm_flop((b * h, d_q, s_q), (b * h, s_q, s_k))
    return total_flops


@register_flop_formula([aten._scaled_dot_product_efficient_attention_backward,
                        aten._scaled_dot_product_flash_attention_backward,
                        aten._scaled_dot_product_cudnn_attention_backward])
def sdpa_backward_flop(grad_out_shape, query_shape, key_shape, value_shape, *args, out_shape=None, **kwargs) -> int:
    """Count flops for self-attention backward."""
    return sdpa_backward_flop_count(grad_out_shape, query_shape, key_shape, value_shape)

@register_flop_formula(aten._flash_attention_backward, get_raw=True)
def _flash_attention_backward_flop(
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
    # in case this is a nested tensor, we unpack the individual batch elements
    # and then sum the flops per batch element
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
        sdpa_backward_flop_count(grad_out_shape, query_shape, key_shape, value_shape)
        for query_shape, key_shape, value_shape, grad_out_shape in shapes
    )


@register_flop_formula(aten._efficient_attention_backward, get_raw=True)
def _efficient_attention_backward_flop(
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
    # in case this is a nested tensor, we unpack the individual batch elements
    # and then sum the flops per batch element
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
        sdpa_backward_flop_count(grad_out_shape, query_shape, key_shape, value_shape)
        for query_shape, key_shape, value_shape, grad_out_shape in shapes
    )


flop_registry = {
    aten.mm: mm_flop,
    aten.addmm: addmm_flop,
    aten.bmm: bmm_flop,
    aten.baddbmm: baddbmm_flop,
    aten._scaled_mm: _scaled_mm_flop,
    aten.convolution: conv_flop,
    aten._convolution: conv_flop,
    aten.cudnn_convolution: conv_flop,
    aten._slow_conv2d_forward: conv_flop,
    aten.convolution_backward: conv_backward_flop,
    aten._scaled_dot_product_efficient_attention: sdpa_flop,
    aten._scaled_dot_product_flash_attention: sdpa_flop,
    aten._scaled_dot_product_cudnn_attention: sdpa_flop,
    aten._scaled_dot_product_efficient_attention_backward: sdpa_backward_flop,
    aten._scaled_dot_product_flash_attention_backward: sdpa_backward_flop,
    aten._scaled_dot_product_cudnn_attention_backward: sdpa_backward_flop,
    aten._flash_attention_forward: _flash_attention_forward_flop,
    aten._efficient_attention_forward: _efficient_attention_forward_flop,
    aten._flash_attention_backward: _flash_attention_backward_flop,
    aten._efficient_attention_backward: _efficient_attention_backward_flop,
}

def normalize_tuple(x):
    if not isinstance(x, tuple):
        return (x,)
    return x


# Define the suffixes for different orders of magnitude
suffixes = ["", "K", "M", "B", "T"]
# Thanks BingChat!
def get_suffix_str(number):
    # Find the index of the appropriate suffix based on the number of digits
    # with some additional overflow.
    # i.e. 1.01B should be displayed as 1001M, not 1.001B
    index = max(0, min(len(suffixes) - 1, (len(str(number)) - 2) // 3))
    return suffixes[index]

def convert_num_with_suffix(number, suffix):
    index = suffixes.index(suffix)
    # Divide the number by 1000^index and format it to two decimal places
    value = f"{number / 1000 ** index:.3f}"
    # Return the value and the suffix as a string
    return value + suffixes[index]

def convert_to_percent_str(num, denom):
    if denom == 0:
        return "0%"
    return f"{num / denom:.2%}"

def _pytreeify_preserve_structure(f):
    @wraps(f)
    def nf(args):
        flat_args, spec = tree_flatten(args)
        out = f(*flat_args)
        return tree_unflatten(out, spec)

    return nf


class FlopCounterMode:
    """
    ``FlopCounterMode`` is a context manager that counts the number of flops within its context.

    It does this using a ``TorchDispatchMode``.

    It also supports hierarchical output by passing a module (or list of
    modules) to FlopCounterMode on construction. If you do not need hierarchical
    output, you do not need to use it with a module.

    Example usage

    .. code-block:: python

        mod = ...
        with FlopCounterMode(mod) as flop_counter:
            mod.sum().backward()

    """

    def __init__(
            self,
            mods: Optional[Union[torch.nn.Module, list[torch.nn.Module]]] = None,
            depth: int = 2,
            display: bool = True,
            custom_mapping: Optional[dict[Any, Any]] = None):
        super().__init__()
        self.flop_counts: dict[str, dict[Any, int]] = defaultdict(lambda: defaultdict(int))
        self.depth = depth
        self.display = display
        self.mode: Optional[_FlopCounterMode] = None
        if custom_mapping is None:
            custom_mapping = {}
        if mods is not None:
            warnings.warn("mods argument is not needed anymore, you can stop passing it", stacklevel=2)
        self.flop_registry = {
            **flop_registry,
            **{k: v if getattr(v, "_get_raw", False) else shape_wrapper(v) for k, v in custom_mapping.items()}
        }
        self.mod_tracker = ModuleTracker()

    def get_total_flops(self) -> int:
        return sum(self.flop_counts['Global'].values())

    def get_flop_counts(self) -> dict[str, dict[Any, int]]:
        """Return the flop counts as a dictionary of dictionaries.

        The outer
        dictionary is keyed by module name, and the inner dictionary is keyed by
        operation name.

        Returns:
            Dict[str, Dict[Any, int]]: The flop counts as a dictionary.
        """
        return {k: dict(v) for k, v in self.flop_counts.items()}

    def get_table(self, depth=None):
        if depth is None:
            depth = self.depth
        if depth is None:
            depth = 999999


        import tabulate

        tabulate.PRESERVE_WHITESPACE = True
        header = ["Module", "FLOP", "% Total"]
        values = []
        global_flops = self.get_total_flops()
        global_suffix = get_suffix_str(global_flops)
        is_global_subsumed = False

        def process_mod(mod_name, depth):
            nonlocal is_global_subsumed

            total_flops = sum(self.flop_counts[mod_name].values())

            is_global_subsumed |= total_flops >= global_flops

            padding = " " * depth
            values = []
            values.append([
                padding + mod_name,
                convert_num_with_suffix(total_flops, global_suffix),
                convert_to_percent_str(total_flops, global_flops)
            ])
            for k, v in self.flop_counts[mod_name].items():
                values.append([
                    padding + " - " + str(k),
                    convert_num_with_suffix(v, global_suffix),
                    convert_to_percent_str(v, global_flops)
                ])
            return values

        for mod in sorted(self.flop_counts.keys()):
            if mod == 'Global':
                continue
            mod_depth = mod.count(".") + 1
            if mod_depth > depth:
                continue

            cur_values = process_mod(mod, mod_depth - 1)
            values.extend(cur_values)

        # We do a bit of messing around here to only output the "Global" value
        # if there are any FLOPs in there that aren't already fully contained by
        # a module.
        if 'Global' in self.flop_counts and not is_global_subsumed:
            for value in values:
                value[0] = " " + value[0]

            values = process_mod('Global', 0) + values

        if len(values) == 0:
            values = [["Global", "0", "0%"]]

        return tabulate.tabulate(values, headers=header, colalign=("left", "right", "right"))

    # NB: This context manager is NOT reentrant
    def __enter__(self):
        self.flop_counts.clear()
        self.mod_tracker.__enter__()
        self.mode = _FlopCounterMode(self)
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

    def _count_flops(self, func_packet, out, args, kwargs):
        if func_packet in self.flop_registry:
            flop_count_func = self.flop_registry[func_packet]
            flop_count = flop_count_func(*args, **kwargs, out_val=out)  # type: ignore[operator]
            for par in set(self.mod_tracker.parents):
                self.flop_counts[par][func_packet] += flop_count

        return out

class _FlopCounterMode(TorchDispatchMode):
    supports_higher_order_operators = True

    def __init__(self, counter: FlopCounterMode):
        self.counter = counter

    def _execute_with_isolated_flop_counting(self, branch_fn, operands):
        """Execute a branch function and capture its FLOP counts without
        affecting self.counter.flop_counts

        Args:
            branch_fn: The branch function to execute
            operands: Arguments to pass to the branch function

        Returns:
            Tuple of (result, flop_counts) where result is the branch output
            and flop_counts is a copy of the FLOP counts after execution
        """
        import copy
        checkpointed_flop_counts = copy.copy(self.counter.flop_counts)
        with self:
            result = branch_fn(*operands)
        flop_counts = copy.copy(self.counter.flop_counts)
        self.counter.flop_counts = checkpointed_flop_counts
        return result, flop_counts

    def _handle_higher_order_ops(self, func, types, args, kwargs):
        if func is not torch.ops.higher_order.cond:
            return NotImplemented

        # The flop counter for cond counts the upper bound of flops.
        # For example, if a matmul is executed 2 times in true branch
        # but only 1 time in the false branch, the flop counter will
        # record the larger number of flops, i.e. 2 times.
        if func is torch.ops.higher_order.cond:

            pred, true_branch, false_branch, operands = args
            # Step 1: Count flops for true branch and false branch separately
            true_out, true_flop_counts = self._execute_with_isolated_flop_counting(
                true_branch, operands
            )
            if true_out is NotImplemented:
                return NotImplemented

            false_out, false_flop_counts = self._execute_with_isolated_flop_counting(
                false_branch, operands
            )
            if false_out is NotImplemented:
                return NotImplemented

            # Step 2: merge flop counts
            all_mod_keys = set(true_flop_counts.keys()) | set(false_flop_counts.keys())
            merged_flop_counts = {}
            for outer_key in all_mod_keys:
                true_func_counts = true_flop_counts[outer_key]
                false_func_counts = false_flop_counts[outer_key]

                merged_func_counts = {}
                all_func_keys = set(true_func_counts.keys()) | set(false_func_counts.keys())

                for func_key in all_func_keys:
                    true_val = true_func_counts.get(func_key, 0)
                    false_val = false_func_counts.get(func_key, 0)
                    merged_func_counts[func_key] = max(true_val, false_val)

                merged_flop_counts[outer_key] = merged_func_counts

            # Step 3: update the counter with merged counts
            for outer_key, inner_dict in merged_flop_counts.items():
                self.counter.flop_counts[outer_key].update(inner_dict)

            # It doesn't matter which one we return since true_fn and false_fn return
            # output with the same structure.
            return true_out

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs if kwargs else {}

        # Skip ops from non-standard dispatch_sizes_strides_policy such as NJT
        if func in {torch.ops.aten.sym_is_contiguous.default,
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
                    torch.ops.prim.layout.default}:

            return NotImplemented

        if isinstance(func, torch._ops.HigherOrderOperator):
            return self._handle_higher_order_ops(func, types, args, kwargs)

        # If we don't have func in flop_registry, see if it can decompose
        if func not in self.counter.flop_registry and func is not torch.ops.prim.device.default:
            with self:
                r = func.decompose(*args, **kwargs)
                if r is not NotImplemented:
                    return r

        # no further decomposition; execute & count flops
        out = func(*args, **kwargs)
        return self.counter._count_flops(func._overloadpacket, out, args, kwargs)
