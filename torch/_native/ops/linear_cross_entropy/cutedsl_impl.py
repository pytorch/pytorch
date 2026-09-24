"""CuTeDSL override registrations for the chunked ``linear_cross_entropy`` ops.

`torch_nn::_linear_cross_entropy_batch_chunked` and its
`..._no_reduction` sibling each put a whole fused loss behind one dispatcher
symbol: an ``(N, F) x (F, C)`` logits matmul, a row-shifted softmax, the
cross-entropy reduction, and -- on the scalar-reduction op when gradients are
requested -- three gradient matmuls, all chunked over the batch. Per chunk the
eager loop walks the ``(B, C)`` logits buffer roughly seven times (matmul write,
row max, subtract, gather, ``exp_``, row sum, scale) before the gradient matmuls
read it again. Collapsing those passes is what a DSL kernel is for.

``_OVERRIDES`` is the single list of what this module registers; the
registration loop and the tests both read it, so adding an override is one row.
"""

import functools
import importlib
from collections.abc import Callable
from typing import Any, NamedTuple

import torch

from ... import cutedsl_utils as cu, variants


def _always(*args: object, **kwargs: object) -> bool:
    """Eligibility of the passthrough variant: anything the op itself accepts.

    Keeping this wide is what lets `passthrough` reproduce routing-only
    behaviour for every input a kernel variant declines, so a plot's routing
    curve means the same thing whatever kernels exist.
    """
    return True


def _batch_chunked_passthrough(
    input: torch.Tensor,
    linear_weight: torch.Tensor,
    target: torch.Tensor,
    linear_bias: torch.Tensor | None,
    weight: torch.Tensor | None,
    reduction: str,
    ignore_index: int,
    label_smoothing: float,
    batch_chunk_size: int,
    acc_policy: str,
    acc_dtype: torch.dtype,
    allow_retain_graph: bool,
    compute_input_grad: bool,
    compute_linear_weight_grad: bool,
    compute_linear_bias_grad: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Installing at a backend key means the op's body never runs, so its checks
    # have to be run here. ``allow_retain_graph`` is a backward-only flag that
    # neither the checks nor the accumulator take.
    #
    # A kernel replacing the accumulator call below also loses the checks
    # *inside* it and must re-enforce those: the unresolved
    # ``acc_policy`` / ``acc_dtype`` check, and ``_ChunkContext.build``'s
    # ``linear_bias`` shape and ``acc_dtype`` compatibility checks.
    from torch.nn.modules.linear_cross_entropy import (
        _check_batch_chunked_grad_flags,
        _linear_cross_entropy_batch_chunked_accumulator,
    )

    _check_batch_chunked_grad_flags(
        input,
        linear_weight,
        target,
        linear_bias,
        compute_input_grad,
        compute_linear_weight_grad,
        compute_linear_bias_grad,
    )
    return _linear_cross_entropy_batch_chunked_accumulator(
        input,
        linear_weight,
        target,
        linear_bias,
        weight,
        reduction,
        ignore_index,
        label_smoothing,
        batch_chunk_size,
        acc_policy,
        acc_dtype,
        compute_input_grad,
        compute_linear_weight_grad,
        compute_linear_bias_grad,
    )


def _no_reduction_passthrough(
    input: torch.Tensor,
    linear_weight: torch.Tensor,
    target: torch.Tensor,
    linear_bias: torch.Tensor | None,
    weight: torch.Tensor | None,
    ignore_index: int,
    batch_chunk_size: int,
    acc_policy: str,
    acc_dtype: torch.dtype,
) -> torch.Tensor:
    # The op's body: the accumulator's loss-only branch. It needs no checks --
    # the op takes no compute_*_grad flags, and its probability-target guard is
    # in its backward, registered above this dispatch key. A kernel replacing
    # this call must re-enforce the accumulator's checks, as the scalar op's
    # impl does.
    from torch.nn.modules.linear_cross_entropy import (
        _linear_cross_entropy_batch_chunked_accumulator,
    )

    return _linear_cross_entropy_batch_chunked_accumulator(
        input,
        linear_weight,
        target,
        linear_bias,
        weight,
        "none",
        ignore_index,
        0.0,
        batch_chunk_size,
        acc_policy,
        acc_dtype,
        compute_input_grad=False,
        compute_linear_weight_grad=False,
        compute_linear_bias_grad=False,
    )[0]


@functools.cache
def _arch_supported(device_index: int = 0) -> bool:
    """Whether the kernel can run on this device: sm_80, the CuTeDSL runtime's
    floor, which is also where cuBLAS gives the logits matmul an fp32 output
    from low-precision inputs.

    Called from `cond`, i.e. at dispatch and never at registration, so the
    `cuInit` this performs cannot poison a fork before torch is used.
    """
    return torch.cuda.get_device_capability(device_index) >= (8, 0)


def _kernel_eligible(
    input: torch.Tensor,
    linear_weight: torch.Tensor,
    target: torch.Tensor,
    linear_bias: torch.Tensor | None,
    weight: torch.Tensor | None,
    reduction: str,
    ignore_index: int,
    label_smoothing: float,
    batch_chunk_size: int,
    acc_policy: str,
    acc_dtype: torch.dtype | None,
    allow_retain_graph: bool,
    compute_input_grad: bool,
    compute_linear_weight_grad: bool,
    compute_linear_bias_grad: bool,
    *,
    dtypes: tuple[torch.dtype, ...] = (torch.bfloat16,),
) -> bool:
    """What the kernel variants implement. Shapes, dtypes and device only -- no
    data reads, so this is safe under FakeTensor tracing. `cond` evaluates the
    selected variant's eligibility, so an ineligible input never enters the
    override: the router falls back, and a profile shows whether the kernel ran.
    `dtypes` is the variant's input-dtype set; fp16 can be in it because eager's
    `compact` logits buffer is fp16 too, and aliasing `g` into it keeps the
    kernel at eager's memory.
    """
    return (
        input.device.type == "cuda"
        and _arch_supported(input.device.index or 0)
        # Every tensor the launch touches must be on that device; declining
        # lets eager raise its own device-mismatch error.
        and linear_weight.device == input.device
        and target.device == input.device
        and (linear_bias is None or linear_bias.device == input.device)
        and (weight is None or weight.device == input.device)
        and input.dtype in dtypes
        and linear_weight.dtype is input.dtype
        and acc_dtype is torch.float32
        and acc_policy == "compact"
        # Class-index targets; a probability target is (N, C) in the input
        # dtype and a different algorithm.
        and target.dtype is torch.int64
        and target.dim() == 1
        and reduction in ("mean", "sum")
        and label_smoothing == 0.0
        and input.dim() == 2
        and linear_weight.dim() == 2
        # Both ends of the class count. With zero classes every target is out
        # of range and the kernel traps, where eager raises on the empty
        # reduction. V crosses the FFI and indexes columns as int32; the kernel
        # strides up to a staging group past V, so the exact safe bound sits
        # slightly lower, but no (C, F) weight near it fits in memory.
        and 1 <= linear_weight.shape[0] <= 2**31 - 1
    )


def _batch_chunked_kernel(
    input: torch.Tensor,
    linear_weight: torch.Tensor,
    target: torch.Tensor,
    linear_bias: torch.Tensor | None,
    weight: torch.Tensor | None,
    reduction: str,
    ignore_index: int,
    label_smoothing: float,
    batch_chunk_size: int,
    acc_policy: str,
    acc_dtype: torch.dtype,
    allow_retain_graph: bool,
    compute_input_grad: bool,
    compute_linear_weight_grad: bool,
    compute_linear_bias_grad: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Chunked loss and gradients with the softmax-gradient transform fused.

    Same four outputs and chunking contract as the op: `batch_chunk_size`
    bounds the per-chunk footprint. Per chunk, one kernel reads the logits
    twice and writes the dense gradient-of-logits `g`, which makes both
    parameter gradients plain GEMMs -- `grad_input = g @ W`,
    `grad_linear_weight = g^T @ X` -- with no one-hot scatter, so the result is
    deterministic. `g` aliases the logits storage, so a chunk costs one
    `(Bc, V)` buffer and no value loses precision. The forward-only path has no
    `g` to write and forms its statistics from eager ops.

    The logits buffer dtype is eager's (fp16 for fp16 input, `acc_dtype`
    otherwise), while `exp()` and the scaled gradient stay in fp32 registers.
    The scalar loss accumulates in `acc_dtype`, where eager rounds it to bf16
    twice per chunk, so for bf16 the two agree to a few ULP of that dtype, not
    bit for bit.
    """
    from torch.nn.modules.linear_cross_entropy import (
        _check_acc_dtype_compatible,
        _check_batch_chunked_grad_flags,
        _check_linear_bias_shape,
        _check_resolved_acc,
        _corrected_target,
        _make_empty,
        _make_zeros,
        _neg_weight_target,
    )

    # Installed at a backend key, so the op's body never runs: all of its
    # checks, including the accumulator's, run here, so the set stays the op's
    # whatever the gate admits.
    _check_batch_chunked_grad_flags(
        input,
        linear_weight,
        target,
        linear_bias,
        compute_input_grad,
        compute_linear_weight_grad,
        compute_linear_bias_grad,
    )
    _check_resolved_acc(acc_policy, acc_dtype)
    _check_linear_bias_shape(linear_weight, linear_bias)
    dtype = input.dtype
    _check_acc_dtype_compatible(dtype, acc_dtype)

    from .fused_grad_logits_kernel import fused_grad_logits_into

    device = input.device
    num_batches, _ = input.shape
    num_classes = linear_weight.shape[0]

    # The op's own per-row prep: clamped target, class weight, mean divisor and
    # ignored rows. Negated, since the kernel writes `s * (softmax - onehot)`;
    # in place is safe, as `_neg_weight_target` returns a fresh tensor.
    # `.contiguous()` because the kernel is compiled for a stride-1 target and
    # `_corrected_target` may return the caller's tensor.
    target_hat = _corrected_target(target, ignore_index, num_classes).contiguous()
    row_scale = _neg_weight_target(
        target_hat, target == ignore_index, weight, acc_dtype, reduction
    ).neg_()

    loss = _make_zeros((), acc_dtype, device)
    grad_input = _make_empty(input.shape, dtype, device, when=compute_input_grad)
    # Uninitialized, since the first chunk writes all of it (beta=0 below); the
    # empty-batch early return has no first chunk, so it gets zeros.
    _make_accumulator = _make_empty if num_batches > 0 else _make_zeros
    grad_linear_weight = _make_accumulator(
        linear_weight.shape, dtype, device, when=compute_linear_weight_grad
    )
    # Uninitialized, since the post-loop `copy_` writes all of it; the
    # empty-batch early return has no loop, so it gets zeros.
    _make_bias = _make_empty if num_batches > 0 else _make_zeros
    grad_linear_bias = _make_bias(
        linear_weight.shape[:-1], dtype, device, when=compute_linear_bias_grad
    )
    bias_grad_acc = _make_zeros(
        linear_weight.shape[:-1], acc_dtype, device, when=compute_linear_bias_grad
    )
    if num_batches == 0:
        if reduction == "mean":
            loss.fill_(torch.nan)
        return (loss.to(dtype), grad_input, grad_linear_weight, grad_linear_bias)

    compute_grads = (
        compute_input_grad or compute_linear_weight_grad or compute_linear_bias_grad
    )
    chunk_rows = min(batch_chunk_size, num_batches)
    # Eager's logits dtype under `compact`; a bf16 buffer would round the softmax
    # input, measured in https://github.com/pytorch/pytorch/pull/195829.
    logits_dtype = dtype if dtype is torch.float16 else acc_dtype
    # Allocated once and reused: the peak is one chunk of each, which is what
    # `batch_chunk_size` promises.
    logits_buf = torch.empty(
        (chunk_rows, num_classes), dtype=logits_dtype, device=device
    )
    # `g` aliases the logits storage, so a chunk is one buffer; the kernel
    # orders its writes against its reads (see its module docstring).
    g_alias = logits_buf.view(dtype).narrow(1, 0, num_classes)
    # Only the forward-only branch uses a row max.
    row_max_buf = _make_empty(
        (chunk_rows, 1), logits_dtype, device, when=not compute_grads
    )
    # One slot per row of the call, filled by the kernel and summed once after
    # the loop; the forward-only path never reads it.
    term_buf = _make_empty((num_batches,), acc_dtype, device, when=compute_grads)
    weight_t = linear_weight.t()
    # `addmm` takes `self` only in `out_dtype` or `mat1`'s dtype, so an fp32
    # bias with fp16 inputs (fp16 buffer) matches neither and is cast; the
    # other combinations pass through unchanged.
    bias_arg = linear_bias
    if linear_bias is not None and linear_bias.dtype not in (logits_dtype, dtype):
        bias_arg = linear_bias.to(logits_dtype)

    for start in range(0, num_batches, chunk_rows):
        rows = min(chunk_rows, num_batches - start)
        input_chunk = input.narrow(0, start, rows)
        target_chunk = target_hat.narrow(0, start, rows)
        scale_chunk = row_scale.narrow(0, start, rows)
        logits = logits_buf.narrow(0, 0, rows)

        # `bias_arg`, not `linear_bias`: it is None exactly when that is, and
        # branching on it is what narrows its type for the `addmm` below.
        if bias_arg is None:
            torch.mm(input_chunk, weight_t, out_dtype=logits_dtype, out=logits)
        else:
            torch.addmm(
                bias_arg, input_chunk, weight_t, out_dtype=logits_dtype, out=logits
            )

        g = g_alias.narrow(0, 0, rows) if compute_grads else None
        if g is not None:
            # This consumes `logits`: on return those bytes hold `g`. Nothing
            # below reads them again, and the next chunk's matmul overwrites
            # the buffer.
            fused_grad_logits_into(
                g,
                term_buf.narrow(0, start, rows),
                logits,
                scale_chunk,
                target_chunk,
            )
        else:
            # Shift in place -- so this pass allocates no (Bc, V) temporary --
            # then read the target logit BEFORE exponentiating, since `exp_`
            # overwrites the shifted logits.
            row_max = row_max_buf.narrow(0, 0, rows)
            torch.amax(logits, dim=1, keepdim=True, out=row_max)
            logits.sub_(row_max)
            target_logit = logits.gather(1, target_chunk.unsqueeze(1)).squeeze(1)
            logits.exp_()
            row_sum = logits.sum(dim=1, dtype=acc_dtype)
            # Both terms shifted by the row max, which keeps their difference
            # from collapsing in fp32 at large row offsets.
            loss.add_((scale_chunk * (row_sum.log() - target_logit)).sum())
            continue

        if compute_linear_bias_grad:
            # Accumulated in acc_dtype like eager's bias-grad scratch, and
            # committed once after the loop.
            bias_grad_acc.add_(g.sum(dim=0, dtype=acc_dtype))
        if compute_input_grad:
            torch.mm(g, linear_weight, out=grad_input.narrow(0, start, rows))
        if compute_linear_weight_grad:
            # The first chunk writes the accumulator (beta=0) rather than adding to it.
            if start == 0:
                torch.mm(g.t(), input_chunk, out=grad_linear_weight)
            else:
                grad_linear_weight.addmm_(g.t(), input_chunk)

    if compute_grads:
        torch.sum(term_buf, dim=0, out=loss)
    if compute_linear_bias_grad:
        grad_linear_bias.copy_(bias_grad_acc)

    return (
        loss.to(dtype),
        grad_input,
        grad_linear_weight,
        grad_linear_bias,
    )


# Declaration read by both the registrar below and the drift-guard test in
# test/python_native/test_override_declarations.py. `aten` ops exist by
# construction, so a bad symbol there dies on any `import torch`; a
# `torch_nn` op exists only once `_DEFINING_MODULE` has executed, which makes
# the binding a runtime property whose failures surface only where the DSL is
# installed. The test resolves these symbols with neither a GPU nor the DSL,
# so drift cannot ship.
_NAMESPACE = "torch_nn"
_DEFINING_MODULE = "torch.nn.modules.linear_cross_entropy"


# Named implementations per op. `cond` answers whether the override applies;
# this answers which of them runs when it does. `PASSTHROUGH` is reserved and
# delegates to the op's own body, so selecting it reproduces routing-only
# behaviour -- the baseline a kernel is measured against -- in any tree,
# however many kernels the table grows.
class _Variant(NamedTuple):
    # `eligible` is what the registry's `cond` evaluates for whichever variant
    # is selected, so an input a kernel cannot take never enters the override
    # and falls back through the router -- rather than entering and delegating
    # internally, which would make "the kernel ran" unobservable.
    eligible: Callable[..., bool]
    impl: Callable[..., Any]


_VARIANTS: dict[str, dict[str, _Variant]] = {
    "_linear_cross_entropy_batch_chunked": {
        # The name is carried over from the variant study that produced it:
        # recorded measurement rows are keyed on it, and labels are data.
        "fused_inplace": _Variant(
            functools.partial(_kernel_eligible, dtypes=(torch.bfloat16, torch.float16)),
            _batch_chunked_kernel,
        ),
        variants.PASSTHROUGH: _Variant(_always, _batch_chunked_passthrough),
    },
    "_linear_cross_entropy_batch_chunked_no_reduction": {
        variants.PASSTHROUGH: _Variant(_always, _no_reduction_passthrough),
    },
}

# What runs when nothing is selected. Promoting a kernel to default is an edit
# here; the OpInfo entries follow it with no test changes, since they exercise
# whichever variant is default.
#
# `passthrough` is reserved and is NOT the fallback path -- an ineligible input
# falls back through the ROUTER, via `cond`, without entering the override at
# all. It earns its keep three other ways: the sweep's `route` stage, which
# separates routing cost from kernel effect; the live registration that the next
# kernels land into as new variants; and the portable routing test plus the
# per-process kill switch.
_DEFAULT_VARIANTS: dict[str, str] = {
    "_linear_cross_entropy_batch_chunked": "fused_inplace",
    "_linear_cross_entropy_batch_chunked_no_reduction": variants.PASSTHROUGH,
}


def _selected_name(op_symbol: str) -> str:
    return variants.get_variant(
        f"{_NAMESPACE}::{op_symbol}", _DEFAULT_VARIANTS[op_symbol]
    )


def _make_variant_cond(op_symbol: str) -> Callable[..., bool]:
    def cond(*args: Any, **kwargs: Any) -> bool:
        selected = _VARIANTS[op_symbol].get(_selected_name(op_symbol))
        # An unknown name routes anyway, so the impl can raise naming the
        # declared variants; returning False here would spend a typo as a
        # silent fall back to the op.
        return True if selected is None else selected.eligible(*args, **kwargs)

    return cond


def _make_variant_impl(op_symbol: str) -> Callable[..., Any]:
    def impl(*args: Any, **kwargs: Any) -> Any:
        name = _selected_name(op_symbol)
        selected = _VARIANTS[op_symbol].get(name)
        if selected is None:
            raise ValueError(
                f"unknown variant {name!r} for {_NAMESPACE}::{op_symbol}; this "
                f"module declares {sorted(_VARIANTS[op_symbol])}"
            )
        return selected.impl(*args, **kwargs)

    return impl


# (op_symbol, cond, impl) for every override this module installs on the
# `torch_nn` namespace. Single source of truth: the registration loop below and
# test/python_native/test_linear_cross_entropy_override.py both read it.
_OVERRIDES = tuple(
    (op_symbol, _make_variant_cond(op_symbol), _make_variant_impl(op_symbol))
    for op_symbol in _VARIANTS
)


def register_linear_cross_entropy_overrides() -> None:
    # Bail out before the import below whenever `cu.register_op_override` would
    # drop the registration anyway -- the DSL missing, disabled, or at a version
    # that is not known-good -- since the import is not free. Don't gate on
    # torch.cuda.is_available() here: it calls cuInit and poisons fork.
    if (
        not cu.runtime_available()
        or cu.check_native_jit_disabled()
        or not cu._version_is_ok()
    ):
        return

    # This import is what defines the ops named in `_OVERRIDES`. torch.nn cannot pull
    # the module in from torch/nn/modules/__init__.py (torch.library does not
    # exist that early in `import torch`), so it is imported lazily on first
    # use -- meaning the ops are absent at registration time unless something
    # asks for them first. Overrides are installed against ops that already
    # exist in the dispatcher, hence the import here.
    importlib.import_module(_DEFINING_MODULE)

    for op_symbol, cond, impl in _OVERRIDES:
        cu.register_op_override(_NAMESPACE, op_symbol, "CUDA", cond=cond, impl=impl)
