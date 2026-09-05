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
    # Mirrors the op's body: the accumulator's loss-only branch, selected by
    # reduction='none' with nothing precomputed. No checks to call -- this op
    # takes no compute_*_grad flags, and its probability-target guard lives in
    # its backward, which is registered above this dispatch key and so is not
    # displaced by the override.
    #
    # A kernel replacing this call inherits the same obligation as the scalar
    # op's impl: re-enforce the checks inside the accumulator.
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


# (op_symbol, cond, impl) for every override this module installs on the
# `torch_nn` namespace. Single source of truth: the registration loop below and
# test/python_native/test_linear_cross_entropy_override.py both read it.
@functools.cache
def _arch_supported(device_index: int = 0) -> bool:
    """Whether the kernel can run on this device: the CuTeDSL runtime's floor.

    8.0 is also where this path's own arithmetic becomes available -- the logits
    matmul asks cuBLAS for an fp32 output from low-precision inputs, which is
    rejected below it. A floor rather than a list of architectures this was
    measured on is deliberate: waiting for a measurement on every capability,
    minor revisions included, would keep the kernel off hardware it runs on
    perfectly well. What is unmeasured below sm_90 is PERFORMANCE, not
    correctness.

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
    """What the kernel variants implement. Shapes, dtypes and device only --
    no data reads, so this is safe under FakeTensor tracing.

    Everything outside this set stays with the op: `cond` evaluates the
    selected variant's eligibility, so an ineligible input never enters the
    override at all and the router falls back, which keeps "the kernel ran"
    observable in a profile.

    The architecture condition is the DSL runtime's floor rather than a list of
    architectures this was measured on -- see `_arch_supported`.

    `dtypes` is a parameter because a future variant may implement a different
    set. fp16 is in this one because eager keeps fp16 logits under `compact`
    (fp32 for bf16), so at fp16 a chunk is two bytes per element -- which this
    kernel matches by aliasing `g` into the logits rather than allocating a
    second buffer.
    """
    return (
        input.device.type == "cuda"
        and _arch_supported(input.device.index or 0)
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

    Same four outputs as the op, and the same chunking contract: the caller's
    `batch_chunk_size` bounds the per-chunk footprint. What differs is the
    middle of the loop. Eager walks the `(Bc, V)` logits buffer five more
    times (row max, subtract, gather, `exp_`, row sum, scale) and then
    scatters the one-hot correction into `grad_linear_weight` with
    `index_add_`; here one kernel reads the logits once and writes the dense
    gradient-of-logits `g`, which turns both parameter gradients into plain
    GEMMs -- `grad_input = g @ W`, `grad_linear_weight = g^T @ X` -- and
    removes the scatter entirely, so the result is also deterministic where
    eager's was not.

    The row statistics follow eager's buffer discipline rather than calling
    `torch.logsumexp`: that is a composite which materializes `self - maxes`,
    a full `(Bc, V)` fp32 temporary per chunk, which showed up as a peak-memory
    regression against eager. Shifting and exponentiating in place costs the
    same traffic and allocates nothing.

    One launch computes the row statistics and `g` together from the raw
    logits, so the shift, the `exp_` and the two reductions never run as
    separate passes -- two reads of the logits buffer and one write of `g`,
    against five more passes over it. The forward-only path (no gradient
    requested) has no `g` to write, so it takes its statistics from eager ops;
    fusing that path is B1's job, when its gate is met.

    The logits buffer stays at `acc_dtype`, eager's parity. A low-precision
    buffer was measured and dropped: it halves the buffer and the kernel's
    reads of it, but bf16 stores the logits with absolute error ~|z| * 2^-9
    while the softmax depends on their differences, so the gradients degrade
    once |z| reaches the tens -- the regime a trained head operates in.

    `g` is a VIEW of the logits storage rather than a second buffer, which
    reaches a low-precision buffer's footprint without its rounding: a chunk
    costs one `(Bc, V)` buffer and no value loses precision. Only a kernel that
    owns a whole row in one block can do this, since it has to order its writes
    against its reads; a row split across blocks would need a grid-wide
    barrier.

    At fp16 the buffer follows eager down to two bytes per element, which makes
    the aliasing overlap exact rather than half-offset, and is also why fp16 is
    only offered where `inplace_g` is set: a separate `g` would double eager's
    chunk footprint. The rounded fp16 logits carry eager's own accuracy
    behaviour, measured and recorded in the plan -- what this path does not
    inherit is the fp16 rounding of `exp()` and of the scaled gradient, which
    stay in fp32 registers here.
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

    # Installed at a backend key, so the op's body never runs and its checks
    # have to run here -- including the ones inside the accumulator this
    # replaces.
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

    # Per-row prep from the op's own functions rather than a restatement of
    # them: same clamped target, same class weight, mean divisor and ignored
    # rows. The sign is the one difference -- the accumulator consumes
    # `neg_weight_target * (onehot - softmax)` while the kernel writes
    # `s * (softmax - onehot)` -- and the negation is in place on a tensor the
    # function allocates fresh.
    target_hat = _corrected_target(target, ignore_index, num_classes)
    row_scale = _neg_weight_target(
        target_hat, target == ignore_index, weight, acc_dtype, reduction
    ).neg_()

    loss = _make_zeros((), acc_dtype, device)
    grad_input = _make_empty(input.shape, dtype, device, when=compute_input_grad)
    # Tier 0: the accumulator is fully written by the first chunk (beta=0
    # below), so zeroing it first is a (V, D) write nothing reads. The
    # empty-batch early return has no first chunk, so it keeps the zeros.
    _make_accumulator = _make_empty if num_batches > 0 else _make_zeros
    grad_linear_weight = _make_accumulator(
        linear_weight.shape, dtype, device, when=compute_linear_weight_grad
    )
    grad_linear_bias = _make_zeros(
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
    # Eager's buffer dtype, verbatim: fp16 input under `compact` keeps its
    # logits at fp16 (two bytes per element, and the softmax input rounded with
    # it); everything else upcasts to acc_dtype for softmax stability.
    logits_dtype = dtype if dtype is torch.float16 else acc_dtype
    # Allocated once and reused: the peak is one chunk of each, which is what
    # `batch_chunk_size` promises.
    logits_buf = torch.empty(
        (chunk_rows, num_classes), dtype=logits_dtype, device=device
    )
    # `g` shares the logits storage: row n's gradients occupy the first half
    # of the bytes that held its logits, so a chunk is one buffer, not two. The
    # kernel orders its writes against its reads, which is what makes that safe.
    g_alias = logits_buf.view(dtype).narrow(1, 0, num_classes)
    row_max_buf = torch.empty((chunk_rows, 1), dtype=logits_dtype, device=device)
    # The fused kernel's two (Bc,) statistics outputs.
    # One slot per row of the call: the kernel writes each row's loss
    # contribution and the reduction happens once, after the loop, instead of
    # four launches per chunk on (Bc,) data.
    term_buf = torch.empty(num_batches, dtype=acc_dtype, device=device)
    weight_t = linear_weight.t()

    for start in range(0, num_batches, chunk_rows):
        rows = min(chunk_rows, num_batches - start)
        input_chunk = input.narrow(0, start, rows)
        target_chunk = target_hat.narrow(0, start, rows)
        scale_chunk = row_scale.narrow(0, start, rows)
        logits = logits_buf.narrow(0, 0, rows)

        if linear_bias is None:
            torch.mm(input_chunk, weight_t, out_dtype=logits_dtype, out=logits)
        else:
            torch.addmm(
                linear_bias, input_chunk, weight_t, out_dtype=logits_dtype, out=logits
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
            # Shift in place by the row max, then read the target logit BEFORE
            # exponentiating -- `exp_` overwrites the shifted logits.
            row_max = row_max_buf.narrow(0, 0, rows)
            torch.amax(logits, dim=1, keepdim=True, out=row_max)
            logits.sub_(row_max)
            target_logit = logits.gather(1, target_chunk.unsqueeze(1)).squeeze(1)
            logits.exp_()
            row_sum = logits.sum(dim=1, dtype=acc_dtype)
            # Shift-invariant: log(sum exp(z - m)) - (z_T - m) == lse - z_T.
            loss.add_((scale_chunk * (row_sum.log() - target_logit)).sum())
            continue

        if compute_linear_bias_grad:
            # Accumulated in acc_dtype like eager's bias-grad scratch, and
            # committed once after the loop.
            bias_grad_acc.add_(g.sum(dim=0, dtype=acc_dtype))
        if compute_input_grad:
            torch.mm(g, linear_weight, out=grad_input.narrow(0, start, rows))
        if compute_linear_weight_grad:
            # The first chunk WRITES the accumulator (beta=0), which is why it
            # is not zeroed above: that fill plus this chunk's read of it were
            # (V, D) of traffic nothing consumed.
            if start == 0:
                torch.mm(g.t(), input_chunk, out=grad_linear_weight)
            else:
                grad_linear_weight.addmm_(g.t(), input_chunk)

    if compute_grads:
        # `out=loss` rather than `loss.add_(...)`: nothing else writes `loss`
        # on this path, so the reduction lands directly in it.
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


_OVERRIDES = tuple(
    (op_symbol, _make_variant_cond(op_symbol), _make_variant_impl(op_symbol))
    for op_symbol in _VARIANTS
)


def register_linear_cross_entropy_overrides() -> None:
    # Bail out before the import below when the DSL is unavailable or disabled;
    # cu.register_op_override would drop the registration anyway, and the
    # import is not free. Don't gate on torch.cuda.is_available() here -- it
    # calls cuInit and poisons fork.
    if not cu.runtime_available() or cu.check_native_jit_disabled():
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
