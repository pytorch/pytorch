import dataclasses
from typing import Literal

import torch


@dataclasses.dataclass(slots=True, frozen=True)
class LinearCrossEntropyOptions:
    """Configuration for the chunked implementation of
    :func:`linear_cross_entropy`.

    The chunked implementation processes the batch dimension in
    pieces, so the full ``(num_batches, num_classes)`` logits tensor
    is never materialized at once. The trade-off is a small amount of
    kernel-launch overhead in exchange for substantially lower peak
    memory — useful when ``num_classes`` is large relative to
    ``in_features`` (e.g. an LLM vocabulary head against a
    comparatively small hidden state).

    Pass ``options=None`` to :func:`linear_cross_entropy` to use the
    reference implementation (full ``logits = linear(input, weight)``
    followed by :func:`cross_entropy`). Pass an instance of this class
    to opt into the chunked path.

    The chunked path supports a subset of
    :func:`linear_cross_entropy`'s features: ``reduction`` must be
    ``"mean"`` or ``"sum"``, ``label_smoothing`` must be ``0.0``,
    ``target`` must contain class indices (``torch.int64``), and the
    loss must be 2-D (no ``out_features``). If any of those does not
    hold, the call falls through to the reference implementation
    regardless of ``options``.

    Example::

        options = LinearCrossEntropyOptions(
            chunking_method="aspect_ratio:2",
            acc_dtype=torch.float32,
        )
        loss = linear_cross_entropy(
            input,
            weight,
            target,
            reduction="sum",
            options=options,
        )
    """

    allow_retain_graph: bool = False
    """Allow ``retain_graph=True`` on backward.

    When ``False`` (default), backward consumes pre-computed gradient
    buffers in place; a second ``.backward()`` raises ``RuntimeError``.

    When ``True``, the buffers are preserved at the cost of one extra
    gradient-sized allocation per call.

    Higher-order autograd (gradgrad, forward-mode AD) is unsupported.
    """

    batch_chunk_size: int | None = None
    """Number of batch rows processed per chunk.

    The op loops over ``ceil(num_batches / batch_chunk_size)`` chunks.
    Smaller values reduce peak memory but launch more kernels; the
    default ``None`` means a single chunk (no actual chunking).

    Cannot be combined with :attr:`chunking_method`: if both are set
    and the heuristic-computed chunk size disagrees with this value, a
    ``ValueError`` is raised. Pass only one.
    """

    chunking_method: str | None = None
    """Heuristic for selecting :attr:`batch_chunk_size` from input sizes.

    Supported methods:

    - "aspect_ratio" — sizes each chunk so its ``(batch_chunk_size,
      num_classes)`` logits buffer uses about the same memory as the
      full ``(num_batches, in_features)`` input.  Computed as
      ``batch_chunk_size = next_pow2(ceil(num_batches /
      ceil(num_classes / in_features)))``.  Suitable when
      ``num_classes`` is much larger than ``in_features`` (LLM
      vocabulary heads); reduces to ``next_pow2(num_batches)`` when
      ``num_classes <= in_features``.
    - ``"aspect_ratio:N"`` for ``N >= 1`` — same heuristic, then
      divides the batch chunk size by ``N``. Reduces peak memory
      roughly by a factor of ``N`` at the cost of more chunks.
    - ``None`` (default) — use :attr:`batch_chunk_size` directly.
    """

    acc_policy: Literal["memory", "accurate"] = "memory"
    """Internal-precision policy when :attr:`acc_dtype` differs from
    the input dtype.

    Controls which intermediate tensors (softmax workspace,
    input-gradient accumulator) are stored in :attr:`acc_dtype` versus
    the input dtype:

    - ``"memory"`` (default) — uses :attr:`acc_dtype` only where it's
      needed for gradient correctness. Lower peak memory.
    - ``"accurate"`` — uses :attr:`acc_dtype` for more intermediates,
      noticeably improving input-gradient accuracy when the chunk size
      is large relative to ``num_classes``. Higher peak memory.

    Has no effect when :attr:`acc_dtype` equals the input dtype.
    """

    acc_dtype: torch.dtype | None = None
    """Dtype for internal accumulation. Defaults to the input dtype.

    Mixed-precision is currently limited to ``torch.float16``
    or ``torch.bfloat16`` input and ``acc_dtype=torch.float32``.
    """

    def __post_init__(self):
        if self.acc_policy not in {"memory", "accurate"}:
            raise ValueError(
                f"acc_policy must be 'memory' or 'accurate', got {self.acc_policy!r}"
            )
        if self.chunking_method is not None:
            if ":" in self.chunking_method:
                name, factor = self.chunking_method.split(":", 1)
            else:
                name, factor = self.chunking_method, "1"
            if not (name == "aspect_ratio" and factor.isdigit() and int(factor) > 0):
                raise ValueError(
                    f"chunking_method must be 'aspect_ratio', 'aspect_ratio:N' for a positive integer N, or None, "
                    f"got {self.chunking_method!r}"
                )

    @staticmethod
    def _ceil_div(a: int, b: int) -> int:
        """ceil(a / b) for non-negative integers."""
        return -(-a // b)

    def _compute_batch_chunk_size(
        self, num_batches: int, in_features: int, num_classes: int
    ) -> int:
        """Compute batch_chunk_size from chunking_method given input shapes.

        To add a new method: extend the if/elif chain here with the
        parsing and size formula, and add the prefix to ``__post_init__``'s
        validation.
        """
        method = str(self.chunking_method)

        if method.startswith("aspect_ratio"):
            factor = int(method.split(":", 1)[1]) if ":" in method else 1
            # Liger-Kernel heuristic: size each chunk so its
            # (chunk_size, num_classes) logits buffer uses about the same
            # memory as the full (num_batches, in_features) input.
            # See LinearCrossEntropyOptions.chunking_method docstring.
            inc_factor = self._ceil_div(num_classes, in_features)
            target = self._ceil_div(num_batches, inc_factor)
            chunk_size = 1 << (target - 1).bit_length()  # next power of 2 >= target
            return min(chunk_size // factor, num_batches)

        # __post_init__ validates the method, so this is unreachable.
        raise AssertionError(f"unhandled chunking_method: {method!r}")

    def _adjust(self, num_batches, in_features, num_classes, dtype):
        """Adjust options to input sizes and dtype.

        Return a new LinearCrossEntropyOptions object with default
        chunk sizes adjusted to the actual input sizes.
        """
        if self.batch_chunk_size is None:
            batch_chunk_size = num_batches
        else:
            batch_chunk_size = min(self.batch_chunk_size, num_batches)

        if self.chunking_method is not None:
            batch_chunk_size = self._compute_batch_chunk_size(
                num_batches, in_features, num_classes
            )
            if (
                self.batch_chunk_size is not None
                and self.batch_chunk_size != batch_chunk_size
            ):
                raise ValueError(
                    f"batch_chunk_size (={self.batch_chunk_size}) and "
                    f"chunking_method ('{self.chunking_method}') give different "
                    f"chunk sizes ({self.batch_chunk_size} vs {batch_chunk_size}); "
                    f"pass only one."
                )

        if self.acc_dtype is None:
            acc_dtype = dtype
        else:
            acc_dtype = self.acc_dtype
        return dataclasses.replace(
            self, batch_chunk_size=max(1, batch_chunk_size), acc_dtype=acc_dtype
        )


def _chunk_iter(total_size, chunk_size):
    for start in range(0, total_size, chunk_size):
        yield start, min(chunk_size, total_size - start)


# Argument count of _linear_cross_entropy_batch_chunked; backward
# returns this many gradient slots. Keep in sync with the signature.
_NUM_OP_INPUTS = 13


def _linear_cross_entropy_batch_chunked_setup_context(ctx, inputs, output):
    *_, allow_retain_graph, compute_input_grad, compute_linear_weight_grad = inputs
    ctx.allow_retain_graph = allow_retain_graph
    ctx.compute_input_grad = compute_input_grad
    ctx.compute_linear_weight_grad = compute_linear_weight_grad
    _, grad_input, grad_linear_weight = output
    ctx._gi = grad_input if compute_input_grad else None
    ctx._gw = grad_linear_weight if compute_linear_weight_grad else None


@torch.library.custom_op(
    "torch_nn::_linear_cross_entropy_batch_chunked", mutates_args=()
)
def _linear_cross_entropy_batch_chunked(
    input: torch.Tensor,
    linear_weight: torch.Tensor,
    target: torch.Tensor,
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns ``(loss, grad_input, grad_linear_weight)``: gradients
    are precomputed during the chunked forward loop and stashed on ctx
    by ``setup_context``; backward is a single multiply by the
    upstream gradient.

    The unusual "compute gradients in forward" design is what makes
    chunking save memory. The standard fprop/bprop split would need to
    retain per-chunk softmax intermediates across the forward pass
    (defeating the chunking) or recompute them in backward
    (slower). Computing the gradient in the same chunk-loop iteration
    that produces the loss term keeps peak memory at one chunk's worth
    of logits.
    """
    # Body uses compute_input_grad / compute_linear_weight_grad as
    # passed; do not derive from input.requires_grad — composite-
    # compliance tests unwrap tensors before reaching the impl, losing
    # requires_grad metadata.
    device = input.device
    dtype = input.dtype
    num_batches, in_features = input.shape
    num_classes, _ = linear_weight.shape
    # CUDA gates the out_dtype= mm fast path; other accelerators take
    # the more conservative CPU path until validated.
    is_cuda = input.device.type == "cuda"

    if dtype != acc_dtype and not (
        dtype in {torch.float16, torch.bfloat16} and acc_dtype == torch.float32
    ):
        raise RuntimeError(
            "linear_cross_entropy supports float32 acc_dtype with"
            f" float16/bfloat16 inputs, but got {acc_dtype} acc_dtype and {dtype} inputs."
        )
    use_acc_dtype = dtype != acc_dtype

    if target.dtype != torch.int64:
        raise TypeError(
            f"linear_cross_entropy: target dtype must be torch.int64, got {target.dtype}."
        )

    if label_smoothing > 0.0:
        raise NotImplementedError(
            "linear_cross_entropy does not support label smoothing"
        )

    if use_acc_dtype:
        dtypes = dict(
            output=acc_dtype if dtype == torch.float16 else dtype,
            grad_input=dtype if acc_policy == "memory" else acc_dtype,
            grad_linear_weight=dtype,
            logits_buf=dtype
            if dtype == torch.float16 and acc_policy == "memory"
            else acc_dtype,
            weight_grad_chunk=acc_dtype,
        )
    else:
        dtypes = dict(
            output=dtype,
            grad_input=dtype,
            grad_linear_weight=dtype,
            logits_buf=dtype,
            weight_grad_chunk=dtype,
        )

    # Inf/NaN at X_[:, ignore_index] propagates through the
    # masked-by-zero multiply (matches cross_entropy; differs from
    # nll_loss):
    mask = target == ignore_index
    if ignore_index < 0 or ignore_index >= num_classes:
        # map out-of-range ignore_index to 0:
        target = torch.where(mask, 0, target)
        # The correctness of this mapping is subtle: mask contains
        # the original target that ensures that selected weights
        # are masked from out-of-range ignore_index mapping
        # correctly.
    if weight is None:
        # Uniform weight=1
        neg_weight_target = (~mask).to(dtypes["logits_buf"])
    else:
        if target.numel() > weight.numel():
            neg_weight_target = torch.where(
                mask, 0, weight.to(dtypes["logits_buf"]).index_select(0, target)
            )
        else:
            neg_weight_target = torch.where(
                mask, 0, weight.index_select(0, target).to(dtypes["logits_buf"])
            )

    if reduction == "mean":
        d = neg_weight_target.sum()
        neg_weight_target.div_(torch.where(d == 0, torch.nan, -d))
    elif reduction == "sum":
        neg_weight_target.neg_()
    else:
        raise NotImplementedError(f"linear_cross_entropy does not support {reduction=}")

    # TODO: per-call allocations cannot be cached across training-loop
    # iterations from inside the op. The largest in practice are the
    # logits chunk logits_buf and the linear_weight_ cast; a user-supplied
    # workspace via LinearCrossEntropyOptions should target those
    # first, with weight_grad_chunk/logits_acc_buf/input_grad_acc_buf/tmp as follow-ups.

    # A chunk buffer used to hold logits, softmax of logits:
    logits_buf = torch.empty(
        (batch_chunk_size, num_classes),
        device=device,
        dtype=dtypes["logits_buf"],
        requires_grad=False,
    )
    logits_acc_buf_shape = (
        (batch_chunk_size, num_classes)
        if (
            compute_linear_weight_grad
            and not is_cuda
            and dtypes["logits_buf"] != acc_dtype
        )
        else (0, 0)
    )
    logits_acc_buf = torch.empty(
        logits_acc_buf_shape,
        device=device,
        dtype=acc_dtype,
        requires_grad=False,
    )

    linear_weight_ = (
        linear_weight.to(logits_buf.dtype)
        if use_acc_dtype and (compute_input_grad or not is_cuda)
        else linear_weight
    )
    grad_input_shape = input.shape if compute_input_grad else (0,)
    grad_input = torch.empty(
        grad_input_shape,
        dtype=dtypes["grad_input"],
        device=device,
        requires_grad=False,
    )
    grad_linear_weight_shape = (
        linear_weight.shape if compute_linear_weight_grad else (0,)
    )
    grad_linear_weight = torch.zeros(
        grad_linear_weight_shape,
        dtype=dtypes["grad_linear_weight"],
        device=device,
        requires_grad=False,
    )
    weight_grad_chunk_shape = (
        (num_classes, in_features) if compute_linear_weight_grad else (0, 0)
    )
    weight_grad_chunk = torch.empty(
        weight_grad_chunk_shape,
        device=device,
        dtype=dtypes["weight_grad_chunk"],
        requires_grad=False,
    )
    input_grad_acc_buf_shape = (
        (batch_chunk_size, in_features)
        if (
            compute_input_grad
            and grad_input.dtype != linear_weight_.dtype
            and not is_cuda
        )
        else (0, 0)
    )
    input_grad_acc_buf = torch.empty(
        input_grad_acc_buf_shape,
        device=device,
        dtype=linear_weight_.dtype,
        requires_grad=False,
    )
    # Chunk-sized scratch reused per iteration: first as logits_max, then as
    # tmp_chunk after logits_max is consumed by logits.sub_(logits_max):
    tmp = torch.empty(
        batch_chunk_size, device=device, dtype=dtypes["logits_buf"], requires_grad=False
    )

    output = torch.zeros((), device=device, dtype=dtypes["output"], requires_grad=False)
    if reduction == "mean" and num_batches == 0:
        output.fill_(torch.nan)

    # chunking along batches dimension:
    for bchunk_start, bchunk_size in _chunk_iter(num_batches, batch_chunk_size):
        input_chunk = input.narrow(0, bchunk_start, bchunk_size)
        target_chunk = target.narrow(0, bchunk_start, bchunk_size)
        weight_chunk = neg_weight_target.narrow(0, bchunk_start, bchunk_size)
        logits = (
            logits_buf.narrow(0, 0, bchunk_size)
            if logits_buf.shape[0] != bchunk_size
            else logits_buf
        )
        input_chunk_acc = (
            input_chunk.to(acc_dtype)
            if use_acc_dtype and compute_linear_weight_grad
            else input_chunk
        )

        # Compute output.

        if use_acc_dtype:
            if is_cuda:
                torch.mm(
                    input_chunk, linear_weight.T, out_dtype=logits_buf.dtype, out=logits
                )
            else:
                torch.mm(input_chunk.to(logits_buf.dtype), linear_weight_.T, out=logits)
        else:
            torch.mm(input_chunk, linear_weight.T, out=logits)

        logits_max = tmp.narrow(0, 0, bchunk_size).unsqueeze(1)
        torch.amax(logits, dim=1, keepdim=True, out=logits_max)

        logits.sub_(logits_max)

        # logit_at_target[i] = logits[i, target_chunk[i]] (logits: (B,
        # C), target_chunk: (B,); unsqueeze+gather+squeeze).
        logit_at_target = logits.gather(1, target_chunk.unsqueeze(1)).squeeze(1)
        # Accumulate sum_i weight_chunk[i] * logit_at_target[i];
        # weight_chunk already carries the reduction sign.
        output.add_(weight_chunk.dot(logit_at_target))

        logits.exp_()

        softmax_denom = logits.sum(dim=1)

        if compute_input_grad or compute_linear_weight_grad:
            tmp_chunk = tmp.narrow(0, 0, bchunk_size)
            torch.div(weight_chunk, softmax_denom, out=tmp_chunk)
            logits.mul_(tmp_chunk.unsqueeze(1))

        softmax_denom.log_()
        output.sub_(weight_chunk.dot(softmax_denom))

        # Compute gradients.

        if compute_input_grad:
            grad_x = grad_input.narrow(0, bchunk_start, bchunk_size)
            if grad_x.dtype == linear_weight_.dtype or is_cuda:
                torch.mul(
                    torch.index_select(linear_weight_, 0, target_chunk),
                    weight_chunk.unsqueeze(1),
                    out=grad_x,
                )
                torch.addmm(grad_x, logits, linear_weight_, alpha=-1, out=grad_x)
            else:
                # CPU: addmm is strict on dtype, and a bf16 subtract
                # loses precision; compute in acc_dtype and cast at
                # the end.
                input_grad_acc_chunk = input_grad_acc_buf.narrow(0, 0, bchunk_size)
                torch.mul(
                    torch.index_select(linear_weight_, 0, target_chunk),
                    weight_chunk.unsqueeze(1),
                    out=input_grad_acc_chunk,
                )
                torch.addmm(
                    input_grad_acc_chunk,
                    logits,
                    linear_weight_,
                    alpha=-1,
                    out=input_grad_acc_chunk,
                )
                grad_x.copy_(input_grad_acc_chunk)

        if compute_linear_weight_grad:
            if logits_buf.dtype == weight_grad_chunk.dtype:
                torch.mm(logits.T, input_chunk_acc, out=weight_grad_chunk)
            elif is_cuda:
                torch.mm(
                    logits.T,
                    input_chunk,
                    out_dtype=weight_grad_chunk.dtype,
                    out=weight_grad_chunk,
                )
            else:
                logits_acc_chunk = logits_acc_buf.narrow(0, 0, bchunk_size)
                logits_acc_chunk.copy_(logits)
                torch.mm(logits_acc_chunk.T, input_chunk_acc, out=weight_grad_chunk)
            if use_acc_dtype:
                # input_chunk_acc is a copy of input slice, so we can
                # change it inplace to reduce memory usage
                input_chunk_acc.mul_(weight_chunk.unsqueeze(1))
                weight_grad_chunk.index_add_(0, target_chunk, input_chunk_acc, alpha=-1)
            else:
                weight_grad_chunk.index_add_(
                    0,
                    target_chunk,
                    input_chunk_acc * weight_chunk.unsqueeze(1),
                    alpha=-1,
                )
            grad_linear_weight.sub_(weight_grad_chunk)

    return output.to(dtype), grad_input.to(dtype), grad_linear_weight


@_linear_cross_entropy_batch_chunked.register_fake
def _(
    input,
    linear_weight,
    target,
    weight,
    reduction,
    ignore_index,
    label_smoothing,
    batch_chunk_size,
    acc_policy: str,
    acc_dtype,
    allow_retain_graph,
    compute_input_grad,
    compute_linear_weight_grad,
):
    if reduction in {"mean", "sum"}:
        result = torch.empty((), dtype=input.dtype, device=input.device)
    else:
        raise NotImplementedError(f"linear_cross_entropy does not support {reduction=}")
    grad_input_shape = input.shape if compute_input_grad else (0,)
    grad_linear_weight_shape = (
        linear_weight.shape if compute_linear_weight_grad else (0,)
    )

    grad_input = torch.empty(
        grad_input_shape,
        dtype=input.dtype,
        device=input.device,
        requires_grad=False,
    )
    grad_linear_weight = torch.empty(
        grad_linear_weight_shape,
        dtype=linear_weight.dtype,
        device=linear_weight.device,
        requires_grad=False,
    )
    return result, grad_input, grad_linear_weight


@_linear_cross_entropy_batch_chunked.register_vmap
def _vmap(info, in_dims, *args):
    """vmap rule: loop over the vmap dimension and stack outputs.

    Each slice runs an independent op invocation, so per-sample input
    and linear_weight gradients flow naturally through autograd — the
    common ``vmap(grad(linear_cross_entropy))`` pattern produces
    per-sample weight gradients without extra plumbing.

    A fold-into-num_batches optimization for the common case (vmap
    over input, shared linear_weight) would require
    ``reduction="none"`` support inside the chunked op, which is not
    currently implemented. Until then, loop.
    """
    batch_size = info.batch_size

    moved_args = [
        arg.movedim(in_dim, 0)
        if in_dim is not None and isinstance(arg, torch.Tensor)
        else arg
        for arg, in_dim in zip(args, in_dims)
    ]

    outputs = [
        _linear_cross_entropy_batch_chunked(
            *[
                arg[i] if in_dim is not None and isinstance(arg, torch.Tensor) else arg
                for arg, in_dim in zip(moved_args, in_dims)
            ]
        )
        for i in range(batch_size)
    ]
    losses = torch.stack([o[0] for o in outputs])
    grad_inputs = torch.stack([o[1] for o in outputs])
    grad_linear_weights = torch.stack([o[2] for o in outputs])
    return (losses, grad_inputs, grad_linear_weights), (0, 0, 0)


def _linear_cross_entropy_batch_chunked_backward(ctx, grad_output, _gi_grad, _gw_grad):
    result = [None] * _NUM_OP_INPUTS
    if ctx.compute_input_grad:
        if ctx.allow_retain_graph:
            result[0] = ctx._gi * grad_output
        else:
            gi, ctx._gi = ctx._gi, None
            if gi is None:
                raise RuntimeError(
                    "linear_cross_entropy chunked backward called twice; "
                    "retain_graph=True / double backward are not supported."
                )
            result[0] = gi.mul_(grad_output)

    if ctx.compute_linear_weight_grad:
        if ctx.allow_retain_graph:
            result[1] = ctx._gw * grad_output
        else:
            gw, ctx._gw = ctx._gw, None
            if gw is None:
                raise RuntimeError(
                    "linear_cross_entropy chunked backward called twice; "
                    "retain_graph=True / double backward are not supported."
                )
            result[1] = gw.mul_(grad_output)
    return tuple(result)


_linear_cross_entropy_batch_chunked.register_autograd(
    _linear_cross_entropy_batch_chunked_backward,
    setup_context=_linear_cross_entropy_batch_chunked_setup_context,
)
