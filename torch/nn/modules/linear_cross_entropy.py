import dataclasses
from functools import cached_property
from typing import Literal

import torch


__all__ = ["LinearCrossEntropyOptions"]


@dataclasses.dataclass(slots=True, frozen=True)
class LinearCrossEntropyOptions:
    """Configuration for the chunked implementation of
    :func:`linear_cross_entropy`.

    The chunked implementation processes the batch dimension in pieces, so
    the full ``(num_batches, num_classes)`` logits tensor is never
    materialized — useful when ``num_classes`` is much larger than
    ``in_features`` (e.g. LLM vocabulary heads). Pass ``options=None`` to
    use the reference path; pass an instance of this class to opt in.

    Zero-argument ``LinearCrossEntropyOptions()`` leaves
    :attr:`acc_policy` and :attr:`chunking_method` set to ``"auto"``,
    resolved at call time from :data:`_AUTO_DEFAULTS` (per-(device, dtype)
    picks measured on A100 / x86 CPU); unlisted pairs fall back to
    ``("compact", "aspect_ratio:2")``.

    The chunked path supports a subset of :func:`linear_cross_entropy`:
    ``reduction in {"mean", "sum"}``, ``label_smoothing == 0.0``, integer
    ``target`` (class indices), and 2-D loss output (no ``out_features``).
    Any other configuration falls through to the reference path.

    Chunking is a win when ``num_batches >= in_features`` and
    ``num_classes > in_features``; below that, the reference path is
    cheaper.

    Example::

        # Device-aware defaults.
        options = LinearCrossEntropyOptions()
        loss = linear_cross_entropy(input, weight, target, options=options)

        # Explicit override.
        options = LinearCrossEntropyOptions(
            chunking_method="aspect_ratio:2",
            acc_dtype=torch.float32,
        )
    """

    allow_retain_graph: bool = False
    """Allow ``retain_graph=True`` on backward.

    When ``False`` (default), backward consumes pre-computed gradient
    buffers in place; a second ``.backward()`` raises ``RuntimeError``.

    When ``True``, the buffers are preserved at the cost of one extra
    gradient-sized allocation per call.

    Higher-order autograd (gradgrad, forward-mode AD) is unsupported.

    Under :func:`torch.compile` this field is auto-promoted to
    ``True`` regardless of what the user passed, because the default
    mode's second-backward guard relies on a Python-level ``ctx``
    mutation that Dynamo's autograd tracing does not preserve.
    :func:`torch.nn.functional.linear_cross_entropy` emits a one-time
    warning on the auto-promotion so the per-call extra-allocation
    cost stays visible.

    Example::

        options = LinearCrossEntropyOptions(allow_retain_graph=True)
        loss = linear_cross_entropy(input, weight, target, options=options)
        loss.backward(retain_graph=True)
        loss.backward()  # second pass is allowed under this option
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

    chunking_method: str | None = "auto"
    """Heuristic for selecting :attr:`batch_chunk_size` from input sizes.

    Supported methods:

    - ``"auto"`` (default) — at call time, resolves to a per-(device,
      dtype) recommendation. Combined with the ``acc_policy="auto"``
      default, zero-argument ``LinearCrossEntropyOptions()`` produces
      a configuration that is sensible for the local device. For
      unlisted (device, dtype) pairs falls back to the conservative
      ``"aspect_ratio:2"``.
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
    - ``None`` — disables the heuristic; uses :attr:`batch_chunk_size`
      directly (single chunk if that is also ``None``).
    """

    acc_policy: Literal[
        "accurate",
        "balanced",
        "compact",
        "auto",
    ] = "auto"
    """Precision/memory trade-off for the chunked path. Controls which
    intermediates are kept in :attr:`acc_dtype` vs. the input dtype, and
    whether the per-chunk weight-gradient scratch buffer is materialized.

    - ``"auto"`` (default) — per-(device, dtype) pick from
      :data:`_AUTO_DEFAULTS`; unlisted pairs fall back to ``"compact"``.
      The fallback assumes a CUDA-like backend with hardware-native
      low-precision matmul; pass ``"accurate"`` explicitly on backends
      that emulate fp16/bf16 GEMMs via fp32 upcast.
    - ``"accurate"`` — broadest use of :attr:`acc_dtype`; noticeably
      better input-grad accuracy when chunk size is large relative to
      ``num_classes``. Highest peak memory and slowest of the chunked
      policies on CUDA. Only chunked policy whose weight-grad matmul
      runs in fp32 on CPU (other policies hit CPU's emulated
      low-precision path, ~20-50x slower).
    - ``"balanced"`` — :attr:`acc_dtype` only where needed for gradient
      correctness; keeps a ``(num_classes, in_features)``
      :attr:`acc_dtype` scratch for cross-chunk weight-grad accumulation.
      Same precision as ``"accurate"`` in bf16, slightly looser in fp16,
      faster than ``"accurate"`` in both.
    - ``"compact"`` — like ``"balanced"`` but drops the weight-grad
      scratch and accumulates per-chunk directly via ``addmm_`` (cuBLAS
      uses an fp32 internal accumulator, so bulk precision matches
      ``"balanced"``). Saves ``num_classes * in_features *
      sizeof(acc_dtype)`` — typically several hundred MB for an LLM
      head. On non-CUDA mixed-precision falls back to ``"balanced"``.

    Policy effects (``"balanced"`` vs ``"accurate"``) are visible only
    when :attr:`acc_dtype` differs from the input dtype; ``"compact"``
    saves memory in both regimes.

    Override ``"auto"`` when you need tighter gradient precision than
    the default (``"accurate"``), pre-``"compact"`` numerics for
    backward-compat (``"balanced"``), or the CUDA-style memory path on
    a non-CUDA backend (``"compact"``, rarely worth it on CPU).
    """

    acc_dtype: torch.dtype | None = None
    """Dtype for internal accumulation. ``None`` (the default visible
    in ``repr`` / :meth:`extra_repr`) is resolved to the input dtype
    at call time — same semantics as passing the input dtype explicitly.

    Mixed-precision is currently limited to ``torch.float16``
    or ``torch.bfloat16`` input and ``acc_dtype=torch.float32``.
    """

    def __post_init__(self):
        if self.acc_policy not in {
            "auto",
            "accurate",
            "balanced",
            "compact",
        }:
            raise ValueError(
                f"acc_policy must be 'auto', 'accurate', 'balanced', "
                f"or 'compact', got {self.acc_policy!r}"
            )
        if self.chunking_method is not None and self.chunking_method != "auto":
            if ":" in self.chunking_method:
                name, factor = self.chunking_method.split(":", 1)
            else:
                name, factor = self.chunking_method, "1"
            if not (name == "aspect_ratio" and factor.isdigit() and int(factor) > 0):
                raise ValueError(
                    f"chunking_method must be 'auto', 'aspect_ratio', "
                    f"'aspect_ratio:N' for a positive integer N, or None, "
                    f"got {self.chunking_method!r}"
                )

    @staticmethod
    def _ceil_div(a: int, b: int) -> int:
        """ceil(a / b) for non-negative integers."""
        return -(-a // b)

    def _compute_batch_chunk_size(
        self,
        num_batches: int,
        in_features: int,
        num_classes: int,
        method: str | None = None,
    ) -> int:
        """Compute batch_chunk_size from chunking_method given input shapes.

        Pass ``method`` explicitly to use a post-``_adjust`` resolved
        value (e.g., when ``self.chunking_method == "auto"``); by
        default falls back to ``self.chunking_method``.

        To add a new method: extend the if/elif chain here with the
        parsing and size formula, and add the prefix to ``__post_init__``'s
        validation.
        """
        method = str(method if method is not None else self.chunking_method)

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

    def _adjust(self, num_batches, in_features, num_classes, dtype, device=None):
        """Resolve ``"auto"`` sentinels and ``None`` defaults against a
        specific call site; returns a fully concrete options instance.

        Internal API consumed by ``F.linear_cross_entropy``'s chunked
        dispatch and a handful of test call sites. ``device=None``
        forces the :data:`_AUTO_FALLBACK` pick instead of the
        per-device one.
        """
        acc_policy = self.acc_policy
        chunking_method = self.chunking_method
        # If the caller passed an explicit ``batch_chunk_size`` (a
        # specific-intent signal), an "auto" chunking heuristic would
        # likely disagree with their chunk size and trigger the
        # batch_chunk_size/chunking_method conflict check below. Honour
        # the explicit chunk size by disabling the heuristic in that
        # case; acc_policy="auto" still resolves normally.
        if chunking_method == "auto" and self.batch_chunk_size is not None:
            chunking_method = None
        if acc_policy == "auto" or chunking_method == "auto":
            if device is not None:
                ap, cm = _AUTO_DEFAULTS.get(
                    (device.type, dtype),
                    _AUTO_FALLBACK,
                )
            else:
                ap, cm = _AUTO_FALLBACK
            if acc_policy == "auto":
                acc_policy = ap
            if chunking_method == "auto":
                chunking_method = cm

        if self.batch_chunk_size is None:
            batch_chunk_size = num_batches
        else:
            batch_chunk_size = min(self.batch_chunk_size, num_batches)

        if chunking_method is not None:
            batch_chunk_size = self._compute_batch_chunk_size(
                num_batches,
                in_features,
                num_classes,
                chunking_method,
            )
            if (
                self.batch_chunk_size is not None
                and self.batch_chunk_size != batch_chunk_size
            ):
                raise ValueError(
                    f"batch_chunk_size (={self.batch_chunk_size}) and "
                    f"chunking_method ('{chunking_method}') give different "
                    f"chunk sizes ({self.batch_chunk_size} vs {batch_chunk_size}); "
                    f"pass only one."
                )

        if self.acc_dtype is None:
            acc_dtype = dtype
        else:
            acc_dtype = self.acc_dtype
        return dataclasses.replace(
            self,
            acc_policy=acc_policy,
            chunking_method=chunking_method,
            batch_chunk_size=max(1, batch_chunk_size),
            acc_dtype=acc_dtype,
        )


# Per-(device_type, input_dtype) "auto" picks. Established by sweeping
# (N, F, V, dtype) against an fp64 reference jacobian, picking the
# Pareto-best (acc_policy, chunking_method) on time x transient_peak.
# CPU picks "accurate" because it is the only policy that stages the
# weight-grad matmul through fp32, avoiding CPU's slow emulated low-
# precision matmul path (20-50x speedup vs. balanced / compact).
_AUTO_DEFAULTS: dict[tuple[str, torch.dtype], tuple[str, str]] = {
    ("cuda", torch.bfloat16): ("compact", "aspect_ratio:2"),
    ("cuda", torch.float16): ("compact", "aspect_ratio:2"),
    ("cpu", torch.bfloat16): ("accurate", "aspect_ratio"),
    ("cpu", torch.float16): ("accurate", "aspect_ratio"),
}
_AUTO_FALLBACK: tuple[str, str] = ("compact", "aspect_ratio:2")


def _make_empty(shape, dtype, device, when=True):
    # When when=False, return a rank-matching empty tensor (shape
    # (0,) * len(shape)) so callers can treat the result uniformly.
    return torch.empty(
        shape if when else (0,) * len(shape),
        device=device,
        dtype=dtype,
        requires_grad=False,
    )


def _make_zeros(shape, dtype, device, when=True):
    return torch.zeros(
        shape if when else (0,) * len(shape),
        device=device,
        dtype=dtype,
        requires_grad=False,
    )


def _linear_cross_entropy_batch_chunked_setup_context(ctx, inputs, output):
    *_, allow_retain_graph, compute_input_grad, compute_linear_weight_grad = inputs
    ctx.allow_retain_graph = allow_retain_graph
    ctx.compute_input_grad = compute_input_grad
    ctx.compute_linear_weight_grad = compute_linear_weight_grad
    _, grad_input, grad_linear_weight = output
    ctx._gi = grad_input if compute_input_grad else None
    ctx._gw = grad_linear_weight if compute_linear_weight_grad else None


@dataclasses.dataclass
class _ChunkViews:
    """Per-iteration tensor views; each property picks the right operand
    variant or scratch destination from ctx dispatch flags so the
    chunked-loop call sites read like raw tensor operations.
    """

    ctx: "_ChunkContext"
    bchunk_start: int
    bchunk_size: int
    input_chunk: torch.Tensor
    target_chunk: torch.Tensor
    weight_chunk: torch.Tensor
    logits: torch.Tensor
    input_chunk_acc: torch.Tensor

    @property
    def input(self) -> torch.Tensor:
        return (
            self.input_chunk_acc
            if self.ctx.forward_uses_acc_input
            else self.input_chunk
        )

    @property
    def linear_weight(self) -> torch.Tensor:
        ctx = self.ctx
        return (
            ctx.linear_weight
            if ctx.forward_uses_cuda_out_dtype
            else ctx.linear_weight_cast
        )

    @property
    def input_grad_logits(self) -> torch.Tensor:
        if self.ctx.input_grad_uses_logits_lw:
            return self.logits_downcast
        return self.logits

    @property
    def input_grad_linear_weight(self) -> torch.Tensor:
        ctx = self.ctx
        return (
            ctx.linear_weight
            if ctx.input_grad_uses_logits_lw
            else ctx.linear_weight_cast
        )

    @property
    def weight_grad_input(self) -> torch.Tensor:
        ctx = self.ctx
        if ctx.is_cuda and not ctx.weight_grad_mm_same_dtype:
            return self.input_chunk
        return self.input_chunk_acc

    @property
    def logits_upcast(self) -> torch.Tensor:
        # On non-CUDA mixed-dtype with logits at input dtype, copy into
        # logits_acc_buf so the weight-grad mm has matching-dtype operands.
        ctx = self.ctx
        if ctx.is_cuda or ctx.weight_grad_mm_same_dtype:
            return self.logits
        return ctx.logits_acc_buf.narrow(0, 0, self.bchunk_size).copy_(self.logits)

    @cached_property
    def grad_input_chunk(self) -> torch.Tensor:
        # Buf-and-copy commit happens in ctx.chunks() post-yield.
        ctx = self.ctx
        if ctx.alloc_input_grad_acc_buf:
            return ctx.input_grad_acc_buf.narrow(0, 0, self.bchunk_size)
        return ctx.grad_input.narrow(0, self.bchunk_start, self.bchunk_size)

    @cached_property
    def logits_downcast(self) -> torch.Tensor:
        # Lazy: first access must come AFTER in-place mods to self.logits
        # in the loop body finish.
        ctx = self.ctx
        if ctx.loop_caches_logits_downcast:
            return self.logits.to(ctx.linear_weight.dtype)
        return self.logits


@dataclasses.dataclass
class _ChunkContext:
    """Per-call state for the chunked loop. Built once via
    ``_ChunkContext.build`` before the loop. Methods (``mm``, ``amax``,
    ``dotgather``, ``sumexp_``, ``div``, ``mul``, ``to``) hide
    dtype/device/acc_policy dispatch behind single math operations;
    per-iteration math without dispatch is inlined into the loop body.

    Buffers that dispatch decided are not needed are present as
    rank-matching empty tensors (``_make_empty``/``_make_zeros`` with
    ``when=False``) so the dataclass surface stays uniform.
    """

    dtype: torch.dtype
    num_batches: int
    in_features: int
    num_classes: int
    is_cuda: bool
    use_acc_dtype: bool

    acc_dtype: torch.dtype
    weight_chunk_dtype: torch.dtype
    grad_input_dtype: torch.dtype
    linear_weight_cast_dtype: torch.dtype

    input: torch.Tensor
    target: torch.Tensor  # uncorrected; per-iter slice is from corrected_target
    weight: torch.Tensor | None
    ignore_index: int
    reduction: str
    linear_weight: torch.Tensor

    # The optional "when=" buffers (weight_grad_chunk, logits_acc_buf,
    # input_grad_acc_buf, input_chunk_acc_buf, grad_input,
    # grad_linear_weight) are cached_properties below.
    logits_buf: torch.Tensor
    tmp: torch.Tensor
    output: torch.Tensor

    compute_input_grad: bool
    compute_linear_weight_grad: bool
    alloc_weight_grad_chunk: bool
    alloc_input_grad_acc_buf: bool
    alloc_input_chunk_acc_buf: bool
    alloc_logits_acc_buf: bool
    forward_uses_acc_input: bool
    forward_uses_cuda_out_dtype: bool
    weight_grad_mm_same_dtype: bool
    input_grad_uses_logits_lw: bool
    loop_caches_logits_downcast: bool
    weight_grad_uses_logits_buf_temp: bool

    @cached_property
    def _mask(self) -> torch.Tensor:
        return self.target == self.ignore_index

    @cached_property
    def corrected_target(self) -> torch.Tensor:
        # Replace out-of-range ignore_index values with 0 lazily so
        # downstream index_select / index_add_ stay in bounds.
        if self.ignore_index < 0 or self.ignore_index >= self.num_classes:
            return torch.where(self._mask, 0, self.target)
        return self.target

    @cached_property
    def neg_weight_target(self) -> torch.Tensor:
        # Per-row weighting: -(weight[target] if weight else 1) / d on
        # unmasked positions, 0 on masked positions. d = sum of unmasked
        # weights for "mean", 1 for "sum".
        mask = self._mask
        target = self.corrected_target
        weight = self.weight
        weight_chunk_dtype = self.weight_chunk_dtype
        if weight is None:
            neg_weight_target = (~mask).to(weight_chunk_dtype)
        elif target.numel() > weight.numel():
            neg_weight_target = torch.where(
                mask, 0, weight.to(weight_chunk_dtype).index_select(0, target)
            )
        else:
            neg_weight_target = torch.where(
                mask, 0, weight.index_select(0, target).to(weight_chunk_dtype)
            )
        if self.reduction == "mean":
            d = neg_weight_target.sum()
            neg_weight_target.div_(torch.where(d == 0, torch.nan, -d))
        else:  # "sum"
            neg_weight_target.neg_()
        return neg_weight_target

    @cached_property
    def linear_weight_cast(self) -> torch.Tensor:
        if self.linear_weight_cast_dtype != self.dtype:
            return self.linear_weight.to(self.linear_weight_cast_dtype)
        return self.linear_weight

    @cached_property
    def weight_grad_chunk(self) -> torch.Tensor:
        # (V, F) acc_dtype accumulator for keep-path weight-grad mm;
        # rank-empty on the direct (compact) path.
        return _make_empty(
            (self.num_classes, self.in_features),
            self.acc_dtype,
            self.input.device,
            when=self.alloc_weight_grad_chunk,
        )

    @cached_property
    def logits_acc_buf(self) -> torch.Tensor:
        # (B, V) scratch for non-CUDA mixed-dtype logits_upcast.
        return _make_empty(
            (self.logits_buf.shape[0], self.num_classes),
            self.acc_dtype,
            self.input.device,
            when=self.alloc_logits_acc_buf,
        )

    @cached_property
    def input_grad_acc_buf(self) -> torch.Tensor:
        # (B, F) scratch for input-grad addmm on the buf-and-copy path
        # (CPU mixed-dtype, MPS).
        return _make_empty(
            (self.logits_buf.shape[0], self.in_features),
            self.linear_weight_cast_dtype,
            self.input.device,
            when=self.alloc_input_grad_acc_buf,
        )

    @cached_property
    def input_chunk_acc_buf(self) -> torch.Tensor:
        return _make_empty(
            (self.logits_buf.shape[0], self.in_features),
            self.acc_dtype,
            self.input.device,
            when=self.alloc_input_chunk_acc_buf,
        )

    @cached_property
    def grad_input(self) -> torch.Tensor:
        return _make_empty(
            self.input.shape,
            self.grad_input_dtype,
            self.input.device,
            when=self.compute_input_grad,
        )

    @cached_property
    def grad_linear_weight(self) -> torch.Tensor:
        return _make_zeros(
            self.linear_weight.shape,
            self.dtype,
            self.input.device,
            when=self.compute_linear_weight_grad,
        )

    @classmethod
    def build(
        cls,
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
        compute_input_grad: bool,
        compute_linear_weight_grad: bool,
    ) -> "_ChunkContext":
        # ===== Validation =====
        if target.dtype != torch.int64:
            raise TypeError(
                f"linear_cross_entropy: target dtype must be torch.int64, got {target.dtype}."
            )
        if label_smoothing > 0.0:
            raise NotImplementedError(
                "linear_cross_entropy does not support label smoothing"
            )
        if reduction not in {"mean", "sum"}:
            raise NotImplementedError(
                f"linear_cross_entropy does not support {reduction=}"
            )

        device = input.device
        dtype = input.dtype
        num_batches, in_features = input.shape
        num_classes, _ = linear_weight.shape
        # CUDA gates the out_dtype= mm fast path; the non-CUDA path
        # (CPU, MPS, XPU, ...) routes mixed-dtype mm through explicit
        # casts until out_dtype= is validated on those backends.
        is_cuda = device.type == "cuda"
        is_mps = device.type == "mps"

        if dtype != acc_dtype and not (
            dtype in {torch.float16, torch.bfloat16} and acc_dtype == torch.float32
        ):
            raise RuntimeError(
                "linear_cross_entropy supports float32 acc_dtype with"
                f" float16/bfloat16 inputs, but got {acc_dtype} acc_dtype and {dtype} inputs."
            )
        use_acc_dtype = dtype != acc_dtype

        # ===== Internal dtype layout =====
        # "compact" follows the same dtype layout as "balanced"; the
        # extra savings come from skipping the weight_grad_chunk
        # scratch later, not from a different dtype layout.
        is_memory_like = acc_policy in {"balanced", "compact"}
        if use_acc_dtype:
            output_dtype = acc_dtype if dtype == torch.float16 else dtype
            grad_input_dtype = dtype if is_memory_like else acc_dtype
            # fp16 + memory-like keeps the per-chunk logits at input
            # dtype (fp16 native matmul is well-served and the buffer
            # is sized to 2BC instead of 4BC). Everything else under
            # mixed precision upcasts for softmax stability.
            logits_buf_dtype = (
                dtype if dtype == torch.float16 and is_memory_like else acc_dtype
            )
            # weight_chunk_dtype is always acc_dtype under mixed
            # precision: 1/N for reduction="mean" is subnormal in fp16
            # at N >= 65536, silently zeroing the loss / grad.
            weight_chunk_dtype = acc_dtype
        else:
            output_dtype = grad_input_dtype = logits_buf_dtype = weight_chunk_dtype = (
                dtype
            )

        # ===== Dispatch flags =====
        # CUDA + balanced uses cuBLAS out_dtype= directly; no cast.
        needs_linear_weight_cast = use_acc_dtype and (
            not is_cuda or (compute_input_grad and grad_input_dtype == logits_buf_dtype)
        )
        linear_weight_cast_dtype = (
            logits_buf_dtype if needs_linear_weight_cast else dtype
        )
        alloc_weight_grad_chunk = compute_linear_weight_grad and not (
            acc_policy == "compact" and (is_cuda or logits_buf_dtype == dtype)
        )
        alloc_input_grad_acc_buf = (
            compute_input_grad
            and not is_cuda
            and (grad_input_dtype != linear_weight_cast_dtype or is_mps)
        )
        alloc_input_chunk_acc_buf = use_acc_dtype and (
            compute_linear_weight_grad or (not is_cuda and dtype != logits_buf_dtype)
        )
        forward_uses_acc_input = (
            use_acc_dtype and not is_cuda and dtype != logits_buf_dtype
        )
        forward_uses_cuda_out_dtype = is_cuda and use_acc_dtype
        weight_grad_mm_same_dtype = logits_buf_dtype == acc_dtype
        # Storage-trick fires when logits' dtype differs from the input-
        # grad accumulator dtype.
        input_grad_uses_logits_lw = (
            is_cuda and compute_input_grad and logits_buf_dtype != grad_input_dtype
        )
        # Per-iter ``logits.to(linear_weight.dtype)`` shared between the
        # inlined input-grad addmm and the inlined direct weight-grad addmm_.
        loop_caches_logits_downcast = input_grad_uses_logits_lw or (
            compute_linear_weight_grad
            and not alloc_weight_grad_chunk
            and logits_buf_dtype != dtype
        )
        alloc_logits_acc_buf = (
            alloc_weight_grad_chunk and not is_cuda and logits_buf_dtype != acc_dtype
        )
        # Direct weight-grad path: reuse logits_buf storage for the
        # acc_dtype->dtype cast when its bytes/row fits the (B, F) cast.
        weight_grad_uses_logits_buf_temp = (
            not alloc_weight_grad_chunk
            and use_acc_dtype
            and num_classes * logits_buf_dtype.itemsize >= in_features * dtype.itemsize
        )

        return cls(
            dtype=dtype,
            num_batches=num_batches,
            in_features=in_features,
            num_classes=num_classes,
            is_cuda=is_cuda,
            use_acc_dtype=use_acc_dtype,
            acc_dtype=acc_dtype,
            weight_chunk_dtype=weight_chunk_dtype,
            grad_input_dtype=grad_input_dtype,
            linear_weight_cast_dtype=linear_weight_cast_dtype,
            input=input,
            target=target,
            weight=weight,
            ignore_index=ignore_index,
            reduction=reduction,
            linear_weight=linear_weight,
            logits_buf=torch.empty(
                (batch_chunk_size, num_classes),
                dtype=logits_buf_dtype,
                device=device,
                requires_grad=False,
            ),
            tmp=torch.empty(
                (batch_chunk_size,),
                dtype=logits_buf_dtype,
                device=device,
                requires_grad=False,
            ),
            output=_make_zeros((), output_dtype, device),
            compute_input_grad=compute_input_grad,
            compute_linear_weight_grad=compute_linear_weight_grad,
            alloc_weight_grad_chunk=alloc_weight_grad_chunk,
            alloc_input_grad_acc_buf=alloc_input_grad_acc_buf,
            alloc_input_chunk_acc_buf=alloc_input_chunk_acc_buf,
            alloc_logits_acc_buf=alloc_logits_acc_buf,
            forward_uses_acc_input=forward_uses_acc_input,
            forward_uses_cuda_out_dtype=forward_uses_cuda_out_dtype,
            weight_grad_mm_same_dtype=weight_grad_mm_same_dtype,
            input_grad_uses_logits_lw=input_grad_uses_logits_lw,
            loop_caches_logits_downcast=loop_caches_logits_downcast,
            weight_grad_uses_logits_buf_temp=weight_grad_uses_logits_buf_temp,
        )

    def chunks(self):
        """Yield a ``_ChunkViews`` per iter. Post-yield, commit the
        buf-and-copy input-grad slice into ``grad_input`` (skipped on
        the fast path where ``grad_input_chunk`` aliases ``grad_input``).
        """
        batch_chunk_size = self.logits_buf.shape[0]
        for bchunk_start in range(0, self.num_batches, batch_chunk_size):
            bchunk_size = min(batch_chunk_size, self.num_batches - bchunk_start)
            chunk = self.bind_chunk(bchunk_start, bchunk_size)
            yield chunk
            if self.alloc_input_grad_acc_buf:
                self.grad_input.narrow(0, bchunk_start, bchunk_size).copy_(
                    chunk.grad_input_chunk
                )

    def bind_chunk(self, bchunk_start: int, bchunk_size: int) -> _ChunkViews:
        input_chunk = self.input.narrow(0, bchunk_start, bchunk_size)
        target_chunk = self.corrected_target.narrow(0, bchunk_start, bchunk_size)
        weight_chunk = self.neg_weight_target.narrow(0, bchunk_start, bchunk_size)
        logits = self.logits_buf.narrow(0, 0, bchunk_size)
        input_chunk_acc = (
            self.input_chunk_acc_buf.narrow(0, 0, bchunk_size).copy_(input_chunk)
            if self.alloc_input_chunk_acc_buf
            else input_chunk
        )
        return _ChunkViews(
            ctx=self,
            bchunk_start=bchunk_start,
            bchunk_size=bchunk_size,
            input_chunk=input_chunk,
            target_chunk=target_chunk,
            weight_chunk=weight_chunk,
            logits=logits,
            input_chunk_acc=input_chunk_acc,
        )

    def mm(self, mat1: torch.Tensor, mat2: torch.Tensor, *, out: torch.Tensor) -> None:
        # Mismatched dtype: cuBLAS ``out_dtype=`` upcasts inside the matmul.
        if mat1.dtype == out.dtype:
            torch.mm(mat1, mat2, out=out)
        else:
            torch.mm(mat1, mat2, out_dtype=out.dtype, out=out)

    def amax(self, x: torch.Tensor) -> torch.Tensor:
        out = self.tmp.narrow(0, 0, x.shape[0]).unsqueeze(1)
        torch.amax(x, dim=1, keepdim=True, out=out)
        return out

    def dotgather(
        self, weight: torch.Tensor, x: torch.Tensor, indices: torch.Tensor
    ) -> torch.Tensor:
        return weight.dot(x.gather(1, indices.unsqueeze(1)).squeeze(1).to(weight.dtype))

    def sumexp_(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        # Result always in acc_dtype: fp16 softmax_denom can underflow
        # on rows with widely-spread logits, after which log(0)=-inf
        # poisons the loss and 1/0 the gradient.
        return x.exp_().sum(dim, dtype=self.acc_dtype)

    def div(self, num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
        factor = self.tmp.narrow(0, 0, num.shape[0])
        # MPS lacks fused cross-dtype torch.div(out=); copy_ fallback.
        if self.input.device.type == "mps" and (
            num.dtype != factor.dtype or den.dtype != factor.dtype
        ):
            factor.copy_(num / den)
        else:
            torch.div(num, den, out=factor)
        return factor

    def mul(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        # Pick a destination from in-flight scratch when possible.
        if self.use_acc_dtype:
            return x.mul_(w.unsqueeze(1))
        if self.num_classes >= self.in_features:
            return torch.mul(
                x,
                w.unsqueeze(1),
                out=self.logits_buf.narrow(0, 0, x.shape[0]).narrow(
                    1, 0, self.in_features
                ),
            )
        return x * w.unsqueeze(1)

    def to(self, x: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
        # Reuse logits_buf storage as cast destination when its bytes/row fits.
        if x.dtype == dtype:
            return x
        if self.weight_grad_uses_logits_buf_temp:
            x = (
                self.logits_buf.view(dtype)
                .narrow(0, 0, x.shape[0])
                .narrow(1, 0, x.shape[1])
                .copy_(x)
            )
            return x
        return x.to(dtype)


# Private op (leading-underscore qualified name). Returns
# ``(loss, grad_input, grad_linear_weight)`` because
# ``torch.library.register_autograd`` has no ``save_for_backward``-
# style hidden ctx state; ``setup_context`` stashes the grad tensors,
# ``backward`` consumes them in place. ``F.linear_cross_entropy``
# discards the tuple's extra slots. Direct callers who capture the
# stashed grads get a loud "consumed buffer" failure, not silent
# corruption.
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
    """Returns ``(loss, grad_input, grad_linear_weight)``. Gradients
    are precomputed during the chunked forward loop and stashed on ctx
    by ``setup_context``; backward is a single multiply by the upstream
    gradient. This "compute gradients in forward" design is what lets
    chunking save memory — the alternative (retain per-chunk softmax
    intermediates or recompute in backward) would defeat the
    chunking.
    """
    # compute_input_grad / compute_linear_weight_grad are read at call
    # time in F.linear_cross_entropy; under AOTAutograd / AOTInductor
    # they are baked at trace time and the runtime tensor's
    # requires_grad is ignored. Catch the silent-corruption direction
    # (bool False at trace, requires_grad True at runtime).
    if not compute_input_grad and input.requires_grad:
        raise RuntimeError(
            "linear_cross_entropy chunked op: compute_input_grad was False at "
            "trace time but input.requires_grad is True at runtime; recompile "
            "the graph with the desired requires_grad."
        )
    if not compute_linear_weight_grad and linear_weight.requires_grad:
        raise RuntimeError(
            "linear_cross_entropy chunked op: compute_linear_weight_grad was "
            "False at trace time but linear_weight.requires_grad is True at "
            "runtime; recompile the graph with the desired requires_grad."
        )
    ctx = _ChunkContext.build(
        input,
        linear_weight,
        target,
        weight,
        reduction,
        ignore_index,
        label_smoothing,
        batch_chunk_size,
        acc_policy,
        acc_dtype,
        compute_input_grad,
        compute_linear_weight_grad,
    )
    dtype = ctx.dtype
    output = ctx.output
    grad_input = ctx.grad_input
    grad_linear_weight = ctx.grad_linear_weight
    weight_grad_chunk = ctx.weight_grad_chunk
    linear_weight_cast = ctx.linear_weight_cast

    if reduction == "mean" and ctx.num_batches == 0:
        output.fill_(torch.nan)

    compute_grads = compute_input_grad or compute_linear_weight_grad

    # Do not ``break`` from this loop -- ``ctx.chunks()`` runs a
    # post-yield grad_input commit that the in-flight chunk needs.
    for chunk in ctx.chunks():
        logits = chunk.logits
        weight_chunk = chunk.weight_chunk
        target_chunk = chunk.target_chunk

        ctx.mm(chunk.input, chunk.linear_weight.T, out=logits)
        logits.sub_(ctx.amax(logits))  # softmax stability
        # output += <weight_chunk, logits[:, target_chunk]>
        output.add_(ctx.dotgather(weight_chunk, logits, target_chunk))
        softmax_denom = ctx.sumexp_(logits, dim=1)
        if compute_grads:
            # logits *= weight_chunk / softmax_denom         (= softmax(logits) * w)
            # MPS: in-place mul_ on a narrow view of logits_buf doesn't
            # propagate to parent storage; the out= form does.
            torch.mul(
                logits,
                ctx.div(weight_chunk, softmax_denom).unsqueeze(1),
                out=logits,
            )
        # output -= <weight_chunk, log(softmax_denom)>
        # softmax_denom is always in acc_dtype (see ``sumexp_``); the
        # dot needs matching dtypes, so promote weight_chunk up
        # rather than rounding the wider denominator down.
        output.sub_(weight_chunk.to(softmax_denom.dtype).dot(softmax_denom.log_()))

        if compute_grads:
            if compute_input_grad:
                grad_input_chunk = chunk.grad_input_chunk
                input_grad_logits = chunk.input_grad_logits
                input_grad_linear_weight = chunk.input_grad_linear_weight

                # grad_input_chunk = linear_weight_cast[target_chunk] * weight_chunk
                # (.to() handles dtype-strict torch.mul on backends
                # that don't support cross-dtype out=)
                torch.mul(
                    torch.index_select(linear_weight_cast, 0, target_chunk),
                    weight_chunk.to(grad_input_chunk.dtype).unsqueeze(1),
                    out=grad_input_chunk,
                )
                # grad_input_chunk -= input_grad_logits @ input_grad_linear_weight
                torch.addmm(
                    grad_input_chunk,
                    input_grad_logits,
                    input_grad_linear_weight,
                    alpha=-1,
                    out=grad_input_chunk,
                )
            if compute_linear_weight_grad:
                input_chunk_acc = chunk.input_chunk_acc

                # grad_linear_weight += per-chunk weight grad
                # (= -logits.T @ input + input*weight scattered at target rows)
                if ctx.alloc_weight_grad_chunk:
                    # Build the per-chunk weight grad in acc_dtype scratch,
                    # then commit to grad_lw with one sub_ -- keeps the
                    # bulk-minus-correction subtraction in fp32 for precision.
                    logits_upcast = chunk.logits_upcast
                    weight_grad_input = chunk.weight_grad_input
                    ctx.mm(
                        logits_upcast.T,
                        weight_grad_input,
                        out=weight_grad_chunk,
                    )
                    temp = ctx.mul(input_chunk_acc, weight_chunk)
                    weight_grad_chunk.index_add_(0, target_chunk, temp, alpha=-1)
                    grad_linear_weight.sub_(weight_grad_chunk)
                else:
                    # Stream bulk + correction directly into grad_lw without
                    # a (V, F) scratch (the compact path).
                    grad_linear_weight.addmm_(
                        chunk.logits_downcast.T, chunk.input_chunk, alpha=-1
                    )
                    temp = ctx.to(
                        ctx.mul(input_chunk_acc, weight_chunk),
                        dtype=grad_linear_weight.dtype,
                    )
                    grad_linear_weight.index_add_(0, target_chunk, temp, alpha=1)

    return (
        output.to(dtype),
        grad_input.to(dtype),
        grad_linear_weight,
    )


# Argument count of _linear_cross_entropy_batch_chunked; backward returns
# this many gradient slots. Hand-maintained because torch.library.custom_op
# has no public arg-introspection path (inspect.signature returns the
# CustomOpDef wrapper's *args/**kwargs). Update on signature change.
_NUM_OP_INPUTS = 13


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
    grad_input_shape = input.shape if compute_input_grad else (0, 0)
    grad_linear_weight_shape = (
        linear_weight.shape if compute_linear_weight_grad else (0, 0)
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
    Per-sample input/linear_weight grads flow through autograd
    naturally. A fold-into-num_batches fast path would need
    ``reduction="none"`` support inside the chunked op.
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
