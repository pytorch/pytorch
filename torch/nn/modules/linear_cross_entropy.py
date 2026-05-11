import dataclasses
from functools import cached_property
from typing import Literal

import torch


__all__ = ["LinearCrossEntropyOptions"]


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

    .. note::

        **Memory characteristics — when chunking is worth it**

        Letting ``B = batch_chunk_size``, ``N = num_batches``, ``C =
        num_classes``, ``F = in_features``, the chunked implementation
        on CUDA has approximate peak memory:

        - ``"memory"``: ``6BC + 6CF + 4BF + 2NF`` bytes
        - ``"accurate"``: ``4BC + 10CF + 4BF + 4NF`` bytes
        - ``"lowmemory"``: ``6BC + 2CF + 4BF + 2NF`` bytes

        The reference path (``options=None``) needs ``4NC + 2NF +
        2CF`` bytes — the full logits tensor + saved softmax + grad
        outputs.

        The chunked path's value is the absence of the ``4NC`` term;
        it pays a fixed ``O(CF)`` overhead instead. With
        ``chunking_method="aspect_ratio"`` the heuristic ties ``B``
        to ``N * F / C`` so the chunked op's memory grows as
        ``O(NF + CF)`` rather than ``O(NC)``. Crossover with the
        reference is around ``num_batches ≈ in_features``:

        - ``num_batches < in_features`` — prefer ``options=None``.
          The reference path is dramatically cheaper for short
          sequences or small batches (e.g., test-sized inputs).
        - ``num_batches ≥ in_features`` — chunking is competitive,
          and the win grows linearly with ``num_batches`` from there.
          For long-context training (``num_batches`` in the 10k-100k
          range), chunking saves several GB.
        - ``num_classes ≤ in_features`` — chunking has no memory
          benefit; the heuristic degenerates to a single chunk and
          the fixed overhead is pure cost.

        Among the chunked policies on CUDA, ``"lowmemory"`` is the
        smallest by ``4CF`` (the dropped ``weight_grad_chunk``
        scratch); ``fp16+lowmemory`` further halves the
        ``logits_buf`` term to ``2BC``.

        For minimum memory in long-context training, combining
        ``acc_policy="lowmemory"`` with
        ``chunking_method="aspect_ratio:N"`` (``N >= 2``) is the
        closest stock configuration to `Liger-Kernel
        <https://github.com/linkedin/Liger-Kernel>`_'s Triton-fused
        linear-cross-entropy. The remaining 30%-50% memory gap to
        Liger is the cost of staying in pure PyTorch eager ops
        without a fused softmax+grad-logits kernel.

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

    acc_policy: Literal["memory", "accurate", "lowmemory", "ultralow"] = "memory"
    """Precision/memory trade-off policy.

    Controls which intermediate tensors are stored in :attr:`acc_dtype`
    versus the input dtype, and whether the per-chunk weight-gradient
    scratch buffer is materialized:

    - ``"memory"`` (default) — uses :attr:`acc_dtype` only where it's
      needed for gradient correctness; keeps a per-call ``(num_classes,
      in_features)`` scratch buffer in :attr:`acc_dtype` for accurate
      cross-chunk weight-gradient accumulation.
    - ``"accurate"`` — uses :attr:`acc_dtype` for more intermediates,
      noticeably improving input-gradient accuracy when the chunk size
      is large relative to ``num_classes``. Higher peak memory than
      ``"memory"``.
    - ``"lowmemory"`` — like ``"memory"`` but additionally drops the
      ``(num_classes, in_features)`` weight-gradient scratch buffer
      and accumulates per-chunk weight gradients directly into
      ``grad_linear_weight`` via ``addmm_``. cuBLAS still uses an
      fp32 internal accumulator inside the matmul, so the per-chunk
      gradient quality matches ``"memory"`` for the bulk of
      ``grad_linear_weight``; rows scattered into by the
      ``ignore_index`` correction take one extra dtype quantization
      per chunk (typically a few extra ULP). Saves
      ``num_classes * in_features * sizeof(acc_dtype)``
      bytes — roughly 800 MB for an LLM head with mixed precision.
      The optimization only applies on CUDA when :attr:`acc_dtype`
      differs from the input dtype, and on any backend when
      :attr:`acc_dtype` equals the input dtype; on non-CUDA backends
      with mixed precision, ``"lowmemory"`` silently falls back to
      ``"memory"`` (the dtype-strict non-CUDA addmm has no
      ``out_dtype=`` to bridge the dtype gap).
    - ``"ultralow"`` — like ``"lowmemory"`` but additionally keeps
      the per-chunk logits buffer at the input dtype (bf16/fp16
      instead of upcasting to fp32) and computes the softmax
      denominator with an fp32 accumulator over the bf16/fp16
      summands. Eliminates the per-iteration ``logits.to(...)``
      cast required by the input-grad addmm under
      ``"memory"``/``"lowmemory"``. Saves ``num_classes *
      batch_chunk_size * sizeof(acc_dtype)`` bytes per call (the
      ``logits_buf`` upcast) plus a similar amount in transient
      per-iteration memory — roughly closes the gap with
      Triton-fused implementations. The trade-off is precision
      regression in the softmax denominator (``log2(num_classes) *
      eps_dtype`` relative error), which propagates linearly into
      the input-grad and weight-grad. Acceptable when models have
      non-degenerate output entropy (typical training); risky for
      sharply-peaked distributions where ``softmax_denom`` is
      close to 1.

    Most policy effects (``"memory"`` vs ``"accurate"``) are visible
    only when :attr:`acc_dtype` differs from the input dtype;
    ``"lowmemory"`` and ``"ultralow"`` save memory in both regimes.
    """

    acc_dtype: torch.dtype | None = None
    """Dtype for internal accumulation. ``None`` (the default visible
    in ``repr`` / :meth:`extra_repr`) is resolved to the input dtype
    at call time — same semantics as passing the input dtype explicitly.

    Mixed-precision is currently limited to ``torch.float16``
    or ``torch.bfloat16`` input and ``acc_dtype=torch.float32``.
    """

    def __post_init__(self):
        if self.acc_policy not in {"memory", "accurate", "lowmemory", "ultralow"}:
            raise ValueError(
                f"acc_policy must be 'memory', 'accurate', 'lowmemory', or "
                f"'ultralow', got {self.acc_policy!r}"
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


def _make_empty(shape, dtype, device, when=True):
    """Allocate an uninitialized tensor (``torch.empty``-like).

    When ``when`` is false, return a rank-matching empty tensor
    (shape ``(0,) * len(shape)``) instead of allocating ``shape``.
    The returned tensor preserves the rank of ``shape`` so callers
    can treat the result uniformly regardless of ``when``.
    """
    return torch.empty(
        shape if when else (0,) * len(shape),
        device=device,
        dtype=dtype,
        requires_grad=False,
    )


def _make_zeros(shape, dtype, device, when=True):
    """Allocate a zero-initialized tensor (``torch.zeros``-like).

    Same ``when=False`` rank-matching-empty behavior as
    :func:`_make_empty`.
    """
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
    """Per-iteration tensor views for the chunked loop.

    The ``input``, ``linear_weight``, ``input_grad_logits``,
    ``input_grad_linear_weight``, ``weight_grad_input``,
    ``logits_upcast``, ``logits_downcast``, and ``grad_input_chunk``
    properties resolve which operand variant or scratch destination
    to use based on ctx dispatch flags, so the chunked-loop call
    sites read like raw tensor operations. See each property's
    docstring for the specific dispatch logic.
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
        """Input operand for the forward ``mm``: ``input_chunk_acc``
        when a non-CUDA mixed-dtype upcast is needed, else
        ``input_chunk``.
        """
        return (
            self.input_chunk_acc
            if self.ctx.forward_uses_acc_input
            else self.input_chunk
        )

    @property
    def linear_weight(self) -> torch.Tensor:
        """``linear_weight`` operand for the forward ``mm``: raw on
        the cuBLAS ``out_dtype=`` path, ``linear_weight_cast``
        otherwise.
        """
        ctx = self.ctx
        return (
            ctx.linear_weight
            if ctx.forward_uses_cuda_out_dtype
            else ctx.linear_weight_cast
        )

    @property
    def input_grad_logits(self) -> torch.Tensor:
        """``logits``-side operand for the input-grad ``addmm``:
        the pre-cast ``logits_downcast`` on CUDA mixed-dtype, else
        ``logits`` itself. ``logits_downcast`` aliases ``logits``
        when no per-iter cast is needed (see :meth:`bind_chunk`).
        """
        if self.ctx.input_grad_uses_logits_lw:
            return self.logits_downcast
        return self.logits

    @property
    def input_grad_linear_weight(self) -> torch.Tensor:
        """``linear_weight``-side operand for the input-grad
        ``addmm``: raw when paired with ``logits_downcast``, else
        ``linear_weight_cast``.
        """
        ctx = self.ctx
        return (
            ctx.linear_weight
            if ctx.input_grad_uses_logits_lw
            else ctx.linear_weight_cast
        )

    @property
    def weight_grad_input(self) -> torch.Tensor:
        """Input operand for the keep-path weight-grad ``mm``: raw
        under the CUDA mixed-dtype path (cuBLAS ``out_dtype=``
        requires matching-dtype mm operands, both at the input
        dtype), accumulator-cast otherwise (so all operands share the
        ``out=`` dtype after the upcast).
        """
        ctx = self.ctx
        if ctx.is_cuda and not ctx.weight_grad_mm_same_dtype:
            return self.input_chunk
        return self.input_chunk_acc

    @property
    def logits_upcast(self) -> torch.Tensor:
        """``logits``-side operand for the keep-path weight-grad
        ``mm``, materialized in ``acc_dtype`` when needed.

        Aliases ``logits`` on CUDA (cuBLAS' ``out_dtype=`` handles the
        upcast inside the matmul) and when ``logits`` is already in
        ``weight_grad_chunk``'s dtype. Otherwise (non-CUDA mixed-dtype
        with ``logits`` at the input dtype) copies into
        ``logits_acc_buf`` so the matmul has matching-dtype operands
        and ``out``.
        """
        ctx = self.ctx
        if ctx.is_cuda or ctx.weight_grad_mm_same_dtype:
            return self.logits
        return ctx.logits_acc_buf.narrow(0, 0, self.bchunk_size).copy_(self.logits)

    @cached_property
    def grad_input_chunk(self) -> torch.Tensor:
        """Destination view for this iter's input-grad ``addmm``:
        ``input_grad_acc_buf`` on the buf-and-copy path, a narrow
        view of ``ctx.grad_input`` on the fast path. The buf-and-copy
        commit happens in :meth:`_ChunkContext.chunks` post-yield.
        """
        ctx = self.ctx
        if ctx.alloc_input_grad_acc_buf:
            return ctx.input_grad_acc_buf.narrow(0, 0, self.bchunk_size)
        return ctx.grad_input.narrow(0, self.bchunk_start, self.bchunk_size)

    @cached_property
    def logits_downcast(self) -> torch.Tensor:
        """``logits`` cast to ``linear_weight.dtype`` for downstream
        consumers (the input-grad addmm via :attr:`input_grad_logits`
        and the inlined direct weight-grad ``addmm_``).

        When ``loop_caches_logits_downcast`` fires, lazily allocates a
        per-iter cast tensor on first access; ``cached_property``
        caches the result on the instance so subsequent accesses in
        the same iteration are O(1). Otherwise aliases ``self.logits``
        directly (no copy).

        First access must happen *after* the in-place modifications
        to ``self.logits`` in the chunked-loop body finish.
        """
        ctx = self.ctx
        if ctx.loop_caches_logits_downcast:
            return self.logits.to(ctx.linear_weight.dtype)
        return self.logits


@dataclasses.dataclass
class _ChunkContext:
    """Per-call state for the chunked loop: shape and device info, the
    full internal dtype layout, the prepared ``target`` and
    ``neg_weight_target``, the (raw and optionally cast) ``linear_weight``,
    all allocated buffers, the persistent outputs, and pre-computed
    dispatch booleans.

    Built once via ``_ChunkContext.build`` before the loop. Methods
    exist only where real dispatch lives (``mm``, ``amax``,
    ``dotgather``, ``sumexp_``, ``div``, ``mul``, ``to``);
    each one hides the dtype/device/acc_policy dispatch behind a
    single mathematical operation. Per-iteration math without
    dispatch (loss accumulation, input-grad mul + addmm) is inlined
    directly in the chunked loop body.

    Buffers that the dispatch decided are not needed are present as
    rank-matching empty tensors (``shape == (0,) * rank``) — `_make_empty`
    / `_make_zeros`'s ``when=False`` behavior. They never get accessed
    on those configs, but having them as fields means the dataclass
    surface stays uniform.
    """

    # ===== Shape and device =====
    dtype: torch.dtype
    num_batches: int
    in_features: int
    num_classes: int
    is_cuda: bool
    use_acc_dtype: bool
    is_ultralow: bool

    # ===== Internal dtype layout =====
    acc_dtype: torch.dtype
    weight_chunk_dtype: torch.dtype
    grad_input_dtype: torch.dtype
    linear_weight_cast_dtype: torch.dtype

    # ===== Per-call tensors =====
    input: torch.Tensor
    # ``target`` is the original (uncorrected) target; the per-iter
    # ``target_chunk`` is sliced from :attr:`corrected_target`, which
    # replaces out-of-range ignore_index values with 0 lazily.
    target: torch.Tensor
    weight: torch.Tensor | None
    ignore_index: int
    reduction: str
    linear_weight: torch.Tensor

    # ===== Always-allocated buffers =====
    # The "when=" buffers (weight_grad_chunk, logits_acc_buf,
    # input_grad_acc_buf, input_chunk_acc_buf, grad_input,
    # grad_linear_weight) are cached_properties below — see each one's
    # body for its dispatch flag and shape/dtype.
    logits_buf: torch.Tensor
    tmp: torch.Tensor

    # ===== Persistent outputs =====
    output: torch.Tensor

    # ===== Per-call dispatch flags =====
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
        """``target == ignore_index`` — positions to exclude from the
        weighted sum. Cached on the instance so :attr:`corrected_target`
        and :attr:`neg_weight_target` share it.
        """
        return self.target == self.ignore_index

    @cached_property
    def corrected_target(self) -> torch.Tensor:
        """``self.target`` with out-of-range ``ignore_index`` values
        replaced by 0 (so downstream ``index_select`` / ``index_add_``
        calls don't index out of bounds). Aliases ``self.target``
        when ``ignore_index`` is in range.
        """
        if self.ignore_index < 0 or self.ignore_index >= self.num_classes:
            return torch.where(self._mask, 0, self.target)
        return self.target

    @cached_property
    def neg_weight_target(self) -> torch.Tensor:
        """Per-row weighting factor for the chunked loss formula,
        sign-adjusted and (for ``reduction="mean"``) normalized: each
        position holds ``-(weight[target] if weight else 1) / d`` for
        unmasked positions, ``0`` for masked positions. ``d`` is the
        sum of unmasked weight magnitudes when ``reduction="mean"``;
        for ``reduction="sum"`` the result is simply negated. The
        chunked loop reads slices of this via ``chunk.weight_chunk``.
        """
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
        """``self.linear_weight`` materialized in
        ``linear_weight_cast_dtype`` when the dispatch demands a cast
        (non-CUDA + ``use_acc_dtype``, or CUDA + ``use_acc_dtype`` +
        accurate-mode input-grad). Otherwise aliases ``linear_weight``.
        """
        if self.linear_weight_cast_dtype != self.dtype:
            return self.linear_weight.to(self.linear_weight_cast_dtype)
        return self.linear_weight

    @cached_property
    def weight_grad_chunk(self) -> torch.Tensor:
        """``(V, F)`` accumulator in ``acc_dtype`` for the keep-path
        weight-grad ``mm``. Rank-empty (and unused) when the dispatch
        chose the direct path (lowmemory / ultralow).
        """
        return _make_empty(
            (self.num_classes, self.in_features),
            self.acc_dtype,
            self.input.device,
            when=self.alloc_weight_grad_chunk,
        )

    @cached_property
    def logits_acc_buf(self) -> torch.Tensor:
        """``(B, V)`` scratch in ``acc_dtype`` for the keep-path
        ``chunk.logits_upcast`` (non-CUDA mixed-dtype only). Rank-empty
        otherwise.
        """
        return _make_empty(
            (self.logits_buf.shape[0], self.num_classes),
            self.acc_dtype,
            self.input.device,
            when=self.alloc_logits_acc_buf,
        )

    @cached_property
    def input_grad_acc_buf(self) -> torch.Tensor:
        """``(B, F)`` scratch for the input-grad ``addmm`` on the
        buf-and-copy path (CPU mixed-dtype, MPS). Rank-empty on the
        fast path where ``grad_input_chunk`` aliases ``grad_input``.
        """
        return _make_empty(
            (self.logits_buf.shape[0], self.in_features),
            self.linear_weight_cast_dtype,
            self.input.device,
            when=self.alloc_input_grad_acc_buf,
        )

    @cached_property
    def input_chunk_acc_buf(self) -> torch.Tensor:
        """``(B, F)`` buffer holding the per-chunk input in
        ``acc_dtype``. Rank-empty when the dispatch keeps the input
        at the user dtype.
        """
        return _make_empty(
            (self.logits_buf.shape[0], self.in_features),
            self.acc_dtype,
            self.input.device,
            when=self.alloc_input_chunk_acc_buf,
        )

    @cached_property
    def grad_input(self) -> torch.Tensor:
        """``input.shape`` gradient accumulator. Rank-empty (and
        ignored by autograd) when ``compute_input_grad`` is False.
        """
        return _make_empty(
            self.input.shape,
            self.grad_input_dtype,
            self.input.device,
            when=self.compute_input_grad,
        )

    @cached_property
    def grad_linear_weight(self) -> torch.Tensor:
        """``linear_weight.shape`` zero-initialized gradient
        accumulator. Rank-empty when ``compute_linear_weight_grad``
        is False.
        """
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
        # "lowmemory" follows the same dtype layout as "memory"; the
        # extra savings come from skipping weight_grad_chunk later, not
        # from a different dtype layout. "ultralow" goes further: it
        # keeps logits in the input dtype (bf16/fp16) and accumulates
        # the softmax denominator into fp32 via sum(dtype=acc_dtype),
        # decoupling the per-row weight tensor from logits_buf to
        # preserve loss/grad accumulator precision.
        is_memory_like = acc_policy in {"memory", "lowmemory", "ultralow"}
        is_ultralow = acc_policy == "ultralow"
        if use_acc_dtype:
            output_dtype = acc_dtype if dtype == torch.float16 else dtype
            grad_input_dtype = dtype if is_memory_like else acc_dtype
            # ultralow keeps logits at input dtype regardless;
            # fp16+memory(-like) does the same for fp16 only;
            # other use_acc_dtype configs upcast for softmax stability.
            logits_buf_dtype = (
                dtype
                if is_ultralow or (dtype == torch.float16 and is_memory_like)
                else acc_dtype
            )
            # weight_chunk_dtype: kept at acc_dtype under ultralow even
            # though logits_buf is at input dtype, so the per-row weight
            # used in loss accumulation and softmax-denom division
            # retains fp32 precision. For other policies, the
            # logits_buf/weight_chunk dtypes coincide.
            weight_chunk_dtype = acc_dtype if is_ultralow else logits_buf_dtype
        else:
            output_dtype = grad_input_dtype = logits_buf_dtype = weight_chunk_dtype = (
                dtype
            )

        # ===== Dispatch flags =====
        # CUDA + use_acc_dtype + memory mode does NOT need linear_weight_cast:
        # the forward mm uses out_dtype= on the original linear_weight, the
        # input-grad goes through the storage-trick path with the original
        # linear_weight, and the input-grad first-term mul gets implicit type
        # promotion through TensorIterator.
        needs_linear_weight_cast = use_acc_dtype and (
            not is_cuda or (compute_input_grad and grad_input_dtype == logits_buf_dtype)
        )
        linear_weight_cast_dtype = (
            logits_buf_dtype if needs_linear_weight_cast else dtype
        )
        alloc_weight_grad_chunk = compute_linear_weight_grad and not (
            acc_policy in {"lowmemory", "ultralow"}
            and (is_cuda or logits_buf_dtype == dtype)
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
        # The direct weight-grad path needs to cast ``temp`` (acc_dtype)
        # to grad_linear_weight.dtype (= input dtype) for index_add_
        # (see ``_ChunkContext.to``). When the per-chunk logits_buf has
        # enough bytes per row to hold the cast result
        # (``num_classes * sizeof(logits_buf_dtype) >= in_features *
        # sizeof(dtype)``), the cast writes into a narrowed view of
        # logits_buf storage instead of allocating a fresh ``(B, F)``
        # tensor per iteration.
        weight_grad_uses_logits_buf_temp = (
            not alloc_weight_grad_chunk
            and use_acc_dtype
            and num_classes * logits_buf_dtype.itemsize >= in_features * dtype.itemsize
        )

        # ``neg_weight_target``, ``linear_weight_cast``,
        # ``weight_grad_chunk``, ``logits_acc_buf``, ``input_grad_acc_buf``,
        # ``input_chunk_acc_buf``, ``grad_input``, and
        # ``grad_linear_weight`` are computed lazily by cached_properties
        # on :class:`_ChunkContext`; see the property bodies above.
        return cls(
            dtype=dtype,
            num_batches=num_batches,
            in_features=in_features,
            num_classes=num_classes,
            is_cuda=is_cuda,
            use_acc_dtype=use_acc_dtype,
            is_ultralow=is_ultralow,
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
        """Iterate the chunked loop, yielding a fresh ``_ChunkViews``
        per iteration.

        Post-yield, finalize the buf-and-copy input-grad path
        (CPU mixed-dtype / MPS): the per-chunk addmm writes into
        ``input_grad_acc_buf``, and we copy that slice into the
        corresponding rows of ``ctx.grad_input``. Skipped on the
        fast path where ``grad_input_chunk`` already aliases
        ``ctx.grad_input``.

        The chunk size is taken from ``self.logits_buf.shape[0]`` —
        i.e., the ``batch_chunk_size`` the buffers were sized for at
        build time.
        """
        batch_chunk_size = self.logits_buf.shape[0]
        for bchunk_start, bchunk_size in _chunk_iter(
            self.num_batches, batch_chunk_size
        ):
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
        """``out = mat1 @ mat2`` with dtype-aware dispatch.

        - same-dtype: plain ``torch.mm``.
        - mismatch: ``torch.mm(out_dtype=out.dtype)`` so cuBLAS
          upcasts inside the matmul (``mat1`` and ``mat2`` must
          still share dtype).
        """
        if mat1.dtype == out.dtype:
            torch.mm(mat1, mat2, out=out)
        else:
            torch.mm(mat1, mat2, out_dtype=out.dtype, out=out)

    def amax(self, x: torch.Tensor) -> torch.Tensor:
        """Returns ``x.amax(dim=1, keepdim=True)``, reusing
        ``self.tmp`` as the scratch buffer for the per-row max
        rather than allocating a fresh tensor.
        """
        out = self.tmp.narrow(0, 0, x.shape[0]).unsqueeze(1)
        torch.amax(x, dim=1, keepdim=True, out=out)
        return out

    def dotgather(
        self, weight: torch.Tensor, x: torch.Tensor, indices: torch.Tensor
    ) -> torch.Tensor:
        """Returns ``<weight, x[i, indices[i]]>`` — the weighted sum
        of per-row gathered values. Equivalent to
        ``weight.dot(x.gather(1, indices.unsqueeze(1)).squeeze(1))``,
        with a final cast to ``weight.dtype`` so the dtype-strict
        ``torch.dot`` accepts mismatched operand dtypes (no-op when
        they match).
        """
        return weight.dot(x.gather(1, indices.unsqueeze(1)).squeeze(1).to(weight.dtype))

    def sumexp_(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        """In-place exp + dim-wise sum: ``x := exp(x); return sum(x, dim)``.

        When ``self.is_ultralow`` is set, the reduction runs
        in ``self.acc_dtype`` (preserves precision when summing many
        low-precision values); otherwise in ``x``'s dtype.
        """
        return x.exp_().sum(dim, dtype=self.acc_dtype if self.is_ultralow else None)

    def div(self, num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
        """Returns ``num / den`` written into ``self.tmp`` rather than
        a fresh tensor.

        CUDA/CPU lower ``torch.div(out=)`` to a fused compute-then-cast
        kernel even when ``num``/``den`` are wider than ``self.tmp``
        (e.g. ``acc_dtype="ultralow"``). MPS lacks the cross-dtype
        ``div_true_dense_<src>_<dst>`` variant, so on MPS we materialize
        the quotient then ``copy_`` it (cross-dtype copy is supported).
        """
        factor = self.tmp.narrow(0, 0, num.shape[0])
        if self.input.device.type == "mps" and num.dtype != factor.dtype:
            factor.copy_(num / den)
        else:
            torch.div(num, den, out=factor)
        return factor

    def mul(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Element-wise ``x * w.unsqueeze(1)`` (per-row weighted x).

        Picks a destination from in-flight scratch to avoid a fresh
        allocation:

        - ``use_acc_dtype``: mutate ``x`` in place.
        - ``num_classes >= in_features``: write into a narrowed view
          of ``logits_buf`` storage.
        - otherwise: allocate a fresh tensor.
        """
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
        """Returns ``x`` cast to ``dtype``, reusing ``logits_buf``
        storage as the cast destination when possible.

        Three sub-paths, in order of work:

        - ``x.dtype == dtype``: no cast needed, return ``x`` directly.
        - ``weight_grad_uses_logits_buf_temp``: write the cast into a
          narrowed view of ``logits_buf`` storage. Saves a fresh
          ``(B, F)`` allocation per iter.
        - else: fresh ``x.to(dtype)`` allocation. Fires only when
          ``logits_buf``'s shape can't hold the cast result.
        """
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

    Per-call buffers, the (raw and optionally cast) ``linear_weight``,
    persistent outputs, and dispatch booleans live on ``_ChunkContext``;
    each iteration's narrow views and shared per-iter casts live on
    ``_ChunkViews``. The loop body just iterates phases; the dispatch
    matrix (device / dtype / acc_policy) is pre-baked into ctx
    booleans and consumed by phase methods.
    """
    # Body uses compute_input_grad / compute_linear_weight_grad as
    # passed; do not derive from input.requires_grad — composite-
    # compliance tests unwrap tensors before reaching the impl, losing
    # requires_grad metadata.
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

    # Do not ``break`` from this loop — ``ctx.chunks()`` runs a
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
        output.sub_(weight_chunk.dot(softmax_denom.log_()))

        if compute_grads:
            if compute_input_grad:
                grad_input_chunk = chunk.grad_input_chunk
                input_grad_logits = chunk.input_grad_logits
                input_grad_linear_weight = chunk.input_grad_linear_weight

                # grad_input_chunk = linear_weight_cast[target_chunk] * weight_chunk
                # (.to() handles dtype-strict torch.mul under ultralow)
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
                    # then commit to grad_lw with one sub_ — keeps the
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
                    # a (V, F) scratch (lowmemory / ultralow paths).
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
# this many gradient slots. Update if the op's signature changes — the
# backward function fails with a tuple-length mismatch if this drifts.
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
