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

    Defaults are device-aware: ``LinearCrossEntropyOptions()`` with no
    arguments leaves :attr:`acc_policy` and :attr:`chunking_method` set
    to the sentinel ``"auto"``, which is resolved at call time to a
    per-(device, dtype) recommendation. The current picks are:

    CUDA:

    - ``bfloat16`` and ``float16`` — ``acc_policy="compact"``,
      ``chunking_method="aspect_ratio:2"``. Best balance of
      throughput, transient memory, and weight-gradient accuracy
      against an fp64 reference jacobian across the swept LLM-scale
      (N, F, V) range we measured on A100.

    CPU:

    - ``bfloat16`` and ``float16`` — ``acc_policy="accurate"``,
      ``chunking_method="aspect_ratio"``. CPU has no hardware
      fp16/bf16 matmul path, so low-precision GEMMs are emulated via
      fp32. ``"accurate"`` is the only chunked policy that stages the
      bulk weight-gradient matmul through an fp32 buffer, so it gets
      native fp32 throughput; the other chunked policies hit the slow
      emulated low-precision path and run 20-50x slower at the same
      memory + accuracy.

    Unlisted (device, dtype) combinations fall back to the
    conservative ``("compact", "aspect_ratio:2")``. Power users can
    override either field explicitly; the other stays ``"auto"`` and
    is filled in at call time.

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

        - ``"balanced"``: ``6BC + 6CF + 4BF + 2NF`` bytes
        - ``"accurate"``: ``4BC + 10CF + 4BF + 4NF`` bytes
        - ``"compact"``: ``6BC + 2CF + 4BF + 2NF`` bytes

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

        Among the chunked policies on CUDA, ``"compact"`` is the
        smallest by ``4CF`` (the dropped ``weight_grad_chunk``
        scratch); ``fp16+compact`` further halves the ``logits_buf``
        term to ``2BC``.

        For minimum memory in long-context training, combine
        ``acc_policy="compact"`` with
        ``chunking_method="aspect_ratio:N"`` (``N >= 2``). The
        residual overhead vs. a fully-fused softmax+grad-logits
        kernel (e.g. a Triton custom op) is the cost of staying in
        pure PyTorch eager ops.

    Example (minimal, device-aware defaults)::

        # acc_policy and chunking_method default to "auto"; resolved
        # per device + dtype at call time.
        options = LinearCrossEntropyOptions()
        loss = linear_cross_entropy(input, weight, target, options=options)

    Example (explicit override)::

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
    """Precision/memory trade-off policy.

    Controls which intermediate tensors are stored in :attr:`acc_dtype`
    versus the input dtype, and whether the per-chunk weight-gradient
    scratch buffer is materialized:

    - ``"auto"`` (default) — resolved at call time from the input's
      device type and dtype. Currently picks:

      - CUDA + bf16 / fp16: ``"compact"`` (best balance of
        throughput, transient memory, and weight-gradient accuracy
        against an fp64 reference jacobian across the swept
        LLM-scale (N, F, V) range we measured on A100).
      - CPU + fp16 / bf16: ``"accurate"`` (the only chunked policy
        whose bulk weight-grad matmul runs in fp32 on CPU; other
        policies fall onto the emulated low-precision matmul path
        and run 20-50x slower).

      Unlisted (device, dtype) pairs fall back to ``"compact"``.
      This treats unknown device types (e.g. ``"mps"``, ``"xpu"``,
      ``"hpu"``, future accelerators) as CUDA-like — i.e. as
      having hardware-native low-precision matmul throughput. The
      assumption is correct for every accelerator backend with a
      tensor-core-equivalent matrix engine, but wrong for any
      device that emulates fp16 / bf16 GEMMs by upcasting to fp32
      (CPU, which is handled explicitly above, is the only such
      device we have benchmarked). If your backend lacks
      hardware-native low-precision matmul, pass
      ``acc_policy="accurate"`` explicitly until an entry is
      added.
    - ``"accurate"`` — uses :attr:`acc_dtype` for the broadest set
      of intermediates, noticeably improving input-gradient accuracy
      when the chunk size is large relative to ``num_classes``.
      Highest peak memory and slowest of the chunked policies.
    - ``"balanced"`` — uses :attr:`acc_dtype` only where it's
      needed for gradient correctness; keeps a per-call
      ``(num_classes, in_features)`` scratch buffer in
      :attr:`acc_dtype` for accurate cross-chunk weight-gradient
      accumulation. Same precision as ``"accurate"`` in bf16,
      noticeably less precise in fp16, but faster than ``"accurate"``
      in both. The original default "fp32-accumulators-without-
      input-grad-fp32-buffers" policy.
    - ``"compact"`` — like ``"balanced"`` but additionally drops the
      ``(num_classes, in_features)`` weight-gradient scratch buffer
      and accumulates per-chunk weight gradients directly into
      ``grad_linear_weight`` via ``addmm_``. cuBLAS still uses an
      fp32 internal accumulator inside the matmul, so the per-chunk
      gradient quality matches ``"balanced"`` for the bulk of
      ``grad_linear_weight``; rows scattered into by the
      ``ignore_index`` correction take one extra dtype quantization
      per chunk (typically a few extra ULP). Saves
      ``num_classes * in_features * sizeof(acc_dtype)``
      bytes — roughly 800 MB for an LLM head with mixed precision.
      The optimization only applies on CUDA when :attr:`acc_dtype`
      differs from the input dtype, and on any backend when
      :attr:`acc_dtype` equals the input dtype; on non-CUDA backends
      with mixed precision, ``"compact"`` silently falls back to
      ``"balanced"`` (the dtype-strict non-CUDA addmm has no
      ``out_dtype=`` to bridge the dtype gap).

    Most policy effects (``"balanced"`` vs ``"accurate"``) are
    visible only when :attr:`acc_dtype` differs from the input
    dtype; ``"compact"`` saves memory in both regimes.

    Quick reference for when to override ``acc_policy="auto"``:

    ============ ====================================================
    Policy       Use when
    ============ ====================================================
    ``accurate`` you want tighter gradient precision than the
                 default — gradient-noise-sensitive training (small
                 LR with many steps, convergence-sensitive eval).
                 Pays roughly 6x time and 1.8x transient memory on
                 CUDA vs. the ``auto`` pick. On CPU this is
                 already the ``auto`` pick.
    ``balanced`` reproducing pre-``"compact"`` numerics for
                 backward-compatibility. Otherwise dominated by
                 ``"compact"`` on CUDA and by ``"accurate"`` on CPU.
    ``compact``  forcing the CUDA-style memory-saving path on a
                 non-CUDA backend — rarely worth it (e.g. on CPU it
                 pays a 20-50x time penalty). On CUDA this is
                 already the ``auto`` pick.
    ============ ====================================================
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
        """Resolve all deferred fields against a specific call site.

        **Internal API contract** (private — leading underscore — but
        consumed at one fixed call site:
        :func:`torch.nn.functional.linear_cross_entropy`'s chunked
        dispatch path, plus a handful of direct unit-test
        invocations in ``test/test_nn.py``). End users do not call
        this; the documented entry point is to construct
        :class:`LinearCrossEntropyOptions` and pass it to
        ``linear_cross_entropy`` / :class:`LinearCrossEntropyLoss`.
        The signature and the post-conditions below are kept stable
        as part of that internal contract; if either has to change,
        update both this method and the
        :func:`torch.nn.functional.linear_cross_entropy` call site
        in the same commit.

        Args:
            num_batches: total ``N`` from the call site (after
                ``input.unsqueeze(0)`` if the user passed an
                unbatched input).
            in_features: ``F`` (the last dim of ``input``).
            num_classes: ``V`` (the leading dim of ``linear_weight``
                after K-dim flattening).
            dtype: ``input.dtype``.
            device: ``input.device``. Optional for backward
                compatibility with call sites and tests that do not
                pass it; when ``None``, ``"auto"`` resolves to
                :data:`_AUTO_FALLBACK` instead of the per-device
                pick.

        Returns:
            A new :class:`LinearCrossEntropyOptions` with:

            - ``"auto"`` sentinels in :attr:`acc_policy` and
              :attr:`chunking_method` replaced by the device +
              dtype specific defaults from :data:`_AUTO_DEFAULTS`,
              or :data:`_AUTO_FALLBACK` if the pair is not listed
              (or ``device`` is ``None``);
            - :attr:`batch_chunk_size` populated: if the caller
              passed it explicitly, it is honoured (and the
              ``"auto"`` chunking heuristic is disabled to avoid
              spurious conflicts); otherwise it is computed from
              the resolved :attr:`chunking_method` via
              :meth:`_compute_batch_chunk_size`;
            - :attr:`acc_dtype` resolved: ``None`` → ``dtype``;
              other values left intact.

            The returned instance is fully concrete (no remaining
            sentinels, no ``None`` defaults that need to be touched
            by the op body).
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


# Per-(device_type, input_dtype) chunked defaults. Looked up by
# ``LinearCrossEntropyOptions._adjust`` when ``acc_policy`` or
# ``chunking_method`` is the sentinel ``"auto"``. Each entry maps to
# ``(acc_policy, chunking_method)``.
#
# Add entries as new platforms get characterized; combinations not
# present here fall through to ``_AUTO_FALLBACK``. Recommendations are
# established by sweeping ``(N, F, V, dtype)`` against an external
# fused-kernel reference (e.g. Liger) and against an fp64
# reference-jacobian computed with ``options=None``, then picking the
# Pareto-best ``(acc_policy, chunking_method)`` on the composite
# ``time * transient_peak_memory`` axis at acceptable weight-grad
# precision.
#
# The chosen configuration should (a) avoid catastrophic weight-grad
# error in any swept (N, F, V) regime, (b) sit at <= 1x liger's
# transient memory, and (c) stay within ~25% of liger's wall time.
#
# CUDA picks: ``compact`` for both bf16 and fp16.
#
# bf16: ``compact`` measurably beats every alternative on the
# composite cost / gradient-quality trade in A100 sweeps (matches
# Liger throughput at ~25% less transient memory and ~10x tighter
# weight-grad error in bf16; sits at the liger noise floor in fp16
# at equivalent cost). The single CUDA policy across dtypes keeps
# the mental model simple.
#
# CPU picks:
#
# CPU has no hardware fp16/bf16 matmul path: low-precision GEMMs are
# emulated by upcasting tiles to fp32, which is dramatically slower
# than a native fp32 GEMM via MKL/oneDNN. ``accurate`` is the only
# chunked policy that explicitly stages the bulk weight-gradient
# matmul through an fp32 buffer (``weight_grad_chunk``), so on CPU it
# runs the GEMM at native fp32 throughput and beats every other
# chunked policy by 20-50x on time. ``balanced`` / ``compact`` do the
# bulk matmul in the input dtype, hitting the emulation slow path.
# ``aspect_ratio`` (no factor suffix) is picked because at the
# CPU-sized sweep points it naturally degenerates to a single chunk
# when V <= F, avoiding loop overhead; when it does split, the chunk
# size is large enough to keep MKL throughput high.
_AUTO_DEFAULTS: dict[tuple[str, torch.dtype], tuple[str, str]] = {
    ("cuda", torch.bfloat16): ("compact", "aspect_ratio:2"),
    ("cuda", torch.float16): ("compact", "aspect_ratio:2"),
    ("cpu", torch.bfloat16): ("accurate", "aspect_ratio"),
    ("cpu", torch.float16): ("accurate", "aspect_ratio"),
}
_AUTO_FALLBACK: tuple[str, str] = ("compact", "aspect_ratio:2")


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
    rank-matching empty tensors (``shape == (0,) * rank``) -- `_make_empty`
    / `_make_zeros`'s ``when=False`` behavior. They never get accessed
    on those configs, but having them as fields means the dataclass
    surface stays uniform.

    TODO (maintainability): the dispatch matrix flowing through the
    boolean flags below has grown to ~14 entries (``is_cuda``,
    ``use_acc_dtype``, ``compute_input_grad``,
    ``compute_linear_weight_grad``, ``alloc_*``, ``forward_uses_*``,
    ``weight_grad_*``, ``input_grad_uses_*``,
    ``loop_caches_*``). Debugging a wrong combination is currently
    a several-property-trace exercise. A strategy / class-hierarchy
    pattern -- one subclass per ``(device_type, dtype, acc_policy)``
    cell of the dispatch matrix, with phase methods (``mm``,
    ``amax``, ``dotgather``, ``sumexp_``, ``div``, ``mul``, ``to``)
    overridden where they differ -- would replace the bool flags
    with named cells of a small enum-keyed table. Deferred from
    this PR to keep the diff scoped to the chunked-op functionality
    + auto-dispatch infrastructure; see review thread for context.
    """

    # ===== Shape and device =====
    dtype: torch.dtype
    num_batches: int
    in_features: int
    num_classes: int
    is_cuda: bool
    use_acc_dtype: bool

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
    # grad_linear_weight) are cached_properties below -- see each one's
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
        chose the direct path (compact).
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
            # ``weight_chunk_dtype`` is always ``acc_dtype`` under
            # mixed precision: the per-row weight is ``1/N`` for
            # ``reduction="mean"``; at large ``num_batches``
            # (e.g. N >= 65536) ``1/N`` is subnormal in fp16 (below
            # the smallest fp16 normal ~6.1e-5) and any fp16-domain
            # dot / mul against it loses orders of magnitude of
            # precision, which silently corrupts both the loss
            # accumulator and the downstream gradient term
            # ``logits *= weight_chunk / softmax_denom``. The
            # acc_dtype storage cost is trivial (B * 4 bytes per
            # chunk).
            weight_chunk_dtype = acc_dtype
        else:
            output_dtype = grad_input_dtype = logits_buf_dtype = weight_chunk_dtype = (
                dtype
            )

        # ===== Dispatch flags =====
        # CUDA + use_acc_dtype + balanced mode does NOT need
        # linear_weight_cast: the forward mm uses out_dtype= on the
        # original linear_weight, the input-grad goes through the
        # storage-trick path with the original linear_weight, and the
        # input-grad first-term mul gets implicit type promotion
        # through TensorIterator.
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

        Always returns the reduced result in :attr:`acc_dtype` so the
        downstream ``log(softmax_denom)`` and
        ``weight_chunk / softmax_denom`` operations carry the wider
        precision even when ``x`` is fp16/bf16. Without this, the
        reduced ``softmax_denom`` can fall below fp16's representable
        range for rows with widely-spread logits (long-vocab inputs
        with extreme per-row distributions); casting back to fp16
        would round those small values to zero, after which
        ``log(0) = -inf`` poisons the loss accumulator and the
        division in the gradient path produces ``inf`` / ``nan``.
        Keeping the result in ``acc_dtype`` lets every subsequent
        op see the precise denominator. Downstream call sites that
        need a same-dtype op with ``weight_chunk`` (e.g. the
        ``weight_chunk.dot(softmax_denom.log_())`` term) cast
        ``weight_chunk`` up to ``softmax_denom.dtype`` at the call
        site rather than rounding the denominator down.
        """
        return x.exp_().sum(dim, dtype=self.acc_dtype)

    def div(self, num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
        """Returns ``num / den`` written into ``self.tmp`` rather than
        a fresh tensor.

        CUDA/CPU lower ``torch.div(out=)`` to a fused compute-then-cast
        kernel even when ``num``/``den`` are wider than ``self.tmp``
        (e.g. ``softmax_denom`` is in ``acc_dtype`` even when
        ``self.tmp`` is in input dtype). MPS lacks the cross-dtype
        ``div_true_dense_<src>_<dst>`` variant, so on MPS we
        materialize the quotient then ``copy_`` it (cross-dtype
        copy is supported).
        """
        factor = self.tmp.narrow(0, 0, num.shape[0])
        # MPS only: fall back to a non-out= div + copy_ when any of
        # the inputs disagrees with the destination dtype. Cross-dtype
        # ``torch.div(out=)`` is fused on CUDA/CPU but missing on MPS.
        if self.input.device.type == "mps" and (
            num.dtype != factor.dtype or den.dtype != factor.dtype
        ):
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


# Private op: leading underscore on the qualified name signals that
# this is an implementation detail of
# ``torch.nn.functional.linear_cross_entropy``. The tuple-return
# shape ``(loss, grad_input, grad_linear_weight)`` is a design
# workaround, not a public API: ``torch.library.register_autograd``
# currently exposes no ``save_for_backward``-style hidden ctx state,
# so the precomputed gradients have to flow through the op's
# returns. ``setup_context`` then stashes ``grad_input`` /
# ``grad_linear_weight`` on ctx, and ``backward`` consumes them
# in-place via ``mul_``. ``functional.linear_cross_entropy``
# immediately discards ``result[1]`` / ``result[2]``.
#
# A direct caller of ``torch.ops.torch_nn._linear_cross_entropy_batch_chunked``
# who captures ``result[1]`` and uses it in a downstream autograd
# computation gets a loud failure once the subsequent backward
# re-enters this op's backward (it hits the "consumed buffer"
# guard there) -- not silent corruption. The ``_`` prefix is the
# convention that documents the privacy boundary; direct calls
# from user code are unsupported.
#
# TODO: when ``torch.library.register_autograd`` grows
# ``save_for_backward`` (or any other way to attach hidden tensor
# state to ctx), tighten the op to return only ``loss`` and move
# ``grad_input`` / ``grad_linear_weight`` off the public return
# tuple onto ctx-private state. That closes the namespace-exposure
# surface entirely.
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
    # passed; do not derive from input.requires_grad -- composite-
    # compliance tests unwrap tensors before reaching the impl, losing
    # requires_grad metadata.
    #
    # The bools are read from the tensors in
    # ``F.linear_cross_entropy`` at call time, which works fine for
    # eager and for Dynamo-traced ``torch.compile`` (Dynamo emits a
    # tensor-property guard on ``requires_grad``). For export-based
    # workflows (AOTAutograd export, AOTInductor) the bools are
    # baked at trace time and the runtime tensor's actual
    # ``requires_grad`` is ignored. The silent-corruption direction
    # there is: bool was traced as False, but the runtime tensor
    # has ``requires_grad=True`` -- the backward would return no
    # gradient for an input the caller expects one for. Convert
    # that into a loud failure here. The opposite direction (bool
    # True, runtime False) merely wastes work and is harmless.
    if not compute_input_grad and input.requires_grad:
        raise RuntimeError(
            "linear_cross_entropy chunked op: compute_input_grad was "
            "captured as False at trace time but input.requires_grad "
            "is True at runtime. This usually indicates a baked "
            "AOTAutograd / AOTInductor graph being replayed with a "
            "different ``input.requires_grad`` than it was exported "
            "with; recompile the graph with the desired "
            "``requires_grad`` setting."
        )
    if not compute_linear_weight_grad and linear_weight.requires_grad:
        raise RuntimeError(
            "linear_cross_entropy chunked op: "
            "compute_linear_weight_grad was captured as False at "
            "trace time but linear_weight.requires_grad is True at "
            "runtime. This usually indicates a baked AOTAutograd / "
            "AOTInductor graph being replayed with a different "
            "``linear_weight.requires_grad`` than it was exported "
            "with; recompile the graph with the desired "
            "``requires_grad`` setting."
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
# this many gradient slots. Update if the op's signature changes -- the
# backward function fails with a tuple-length mismatch on the first
# invocation if this drifts.
#
# Kept as a hand-maintained constant rather than derived via
# ``inspect.signature`` for a reason: there is no public introspection
# path on ``torch.library.custom_op``. The decorator wraps the user
# function in a ``CustomOpDef.__call__(*args, **kwargs)`` shim, so
# ``inspect.signature(_linear_cross_entropy_batch_chunked)`` returns
# ``(*args, **kwargs)`` and is unusable. Every working alternative
# (``._init_fn``, ``._opoverload._schema.arguments``, etc.) reaches
# into the private ``torch._library.custom_ops`` namespace -- the prior
# review iteration explicitly rejected that coupling in favour of this
# constant. Don't reopen unless ``CustomOpDef`` grows a public
# argument-introspection API.
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
