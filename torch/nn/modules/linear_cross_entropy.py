import dataclasses
from typing import Literal, NamedTuple

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
      ``num_classes * in_features * sizeof(weight_grad_chunk_dtype)``
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
    """Dtype for internal accumulation. Defaults to the input dtype.

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


class _LinearCrossEntropyChunkedSetup(NamedTuple):
    """Pre-loop state computed by ``_linear_cross_entropy_setup``:
    shape/device info, the dtype layout for every internal buffer,
    and the prepared ``neg_weight_target`` plus its (possibly
    re-mapped) ``target`` indices.

    The fields named ``*_dtype`` drive the buffer allocations and
    dispatch flags in ``_linear_cross_entropy_batch_chunked``. The
    boolean flags (``is_cuda``, ``is_mps``, ``use_acc_dtype``,
    ``is_memory_like``, ``is_ultralow``) are reused throughout the
    chunked loop's dispatch. ``linear_weight_cast`` is computed by
    the caller after this setup since it depends on
    ``compute_input_grad`` dispatch and is naturally co-located
    with the buffer allocation block.
    """

    device: torch.device
    dtype: torch.dtype
    num_batches: int
    in_features: int
    num_classes: int
    is_cuda: bool
    is_mps: bool
    use_acc_dtype: bool
    is_memory_like: bool
    is_ultralow: bool
    output_dtype: torch.dtype
    grad_input_dtype: torch.dtype
    grad_linear_weight_dtype: torch.dtype
    logits_buf_dtype: torch.dtype
    weight_grad_chunk_dtype: torch.dtype
    weight_chunk_dtype: torch.dtype
    target: torch.Tensor
    neg_weight_target: torch.Tensor


def _linear_cross_entropy_setup(
    input: torch.Tensor,
    linear_weight: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None,
    reduction: str,
    ignore_index: int,
    label_smoothing: float,
    acc_policy: str,
    acc_dtype: torch.dtype,
) -> _LinearCrossEntropyChunkedSetup:
    """Validate inputs, compute the internal dtype layout, and build
    ``neg_weight_target``.

    Pulled out of ``_linear_cross_entropy_batch_chunked`` so the main
    function focuses on buffer allocation and the chunked loop.
    """
    # ===== Setup =====
    device = input.device
    dtype = input.dtype
    num_batches, in_features = input.shape
    num_classes, _ = linear_weight.shape
    # CUDA gates the out_dtype= mm fast path; the non-CUDA path
    # (CPU, MPS, XPU, ...) routes mixed-dtype mm through explicit casts
    # until out_dtype= is validated on those backends.
    is_cuda = input.device.type == "cuda"
    is_mps = input.device.type == "mps"

    # ===== Validation =====
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

    if reduction not in {"mean", "sum"}:
        raise NotImplementedError(f"linear_cross_entropy does not support {reduction=}")

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
        grad_linear_weight_dtype = dtype
        # ultralow keeps logits at input dtype regardless;
        # fp16+memory(-like) does the same for fp16 only;
        # other use_acc_dtype configs upcast for softmax stability.
        logits_buf_dtype = (
            dtype
            if is_ultralow or (dtype == torch.float16 and is_memory_like)
            else acc_dtype
        )
        weight_grad_chunk_dtype = acc_dtype
        # weight_chunk_dtype: kept at acc_dtype under ultralow even
        # though logits_buf is at input dtype, so the per-row weight
        # used in loss accumulation and softmax-denom division
        # retains fp32 precision. For other policies, the
        # logits_buf/weight_chunk dtypes coincide.
        weight_chunk_dtype = acc_dtype if is_ultralow else logits_buf_dtype
    else:
        output_dtype = grad_input_dtype = grad_linear_weight_dtype = (
            logits_buf_dtype
        ) = weight_grad_chunk_dtype = weight_chunk_dtype = dtype

    # ===== Build neg_weight_target =====
    # Inf/NaN at X_[:, ignore_index] propagates through the
    # masked-by-zero multiply (matches cross_entropy; differs from
    # nll_loss). The mask is built from the original target so that
    # out-of-range ignore_index values (mapped to 0 below) still get
    # zeroed out correctly.
    mask = target == ignore_index
    if ignore_index < 0 or ignore_index >= num_classes:
        target = torch.where(mask, 0, target)
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

    if reduction == "mean":
        d = neg_weight_target.sum()
        neg_weight_target.div_(torch.where(d == 0, torch.nan, -d))
    else:  # "sum"
        neg_weight_target.neg_()

    return _LinearCrossEntropyChunkedSetup(
        device=device,
        dtype=dtype,
        num_batches=num_batches,
        in_features=in_features,
        num_classes=num_classes,
        is_cuda=is_cuda,
        is_mps=is_mps,
        use_acc_dtype=use_acc_dtype,
        is_memory_like=is_memory_like,
        is_ultralow=is_ultralow,
        output_dtype=output_dtype,
        grad_input_dtype=grad_input_dtype,
        grad_linear_weight_dtype=grad_linear_weight_dtype,
        logits_buf_dtype=logits_buf_dtype,
        weight_grad_chunk_dtype=weight_grad_chunk_dtype,
        weight_chunk_dtype=weight_chunk_dtype,
        target=target,
        neg_weight_target=neg_weight_target,
    )


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
    setup = _linear_cross_entropy_setup(
        input,
        linear_weight,
        target,
        weight,
        reduction,
        ignore_index,
        label_smoothing,
        acc_policy,
        acc_dtype,
    )
    device = setup.device
    dtype = setup.dtype
    num_batches = setup.num_batches
    in_features = setup.in_features
    num_classes = setup.num_classes
    is_cuda = setup.is_cuda
    is_mps = setup.is_mps
    use_acc_dtype = setup.use_acc_dtype
    is_ultralow = setup.is_ultralow
    output_dtype = setup.output_dtype
    grad_input_dtype = setup.grad_input_dtype
    grad_linear_weight_dtype = setup.grad_linear_weight_dtype
    logits_buf_dtype = setup.logits_buf_dtype
    weight_grad_chunk_dtype = setup.weight_grad_chunk_dtype
    target = setup.target
    neg_weight_target = setup.neg_weight_target

    # linear_weight_cast: linear_weight materialized in
    # logits_buf_dtype when actually needed:
    # - non-CUDA + use_acc_dtype: forward mm operands and the
    #   buf-and-copy input-grad addmm need matching dtypes.
    # - CUDA + use_acc_dtype + accurate-mode input-grad
    #   (grad_input_dtype == logits_buf_dtype): the fast-path addmm
    #   needs self/mat2 to match logits' dtype.
    # CUDA + use_acc_dtype + memory mode does NOT need the cast: the
    # forward mm uses out_dtype= on the original linear_weight, the
    # input-grad goes through the storage-trick path with the
    # original linear_weight, and the input-grad first-term mul gets
    # implicit type promotion through TensorIterator. Kept here
    # (rather than in _linear_cross_entropy_setup) because the gate
    # depends on compute_input_grad and the result is naturally
    # co-located with the buffer-allocation block below.
    if use_acc_dtype and (
        not is_cuda or (compute_input_grad and grad_input_dtype == logits_buf_dtype)
    ):
        linear_weight_cast = linear_weight.to(logits_buf_dtype)
    else:
        linear_weight_cast = linear_weight

    # ===== Buffer allocations =====
    # TODO: per-call allocations cannot be cached across training-loop
    # iterations from inside the op. The largest in practice are
    # logits_buf and linear_weight_cast; a user-supplied workspace via
    # LinearCrossEntropyOptions should target those first, with
    # weight_grad_chunk/logits_acc_buf/input_grad_acc_buf as follow-ups.
    #
    # make_empty / make_zeros wrap torch.empty / torch.zeros with a
    # `when=False` fast path that returns a rank-matching empty tensor
    # ((0,) * len(shape)), so persistent outputs returned to the
    # caller still have the expected number of dimensions even when
    # the corresponding gradient is not being computed.
    def make_empty(shape, dtype, when=True):
        return torch.empty(
            shape if when else (0,) * len(shape),
            device=device,
            dtype=dtype,
            requires_grad=False,
        )

    def make_zeros(shape, dtype, when=True):
        return torch.zeros(
            shape if when else (0,) * len(shape),
            device=device,
            dtype=dtype,
            requires_grad=False,
        )

    # Chunk buffer used to hold logits, then softmax of logits.
    logits_buf = make_empty((batch_chunk_size, num_classes), logits_buf_dtype)
    # weight_grad_chunk: per-chunk weight-gradient scratch.
    # Skipped under acc_policy="lowmemory" when an addmm-into-grad_lw
    # path is available (CUDA, or any backend with same-dtype mm).
    keep_weight_grad_chunk = compute_linear_weight_grad and not (
        acc_policy in {"lowmemory", "ultralow"}
        and (is_cuda or logits_buf_dtype == grad_linear_weight_dtype)
    )
    weight_grad_chunk = make_empty(
        (num_classes, in_features),
        weight_grad_chunk_dtype,
        when=keep_weight_grad_chunk,
    )
    # logits_acc_buf: upcast scratch for the weight-grad mm in the
    # fp16+memory non-CUDA path (where logits_buf is fp16 but
    # weight_grad_chunk is fp32). Tied to keep_weight_grad_chunk —
    # lowmemory's addmm path doesn't need the upcast scratch.
    logits_acc_buf = make_empty(
        (batch_chunk_size, num_classes),
        acc_dtype,
        when=keep_weight_grad_chunk and not is_cuda and logits_buf_dtype != acc_dtype,
    )
    # input_grad_acc_buf: non-narrow scratch for the buf-and-copy
    # input-grad path on:
    # - CPU mixed dtype (avoids dtype-strict addmm + bf16 subtract).
    # - MPS (avoids an addmm bug where M aliasing out= on a narrow
    #   view of grad_input gives ~48% relative error vs fp64).
    use_input_grad_acc = (
        compute_input_grad
        and not is_cuda
        and (grad_input_dtype != linear_weight_cast.dtype or is_mps)
    )
    input_grad_acc_buf = make_empty(
        (batch_chunk_size, in_features),
        linear_weight_cast.dtype,
        when=use_input_grad_acc,
    )
    # input_chunk_acc_buf: pre-cast input chunk. Replaces a per-
    # iteration input_chunk.to(acc_dtype) allocation with a copy_
    # into this buffer. Used by the weight-grad path and (for non-CUDA
    # mixed dtype) by the forward mm; fp16+memory has dtype ==
    # logits_buf_dtype so the forward mm cast is a no-op there.
    use_input_chunk_acc = use_acc_dtype and (
        compute_linear_weight_grad or (not is_cuda and dtype != logits_buf_dtype)
    )
    input_chunk_acc_buf = make_empty(
        (batch_chunk_size, in_features), acc_dtype, when=use_input_chunk_acc
    )
    # Chunk-sized scratch reused per iteration: first as logits_max
    # (must match logits.dtype = logits_buf_dtype for amax's out=),
    # then as tmp_chunk after logits.sub_(logits_max). Under
    # ultralow's bf16-logits layout the weight_chunk/softmax_denom
    # division still casts to bf16 for tmp_chunk, but the
    # subsequent logits.mul_ promotes through fp32 internally so
    # the final bf16 logits matches what an fp32 tmp_chunk would
    # have produced.
    tmp = make_empty((batch_chunk_size,), logits_buf_dtype)

    # Persistent outputs:
    grad_input = make_empty(input.shape, grad_input_dtype, when=compute_input_grad)
    grad_linear_weight = make_zeros(
        linear_weight.shape,
        grad_linear_weight_dtype,
        when=compute_linear_weight_grad,
    )
    output = make_zeros((), output_dtype)
    if reduction == "mean" and num_batches == 0:
        output.fill_(torch.nan)

    # ===== Loop dispatch flags (constant across iterations) =====
    compute_grads = compute_input_grad or compute_linear_weight_grad
    forward_use_acc_input = use_acc_dtype and not is_cuda and dtype != logits_buf_dtype
    use_cuda_out_dtype = is_cuda and use_acc_dtype
    weight_grad_mm_same_dtype = logits_buf_dtype == weight_grad_chunk_dtype
    # Storage-trick fires when logits' dtype differs from the input-
    # grad accumulator dtype: addmm needs operands in grad_input_dtype,
    # so logits is cast down via a bf16 view of logits_buf storage.
    # Uses logits_buf_dtype (not linear_weight_cast.dtype) because
    # linear_weight_cast may now be the original bf16 tensor in CUDA
    # memory mode.
    input_grad_uses_logits_lw = (
        is_cuda and compute_input_grad and logits_buf_dtype != grad_input_dtype
    )
    # Per-iter `logits.to(linear_weight.dtype)` is needed by:
    # - The input-grad addmm (input_grad_uses_logits_lw).
    # - The lowmemory weight-grad mm Part 1 when logits is in a
    #   wider dtype than grad_linear_weight (cuBLAS rejects fp32
    #   inputs with bf16 out_dtype).
    # In every config where both fire, the target dtype is the same
    # (linear_weight.dtype == grad_linear_weight.dtype in memory-like
    # layouts), so a single cast is shared.
    need_logits_cast_per_iter = input_grad_uses_logits_lw or (
        compute_linear_weight_grad
        and not keep_weight_grad_chunk
        and logits_buf_dtype != grad_linear_weight_dtype
    )

    # chunking along batches dimension:
    for bchunk_start, bchunk_size in _chunk_iter(num_batches, batch_chunk_size):
        # ----- Per-chunk views -----
        input_chunk = input.narrow(0, bchunk_start, bchunk_size)
        target_chunk = target.narrow(0, bchunk_start, bchunk_size)
        weight_chunk = neg_weight_target.narrow(0, bchunk_start, bchunk_size)
        if logits_buf.shape[0] != bchunk_size:
            logits = logits_buf.narrow(0, 0, bchunk_size)
        else:
            logits = logits_buf
        if use_input_chunk_acc:
            input_chunk_acc = input_chunk_acc_buf.narrow(0, 0, bchunk_size).copy_(
                input_chunk
            )
        else:
            input_chunk_acc = input_chunk

        # ----- Forward: logits = input_chunk @ linear_weight.T -----
        # CUDA + use_acc_dtype lets cuBLAS upcast via out_dtype= on
        # original tensors; non-CUDA mixed-dtype needs the pre-cast
        # input_chunk_acc and linear_weight_cast.
        mat1 = input_chunk_acc if forward_use_acc_input else input_chunk
        mat2 = linear_weight.T if use_cuda_out_dtype else linear_weight_cast.T
        if use_cuda_out_dtype:
            torch.mm(mat1, mat2, out_dtype=logits.dtype, out=logits)
        else:
            torch.mm(mat1, mat2, out=logits)

        # ----- Softmax + loss accumulation -----
        logits_max = tmp.narrow(0, 0, bchunk_size).unsqueeze(1)
        torch.amax(logits, dim=1, keepdim=True, out=logits_max)
        logits.sub_(logits_max)
        # output += sum_i weight_chunk[i] * logits[i, target_chunk[i]];
        # weight_chunk already carries the reduction sign. torch.dot
        # is dtype-strict, so cast the gather to weight_chunk's dtype
        # — under ultralow weight_chunk is acc_dtype while logits is
        # the input dtype; the .to() is a no-op when they match.
        output.add_(
            weight_chunk.dot(
                logits.gather(1, target_chunk.unsqueeze(1))
                .squeeze(1)
                .to(weight_chunk.dtype)
            )
        )
        # Under ultralow, logits is in dtype (bf16/fp16); upcast the
        # softmax denominator's accumulator so summing C low-dtype
        # values doesn't lose precision (cuBLAS reductions roughly
        # add log2(C) * eps cumulative error from the input dtype).
        if is_ultralow:
            softmax_denom = logits.exp_().sum(dim=1, dtype=acc_dtype)
        else:
            softmax_denom = logits.exp_().sum(dim=1)

        if compute_grads:
            if is_ultralow:
                # weight_chunk and softmax_denom are acc_dtype while
                # tmp/logits are dtype; torch.div with out=dtype is
                # dtype-strict, so compute the factor in acc_dtype
                # and cast to logits.dtype for the in-place multiply.
                factor = (weight_chunk / softmax_denom).to(logits.dtype)
            else:
                tmp_chunk = tmp.narrow(0, 0, bchunk_size)
                torch.div(weight_chunk, softmax_denom, out=tmp_chunk)
                factor = tmp_chunk
            # MPS: in-place mul_ on a narrow view of logits_buf
            # doesn't propagate to parent storage; the out= form does.
            torch.mul(logits, factor.unsqueeze(1), out=logits)

        output.sub_(weight_chunk.dot(softmax_denom.log_()))

        # Pre-cast logits to linear_weight.dtype if any of the
        # downstream addmms (lowmemory weight-grad mm, or input-grad
        # storage-trick replacement) needs it. Doing this once and
        # sharing avoids two separate per-iteration casts.
        logits_cast = (
            logits.to(linear_weight.dtype) if need_logits_cast_per_iter else logits
        )

        # ----- Weight-grad mm (Part 1): consumes logits in logits_buf.dtype -----
        # Split from the accumulation step so the input-grad branch
        # below can reuse logits_buf's storage as a bf16 view.
        # Two top-level dispatches: keep_weight_grad_chunk drives the
        # acc_policy="lowmemory" path that addmm_'s straight into
        # grad_linear_weight, skipping the (V, F) scratch.
        if compute_linear_weight_grad:
            if keep_weight_grad_chunk:
                if weight_grad_mm_same_dtype:
                    torch.mm(logits.T, input_chunk_acc, out=weight_grad_chunk)
                elif is_cuda:
                    torch.mm(
                        logits.T,
                        input_chunk,
                        out_dtype=weight_grad_chunk.dtype,
                        out=weight_grad_chunk,
                    )
                else:
                    logits_acc_chunk = logits_acc_buf.narrow(0, 0, bchunk_size).copy_(
                        logits
                    )
                    torch.mm(logits_acc_chunk.T, input_chunk_acc, out=weight_grad_chunk)
            else:
                # Direct in-place addmm into grad_linear_weight.
                # cuBLAS Tensor Cores use an fp32 internal accumulator
                # regardless of operand dtype, giving the same
                # precision as the keep_weight_grad_chunk path's
                # fp32 mm + cast. logits_cast already matches
                # grad_linear_weight.dtype when a cast is needed.
                grad_linear_weight.addmm_(logits_cast.T, input_chunk, alpha=-1)

        # ----- Input-grad: lw_cast[target]*w - logits @ lw_cast -----
        if compute_input_grad:
            if use_input_grad_acc:
                grad_input_chunk = input_grad_acc_buf.narrow(0, 0, bchunk_size)
            else:
                grad_input_chunk = grad_input.narrow(0, bchunk_start, bchunk_size)

            # First term: linear_weight_cast[target] * weight_chunk.
            # Under ultralow, weight_chunk is acc_dtype while
            # linear_weight_cast and grad_input_chunk are dtype;
            # torch.mul with out= is dtype-strict, so cast
            # weight_chunk down to match. The .to() is a no-op when
            # dtypes already match.
            torch.mul(
                torch.index_select(linear_weight_cast, 0, target_chunk),
                weight_chunk.to(grad_input_chunk.dtype).unsqueeze(1),
                out=grad_input_chunk,
            )

            # Second term: subtract logits @ linear_weight_cast.
            # On CUDA mixed-dtype, logits_cast is the pre-cast bf16
            # view of logits (shared with Part 1 above). cuBLAS is
            # dtype-strict, so the cast is required; the caching
            # allocator reuses the same slot across iterations, so
            # peak memory does not grow with the loop.
            if input_grad_uses_logits_lw:
                mat1, mat2 = logits_cast, linear_weight
            else:
                mat1, mat2 = logits, linear_weight_cast
            torch.addmm(grad_input_chunk, mat1, mat2, alpha=-1, out=grad_input_chunk)

            if use_input_grad_acc:
                # Buf-and-copy: cast/copy scratch back into grad_input.
                # Used for CPU mixed-dtype (avoids dtype-strict addmm
                # and bf16 subtract precision loss) and MPS (avoids
                # an addmm bug where M aliasing out= on a narrow view
                # of grad_input gives ~48% relative error vs fp64).
                grad_input.narrow(0, bchunk_start, bchunk_size).copy_(grad_input_chunk)

        # ----- Weight-grad accumulation (Part 2): does not touch logits_buf -----
        if compute_linear_weight_grad:
            if use_acc_dtype:
                # Safe to mutate: input_chunk_acc is a fresh copy in
                # input_chunk_acc_buf; next iteration's copy_ overwrites.
                temp = input_chunk_acc.mul_(weight_chunk.unsqueeze(1))
            elif num_classes >= in_features:
                # input_chunk_acc is a view of user's input (cannot
                # mutate). Reuse logits storage — its values are no
                # longer needed after the weight-grad mm above.
                temp = torch.mul(
                    input_chunk_acc,
                    weight_chunk.unsqueeze(1),
                    out=logits.narrow(1, 0, in_features),
                )
            else:
                # No suitable buffer to reuse; allocate per chunk.
                temp = input_chunk_acc * weight_chunk.unsqueeze(1)
            if keep_weight_grad_chunk:
                # Original: scatter into scratch (alpha=-1), then sub_
                # the whole chunk into grad_lw (one fp-cast per element).
                weight_grad_chunk.index_add_(0, target_chunk, temp, alpha=-1)
                grad_linear_weight.sub_(weight_grad_chunk)
            else:
                # lowmemory: the mm part was already subtracted into
                # grad_linear_weight via addmm_(alpha=-1) above; the
                # ignore-index correction adds (sign flipped from -1
                # to +1) directly into grad_lw at scattered rows.
                # index_add_ is dtype-strict (unlike sub_), so cast
                # temp to grad_lw.dtype first; .to() is a no-op when
                # dtypes already match (non-acc_dtype paths).
                grad_linear_weight.index_add_(
                    0, target_chunk, temp.to(grad_linear_weight.dtype), alpha=1
                )

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
