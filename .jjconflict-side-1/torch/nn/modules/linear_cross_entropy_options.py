import dataclasses
from typing import Literal

import torch


__all__ = ["LinearCrossEntropyOptions"]


_VALID_ACC_POLICIES = ("auto", "accurate", "compact")


def _auto_acc_policy(device_type: str | None, dtype: torch.dtype) -> str:
    """Resolve the ``acc_policy`` ``"auto"`` sentinel from device and dtype.

    Only CPU + low-precision input (fp16/bf16) picks ``"accurate"``: there
    ``compact`` runs the weight-grad GEMM in the input dtype,
    hitting CPU's emulated low-precision matmul (~20-50x slower), so
    ``"accurate"`` keeps it in fp32. Everything else picks ``"compact"`` --
    accelerators have hardware-native low-precision GEMMs, and for fp32/fp64
    the GEMM is native at any policy (``"accurate"`` would only add a
    weight-grad scratch, no benefit).
    """
    if device_type == "cpu" and dtype in (torch.float16, torch.bfloat16):
        return "accurate"
    return "compact"


@dataclasses.dataclass(slots=True, frozen=True)
class LinearCrossEntropyOptions:
    """Configuration for the chunked implementation of
    :func:`linear_cross_entropy`.

    The chunked implementation processes the batch dimension in pieces, so
    the full ``(num_batches, num_classes)`` logits tensor is never
    materialized -- useful when ``num_classes`` is much larger than
    ``in_features`` (e.g. LLM vocabulary heads). Pass ``options=None`` to
    use the reference path; pass an instance of this class to opt in.

    Zero-argument ``LinearCrossEntropyOptions()`` leaves :attr:`acc_policy`
    and :attr:`chunking_method` at ``"auto"``, resolved per (device, dtype) at
    call time -- see the field docs below.

    Supports a subset of :func:`linear_cross_entropy`; unsupported
    configurations fall through to the reference path with a warning.

    Chunking is a win when ``num_batches >= in_features`` and
    ``num_classes > in_features``; below that, the reference path is
    cheaper.
    """

    allow_retain_graph: bool = False
    """Allow ``retain_graph=True`` on backward. Applies only to the scalar
    reductions (``"mean"`` / ``"sum"``).

    When ``False`` (default), their backward consumes pre-computed gradient
    buffers in place; a second ``.backward()`` raises ``RuntimeError``.

    When ``True``, the buffers are preserved at the cost of one extra
    gradient-sized allocation per call.

    ``reduction="none"`` ignores this field: its backward recomputes the
    chunked gradients from the saved inputs, so ``retain_graph=True`` works
    unconditionally with no extra allocation.

    Higher-order autograd (gradgrad, forward-mode AD) is unsupported.

    Under :func:`torch.compile` this field is auto-promoted to ``True`` for
    the scalar reductions because the default-mode second-backward guard
    relies on a ctx mutation Dynamo doesn't preserve; the wrapper warns on
    the promotion.
    """

    batch_chunk_size: int | None = None
    """Batch rows per chunk. The op loops over
    ``ceil(num_batches / batch_chunk_size)`` chunks; smaller values cut
    peak memory but launch more kernels. Default ``None`` means a single
    chunk. Cannot be combined with :attr:`chunking_method` -- if both are
    set and disagree, ``ValueError`` is raised.
    """

    chunking_method: str | None = "auto"
    """Heuristic for picking :attr:`batch_chunk_size`.

    - ``"auto"`` (default) -- resolves to ``"aspect_ratio"`` (factor 1),
      with the compact path additionally capped at a per-target ``B_ref``
      in :meth:`_adjust` so the chunked peak never exceeds the unchunked
      reference in the budget regime (``in_features >= num_classes``).
    - ``"aspect_ratio"`` -- sizes each chunk so its
      ``(batch_chunk_size, num_classes)`` logits buffer matches the
      ``(num_batches, in_features)`` input in memory:
      ``next_pow2(ceil(num_batches / ceil(num_classes / in_features)))``.
      Best when ``num_classes >> in_features`` (LLM vocab heads).
    - ``"aspect_ratio:N"`` (``N >= 1``) -- same, divided by ``N``.
      ~N times less peak memory at the cost of N times more chunks.
    - ``None`` -- disables the heuristic; uses :attr:`batch_chunk_size`.
    """

    acc_policy: Literal[
        "accurate",
        "compact",
        "auto",
    ] = "auto"
    """Precision/memory trade-off for the chunked path. Controls which
    intermediates are kept in :attr:`acc_dtype` vs. the input dtype, and
    whether the per-chunk weight-gradient scratch buffer is materialized.

    - ``"auto"`` (default) -- ``"accurate"`` for CPU low-precision input
      (fp16/bf16), ``"compact"`` otherwise (:func:`_auto_acc_policy`). Pass
      ``"accurate"`` explicitly on any other backend that emulates fp16/bf16
      GEMMs via fp32 upcast.
    - ``"accurate"`` -- broadest use of :attr:`acc_dtype`; noticeably
      better input-grad accuracy when chunk size is large relative to
      ``num_classes``. Highest peak memory and slowest of the chunked
      policies on CUDA. Only chunked policy whose weight-grad matmul
      runs in fp32 on CPU (other policies hit CPU's emulated
      low-precision path, ~20-50x slower).
    - ``"compact"`` -- :attr:`acc_dtype` only where needed for gradient
      correctness; accumulates the weight gradient per chunk directly via
      ``addmm_`` instead of a ``(num_classes, in_features)``
      :attr:`acc_dtype` scratch (on CUDA cuBLAS uses an fp32 internal
      accumulator, so bulk precision is unchanged). Saves
      ``num_classes * in_features * sizeof(acc_dtype)`` -- typically
      several hundred MB for an LLM head. On non-CUDA mixed-precision it
      retains that scratch for the cross-chunk accumulation.

    The precision difference between ``"compact"`` and ``"accurate"`` is
    visible only when :attr:`acc_dtype` differs from the input dtype;
    ``"compact"`` saves memory in both regimes.
    """

    acc_dtype: torch.dtype | None = None
    """Dtype for internal accumulation. ``None`` resolves at call time
    to ``torch.float32`` under ``acc_policy="auto"`` with fp16/bf16
    input on hardware with mixed-precision mm (CUDA SM 7.0+ for fp16,
    SM 8.0+ for bf16, and CPU); otherwise to the input dtype.
    Mixed-precision currently requires fp16/bf16 input with
    ``acc_dtype=torch.float32``.
    """

    def __post_init__(self):
        if self.acc_policy not in _VALID_ACC_POLICIES:
            raise ValueError(
                f"invalid acc_policy: {self.acc_policy!r}; expected one of "
                f"{', '.join(map(repr, _VALID_ACC_POLICIES))}"
            )
        if self.chunking_method is not None and self.chunking_method != "auto":
            name, sep, factor = self.chunking_method.partition(":")
            if not sep:
                factor = "1"
            if not (name == "aspect_ratio" and factor.isdigit() and int(factor) > 0):
                raise ValueError(f"invalid chunking_method: {self.chunking_method!r}")
        if not (
            self.batch_chunk_size is None
            or (isinstance(self.batch_chunk_size, int) and self.batch_chunk_size > 0)
        ):
            raise ValueError(
                f"batch_chunk_size must be positive int or None, got {self.batch_chunk_size!r}"
            )
        # fp64 is allowed for the internal ``_adjust`` path (fp64 inputs).
        _SUPPORTED_ACC_DTYPES = {
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        }
        if self.acc_dtype is not None and self.acc_dtype not in _SUPPORTED_ACC_DTYPES:
            raise ValueError(
                f"acc_dtype must be one of {{None, torch.float16, torch.bfloat16, "
                f"torch.float32, torch.float64}}, got {self.acc_dtype!r}. Pass "
                "acc_dtype=None to let the op pick automatically."
            )

    @staticmethod
    def _ceil_div(a: int, b: int) -> int:
        """ceil(a / b) for non-negative integers."""
        return -(-a // b)

    @staticmethod
    def _resolve_auto_acc_dtype(
        dtype: torch.dtype, device: torch.device | None
    ) -> torch.dtype | None:
        """Return fp32 for fp16/bf16 input on hardware with a working
        mixed-precision mm path (CUDA SM 7.0+ for fp16, SM 8.0+ for bf16,
        and CPU); ``None`` otherwise (caller falls back to input dtype).
        """
        if device is None or dtype not in (torch.float16, torch.bfloat16):
            return None
        if device.type == "cuda":
            if not torch.cuda.is_available():
                return None
            major, _ = torch.cuda.get_device_capability(device)
            if dtype == torch.bfloat16 and major < 8:
                return None
            if dtype == torch.float16 and major < 7:
                return None
            return torch.float32
        if device.type == "cpu":
            return torch.float32
        return None

    def _compute_batch_chunk_size(
        self,
        num_batches: int,
        in_features: int,
        num_classes: int,
        method: str | None = None,
    ) -> int:
        """Compute batch_chunk_size from chunking_method given input shapes.

        Pass ``method`` to use a post-``_adjust`` resolved value (e.g. when
        ``self.chunking_method == "auto"``); otherwise uses ``self.chunking_method``.
        To add a method: extend the if/elif chain plus ``__post_init__`` validation.
        """
        method = str(method if method is not None else self.chunking_method)

        if method.startswith("aspect_ratio"):
            factor = int(method.split(":", 1)[1]) if ":" in method else 1
            # See LinearCrossEntropyOptions.chunking_method docstring.
            inc_factor = self._ceil_div(num_classes, in_features)
            target = self._ceil_div(num_batches, inc_factor)
            chunk_size = 1 << (target - 1).bit_length()  # next power of 2 >= target
            return min(chunk_size // factor, num_batches)

        # __post_init__ validates the method, so this is unreachable.
        raise AssertionError(f"unhandled chunking_method: {method!r}")

    def _adjust(
        self,
        num_batches,
        in_features,
        num_classes,
        dtype,
        device=None,
        prob_target=False,
    ):
        """Resolve ``"auto"`` sentinels and ``None`` defaults against a
        specific call site; returns a fully concrete options instance.

        Internal API consumed by ``F.linear_cross_entropy``'s chunked
        dispatch and a handful of test call sites. ``device=None`` forces the
        ``"compact"`` fallback instead of the per-device acc_policy pick.
        ``prob_target`` selects the per-target memory cap (``N/2`` vs
        ``N*V/4D``).
        """
        acc_policy = self.acc_policy
        chunking_method = self.chunking_method
        chunking_was_auto = chunking_method == "auto"
        # Honour an explicit batch_chunk_size by disabling auto chunking
        # (it would conflict with the user's size); acc_policy="auto" is
        # unaffected.
        if chunking_method == "auto" and self.batch_chunk_size is not None:
            chunking_method = None
        if acc_policy == "auto":
            acc_policy = _auto_acc_policy(
                device.type if device is not None else None, dtype
            )
        if chunking_method == "auto":
            chunking_method = "aspect_ratio"

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
            # Memory cap for the auto compact path. aspect_ratio degenerates to
            # a single chunk when in_features >= num_classes, where the chunked
            # peak exceeds the unchunked reference. ``b_ref`` is the largest
            # chunk staying at or below reference -- ``N*V/(4*D)`` for index
            # targets, ``N/2`` for prob (whose reference also materializes the
            # soft target, so the single-chunk excess clears with two chunks);
            # both device- and dtype-independent. Floored to a power of two so
            # the cap never lands above ``b_ref``.
            if chunking_was_auto and acc_policy == "compact":
                if prob_target:
                    b_ref = num_batches // 2
                else:
                    b_ref = num_batches * num_classes // (4 * in_features)
                b_ref = 1 << (b_ref.bit_length() - 1) if b_ref >= 1 else 1
                batch_chunk_size = max(1, min(batch_chunk_size, b_ref))
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
            # Under "auto" prefer fp32 on hardware that supports the
            # mixed-precision mm path; otherwise fall back to input dtype.
            auto_acc = (
                self._resolve_auto_acc_dtype(dtype, device)
                if self.acc_policy == "auto"
                else None
            )
            acc_dtype = auto_acc if auto_acc is not None else dtype
        else:
            acc_dtype = self.acc_dtype
        return dataclasses.replace(
            self,
            acc_policy=acc_policy,
            chunking_method=chunking_method,
            batch_chunk_size=max(1, batch_chunk_size),
            acc_dtype=acc_dtype,
        )
