"""
Custom encoder functions

This module provides reusable encoder functions that convert function parameters
into JSON-serializable dictionaries for caching purposes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import ParamSpec, TypedDict


if TYPE_CHECKING:
    from collections.abc import Callable

import torch
from torch import Tensor
from torch._inductor.runtime.caching.interfaces import DeferredRecording


if TYPE_CHECKING:
    from torch._inductor.ir import (
        Buffer,
        IRNode,
        Layout,
        MultiTemplateBuffer,
        TemplateBuffer,
        TensorBox,
    )
    from torch._inductor.kernel_inputs import MMKernelInputs
    from torch._inductor.kernel_template_choice import KernelTemplateChoice
    from torch._inductor.pattern_matcher import Match
    from torch._inductor.select_algorithm import ChoiceCaller


# Type variable for function parameters
_P = ParamSpec("_P")


# =============================================================================
# Encoded Types
# =============================================================================


class EncodedTensor(TypedDict):
    """TypedDict for encoded tensor metadata."""

    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: str


class EncodedNode(TypedDict):
    """TypedDict for encoded input node information (dtype, shape, stride)."""

    dtype: str
    shape: list[int]
    stride: list[int]


class EncodedChoice(TypedDict, total=False):
    """TypedDict for an encoded choice from a KernelTemplateChoice.

    Fields:
        template_id: identifies which template (e.g., "aten::mm", "mm")
        params: serialized params from ktc.params.to_serializeable_dict()
        rank: optional rank for multi-choice encoding (lower is better)
    """

    template_id: str
    params: dict[str, str]
    rank: int


class TunedKernelEncodedParams(TypedDict, total=False):
    """TypedDict for encoded tuned kernel parameters (mm, addmm, etc.).

    This structure mirrors the key format from LookupTableChoices._generate_kernel_inputs_key:
    - nodes: list of (dtype, shape, stride) for each input node
    - scalars: optional dict of scalar key=value pairs (e.g., alpha, beta for addmm)
    """

    nodes: list[EncodedNode]
    scalars: dict[str, float | int]


class TunedKernelEncodedResult(TypedDict, total=False):
    """TypedDict for encoded tuned kernel result (mm, addmm, etc.).

    The `_type` field determines which other fields are present:
    - "single_choice": `choice` is present
    - "multi_template_buffer": `choices` is present
    - "unknown": encoding failed, decoder should recompute
    """

    _type: str
    choice: EncodedChoice
    choices: list[EncodedChoice]


class ShouldPadEncodedParams(TypedDict):
    """TypedDict for encoded should_pad parameters."""

    mat1: EncodedTensor
    mat2: EncodedTensor
    op: str
    input: EncodedTensor | None
    mat1_exclude_padding_time: bool
    mat2_exclude_padding_time: bool
    tf32: bool


# =============================================================================
# Encoder Helper Functions
# =============================================================================


def _encode_tensor(t: Tensor) -> EncodedTensor:
    """Encode a tensor's metadata into a JSON-serializable dict."""
    return EncodedTensor(
        shape=tuple(t.shape),
        stride=tuple(t.stride()),
        dtype=str(t.dtype),
    )


def _encode_kernel_inputs(kernel_inputs: MMKernelInputs) -> TunedKernelEncodedParams:
    """Encode MMKernelInputs into a human-readable dict."""
    dtypes = kernel_inputs.dtypes()
    shapes = kernel_inputs.shapes_hinted()
    strides = kernel_inputs.strides_hinted()

    nodes: list[EncodedNode] = [
        EncodedNode(
            dtype=str(dtype),
            shape=list(shape),
            stride=list(stride),
        )
        for dtype, shape, stride in zip(dtypes, shapes, strides)
    ]

    result = TunedKernelEncodedParams(nodes=nodes)

    if kernel_inputs._scalars:
        result["scalars"] = dict(kernel_inputs._scalars)

    return result


def _encode_choice_from_ktc(
    ktc: KernelTemplateChoice,
    rank: int | None = None,
) -> EncodedChoice | None:
    """Encode a choice from a KernelTemplateChoice."""
    if not hasattr(ktc, "template") or ktc.template is None:
        return None
    if not hasattr(ktc, "params") or ktc.params is None:
        return None

    # Convert params values to strings to ensure JSON serializability
    # (e.g., torch.dtype objects are not JSON serializable)
    raw_params = ktc.params.to_serializeable_dict()
    serializable_params = {k: str(v) for k, v in raw_params.items()}

    result = EncodedChoice(
        template_id=ktc.template.uid,
        params=serializable_params,
    )

    if rank is not None:
        result["rank"] = rank

    return result


def _encode_choice_from_caller_or_node(
    obj: ChoiceCaller | TemplateBuffer | IRNode,
    rank: int | None = None,
) -> EncodedChoice | None:
    """Encode a choice from a ChoiceCaller or buffer node by extracting its KTC annotation.

    This function is general and works with any object that has an annotations dict
    containing a "ktc" key. This includes:
    - ChoiceCaller instances (from autotune_select_algorithm)
    - TemplateBuffer nodes (from output_node())
    - Other IRNode types with annotations

    Args:
        obj: A ChoiceCaller, TemplateBuffer, IRNode, or any object with annotations["ktc"]
        rank: Optional rank for multi-choice encoding (lower is better)

    Returns:
        EncodedChoice if encoding succeeded, None otherwise
    """
    annotations = getattr(obj, "annotations", None)
    if not annotations or "ktc" not in annotations:
        return None

    ktc = annotations["ktc"]
    return _encode_choice_from_ktc(ktc, rank=rank)


def _encode_multi_template_buffer(
    buffer: MultiTemplateBuffer,
    fn: Callable[..., TensorBox],
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> DeferredRecording[TensorBox, TunedKernelEncodedResult]:
    """Encode a MultiTemplateBuffer using deferred recording.

    This function wraps the buffer's choice_timings method to capture timing
    results and encode them when the timings are computed.

    Memory management:
    To avoid a reference cycle that would keep the DeferredRecording alive
    as long as the buffer exists, we use a mutable holder for the deferred
    reference and clear it after finalization.

    Interim result handling:
    When make_interim_result creates a new buffer, we need to map the cached
    timings to the new buffer's ChoiceCaller objects. Since each call to the
    underlying function produces different ChoiceCaller objects, we use their
    hash to match corresponding choices between the original and new buffers.
    """
    from torch._inductor.ir import MultiTemplateBuffer, StorageBox, TensorBox

    deferred: DeferredRecording[TensorBox, TunedKernelEncodedResult] = (
        DeferredRecording()
    )

    original_choice_timings = buffer.choice_timings  # type: ignore[union-attr]
    finalized = False

    # Store deferred in a mutable holder so we can clear it after finalization.
    deferred_holder: list[
        DeferredRecording[TensorBox, TunedKernelEncodedResult] | None
    ] = [deferred]

    # Cache timing results by ChoiceCaller hash_key for mapping to new buffers in interim results.
    # When make_interim_result creates a new buffer, its ChoiceCallers are different objects
    # but have the same hash_key as the corresponding original ChoiceCallers.
    cached_timings_by_hash: dict[str, float] | None = None

    def wrapped_choice_timings(
        hint_override: int | None = None,
    ) -> dict[ChoiceCaller, float]:
        nonlocal finalized, cached_timings_by_hash
        timings_result = original_choice_timings(hint_override)

        if hint_override is None and not finalized:
            finalized = True

            # Cache timings by hash_key for use by interim results
            cached_timings_by_hash = {
                caller.hash_key(): timing for caller, timing in timings_result.items()
            }

            # Get deferred from holder and clear it to break the reference cycle.
            deferred_obj = deferred_holder[0]
            deferred_holder[0] = None
            assert deferred_obj is not None

            sorted_choices = sorted(timings_result.items(), key=lambda x: x[1])
            encoded_choices: list[EncodedChoice] = []
            encoding_failed = False

            for rank, (choice_caller, _timing) in enumerate(sorted_choices):
                encoded = _encode_choice_from_caller_or_node(choice_caller, rank=rank)
                if encoded is None:
                    encoding_failed = True
                    break
                encoded_choices.append(encoded)

            if encoding_failed or not encoded_choices:
                encoded_result = TunedKernelEncodedResult(_type="unknown")
            else:
                encoded_result = TunedKernelEncodedResult(
                    _type="multi_template_buffer",
                    choices=encoded_choices,
                )

            deferred_obj.finalize(encoded_result)

        return timings_result

    buffer.choice_timings = wrapped_choice_timings  # type: ignore[union-attr]

    def make_interim_result() -> TensorBox:
        new_result = fn(*args, **kwargs)

        if isinstance(new_result, TensorBox) and isinstance(
            new_result.data, StorageBox
        ):
            new_buffer = new_result.data.data
            if isinstance(new_buffer, MultiTemplateBuffer):
                # Save the new buffer's original choice_timings for fallback
                new_choice_timings = new_buffer.choice_timings  # type: ignore[union-attr]

                def mapped_choice_timings(
                    hint_override: int | None = None,
                ) -> dict[ChoiceCaller, float]:
                    # Check if cached timings are available (set when original choice_timings completes)
                    if cached_timings_by_hash is not None:
                        # Get the new buffer's choices by calling its original choice_timings.
                        # This returns dict[ChoiceCaller, float] - we need the keys.
                        new_timings = new_choice_timings(hint_override)

                        # Safety check: verify all choices have cached timings.
                        # If any choice's hash_key is missing, fall back to actual timings.
                        all_choices_cached = all(
                            choice.hash_key() in cached_timings_by_hash
                            for choice in new_timings
                        )

                        if all_choices_cached:
                            # Map cached timings to new buffer's choices by hash_key.
                            # Each ChoiceCaller has a hash_key that identifies the choice,
                            # so we can match new ChoiceCallers to their cached timings.
                            return {
                                choice: cached_timings_by_hash[choice.hash_key()]
                                for choice in new_timings
                            }
                        else:
                            # Some choices weren't cached - use the actual timings we just computed
                            return new_timings
                    else:
                        # Cached timings not ready yet, fall back to actual timing computation
                        return new_choice_timings(hint_override)

                new_buffer.choice_timings = mapped_choice_timings  # type: ignore[union-attr]

        return new_result

    deferred.make_interim_result = make_interim_result

    return deferred


# =============================================================================
# Encoders
# =============================================================================


def should_pad_params_encoder(
    match: Match,
    mat1: Tensor,
    mat2: Tensor,
    op: torch._ops.OpOverloadPacket,
    input: Tensor | None = None,
) -> ShouldPadEncodedParams:
    """Encode parameters for _should_pad into a human-readable dict.

    This encoder extracts only the information needed for caching:
    - Tensor shape, stride, and dtype (not the actual data)
    - Whether padding time should be excluded for mat1 and mat2
    - The operation as a string

    Args:
        match: The pattern match object
        mat1: First matrix tensor
        mat2: Second matrix tensor
        op: The operation being performed
        input: Optional input tensor for addmm

    Returns:
        A dict containing the encoded parameters in human-readable form
    """
    # Import here to avoid circular dependency
    from torch._inductor.fx_passes.pad_mm import should_exclude_padding_time

    return ShouldPadEncodedParams(
        mat1=_encode_tensor(mat1),
        mat2=_encode_tensor(mat2),
        op=str(op),
        input=_encode_tensor(input) if input is not None else None,
        mat1_exclude_padding_time=should_exclude_padding_time(match, "mat1"),
        mat2_exclude_padding_time=should_exclude_padding_time(match, "mat2"),
        tf32=False
        if mat1.dtype != torch.float32
        else bool(
            torch.backends.cuda.matmul.allow_tf32 or torch.backends.mkldnn.allow_tf32
        ),
    )


def tuned_mm_params_encoder(
    mat1: Buffer,
    mat2: Buffer,
    out_dtype: torch.dtype | None = None,
    *,
    layout: Layout | None = None,
) -> TunedKernelEncodedParams:
    """Encode parameters for tuned_mm into a human-readable dict.

    This encoder mirrors the behavior of tuned_mm:
    1. First calls mm_args to realize the matrices (just like tuned_mm does)
    2. Creates MMKernelInputs with the realized matrices
    3. Extracts the same information used by _generate_kernel_inputs_key

    The encoding includes:
    - nodes: dtype, shape (hinted), and stride (hinted) for each input node
    - scalars: any scalar values (not used in basic mm, but used in addmm)

    Args:
        mat1: First matrix buffer
        mat2: Second matrix buffer
        out_dtype: Optional output dtype
        layout: Optional layout

    Returns:
        A dict containing the encoded parameters in human-readable form
    """
    from torch._inductor.kernel.mm_common import mm_args
    from torch._inductor.kernel_inputs import MMKernelInputs

    # First call mm_args to realize the matrices, exactly as done in tuned_mm
    _m, _n, _k, _layout, mat1_realized, mat2_realized = mm_args(
        mat1, mat2, layout=layout, out_dtype=out_dtype
    )

    # Create MMKernelInputs with the realized matrices
    kernel_inputs = MMKernelInputs([mat1_realized, mat2_realized], out_dtype=out_dtype)

    return _encode_kernel_inputs(kernel_inputs)


def tuned_addmm_params_encoder(
    inp: Buffer,
    mat1: Buffer,
    mat2: Buffer,
    *,
    alpha: float | int = 1,
    beta: float | int = 1,
    layout: Layout | None = None,
) -> TunedKernelEncodedParams:
    """Encode parameters for tuned_addmm into a human-readable dict.

    This encoder mirrors the behavior of tuned_addmm:
    1. First calls mm_args to realize the matrices (just like tuned_addmm does)
    2. Creates MMKernelInputs with the realized matrices and scalars
    3. Extracts the same information used by _generate_kernel_inputs_key

    The encoding includes:
    - nodes: dtype, shape (hinted), and stride (hinted) for each input node
    - scalars: alpha and beta values

    Args:
        inp: Input bias buffer
        mat1: First matrix buffer
        mat2: Second matrix buffer
        alpha: Scalar multiplier for mat1 @ mat2
        beta: Scalar multiplier for inp
        layout: Optional layout

    Returns:
        A dict containing the encoded parameters in human-readable form
    """
    from torch._inductor.kernel.mm_common import mm_args
    from torch._inductor.kernel_inputs import MMKernelInputs

    # First call mm_args to realize the matrices, exactly as done in tuned_addmm
    _m, _n, _k, _layout, mat1_realized, mat2_realized, inp_expanded = mm_args(
        mat1, mat2, inp, layout=layout
    )

    # Create MMKernelInputs with the realized matrices
    kernel_inputs = MMKernelInputs(
        [inp_expanded, mat1_realized, mat2_realized],
        scalars=dict(alpha=alpha, beta=beta),
    )

    return _encode_kernel_inputs(kernel_inputs)


def tuned_bmm_params_encoder(
    mat1: Buffer,
    mat2: Buffer,
    out_dtype: torch.dtype | None = None,
    *,
    layout: Layout | None = None,
) -> TunedKernelEncodedParams:
    """Encode parameters for tuned_bmm into a human-readable dict.

    This encoder mirrors the behavior of tuned_bmm:
    1. First calls mm_args to realize the matrices (just like tuned_bmm does)
    2. Creates MMKernelInputs with the realized matrices
    3. Extracts the same information used by _generate_kernel_inputs_key

    The encoding includes:
    - nodes: dtype, shape (hinted), and stride (hinted) for each input node
    - scalars: any scalar values (not used in basic bmm)

    Args:
        mat1: First matrix buffer (batched)
        mat2: Second matrix buffer (batched)
        out_dtype: Optional output dtype
        layout: Optional layout

    Returns:
        A dict containing the encoded parameters in human-readable form
    """
    from torch._inductor.kernel.mm_common import mm_args
    from torch._inductor.kernel_inputs import MMKernelInputs

    # First call mm_args to realize the matrices, exactly as done in tuned_bmm
    _m, _n, _k, _layout, mat1_realized, mat2_realized = mm_args(
        mat1, mat2, layout=layout, out_dtype=out_dtype
    )

    # Create MMKernelInputs with the realized matrices
    kernel_inputs = MMKernelInputs([mat1_realized, mat2_realized], out_dtype=out_dtype)

    return _encode_kernel_inputs(kernel_inputs)


def tuned_baddbmm_params_encoder(
    inp: Buffer,
    mat1: Buffer,
    mat2: Buffer,
    *,
    alpha: float | int = 1,
    beta: float | int = 1,
    layout: Layout | None = None,
) -> TunedKernelEncodedParams:
    """Encode parameters for tuned_baddbmm into a human-readable dict.

    This encoder mirrors the behavior of tuned_baddbmm:
    1. First calls mm_args to realize the matrices (just like tuned_baddbmm does)
    2. Creates MMKernelInputs with the realized matrices and scalars
    3. Extracts the same information used by _generate_kernel_inputs_key

    The encoding includes:
    - nodes: dtype, shape (hinted), and stride (hinted) for each input node
    - scalars: alpha and beta values

    Args:
        inp: Input bias buffer (batched)
        mat1: First matrix buffer (batched)
        mat2: Second matrix buffer (batched)
        alpha: Scalar multiplier for mat1 @ mat2
        beta: Scalar multiplier for inp
        layout: Optional layout

    Returns:
        A dict containing the encoded parameters in human-readable form
    """
    from torch._inductor.kernel.mm_common import mm_args
    from torch._inductor.kernel_inputs import MMKernelInputs

    # First call mm_args to realize the matrices, exactly as done in tuned_baddbmm
    _m, _n, _k, _layout, mat1_realized, mat2_realized, inp_expanded = mm_args(
        mat1, mat2, inp, layout=layout
    )

    # Create MMKernelInputs with the realized matrices
    kernel_inputs = MMKernelInputs(
        [inp_expanded, mat1_realized, mat2_realized],
        scalars=dict(alpha=alpha, beta=beta),
    )

    return _encode_kernel_inputs(kernel_inputs)


def tuned_mm_plus_mm_params_encoder(
    mat1: Buffer,
    mat2: Buffer,
    mat3: Buffer,
    mat4: Buffer,
    *,
    layout: Layout | None = None,
) -> TunedKernelEncodedParams:
    """Encode parameters for tuned_mm_plus_mm into a human-readable dict.

    This encoder mirrors the behavior of tuned_mm_plus_mm:
    1. First calls mm_args to realize all four matrices
    2. Creates MMKernelInputs with all four realized matrices
    3. Extracts the same information used by _generate_kernel_inputs_key

    The encoding includes:
    - nodes: dtype, shape (hinted), and stride (hinted) for each of the 4 input nodes

    Args:
        mat1: First matrix buffer (for first mm)
        mat2: Second matrix buffer (for first mm)
        mat3: Third matrix buffer (for second mm)
        mat4: Fourth matrix buffer (for second mm)
        layout: Optional layout

    Returns:
        A dict containing the encoded parameters in human-readable form
    """
    from torch._inductor.kernel.mm_common import mm_args
    from torch._inductor.kernel_inputs import MMKernelInputs

    # First call mm_args to realize the matrices, exactly as done in tuned_mm_plus_mm
    _m1, _n1, _k1, _layout1, mat1_realized, mat2_realized = mm_args(
        mat1, mat2, layout=layout
    )
    _m2, _n2, _k2, _layout2, mat3_realized, mat4_realized = mm_args(
        mat3, mat4, layout=layout
    )

    # Create MMKernelInputs with all 4 realized matrices
    kernel_inputs = MMKernelInputs(
        [mat1_realized, mat2_realized, mat3_realized, mat4_realized],
        mat1_idx=0,
        mat2_idx=1,
    )

    return _encode_kernel_inputs(kernel_inputs)


def tuned_kernel_result_encoder(
    fn: Callable[_P, TensorBox],
) -> Callable[
    _P,
    Callable[
        [TensorBox],
        TunedKernelEncodedResult
        | DeferredRecording[TensorBox, TunedKernelEncodedResult],
    ],
]:
    """Factory factory that returns a params-to-encoder factory for tuned kernel results.

    This is a generic result encoder that works with any tuned kernel function
    (tuned_mm, tuned_addmm, etc.). It encodes choices using the KernelTemplateChoice
    (KTC) annotations on ChoiceCallers and/or the buffer's annotations dict.

    Encoding strategy:
    1. MultiTemplateBuffer → deferred recording (choice_timings is expensive)
       - When choice_timings completes, encode each ChoiceCaller via its KTC annotation
    2. Single output node (TemplateBuffer or ExternKernelOut) → extract KTC from
       buffer.annotations["ktc"]
    3. Unknown types → return "unknown" for recomputation on decode

    Args:
        fn: The underlying unwrapped function (passed by the memoizer)

    Returns:
        A factory that takes (*args, **kwargs) and returns an encoder function
    """

    def params_to_encoder(
        *args: _P.args,
        **kwargs: _P.kwargs,
    ) -> Callable[
        [TensorBox],
        TunedKernelEncodedResult
        | DeferredRecording[TensorBox, TunedKernelEncodedResult],
    ]:
        """Factory that returns an encoder function for the given params."""

        def encode_result(
            result: TensorBox,
        ) -> (
            TunedKernelEncodedResult
            | DeferredRecording[TensorBox, TunedKernelEncodedResult]
        ):
            # Import at runtime to avoid circular imports
            from torch._inductor.ir import (
                MultiTemplateBuffer,
                StorageBox,
                TensorBox as TensorBoxType,
            )

            # Cases 1-2: TensorBox containing a StorageBox
            if isinstance(result, TensorBoxType) and isinstance(
                result.data, StorageBox
            ):
                buffer = result.data.data

                # Case 1: MultiTemplateBuffer - use deferred recording
                if isinstance(buffer, MultiTemplateBuffer):
                    return _encode_multi_template_buffer(buffer, fn, args, kwargs)

                # Case 2: Single output node - encode via annotations["ktc"] if available
                encoded = _encode_choice_from_caller_or_node(buffer)
                if encoded is not None:
                    return TunedKernelEncodedResult(
                        _type="single_choice",
                        choice=encoded,
                    )

            # Fallback for unknown types - mark as unknown
            return TunedKernelEncodedResult(_type="unknown")

        return encode_result

    return params_to_encoder
