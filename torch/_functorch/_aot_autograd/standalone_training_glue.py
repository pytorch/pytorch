"""Source template for the autograd bridge of a standalone TRAINING artifact.

``to_standalone_python._compose_training_module`` splices the functions and the class
listed in ``SPLICED`` into the emitted module verbatim (``inspect.getsource``), right
after the per-artifact bindings they read as module globals. This file exists so that
code is real, linted and type-checked Python instead of an f-string; nothing here is
meant to be CALLED from this module. The ``_BOUND_BY_ARTIFACT`` declarations below name
the globals the artifact binds before the splice point (here they are only typed
placeholders that fail loudly if touched), and ``_compose_training_module`` checks
that it emits exactly that set.

The artifact binds, in order: the baked metadata (``_fw_metadata`` with its derived
state restored, ``_saved_state``, ``_rng_state``, the tangent dependency tables,
``_NUM_FORWARD_RETURNS``, ``_DISABLE_AMP``); ``_FORWARD_CALL``, the one name that
varies per artifact (the inner Inductor forward, or the inner-chain subclass /
functionalized-RNG wrapper around it); the ``_AOT_BACKWARD_VARIANTS`` table, which
always holds mask 0; ``_AOT_BACKWARD_VARIANT_COMPILER``, None in a finished artifact
and bound to ``_CompileToPythonState.compile_mask`` while capturing; and AOTAutograd's
codegen'd ``_transform_raw_returns`` / ``_compiled_forward`` / ``_compiled_backward`` /
``_backward_prologue`` / ``_backward_epilogue``.
"""

from __future__ import annotations

import contextlib
import weakref
from typing import Any, TYPE_CHECKING

import torch
from torch._functorch._aot_autograd.standalone_runtime import (
    _grad_output_prototypes,
    _mask_pruned_backward_outputs,
    _pruned_backward_output_indices_from_dependencies,
    _snapshot_external_objects,
    index_to_external_object_weakref,
    normalize_as_list,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from torch._functorch._aot_autograd.standalone_runtime import (
        _AutogradRngStateTracker,
        _AutogradSavedState,
        _BackwardVariant,
        ViewAndMutationMeta,
    )


class _BoundByArtifact:
    """Stands in for a global the emitted artifact binds before the spliced code."""

    def __getattr__(self, name: str) -> Any:
        raise RuntimeError(
            "standalone_training_glue is a source template; its functions only run "
            "inside an emitted training artifact"
        )

    __call__ = __getattr__


_BOUND_BY_ARTIFACT: Any = _BoundByArtifact()

_fw_metadata: ViewAndMutationMeta = _BOUND_BY_ARTIFACT
_saved_state: _AutogradSavedState = _BOUND_BY_ARTIFACT
_rng_state: _AutogradRngStateTracker = _BOUND_BY_ARTIFACT
_BACKWARD_OUTPUT_DEPENDENCIES: tuple[tuple[int, ...] | None, ...] = _BOUND_BY_ARTIFACT
_BACKWARD_OUTPUT_PROVABLY_ZERO: frozenset[int] = _BOUND_BY_ARTIFACT
_AOT_SPECIALIZABLE_GRAD_OUT_INDICES: frozenset[int] = _BOUND_BY_ARTIFACT
_NUM_FORWARD_RETURNS: int = _BOUND_BY_ARTIFACT
_DISABLE_AMP: bool = _BOUND_BY_ARTIFACT
_FORWARD_CALL: Callable[[list[Any]], Any] = _BOUND_BY_ARTIFACT
_AOT_BACKWARD_VARIANTS: dict[int, _BackwardVariant] = _BOUND_BY_ARTIFACT
_AOT_BACKWARD_VARIANT_COMPILER: Callable[[int], _BackwardVariant] | None = (
    _BOUND_BY_ARTIFACT
)
_transform_raw_returns: Callable[[list[Any]], list[Any]] = _BOUND_BY_ARTIFACT
_compiled_forward: Callable[..., Any] = _BOUND_BY_ARTIFACT
_compiled_backward: Callable[..., Any] = _BOUND_BY_ARTIFACT
_backward_prologue: Callable[..., Any] = _BOUND_BY_ARTIFACT
_backward_epilogue: Callable[..., Any] = _BOUND_BY_ARTIFACT


def _finalize(ctx: Any, fw_outs: Sequence[Any]) -> tuple[Any, ...]:
    raw_returns = list(fw_outs[:_NUM_FORWARD_RETURNS])
    # Undefined-tangent handling is baked in (the live runtime reads the
    # aot_autograd_prune_unused_outputs config here): grads reach backward()
    # unmaterialized and the prologue materializes only those the selected
    # variant did not specialize away. Prototypes are built even for a single
    # differentiable output: a downstream custom Function whose backward returns
    # None hands that sole tangent in as undefined.
    ctx._aot_prune_unused_outputs_enabled = True
    ctx.set_materialize_grads(False)
    prototypes, prototype_objects = _grad_output_prototypes(raw_returns, _fw_metadata)
    ctx._aot_grad_output_prototypes = prototypes
    ctx._aot_grad_output_prototype_objects = prototype_objects
    ctx.mark_non_differentiable(*_transform_raw_returns(raw_returns))
    ctx._materialize_non_diff_grads = False
    _snapshot_external_objects(ctx)
    return tuple(raw_returns)


def _select_backward_variant(undefined_grad_out_indices: Sequence[int]) -> Any:
    # Canonical mask: only specializable (surviving user-output) indices key the
    # table. A non-differentiable output's tangent is ALWAYS undefined and must
    # not fork variants (mirrors the live _specializable_user_grad_output_mask).
    mask = 0
    for index in undefined_grad_out_indices:
        if index in _AOT_SPECIALIZABLE_GRAD_OUT_INDICES:
            mask |= 1 << index
    if _AOT_BACKWARD_VARIANT_COMPILER is not None:
        # Capturing: the compile state records the mask and compiles or reuses
        # its variant; the table below is never mutated by capture.
        return _AOT_BACKWARD_VARIANT_COMPILER(mask)
    variant = _AOT_BACKWARD_VARIANTS.get(mask)
    if variant is None:
        # An unseen pattern is served by the all-tangents-defined backward: the
        # prologue materializes the undefined tangents from the prototypes saved
        # at forward time and _backward_impl prunes the provably-zero grads,
        # exactly like the live runtime's fallback. Nothing compiles at serve time.
        variant = _AOT_BACKWARD_VARIANTS[0]
    return variant


def _backward_impl(ctx: Any, all_args: list[Any]) -> list[Any]:
    ctx.maybe_clear_saved_tensors()
    for idx, obj in ctx._external_objects.items():
        index_to_external_object_weakref[idx] = weakref.ref(obj)
    variant = ctx._aot_backward_variant
    if variant.kept_arg_indices is not None:
        kept_args = [all_args[index] for index in variant.kept_arg_indices]
        all_args.clear()
        all_args.extend(kept_args)
    amp = torch._C._DisableAutocast if _DISABLE_AMP else contextlib.nullcontext
    with amp():
        out = normalize_as_list(variant.inner_call(all_args))
    pruned = variant.pruned_output_indices
    if pruned is None:
        # Dependency alone is not enough to null a grad: only outputs that are
        # ALSO provably zero with their tangents undefined may be masked (an
        # affine custom backward yields a nonzero grad from a zero tangent).
        candidates = _pruned_backward_output_indices_from_dependencies(
            _BACKWARD_OUTPUT_DEPENDENCIES, ctx._undefined_grad_out_indices
        )
        pruned = tuple(i for i in candidates if i in _BACKWARD_OUTPUT_PROVABLY_ZERO)
    return _mask_pruned_backward_outputs(out, pruned)


def _double_backward(ctx: Any, impl_fn: Callable[..., Any], all_args: list[Any]) -> Any:
    class _DoubleBackward(torch.autograd.Function):
        @staticmethod
        # pyrefly: ignore [bad-override]
        def forward(double_ctx: Any, *unused_args: Any) -> Any:
            return impl_fn(double_ctx)

        @staticmethod
        def backward(ctx: Any, *args: Any) -> None:
            raise RuntimeError(
                "torch.compile with aot_autograd does not currently support "
                "double backward"
            )

    # Saved tensors are detached (the forward ran under no-grad), so prepend a
    # dummy requires_grad input to attach a grad_fn for create_graph=True.
    if not any(t.requires_grad for t in all_args if isinstance(t, torch.Tensor)):
        all_args = [torch.empty(0, requires_grad=True)] + all_args
    return _DoubleBackward.apply(*all_args)


class _CompiledFunction(torch.autograd.Function):
    boxed_grads_call = True

    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx: Any, *deduped_flat_tensor_args: Any) -> Any:
        return _compiled_forward(
            ctx,
            deduped_flat_tensor_args,
            _rng_state.add_forward_args,
            _saved_state.save_from_forward,
            _finalize,
            _FORWARD_CALL,
        )

    @staticmethod
    def backward(ctx: Any, *flat_args: Any) -> Any:
        if len(flat_args) == 1 and isinstance(flat_args[0], list):
            ctx._undefined_grad_out_indices = tuple(
                index for index, grad in enumerate(flat_args[0]) if grad is None
            )
        else:
            ctx._undefined_grad_out_indices = ()
        variant = _select_backward_variant(ctx._undefined_grad_out_indices)
        ctx._aot_backward_variant = variant
        ctx._aot_skip_materialize_grad_output_indices = variant.skip_materialize_indices
        return _compiled_backward(
            flat_args,
            ctx,
            _backward_prologue,
            _rng_state.add_backward_args,
            _backward_impl,
            _backward_epilogue,
            _double_backward,
        )


def _boxed_autograd_apply(args: list[Any]) -> Any:
    return _CompiledFunction.apply(*args)


# Spliced into the artifact in this order; the composer reserves these names.
SPLICED: tuple[Any, ...] = (
    _finalize,
    _select_backward_variant,
    _backward_impl,
    _double_backward,
    _CompiledFunction,
    _boxed_autograd_apply,
)
