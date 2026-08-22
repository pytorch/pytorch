# mypy: allow-untyped-defs
import dataclasses
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

# This module is imported during torch initialization (via pytree ->
# opaque_object -> _library). Keep top-level imports free of torch.autograd /
# torch._functorch.pyfunctorch, which are not finished initializing yet.
from torch import _C, _ops, autograd, Tensor
from torch._C._functorch import (
    _unwrap_for_grad,
    _wrap_for_grad,
    get_dynamic_layer_stack_depth,
    TransformType,
)
from torch._functorch.utils import enable_single_level_autograd_function
from torch.utils import _pytree

from . import utils


class InfoProtocol(Protocol):
    _backward_fn: Callable | None
    _setup_context_fn: Callable | None


@dataclasses.dataclass
class Info:
    _backward_fn: Callable | None
    _setup_context_fn: Callable | None


def _backend_keyset_for_args(args: tuple[Any, ...]) -> Any:
    """DispatchKeySet for the tensor device backend (no FuncTorch / Autograd)."""
    DK = _C.DispatchKey
    for arg in args:
        if not isinstance(arg, Tensor):
            continue
        device_type = arg.device.type
        if device_type == "cuda":
            return _C.DispatchKeySet(DK.CUDA)
        if device_type == "cpu":
            return _C.DispatchKeySet(DK.CPU)
        if device_type == "meta":
            return _C.DispatchKeySet(DK.Meta)
        if device_type == "mps":
            return _C.DispatchKeySet(DK.MPS)
        if device_type == "xla":
            return _C.DispatchKeySet(DK.XLA)
        if device_type == "xpu":
            return _C.DispatchKeySet(DK.XPU)
    # Fallback: CPU (matches stock when no tensor args).
    return _C.DispatchKeySet(DK.CPU)


def _keyset_after_nest(metadata_keyset: Any) -> Any:
    """Backend keyset for depth-0 redispatch after Grad nesting.

    Like ``metadata_keyset & _after_autograd_keyset``, but strips FuncTorch
    dynamic-layer keys (nesting already ``lower()``'d to depth 0; leaving
    those bits set trips TLS asserts). Keeps ``Fake`` / ``Python`` so Dynamo
    fake-mode still hits ``register_fake`` instead of the real CUDA kernel.
    """
    DK = _C.DispatchKey
    keyset = metadata_keyset & _C._after_autograd_keyset
    for name in (
        "FuncTorchDynamicLayerBackMode",
        "FuncTorchDynamicLayerFrontMode",
        "FuncTorchGradWrapper",
        "FuncTorchBatched",
        "FuncTorchVmapMode",
        "FuncTorchBatchedDecomposition",
    ):
        if hasattr(DK, name):
            keyset = keyset.remove(getattr(DK, name))
    return keyset


def _fake_tensor_mode_active() -> bool:
    """True while Dynamo / FakeTensorMode is computing example values."""
    try:
        key = _C._TorchDispatchModeKey.FAKE
    except AttributeError:
        return False
    return _C._get_dispatch_mode(key) is not None


def make_autograd_impl(op: _ops.OpOverload, info: InfoProtocol) -> Callable:
    name: str = f"GeneratedBackwardFor_{op._namespace}_{op._opname}_{op._overloadname}"
    schema = op._schema
    is_out_op = utils.is_out(op)
    has_kwarg_only_args = utils.has_kwarg_only_args(schema)
    num_positional_args = sum(not a.kwarg_only for a in schema.arguments)
    has_tensorlist_like_args = any(
        utils.is_tensorlist_like_type(a.type)
        for a in (*schema.arguments, *schema.returns)
    )

    @dataclass
    class Metadata:
        keyset: _C.DispatchKeySet
        keyword_only_args: dict[str, Any]

    def forward_no_grad(*args):
        metadata = args[-1]
        args = args[:-1]

        with _C._AutoDispatchBelowAutograd():
            keyset = metadata.keyset
            kwargs = metadata.keyword_only_args
            result = op.redispatch(keyset & _C._after_autograd_keyset, *args, **kwargs)
            return result

    def _run_kernel(args, metadata, *, backend_only: bool):
        with _C._AutoDispatchBelowAutograd():
            if backend_only:
                # Prefer nest-safe keyset that still carries Fake/Python for
                # Dynamo. Fall back to device backend if metadata is empty.
                # DispatchKeySet has no __bool__/__len__ (a bare truthiness
                # test is always True) and ``remove`` leaves residual backend
                # bits in raw_repr, so "empty" means: no runtime dispatch key
                # left for redispatch to target.
                keyset = _keyset_after_nest(metadata.keyset)
                if keyset.highestPriorityTypeId() == _C.DispatchKey.Undefined:
                    keyset = _backend_keyset_for_args(args)
            else:
                keyset = metadata.keyset & _C._after_autograd_keyset
            if backend_only and _fake_tensor_mode_active():
                # Depth-0 kernel after Grad nesting under AOT tracing.
                # ``op.redispatch`` with ``Python`` in the keyset trips
                # PythonFallbackKernel's ``tls_on_entry`` assert: no
                # dispatcher-entry snapshot exists because functorch's
                # dynamic-layer kernel put ``PythonTLSSnapshot`` in the TLS
                # exclude set while sanitizing. Issue a FRESH op call with
                # that single key un-excluded: the entry then stashes the
                # snapshot normally, the surrounding ``interpreter.lower()``
                # and ``_AutoDispatchBelowAutograd`` guards keep functorch
                # and autograd out of the way, and the call lands on the
                # proxy/fake modes — recording the op as an OPAQUE graph
                # node with ``register_fake`` metadata (never an inlined
                # surrogate body). Grad structure was already recorded by
                # the per-level Nested functions above, so no grad_fn is
                # needed here (mirrors ``forward_backend``).
                include = _C._dispatch_tls_local_include_set()
                exclude = _C._dispatch_tls_local_exclude_set().remove(
                    _C.DispatchKey.PythonTLSSnapshot,
                )
                with _C._ForceDispatchKeyGuard(include, exclude):
                    return op(*args, **metadata.keyword_only_args)
            return op.redispatch(keyset, *args, **metadata.keyword_only_args)

    def _setup_ctx(ctx, args, kwargs, result):
        if not info._setup_context_fn:
            return

        args, kwargs = utils.fill_defaults(op._schema, args, kwargs)
        if has_kwarg_only_args:
            info._setup_context_fn(
                ctx=ctx,
                inputs=args,
                keyword_only_inputs=kwargs,
                output=result,
            )
        else:
            info._setup_context_fn(ctx=ctx, inputs=args, output=result)

    def forward(ctx, *args):
        metadata = args[-1]
        args = args[:-1]
        result = _run_kernel(args, metadata, backend_only=False)
        _setup_ctx(ctx, args, metadata.keyword_only_args, result)
        return result

    def forward_backend(ctx, *args):
        """Depth-0 kernel under nested Grad: nest-safe backend keyset."""
        metadata = args[-1]
        args = args[:-1]
        result = _run_kernel(args, metadata, backend_only=True)
        _setup_ctx(ctx, args, metadata.keyword_only_args, result)
        return result

    def backward(ctx, *grads):
        if info._backward_fn:
            try:
                prev_needs_input_grad = ctx.needs_input_grad
                ctx.needs_input_grad = prev_needs_input_grad[:-1]
                result = info._backward_fn(ctx, *grads)
            finally:
                ctx.needs_input_grad = prev_needs_input_grad
            num_actual_inputs = len(prev_needs_input_grad) - 1
            valid_return_counts = {num_actual_inputs, num_positional_args}
            actual = len(result) if isinstance(result, tuple) else 1
            if actual not in valid_return_counts:
                expected = (
                    str(num_actual_inputs)
                    if num_actual_inputs == num_positional_args
                    else f"{num_actual_inputs} or {num_positional_args}"
                )
                raise RuntimeError(
                    f"The backward formula for {op} returned an incorrect "
                    f"number of gradients (expected {expected}, got {actual}). "
                    f"Expected one gradient for each forward input, or for "
                    f"each positional input to the operator. Use None for "
                    f"inputs that do not require a gradient."
                )
            if isinstance(result, tuple):
                extra_grads = result[num_actual_inputs:]
                if any(grad is not None for grad in extra_grads):
                    raise RuntimeError(
                        f"The backward formula for {op} returned a non-None "
                        f"gradient for an input that was not passed to the "
                        f"operator. Defaulted inputs that were not passed "
                        f"through autograd must return None."
                    )
                if has_tensorlist_like_args:
                    result = result[:num_actual_inputs]
                return (*result, None)
            return result, None
        raise RuntimeError(
            f"Trying to backward through {op} but no autograd "
            f"formula was registered. "
            f"Please use register_autograd to add one."
        )

    Generated = type(
        name,
        (autograd.function._SingleLevelFunction,),
        {
            "forward": staticmethod(forward),
            "backward": staticmethod(backward),
        },
    )

    GeneratedBackend = type(
        name + "_Backend",
        (autograd.function._SingleLevelFunction,),
        {
            "forward": staticmethod(forward_backend),
            "backward": staticmethod(backward),
        },
    )

    schema = op._schema
    needs_tensorlist = has_tensorlist_like_args
    if needs_tensorlist:
        Generated = supports_tensorlist(Generated)
        GeneratedBackend = supports_tensorlist(GeneratedBackend)

    def _apply_single_level(*operands: Any, backend_only: bool = False) -> Any:
        cls = GeneratedBackend if backend_only else Generated
        with enable_single_level_autograd_function():
            return cls.apply(*operands)  # type: ignore[attr-defined]  # pyrefly: ignore[missing-attribute]

    def _generate_nested(interpreter: Any) -> Any:
        """One Grad/Jvp layer: unwrap → lower → redispatch → wrap.

        Matches ``torch._functorch.autograd_function.generate_single_level_function``.
        Old-style ``forward(ctx, ...)`` so ``supports_tensorlist`` works for PE.
        """
        # Lazy: this module loads during torch init; forward_ad / enable_grad
        # are not safe to import at module scope.
        import torch
        from torch.autograd.forward_ad import _set_fwd_grad_enabled

        level = interpreter.level()

        def nested_forward(ctx, *operands: Any) -> Any:
            unwrapped = _pytree.tree_map_only(
                Tensor,
                lambda t: _unwrap_for_grad(t, level),
                operands,
            )
            with (
                torch.enable_grad(),
                _set_fwd_grad_enabled(True),
                interpreter.lower(),
            ):
                out = _dispatch_functorch(*unwrapped)
            wrapped = _pytree.tree_map_only(
                Tensor,
                lambda t: _wrap_for_grad(t, level),
                out,
            )
            metadata = operands[-1]
            _setup_ctx(
                ctx,
                operands[:-1],
                metadata.keyword_only_args,
                wrapped,
            )
            return wrapped

        Nested = type(
            f"{name}_Nested_L{level}",
            (autograd.function._SingleLevelFunction,),
            {
                "forward": staticmethod(nested_forward),
                "backward": staticmethod(backward),
            },
        )
        if needs_tensorlist:
            Nested = supports_tensorlist(Nested)
        return Nested

    def _dispatch_functorch(*operands: Any) -> Any:
        """Nest Grad/Jvp layers until depth 0, then run the backend kernel."""
        from torch._functorch.pyfunctorch import retrieve_current_functorch_interpreter

        if not _C._are_functorch_transforms_active():
            return _apply_single_level(*operands, backend_only=True)

        interpreter = retrieve_current_functorch_interpreter()
        if interpreter.key() not in (TransformType.Grad, TransformType.Jvp):
            return _apply_single_level(*operands, backend_only=True)

        Nested = _generate_nested(interpreter)
        with enable_single_level_autograd_function():
            return Nested.apply(*operands)  # type: ignore[attr-defined]  # pyrefly: ignore[missing-attribute]

    def autograd_impl(keyset, *args, **keyword_only_args):
        from torch._functorch.pyfunctorch import retrieve_current_functorch_interpreter

        if is_out_op:
            if _C.is_grad_enabled() and _C._any_requires_grad(
                *args, **keyword_only_args
            ):
                raise RuntimeError(
                    f"{op._opname}(): functions with out=... arguments don't "
                    "support automatic differentiation, but one of the arguments "
                    "requires grad."
                )
            return forward_no_grad(*args, Metadata(keyset, keyword_only_args))

        meta = Metadata(keyset, keyword_only_args)
        if not (_C.is_grad_enabled() and _C._any_requires_grad(*args)):
            return forward_no_grad(*args, meta)

        operands = (*args, meta)
        if not _C._are_functorch_transforms_active():
            return _apply_single_level(*operands)

        interpreter = retrieve_current_functorch_interpreter()
        if interpreter.key() not in (TransformType.Grad, TransformType.Jvp):
            return _apply_single_level(*operands)

        # depth 1: existing SingleLevel path (first-order torch.func).
        if get_dynamic_layer_stack_depth() <= 1:
            return _apply_single_level(*operands)

        # depth >= 2 (eager AND Fake/AOT tracing): leave Autograd boxed-kernel
        # TLS, then nest like ``custom_function_call_grad`` down to a
        # backend-only redispatch. Under FakeTensorMode the depth-0 kernel in
        # ``_run_kernel`` issues a fresh op call (with ``PythonTLSSnapshot``
        # un-excluded) so proxy tracing records the op as an OPAQUE node —
        # the op is NEVER inlined as a fake/surrogate body. See _run_kernel.
        with _C._AutoDispatchBelowAutograd():
            return _dispatch_functorch(*operands)

    return autograd_impl


def supports_tensorlist(cls: Any) -> Any:
    """Allows a given autograd.Function class to support List[Tensor] inputs/outputs.

    Regular autograd.Function has a constraint that it only directly supports autograd for
    Tensors. Applying @supports_tensorlist enables an autograd.Function to support
    autograd for List[Tensor] inputs and outputs.
    """
    orig_forward = cls.forward
    orig_backward = cls.backward
    orig_apply = cls.apply

    @dataclass
    class TensorListMetadata:
        input_spec: _pytree.TreeSpec
        output_spec: _pytree.TreeSpec | None = None
        result_is_tuple: bool | None = None

    def new_forward(ctx, *args):
        metadata = args[-1]
        args = args[:-1]
        if not isinstance(metadata, TensorListMetadata):
            raise NotImplementedError(
                "NYI: calling supports_tensorlist autograd.Function.forward directly. "
                "You should probably be calling .apply instead. "
                "Please file an issue if not."
            )
        args = _pytree.tree_unflatten(list(args), metadata.input_spec)
        result = orig_forward(ctx, *args)
        metadata.result_is_tuple = isinstance(result, tuple)
        if not metadata.result_is_tuple:
            result = (result,)
        flat_result, output_spec = _pytree.tree_flatten(result, not_list_of_tensor)
        metadata.output_spec = output_spec

        if hasattr(ctx, "_pt_metadata"):
            raise RuntimeError(
                "Please don't set ctx._pt_metadata; PyTorch uses it to store info"
            )
        ctx._pt_metadata = metadata

        return tuple(flat_result)

    def new_backward(ctx, *grads):
        if not hasattr(ctx, "_pt_metadata"):
            raise NotImplementedError(
                "NYI: calling supports_tensorlist autograd.Function.backward directly. "
                "This will automatically get called by PyTorch autograd. "
                "Please file an issue if you need this."
            )

        metadata = ctx._pt_metadata
        grads = _pytree.tree_unflatten(list(grads), metadata.output_spec)

        # If the user's input is ([x, y, z], w),
        # then needs_input_grad is (bool, bool, bool, bool, bool).
        # We need to
        # 1. get rid of the additional bool (which comes from the extra
        # `metadata input`)
        # 2. _pytree.tree_unflatten to get the right structure.
        prev_needs_input_grad = ctx.needs_input_grad
        try:
            ctx.needs_input_grad = _pytree.tree_unflatten(
                list(ctx.needs_input_grad[:-1]), metadata.input_spec
            )
            grad_inputs = orig_backward(ctx, *grads)
        finally:
            ctx.needs_input_grad = prev_needs_input_grad

        if not isinstance(grad_inputs, tuple):
            grad_inputs = (grad_inputs,)
        # Assume that any Nones in the backward are Tensors.
        # If the forward has an arg that is [1, 2, 3], the backward should
        # return None as the grad.
        # If the forward has an arg that is [tensor, tensor], the backward
        # may return [None, None], [grad, None], [None, grad], or [grad, grad].
        flat_grad_inputs, grad_inputs_spec = _pytree.tree_flatten(
            grad_inputs, not_list_of_optional_tensor
        )
        if grad_inputs_spec != metadata.input_spec:
            raise RuntimeError(
                f"Expected the return from backward to be of the same structure "
                f"as the inputs. Got: {grad_inputs_spec} (return from backward), "
                f"{metadata.input_spec} (inputs)"
            )
        return tuple(flat_grad_inputs + [None])

    def new_apply(*args):
        flat_args, input_spec = _pytree.tree_flatten(args, is_leaf=not_list_of_tensor)
        metadata = TensorListMetadata(input_spec)
        result = orig_apply(*flat_args, metadata)  # type: ignore[misc]
        if metadata.output_spec is None:
            raise AssertionError("metadata.output_spec must not be None")
        result = _pytree.tree_unflatten(list(result), metadata.output_spec)
        if not metadata.result_is_tuple:
            if not isinstance(result, tuple):
                raise AssertionError(f"result must be tuple, got {type(result)}")
            if len(result) != 1:
                raise AssertionError(
                    f"result tuple must have length 1, got {len(result)}"
                )
            return result[0]
        return result

    cls.forward = new_forward
    cls.backward = new_backward
    cls.apply = new_apply
    return cls


def not_list_of_tensor(tree):
    if isinstance(tree, tuple):
        return False
    if isinstance(tree, list):
        return any(not isinstance(l, Tensor) for l in tree)
    return True


def not_list_of_optional_tensor(tree):
    if isinstance(tree, tuple):
        return False
    if isinstance(tree, list):
        return any(l is not None and not isinstance(l, Tensor) for l in tree)
    return True
