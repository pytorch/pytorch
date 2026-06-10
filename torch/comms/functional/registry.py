# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Collective operation registration system for torch.comms."""

import logging
from collections.abc import Callable
from typing import Any

import torch
from torch._subclasses.fake_tensor import FakeTensor, in_kernel_invocation_manager
from torch.comms.functional.async_tensor import (
    _wrap_result_with_registered_work,
    FakeWork,
)
from torch.comms.functional.param_parsing import (
    _TYPE_NAME_TO_CLASS,
    CollectiveParamSchema,
    ParamSpec,
)


__all__ = ["finalize_registration", "register_collective"]

logger = logging.getLogger(__name__)

_REGISTERED_COLLECTIVES: dict[str, dict[str, Any]] = {}


def _pack_result(results: list) -> Any:
    """Pack a list of results into the appropriate return value.

    Returns:
        None if empty, single item if one element, tuple if multiple.
    """
    if len(results) == 0:
        return None
    elif len(results) == 1:
        return results[0]
    else:
        return tuple(results)


def register_collective(
    target_class: type,
    method: Callable,
    param_specs: list[ParamSpec],
    meta_fn: Callable | None = None,
    backward_fn: Callable | None = None,
    setup_context_fn: Callable | None = None,
):
    """Register a collective operation with torch.compile support.

    Args:
        target_class: The class this method belongs to (e.g., TorchComm, TorchCommWindow).
        method: The method to register as a collective op.
        param_specs: List of ParamSpec describing the op's parameters.
        meta_fn: Optional meta/abstract implementation for tracing.
        backward_fn: Optional backward function for autograd support.
            Signature: backward(ctx, grad_output, ...) -> tuple of gradients
        setup_context_fn: Optional setup_context function for autograd.
            Signature: setup_context(ctx, inputs, output) -> None
            Called during forward to save tensors/values needed for backward.
    """
    name = method.__name__
    # Derive op prefix from class name (e.g., TorchComm -> torchcomm_)
    torch_op_prefix = target_class.__name__.lower() + "_"
    op_name = f"{torch_op_prefix}{name}"

    # Create the param schema - this handles all type processing and categorization
    param_schema = CollectiveParamSchema.from_raw_specs(target_class, param_specs)

    collective_info = {
        "name": name,
        "meta_fn": meta_fn,
        "method": method,
        "param_schema": param_schema,
        "target_class": target_class,
        "backward_fn": backward_fn,
        "setup_context_fn": setup_context_fn,
    }
    _REGISTERED_COLLECTIVES[op_name] = collective_info

    logger.info(
        "Registered collective: %s (method=%s, inputs=%s, outputs=%s, target_class=%s, has_autograd=%s)",
        op_name,
        name,
        [p.name for p in param_schema.input_params],
        [p.name for p in param_schema.output_params],
        target_class.__name__,
        backward_fn is not None,
    )

    return method


# Track functional ops created for autograd (for effectful registration)
_FUNCTIONAL_OP_NAMES: list[str] = []


def _expand_backward_grads(
    backward_result: Any,
    num_outputs: int,
    tensor_positions: list[int],
    list_tensor_positions: dict[int, int],
) -> list:
    """Expand backward gradients to full gradient tuple.

    Takes the result from a user's backward function (which only returns gradients
    for input tensors) and expands it to the full tuple expected by autograd,
    filling in None for non-tensor inputs.

    Args:
        backward_result: Result from backward_fn, can be single tensor, tuple, or list
        num_outputs: Total number of gradient outputs needed
        tensor_positions: List of positions for single tensor gradients (consumed in order)
        list_tensor_positions: Dict mapping position -> list length for tensor list gradients

    Returns:
        List of gradients with None for non-tensor positions
    """
    # Normalize result to tuple
    if isinstance(backward_result, list):
        result = tuple(backward_result)
    elif not isinstance(backward_result, tuple):
        result = (backward_result,)
    else:
        result = backward_result

    # Build full gradient list with Nones
    full_grads: list = [None] * num_outputs

    # Initialize list tensor positions with [None] * length
    for pos, length in list_tensor_positions.items():
        full_grads[pos] = [None] * length

    # Fill in gradients sequentially: first single tensors, then list tensors
    result_idx = 0

    # Single tensor positions
    for pos in tensor_positions:
        if result_idx < len(result):
            full_grads[pos] = result[result_idx]
            result_idx += 1

    # List tensor positions
    for pos, length in list_tensor_positions.items():
        for i in range(length):
            if result_idx < len(result):
                full_grads[pos][i] = result[result_idx]
                result_idx += 1

    return full_grads


def _wrap_backward_fn(
    backward_fn: Callable,
    schema: CollectiveParamSchema,
) -> Callable:
    """Wrap a backward function to expand tensor-only gradients to full gradient tuple.

    The user's backward function only needs to return gradients for input tensors.
    This wrapper expands those gradients to the full tuple expected by autograd,
    filling in None for non-tensor inputs and output buffers.

    For ops with both mutable and non-mutable tensors (e.g., all_gather_single):
      - Only non-mutable tensors are true inputs that need gradients
      - Mutable tensors are output buffers

    For ops with only mutable tensors (e.g., all_reduce):
      - The mutable tensor is both input and output (in-place op)
      - It needs a gradient

    Args:
        backward_fn: User's backward function that returns only input tensor gradients.
            Expected signature: backward(ctx, *grad_outputs) -> tuple of tensor grads
        schema: The param schema for this op

    Returns:
        Wrapped backward function that returns full gradient tuple
    """
    all_params = schema.all_params

    # Find tensor params
    tensor_params = [(i, p) for i, p in enumerate(all_params) if p.is_tensor_like()]
    non_mutable_tensors = [(i, p) for i, p in tensor_params if not p.mutable]

    # Determine which tensors need gradients:
    # - If there are non-mutable tensors, only those need gradients (they are inputs)
    # - If ALL tensors are mutable (in-place ops like all_reduce), all need gradients
    if non_mutable_tensors:
        input_tensor_indices = [
            i for i, p in non_mutable_tensors if not p.is_tensor_list()
        ]
        # Track non-mutable tensor list params separately - they need structure preservation
        input_tensor_list_indices = [
            i for i, p in non_mutable_tensors if p.is_tensor_list()
        ]
    else:
        input_tensor_indices = [i for i, p in tensor_params if not p.is_tensor_list()]
        input_tensor_list_indices = [i for i, p in tensor_params if p.is_tensor_list()]

    # Total number of gradient outputs needed (all_params already includes object_param)
    total_grads = len(all_params)

    # Identify list tensor params (mutable) that need structure preservation
    # These are output buffers like output_tensor_list in gather
    mutable_list_tensor_param_indices = [
        i for i, p in enumerate(all_params) if p.mutable and p.is_tensor_list()
    ]

    def wrapped_backward(ctx, *grad_outputs):
        # Call user's backward function
        result = backward_fn(ctx, *grad_outputs)

        # Build list_tensor_positions dict from grad_outputs
        # For mutable list tensor params, get length from grad_outputs
        list_tensor_positions: dict[int, int] = {}
        for param_idx in mutable_list_tensor_param_indices:
            for go in grad_outputs:
                if isinstance(go, list):
                    list_tensor_positions[param_idx] = len(go)
                    break

        # For non-mutable tensor list inputs, determine length from the backward result
        # The backward function returns a list for tensor list inputs
        if input_tensor_list_indices:
            # If backward result is a list, it corresponds to the tensor list input
            if isinstance(result, list):
                for idx in input_tensor_list_indices:
                    list_tensor_positions[idx] = len(result)
            elif isinstance(result, tuple):
                # Check if any element is a list (for multi-output backward)
                for r in result:
                    if isinstance(r, list):
                        for idx in input_tensor_list_indices:
                            list_tensor_positions[idx] = len(r)
                        break

        # Use shared helper to expand gradients
        full_grads = _expand_backward_grads(
            result,
            total_grads,
            input_tensor_indices,
            list_tensor_positions,
        )

        return tuple(full_grads)

    return wrapped_backward


def _generate_lib_ops(lib: Any) -> None:
    """Generate torch.library op definitions for all registered collectives.

    For ops with mutable inputs, registers both:
    - Inplace version: op_name_ with Tensor(a!) and -> ()
    - Functional version: op_name with Tensor and -> Tensor

    The functional version clones inputs, calls inplace, and returns clones.
    This enables the reinplace pass to convert back when safe.
    """
    from torch.comms.functional import collectives

    for base_op_name, info in _REGISTERED_COLLECTIVES.items():
        schema = info["param_schema"]
        method = info["method"]
        meta_fn = info["meta_fn"]

        # Check if this op has mutable inputs
        has_mutable_inputs = len(schema.mutable_params) > 0

        if has_mutable_inputs:
            # Register both inplace (op_) and functional (op) versions
            inplace_op_name = f"{base_op_name}_"
            functional_op_name = base_op_name

            # === INPLACE VERSION (op_) ===
            # Use inplace_return_type to return aliased tensors (e.g., Tensor(a!))
            # This matches PyTorch native inplace ops and enables mutation tracking
            inplace_signature = (
                f"{inplace_op_name}({schema.signature}) -> {schema.inplace_return_type}"
            )
            logger.info("Defining inplace lib op: %s", inplace_signature)
            lib.define(inplace_signature, tags=[torch.Tag.pt2_compliant_tag])

            # Inplace meta kernel - returns the input tensors (aliased)
            # This matches PyTorch native inplace ops (e.g., add_ returns self)
            def _create_inplace_meta_kernel(captured_schema: CollectiveParamSchema):
                def _inplace_meta(*args, **kwargs):
                    # Parse args to get mutable outputs (the tensors being mutated)
                    parsed = captured_schema.parse_lib_args(args)
                    mutable_outputs = parsed.get_mutable_outputs()

                    # Return the input tensors (they are mutated in place)
                    return _pack_result(mutable_outputs)

                return _inplace_meta

            inplace_meta_kernel = _create_inplace_meta_kernel(schema)
            torch.library.impl(lib, inplace_op_name, "Meta")(inplace_meta_kernel)

            # Inplace eager kernel - calls method, mutates in-place, returns mutated tensors
            def _create_inplace_eager_kernel(
                captured_method: Callable,
                captured_schema: CollectiveParamSchema,
            ):
                def _inplace_eager(*args):
                    parsed = captured_schema.parse_lib_args(args)
                    async_op = parsed.get_value("async_op") or False

                    result_or_work = captured_method(*parsed.to_values())

                    # Get the mutable outputs (the tensors that were mutated)
                    mutable_outputs = parsed.get_mutable_outputs()

                    # Register work handle for async ops
                    if async_op:
                        work = result_or_work
                        mutable_indices = parsed.get_mutable_tensor_indices()
                        if mutable_indices:
                            anchor = parsed.values[mutable_indices[0]]
                            if isinstance(anchor, list):
                                anchor = anchor[0]
                            collectives._register_tensor_work(anchor, work)

                    # Return the mutated tensors (like native PyTorch inplace ops)
                    return _pack_result(mutable_outputs)

                return _inplace_eager

            inplace_eager_kernel = _create_inplace_eager_kernel(method, schema)
            torch.library.impl(lib, inplace_op_name, "CompositeExplicitAutograd")(
                inplace_eager_kernel
            )

            # === FUNCTIONAL VERSION (op) ===
            functional_signature = f"{functional_op_name}({schema.functional_signature}) -> {schema.functional_return_type}"
            logger.info("Defining functional lib op: %s", functional_signature)
            lib.define(functional_signature, tags=[torch.Tag.pt2_compliant_tag])

            # Track functional op for effectful registration
            _FUNCTIONAL_OP_NAMES.append(functional_op_name)

            # Functional meta kernel - returns empty_like of mutable inputs
            # IMPORTANT: Must preserve requires_grad for autograd to work
            def _create_functional_meta_kernel(captured_schema: CollectiveParamSchema):
                def _functional_meta(*args):
                    parsed = captured_schema.parse_lib_args(args)
                    # Get mutable outputs preserving structure (list vs single tensor)
                    mutable_outputs = parsed.get_mutable_outputs()

                    # Check if any input requires grad - propagate to outputs
                    any_requires_grad = parsed.has_requires_grad()

                    results = []
                    for out in mutable_outputs:
                        if isinstance(out, (list, tuple)):
                            # For tensor lists, return a list (not tuple)
                            # This is required for Tensor[] return type
                            # Propagate requires_grad from inputs or from each tensor
                            results.append(
                                [
                                    torch.empty_like(
                                        t,
                                        requires_grad=(
                                            any_requires_grad or t.requires_grad
                                        ),
                                    )
                                    for t in out
                                ]
                            )
                        else:
                            results.append(
                                torch.empty_like(  # pyrefly: ignore[bad-argument-type]
                                    out,
                                    requires_grad=(
                                        any_requires_grad or out.requires_grad
                                    ),
                                )
                            )

                    return _pack_result(results)

                return _functional_meta

            functional_meta_kernel = _create_functional_meta_kernel(schema)
            torch.library.impl(lib, functional_op_name, "Meta")(functional_meta_kernel)

            # Functional eager kernel - clones, calls inplace, returns clones
            def _create_functional_eager_kernel(
                captured_schema: CollectiveParamSchema,
                captured_inplace_op_name: str,
            ):
                def _functional_eager(*args):
                    parsed = captured_schema.parse_lib_args(args)

                    # Clone mutable inputs (or empty_like for write-only outputs)
                    cloned_values = parsed.to_values()
                    mutable_indices = parsed.get_mutable_tensor_indices()
                    clones = []

                    for idx in mutable_indices:
                        orig = parsed.values[idx]
                        spec = parsed.all_params[idx]
                        if spec.write_only:
                            # Write-only buffer, empty_like is fine
                            if isinstance(orig, list):
                                clone = [torch.empty_like(t) for t in orig]
                            else:
                                clone = torch.empty_like(orig)
                        else:
                            # Read+write, need actual clone to preserve input data
                            if isinstance(orig, list):
                                clone = [t.clone() for t in orig]
                            else:
                                clone = orig.clone()
                        cloned_values[idx] = clone
                        clones.append(clone)

                    # Call inplace op on clones (skip index 0 which is the object)
                    getattr(torch.ops.torchcomms, captured_inplace_op_name)(
                        *cloned_values
                    )

                    # Transfer work from original to clone for async ops
                    async_op = parsed.get_value("async_op") or False
                    if async_op and mutable_indices:
                        # Work was registered on the clone by the inplace op
                        # No need to transfer - the clone already has the work
                        pass

                    return _pack_result(clones)

                return _functional_eager

            functional_eager_kernel = _create_functional_eager_kernel(
                schema, inplace_op_name
            )
            torch.library.impl(lib, functional_op_name, "CompositeExplicitAutograd")(
                functional_eager_kernel
            )

            # === FUNCTIONALIZE IMPL FOR INPLACE OP ===
            # Register py_functionalize_impl to convert inplace -> functional
            # and wrap with with_effects for proper effect token tracking.
            # Also update the original tensors to reference the new results (inplace semantics).
            def _create_functionalize_impl(
                captured_functional_op_name: str,
                captured_schema: CollectiveParamSchema,
            ):
                def _functionalize_impl(ctx, *args):
                    from torch._subclasses.functional_tensor import (
                        FunctionalTensor,
                        PythonFunctionalizeAPI,
                    )

                    # Get the mutable tensor indices in all_params (obj + inputs + extras)
                    mutable_indices = captured_schema.mutable_indices

                    # Unwrap args to get raw tensors
                    unwrapped_args = ctx.unwrap_tensors(args)

                    # Get the functional op
                    functional_op = getattr(
                        torch.ops.torchcomms, captured_functional_op_name
                    )

                    # Check if we have access to the mode's tokens for effects
                    assert (
                        isinstance(ctx, PythonFunctionalizeAPI)
                        and hasattr(ctx, "mode")
                        and ctx.mode is not None
                        and hasattr(ctx.mode, "_tokens")
                    )

                    from torch._higher_order_ops.effects import (
                        handle_effects,
                        has_effects,
                    )

                    assert has_effects(functional_op.default)

                    with ctx.redispatch_to_next():
                        result = handle_effects(
                            ctx.mode._allow_token_discovery,
                            ctx.mode._tokens,
                            functional_op.default,
                            tuple(unwrapped_args),
                            {},
                        )

                    # Unwrap the result - handle_effects may return FunctionalTensors
                    # but ctx.replace expects unwrapped tensors
                    unwrapped_result = ctx.unwrap_tensors(result)

                    # Update original mutable tensors to reference the new results.
                    # This maintains inplace semantics - the original tensor now wraps
                    # the new result from the functional op.
                    if not isinstance(unwrapped_result, (list, tuple)):
                        unwrapped_result = (unwrapped_result,)

                    for i, idx in enumerate(mutable_indices):
                        if i < len(unwrapped_result) and idx < len(args):
                            orig_wrapped = args[idx]
                            new_unwrapped = unwrapped_result[i]
                            if isinstance(
                                orig_wrapped, FunctionalTensor
                            ) and isinstance(new_unwrapped, torch.Tensor):
                                ctx.replace(orig_wrapped, new_unwrapped)
                            elif isinstance(orig_wrapped, (list, tuple)) and isinstance(
                                new_unwrapped, (list, tuple)
                            ):
                                for ow, nu in zip(orig_wrapped, new_unwrapped):
                                    if isinstance(ow, FunctionalTensor) and isinstance(
                                        nu, torch.Tensor
                                    ):
                                        ctx.replace(ow, nu)

                    return result
                    """
                    # Return the original wrapped tensors (which now reference the new results)
                    mutable_outputs = [args[idx] for idx in mutable_indices]
                    return _pack_result(mutable_outputs)
                    """

                return _functionalize_impl

            functionalize_impl = _create_functionalize_impl(functional_op_name, schema)

            # Get the inplace op object and register py_functionalize_impl
            inplace_op = getattr(torch.ops.torchcomms, inplace_op_name)
            inplace_op.default.py_functionalize_impl(functionalize_impl)

            logger.info(
                "Registered py_functionalize_impl: %s -> %s",
                inplace_op_name,
                functional_op_name,
            )

            # === AUTOGRAD KERNEL FOR INPLACE OP ===
            # Use make_autograd_impl to create an autograd kernel for the inplace op.
            # This allows eager execution with requires_grad to work correctly.
            backward_fn = info.get("backward_fn")
            if backward_fn is not None:
                setup_context_fn = info.get("setup_context_fn")
                inplace_wrapped_backward = _wrap_backward_fn(backward_fn, schema)

                from torch._library import autograd as library_autograd

                inplace_info = library_autograd.Info(
                    _backward_fn=inplace_wrapped_backward,
                    _setup_context_fn=setup_context_fn,
                )

                inplace_autograd_kernel = library_autograd.make_autograd_impl(
                    inplace_op.default, inplace_info
                )
                lib.impl(
                    inplace_op_name,
                    inplace_autograd_kernel,
                    "Autograd",
                    with_keyset=True,
                )
                logger.info(
                    "Registered Autograd kernel for inplace op %s", inplace_op_name
                )

        else:
            # No mutable inputs - just register the regular op
            full_signature = (
                f"{base_op_name}({schema.signature}) -> {schema.return_type}"
            )
            logger.info("Defining lib op: %s", full_signature)
            lib.define(full_signature, tags=[torch.Tag.pt2_compliant_tag])

            # Meta kernel
            def _create_meta_kernel(
                meta_func: Callable | None,
                captured_schema: CollectiveParamSchema,
            ):
                def _meta_kernel(*args):
                    parsed = captured_schema.parse_lib_args(args)
                    result_tensors = []

                    if meta_func:
                        custom_result = meta_func(*args)
                        if custom_result is not None:
                            if not isinstance(custom_result, (list, tuple)):
                                result_tensors = [custom_result]
                            else:
                                result_tensors = list(custom_result)
                    elif captured_schema.has_tensor_outputs:
                        raise RuntimeError(
                            "Operation has OUTPUT params but no meta_func provided"
                        )

                    if captured_schema.needs_async_dummy_return:
                        async_op = parsed.get_value("async_op") or False
                        if async_op:
                            # Use CPU device for the dummy tensor since it's just a placeholder
                            # for work tracking and doesn't need to match input device
                            dummy = torch.empty(0, device="cpu", dtype=torch.int)
                            result_tensors.append(dummy)
                        else:
                            result_tensors.append(None)

                    return _pack_result(result_tensors)

                return _meta_kernel

            meta_kernel = _create_meta_kernel(meta_fn, schema)
            torch.library.impl(lib, base_op_name, "Meta")(meta_kernel)

            # Eager kernel
            def _create_eager_kernel(
                captured_method: Callable,
                captured_schema: CollectiveParamSchema,
            ):
                def _eager_kernel(*args):
                    parsed = captured_schema.parse_lib_args(args)
                    async_op = parsed.get_value("async_op") or False
                    result_or_work = captured_method(*parsed.to_values())

                    result_tensors = []
                    if captured_schema.has_tensor_outputs:
                        if async_op:
                            raise RuntimeError(
                                "Async operations with OUTPUT params not yet supported."
                            )
                        if isinstance(result_or_work, (list, tuple)):
                            result_tensors.extend(result_or_work)
                        elif result_or_work is not None:
                            result_tensors.append(result_or_work)

                    if async_op:
                        work = result_or_work
                        if captured_schema.needs_async_dummy_return:
                            dummy = torch.empty(0, device="cpu", dtype=torch.int)
                            collectives._register_tensor_work(dummy, work)
                            result_tensors.append(dummy)

                    return _pack_result(result_tensors)

                return _eager_kernel

            eager_kernel = _create_eager_kernel(method, schema)
            torch.library.impl(lib, base_op_name, "CompositeExplicitAutograd")(
                eager_kernel
            )

            # For ops without tensor inputs (like barrier), register a FakeTensorMode
            # py_impl to bypass the default FakeTensor dispatch which fails to find
            # a device from tensor inputs.
            if (
                schema.num_output_tensors > 0 and schema.num_input_tensors == 0
            ) or schema.needs_async_dummy_return:

                def _create_fake_impl(captured_meta_kernel, captured_schema):
                    def _fake_impl(fake_mode, func, *args, **kwargs):
                        with in_kernel_invocation_manager(fake_mode):
                            result = captured_meta_kernel(*args, **kwargs)

                        if result is None:
                            return None
                        if isinstance(result, torch.Tensor):
                            # FakeTensor expects a meta tensor - convert if needed
                            if result.device.type != "meta":
                                meta_result = torch.empty_like(result, device="meta")
                            else:
                                meta_result = result
                            # For async dummy tensors, always use CPU device
                            # For other tensors, get device from the opaque object
                            if captured_schema.needs_async_dummy_return:
                                target_device = torch.device("cpu")
                            elif result.device.type != "meta":
                                target_device = result.device
                            else:
                                raise RuntimeError("Unable to determine target device")
                            return FakeTensor(fake_mode, meta_result, target_device)
                        return result

                    return _fake_impl

                torch_op = getattr(torch.ops.torchcomms, base_op_name)
                fake_impl = _create_fake_impl(meta_kernel, schema)
                from torch._subclasses.fake_impls import register_op_impl

                register_op_impl(torch_op.default)(fake_impl)
                logger.info(
                    "Registered FakeTensorMode py_impl for %s (no tensor inputs)",
                    base_op_name,
                )

        # Register autograd if backward_fn is provided
        # For mutating ops, register autograd on the functional version (base_op_name)
        # For non-mutating ops, register autograd on the regular op
        backward_fn = info.get("backward_fn")
        if backward_fn is not None:
            setup_context_fn = info.get("setup_context_fn")
            wrapped_backward = _wrap_backward_fn(backward_fn, schema)

            if has_mutable_inputs:
                # Register autograd on the functional version
                try:
                    torch.library.register_autograd(
                        f"torchcomms::{base_op_name}",
                        wrapped_backward,
                        setup_context=setup_context_fn,
                        lib=lib,
                    )
                    logger.info(
                        "Registered autograd for functional op %s", base_op_name
                    )
                except RuntimeError as e:
                    logger.warning(
                        "Failed to register autograd for %s: %s", base_op_name, e
                    )
            else:
                # Non-mutating op - register autograd directly
                try:
                    torch.library.register_autograd(
                        f"torchcomms::{base_op_name}",
                        wrapped_backward,
                        setup_context=setup_context_fn,
                        lib=lib,
                    )
                    logger.info("Registered autograd for %s", base_op_name)
                except RuntimeError as e:
                    logger.warning(
                        "Failed to register autograd for %s: %s", base_op_name, e
                    )

        logger.info("Registered ops for %s", base_op_name)


def _register_lowerings() -> None:
    """Register inductor lowerings for all registered collectives."""
    try:
        from torch._inductor import ir
        from torch._inductor.lowering import register_lowering
    except ImportError:
        logger.info("torch._inductor not available, skipping lowering registration")
        return

    for op_name, collective_info in _REGISTERED_COLLECTIVES.items():
        schema = collective_info["param_schema"]

        try:
            torch_op = getattr(torch.ops.torchcomms, op_name)

            def _create_lowering(
                op_name: str,
                captured_torch_op,
                lowering_schema: CollectiveParamSchema,
            ):
                num_output_tensors = lowering_schema.num_output_tensors
                needs_async_dummy = lowering_schema.needs_async_dummy_return

                def _lowering(*args):
                    from torch._inductor.virtualized import V

                    logger.info(
                        "Lowering torch.comms.%s with %s args", op_name, len(args)
                    )

                    # Parse args using schema
                    parsed = lowering_schema.parse_lib_args(args)

                    # Get flattened inputs with mutable mask
                    flat_inputs, flat_mutable_mask = (
                        parsed.get_tensor_inputs_flat_with_mutable_mask()
                    )

                    # Get device from first tensor input
                    device = None
                    tensor_inputs = parsed.get_tensor_inputs()
                    if tensor_inputs:
                        first_input = tensor_inputs[0]
                        if isinstance(first_input, (list, tuple)):
                            if first_input:
                                device = first_input[0].get_device()
                        else:
                            device = first_input.get_device()

                    # Realize all inputs and mark mutable ones as mutated
                    for inp, is_mutable in zip(flat_inputs, flat_mutable_mask):
                        inp.realize()
                        if is_mutable:
                            V.graph.mark_buffer_mutated(inp.get_name())

                    # Process kernel args - pass opaque object and remaining args separately
                    # to avoid process_kernel trying to extract shapes from the opaque object
                    with V.graph.fake_mode:
                        (
                            example_output,
                            tensor_args,
                            non_tensor_args,
                            unflatten_args,
                            unbacked_bindings,
                        ) = ir._CollectiveKernel.process_kernel(
                            captured_torch_op.default, *args
                        )
                    assert not unbacked_bindings

                    # Check if the op returns an async dummy tensor
                    # example_output will be a tensor if async_op=True and needs_async_dummy=True
                    has_async_dummy_output = (
                        needs_async_dummy
                        and example_output is not None
                        and isinstance(example_output, torch.Tensor)
                    )

                    if num_output_tensors == 0 and not has_async_dummy_output:
                        # No outputs - just create collective for side effect
                        layout = ir.NoneLayout(device=device)
                        packed = ir._CollectiveKernel(
                            layout,
                            captured_torch_op.default,
                            tensor_args,
                            non_tensor_args,
                            unflatten_args,
                        )

                        # Set up mutation outputs for mutable inputs
                        for inp, is_mutable in zip(flat_inputs, flat_mutable_mask):
                            if is_mutable:
                                packed.mutation_outputs.append(
                                    ir.MutationOutput(
                                        ir.NoneLayout(device=device), inp, packed
                                    )
                                )
                                packed.alias_names.append(inp.get_name())

                        return None

                    elif num_output_tensors == 1 or has_async_dummy_output:
                        # Single output - use its layout directly
                        if isinstance(example_output, (list, tuple)):
                            ex_out = example_output[0]
                        else:
                            ex_out = example_output

                        layout = ir._CollectiveKernel.tensor_to_layout(ex_out)
                        packed = ir._CollectiveKernel(
                            layout,
                            captured_torch_op.default,
                            tensor_args,
                            non_tensor_args,
                            unflatten_args,
                        )

                        # Set up mutation outputs for mutable inputs
                        for inp, is_mutable in zip(flat_inputs, flat_mutable_mask):
                            if is_mutable:
                                packed.mutation_outputs.append(
                                    ir.MutationOutput(
                                        ir.NoneLayout(device=device), inp, packed
                                    )
                                )
                                packed.alias_names.append(inp.get_name())

                        return ir.TensorBox.create(packed)

                    else:
                        # Multiple outputs - use MultiOutputLayout
                        # pyrefly: ignore[bad-argument-type]
                        layout = ir.MultiOutputLayout(device=device)
                        packed = ir._CollectiveKernel(
                            layout,
                            captured_torch_op.default,
                            tensor_args,
                            non_tensor_args,
                            unflatten_args,
                        )

                        # Set up mutation outputs for mutable inputs
                        for inp, is_mutable in zip(flat_inputs, flat_mutable_mask):
                            if is_mutable:
                                packed.mutation_outputs.append(
                                    ir.MutationOutput(
                                        ir.NoneLayout(device=device), inp, packed
                                    )
                                )
                                packed.alias_names.append(inp.get_name())

                        # Create MultiOutput for each output tensor
                        packed.outputs = []
                        result_tensors = []

                        if isinstance(example_output, (list, tuple)):
                            for i, ex_out in enumerate(example_output):
                                if hasattr(ex_out, "shape"):
                                    out = ir.MultiOutput(
                                        ir._CollectiveKernel.tensor_to_layout(ex_out),
                                        packed,
                                        [(list, i)],
                                    )
                                    packed.outputs.append(out)
                                    result_tensors.append(ir.TensorBox.create(out))

                        return tuple(result_tensors)

                return _lowering

            lowering_fn = _create_lowering(op_name, torch_op, schema)
            register_lowering(torch_op)(lowering_fn)

            logger.info("Registered lowering for %s", op_name)
        except AttributeError as e:
            logger.warning("Failed to register lowering for %s: %s", op_name, e)


def _patch_eager_methods() -> None:
    """Patch TorchComm methods to dispatch to functional ops for tracing and autograd.

    This patches the native eager API (e.g., comm.all_reduce(tensor, op)) to dispatch
    to functional ops when:
    - Any input tensor has requires_grad=True (for autograd support)
    - Any input tensor is a FakeTensor or meta tensor (for torch.compile tracing)

    This ensures proper behavior during tracing where meta/fake tensors should not
    hit the real C++ collective implementations.
    """
    # Group registered collectives by target class
    class_methods: dict[type, list[tuple[str, dict]]] = {}
    for op_name, info in _REGISTERED_COLLECTIVES.items():
        schema = info["param_schema"]

        # Only patch mutating ops (they have functional versions)
        if len(schema.mutable_params) == 0:
            continue

        target_class = info["target_class"]
        if target_class not in class_methods:
            class_methods[target_class] = []
        class_methods[target_class].append((op_name, info))

    def _create_dispatch_wrapper(original_method, op_name, info):
        """Create a wrapper that dispatches to the functional op for tracing and autograd.

        When tensors require grad or are fake/meta tensors (during torch.compile),
        we dispatch to the inplace op which has proper kernels registered for
        autograd, FakeTensor, and Meta modes.
        """
        wrapper_schema = info["param_schema"]

        def wrapper(self, *args, **kwargs):
            # Parse args using schema
            parsed = wrapper_schema.parse_method_args(self, args, kwargs)

            # Check if we need autograd
            has_requires_grad = parsed.has_requires_grad()
            fake_mode = torch._C._get_dispatch_mode(torch._C._TorchDispatchModeKey.FAKE)
            in_tracing_context = (
                fake_mode is not None
                or parsed.has_fake_or_functional_tensor()
                or parsed.has_meta()
            )

            if not has_requires_grad and not in_tracing_context:
                return original_method(self, *args, **kwargs)

            # Use the inplace op - it has Autograd kernel registered that handles
            # gradient tracking for all tensor types including tensor lists
            inplace_op_name = op_name + "_"
            result = getattr(torch.ops.torchcomms, inplace_op_name)(*parsed.to_values())

            async_op = parsed.get_value("async_op") or False

            # For sync ops with requires_grad, return the result tensors (which have grad_fn)
            # This allows callers like funcol._gather to use tensors with proper autograd tracking
            if not async_op:
                if has_requires_grad and result is not None:
                    return result
                return None

            # For async ops, wrap result with work handle
            if in_tracing_context:
                return FakeWork(result)
            else:
                # Eager context - wrap with registered work handle
                return _wrap_result_with_registered_work(result)

        return wrapper

    # Patch each class
    for target_class, methods in class_methods.items():
        for op_name, info in methods:
            method_name = info["name"]
            try:
                original_method = getattr(target_class, method_name)
                wrapper = _create_dispatch_wrapper(original_method, op_name, info)
                setattr(target_class, method_name, wrapper)
                logger.info(
                    "Patched %s.%s for tracing and autograd",
                    target_class.__name__,
                    method_name,
                )
            except Exception as e:
                logger.warning(
                    "Failed to patch %s.%s: %s", target_class.__name__, method_name, e
                )


_EFFECTFUL_HANDLES: list = []  # Store handles to prevent GC


def _register_effectful_ops() -> None:
    """Register torchcomms ops as effectful with ORDERED effect type.

    Both functional and inplace ops with global side effects (collective communication)
    need effect ordering to ensure correct execution order across ranks.
    The with_effects HOP wraps these ops during tracing.

    We register BOTH versions so that:
    - If we trace inplace ops, they get wrapped with with_effects
    - Functionalization then converts inplace to functional inside the wrapper
    - The effect ordering is preserved throughout
    """
    try:
        from torch._higher_order_ops.effects import _get_effect, _register_effectful_op
        from torch._library.effects import EffectType
    except ImportError:
        logger.info("Effect system not available, skipping effectful registration")
        return

    # Collect all op names to register (both functional and inplace versions)
    ops_to_register = []

    ops_to_register.extend(_REGISTERED_COLLECTIVES)
    ops_to_register.append("torchcomm_wait_tensors")

    logger.info(
        "_register_effectful_ops: Registering %s ops: %s",
        len(ops_to_register),
        ops_to_register,
    )

    for op_name in ops_to_register:
        try:
            op_packet = getattr(torch.ops.torchcomms, op_name, None)
            if op_packet is None:
                logger.warning("Op torch.comms.%s not found, skipping", op_name)
                continue
            # Get the default overload
            if hasattr(op_packet, "default"):
                torch_op = op_packet.default
                handle = _register_effectful_op(torch_op, EffectType.ORDERED)
                _EFFECTFUL_HANDLES.append(handle)  # Keep handle alive
                # Verify registration worked
                effect = _get_effect(torch_op)
                logger.info("Registered torch.comms.%s: effect=%s", op_name, effect)
        except Exception as e:
            logger.warning(
                "Failed to register torch.comms.%s as effectful: %s", op_name, e
            )


def finalize_registration(lib: Any) -> None:
    """Finalize registration of all collectives."""
    from torch.comms.functional.dynamo import (
        _patch_dynamo_for_opaque_methods,
        register_with_dynamo,
    )
    from torch.comms.functional.inductor_lowering import register_torchcomms_lowerings

    logger.info(
        "Finalizing registration for %s collectives", len(_REGISTERED_COLLECTIVES)
    )
    _generate_lib_ops(lib)
    _register_effectful_ops()
    _patch_dynamo_for_opaque_methods(
        _REGISTERED_COLLECTIVES,
        _TYPE_NAME_TO_CLASS,
    )
    register_torchcomms_lowerings()
    register_with_dynamo()

    # Patch eager methods for tracing and autograd support
    _patch_eager_methods()

    logger.info("Collective registration finalized")
