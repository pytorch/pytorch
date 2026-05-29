# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Dynamo integration for torch.comms.

This module provides utilities for integrating torchcomms types with
torch.compile/dynamo. It patches dynamo to handle TorchComm method calls
like get_rank() and get_size() as constants during tracing, and collective
methods like all_reduce() as torch ops.

Usage:
    from torch.comms.functional.dynamo import register_with_dynamo
    register_with_dynamo()  # Call once at startup to enable dynamo support
"""

import logging
from collections.abc import Callable, Sequence
from typing import Any, TYPE_CHECKING

import torch
from torch._dynamo.variables.base import VariableTracker
from torch._dynamo.variables.constant import ConstantVariable
from torch._dynamo.variables.lists import ListVariable


if TYPE_CHECKING:
    from torch._dynamo.symbolic_convert import InstructionTranslator
    from torch._dynamo.variables.script_object import TorchScriptObjectVariable

__all__ = [
    "AsyncWorkVariable",
    "AsyncWorkWaitMethod",
    "TorchCommMethodVariable",
    "register_with_dynamo",
]

logger = logging.getLogger(__name__)


# Mapping from opaque type name to class, shared with registry.py
_TYPE_NAME_TO_CLASS: dict[str, type] = {}

# Registry mapping (target_class, method_name) to op_info for collective methods
# Populated by _patch_dynamo_for_opaque_methods from _REGISTERED_COLLECTIVES
_METHOD_TO_OP: dict[tuple[type, str], dict[str, Any]] = {}


class TorchCommMethodVariable(VariableTracker):
    """Variable that directly generates torch op calls for collective methods."""

    def __init__(
        self,
        obj_var: "TorchScriptObjectVariable",
        method_name: str,
        target_class: type,
        op_info: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.obj_var = obj_var
        self.method_name = method_name
        self.target_class = target_class
        self.op_info = op_info

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: Sequence[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # Directly call the torch op - no tracing through patched method
        from torch._dynamo.variables import TensorVariable
        from torch._dynamo.variables.builder import wrap_fx_proxy
        from torch._dynamo.variables.script_object import TorchScriptObjectVariable

        op_name = self.op_info["op_name"]
        schema = self.op_info["param_schema"]
        input_params = schema.input_params
        extra_params = schema.extra_params
        output_params = schema.output_params

        # Check if this op has mutable inputs (collective ops that need effect ordering)
        has_mutable_inputs = len(schema.mutable_params) > 0

        # Use in-place ops for mutating collectives - they now return the mutated tensors
        # like native PyTorch in-place ops. This enables proper mutation tracking in
        # functionalization, which will convert to functional ops and update data flow.
        if has_mutable_inputs:
            inplace_op_name = f"{op_name}_"
            torch_op = getattr(torch.ops.torchcomms, inplace_op_name).default
            logger.info("Using inplace op %s", inplace_op_name)
        else:
            torch_op = getattr(torch.ops.torchcomms, op_name).default
            logger.info("Using op %s", op_name)

        # Convert kwargs to positional args based on param order
        all_params = input_params + extra_params
        arg_list = list(args)

        # Fill in missing args from kwargs or defaults
        for i in range(len(arg_list), len(all_params)):
            param = all_params[i]
            if param.name in kwargs:
                arg_list.append(kwargs[param.name])
            elif param.has_default():
                arg_list.append(ConstantVariable.create(param.default_value))
            else:
                raise ValueError(f"Missing required argument: {param.name}")

        # Check if this is an async operation
        async_op = False
        extra_param_names = [p.name for p in extra_params]
        if "async_op" in extra_param_names:
            async_idx = len(input_params) + extra_param_names.index("async_op")
            if async_idx < len(arg_list):
                async_var = arg_list[async_idx]
                if isinstance(async_var, ConstantVariable):
                    async_op = async_var.value

        # Build proxy args: obj, *all_args
        proxy_args: list[Any] = [self.obj_var.as_proxy()]
        # Track mutable tensor variables for async work
        mutable_tensor_vars: list[VariableTracker] = []
        for i, arg_var in enumerate(arg_list):
            if isinstance(arg_var, ConstantVariable):
                proxy_args.append(arg_var.value)
            elif isinstance(arg_var, TensorVariable):
                proxy_args.append(arg_var.as_proxy())
                # Track mutable inputs for async work
                if async_op and i < len(input_params) and input_params[i].mutable:
                    mutable_tensor_vars.append(arg_var)
            elif isinstance(arg_var, TorchScriptObjectVariable):
                proxy_args.append(arg_var.as_proxy())
            elif isinstance(arg_var, ListVariable):
                # List of tensors - use .items which is a list of VariableTrackers
                proxy_args.append([t.as_proxy() for t in arg_var.items])
                if async_op and i < len(input_params) and input_params[i].mutable:
                    for item in arg_var.items:
                        if isinstance(item, TensorVariable):
                            mutable_tensor_vars.append(item)
            elif hasattr(arg_var, "as_proxy"):
                proxy_args.append(arg_var.as_proxy())
            else:
                # For other types (dicts, etc.), try to get the underlying value
                if hasattr(arg_var, "value"):
                    proxy_args.append(arg_var.value)  # type: ignore[attr-defined]
                else:
                    raise ValueError(
                        f"Cannot convert argument {i} of type {type(arg_var)} to proxy"
                    )

        # Create the proxy call
        proxy = tx.output.create_proxy(
            "call_function",
            torch_op,
            tuple(proxy_args),
            {},
        )

        # Check if this op returns a dummy tensor for async work tracking
        needs_async_dummy = schema.needs_async_dummy_return

        # Return appropriate variable based on output
        if has_mutable_inputs:
            # Functional op mutates in-place and returns the mutated input tensor(s).
            # We need to update the mutable input's proxy to point to the result
            # so that subsequent uses get the right tensor in the graph.
            result_var = wrap_fx_proxy(tx=tx, proxy=proxy)

            # Find mutable input params and their corresponding arg variables
            # mutable_indices gives indices in all_params (obj + inputs + extras)
            # but args here is just (inputs + extras), so we subtract 1
            mutable_arg_indices = [idx - 1 for idx in schema.mutable_indices]

            # Handle single vs multiple mutable outputs
            # result_var is TensorVariable for single output, TupleVariable for multiple
            from torch._dynamo.variables.lists import BaseListVariable

            if isinstance(result_var, BaseListVariable):
                # Multiple mutable outputs - unpack the tuple
                result_tensors = list(result_var.items)
            else:
                # Single mutable output
                result_tensors = [result_var]

            logger.debug(
                "Functional op %s: mutable_arg_indices=%s, len(args)=%s, len(result_tensors)=%s",
                op_name,
                mutable_arg_indices,
                len(args),
                len(result_tensors),
            )

            # Helper to unwrap LazyVariableTracker and get the actual variable
            def unwrap_lazy(var: VariableTracker) -> VariableTracker:
                """Realize LazyVariableTracker to get the actual underlying variable."""
                if hasattr(var, "realize") and callable(var.realize):
                    return var.realize()
                return var

            # Helper to find parent ListVariable if tensor is from a list
            def find_parent_list_and_index(
                tensor_var: TensorVariable, tx: "InstructionTranslator"
            ) -> tuple[ListVariable | None, int | None]:
                """Find parent ListVariable if this tensor came from list indexing."""
                from torch._dynamo.source import GetItemSource

                source = tensor_var.source
                if source is None:
                    return None, None

                # Check if source is from list indexing (GetItemSource)
                if isinstance(source, GetItemSource):
                    base_source = source.base
                    index = source.index

                    # Try to find the ListVariable from the base source
                    # Look it up in tx.output.input_source_to_var or symbolic_locals
                    if hasattr(tx.output, "input_source_to_var"):
                        if base_source in tx.output.input_source_to_var:
                            parent = tx.output.input_source_to_var[base_source]
                            if isinstance(parent, ListVariable):
                                return parent, index

                    # Also check symbolic_locals by name
                    if hasattr(base_source, "local_name"):
                        local_name = base_source.local_name  # pyre-ignore[16]
                        if local_name in tx.symbolic_locals:
                            parent = tx.symbolic_locals[local_name]
                            parent = unwrap_lazy(parent)
                            if isinstance(parent, ListVariable):
                                return parent, index

                return None, None

            # Update each mutable input variable's proxy to the corresponding result
            # This ensures subsequent uses of the input tensors use the op's outputs.
            # Also collect the original mutable vars for async work tracking
            collected_mutable_vars: list[VariableTracker] = []
            for i, mutable_idx in enumerate(mutable_arg_indices):
                if mutable_idx < len(args):
                    mutable_var = args[mutable_idx]
                    # Unwrap lazy variable to get actual ListVariable or TensorVariable
                    mutable_var = unwrap_lazy(mutable_var)
                    if isinstance(mutable_var, TensorVariable):
                        # Single tensor input - collect for async tracking
                        collected_mutable_vars.append(mutable_var)
                        if i < len(result_tensors):
                            result_tensor = result_tensors[i]
                            if isinstance(result_tensor, TensorVariable):
                                logger.debug(
                                    "Updating tensor proxy: %s -> %s",
                                    mutable_var.proxy,
                                    result_tensor.proxy,
                                )
                                mutable_var.proxy = result_tensor.proxy

                                # If this tensor is an element of a list, we need to:
                                # 1. Update the list's items to include the result tensor
                                # 2. Mark the list as mutated
                                parent_list, index = find_parent_list_and_index(
                                    mutable_var, tx
                                )
                                if parent_list is not None and index is not None:
                                    logger.debug(
                                        "Found parent list, updating item at index %s",
                                        index,
                                    )
                                    # Replace the item in the parent list
                                    if index < len(parent_list.items):
                                        new_items = list(parent_list.items)
                                        new_items[index] = result_tensor
                                        parent_list.items = new_items
                                        # Mark the list as mutated
                                        tx.output.side_effects.mutation(parent_list)
                    elif isinstance(mutable_var, ListVariable):
                        # List of tensors - collect individual items for async tracking
                        for item in mutable_var.items:
                            if isinstance(item, TensorVariable):
                                collected_mutable_vars.append(item)
                        # IMPORTANT: We replace the items entirely (not just update proxies)
                        # because Dynamo may track source information that we need to update
                        new_items = list(mutable_var.items)  # Make a copy to modify
                        for j, _ in enumerate(mutable_var.items):
                            if j < len(result_tensors):
                                result_tensor = result_tensors[j]
                                if isinstance(result_tensor, TensorVariable):
                                    new_items[j] = result_tensor
                        # Update the ListVariable's items to use the result tensors
                        mutable_var.items = new_items
                        # CRITICAL: Mark the ListVariable as mutated so Dynamo knows
                        # to regenerate it from items instead of loading from source.
                        # This triggers the side_effects system to rebuild the list
                        # with the new contents after the graph runs.
                        tx.output.side_effects.mutation(mutable_var)

            # Return appropriate variable
            if async_op:
                # For async ops, pass both result_tensors (for wait_tensors call)
                # and mutable_vars (for updating proxies after wait)
                return AsyncWorkVariable(
                    result_tensors, mutable_vars=collected_mutable_vars
                )
            else:
                return ConstantVariable.create(None)

        elif output_params:
            # Has tensor output - use wrap_fx_proxy to create TensorVariable
            return wrap_fx_proxy(tx=tx, proxy=proxy)
        elif async_op:
            # Async op without mutable inputs - use dummy tensor for work tracking
            if needs_async_dummy:
                dummy_var = wrap_fx_proxy(tx=tx, proxy=proxy)
                mutable_tensor_vars = [dummy_var]
            return AsyncWorkVariable(mutable_tensor_vars)
        else:
            # Sync non-mutating op (like barrier) - return None
            return ConstantVariable.create(None)


class AsyncWorkVariable(VariableTracker):
    """Variable tracker for async work handles.

    When wait() is called, generates the torchcomm_wait_tensors op.
    Tracks both the result tensors (for wait input) and the original mutable
    tensor variables (for proxy updates after wait).
    """

    def __init__(
        self,
        tensor_vars: list[VariableTracker],
        mutable_vars: list[VariableTracker] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.tensor_vars = tensor_vars
        self.mutable_vars = mutable_vars if mutable_vars is not None else []

    def as_proxy(self) -> None:
        # AsyncWorkVariable doesn't have a direct graph representation.
        # The work is tracked through the tensor mutations and wait calls.
        # Return None as proxy since the work handle itself isn't in the graph.
        return None

    def python_type(self) -> type:
        # Return a placeholder type for the work handle
        return type(None)

    def var_getattr(self, tx: "InstructionTranslator", name: str) -> VariableTracker:
        if name == "wait":
            return AsyncWorkWaitMethod(self)
        raise AttributeError(f"AsyncWorkVariable has no attribute {name}")

    def call_method(
        self,
        tx: "InstructionTranslator",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        if name == "wait":
            return self._do_wait(tx)
        raise AttributeError(f"AsyncWorkVariable has no method {name}")

    # for simplicity, right now we always use wait_tensors, which takes a list of all mutable
    # inputs to the waited function, if applicable.
    #
    # however, we can also use wait_tensor, which takes a single tensor, if the waited function
    # only takes a single mutable tensor as input. then we don't have to deal with the list semantics
    # of wait_tensors.
    def _do_wait(self, tx: "InstructionTranslator") -> VariableTracker:
        from torch._dynamo.variables import TensorVariable
        from torch._dynamo.variables.builder import wrap_fx_proxy
        from torch._dynamo.variables.lists import BaseListVariable

        logger.debug(
            "_do_wait called: tensor_vars=%s, mutable_vars=%s",
            len(self.tensor_vars),
            len(self.mutable_vars),
        )

        # Generate wait_tensors call with all tensors (mutable inputs or dummy)
        tensor_proxies = [tv.as_proxy() for tv in self.tensor_vars]
        proxy = tx.output.create_proxy(
            "call_function",
            torch.ops.torchcomms.torchcomm_wait_tensors_.default,
            (tensor_proxies,),
            {},
        )

        # Wrap the result - wait_tensors returns the waited tensors
        result_var = wrap_fx_proxy(tx=tx, proxy=proxy)

        logger.debug(
            "wrap_fx_proxy returned: %s, type=%s", result_var, type(result_var).__name__
        )

        # Extract result tensors
        if isinstance(result_var, BaseListVariable):
            result_tensors = list(result_var.items)
            logger.debug(
                "result_var is BaseListVariable with %s items", len(result_tensors)
            )
        else:
            result_tensors = [result_var]
            logger.debug("result_var is not a list, wrapping as single item")

        # Update each tensor variable's proxy to point to the waited result
        # This ensures that subsequent uses of the tensors (including returns)
        # use the waited outputs, not the pre-wait tensors
        for i, tensor_var in enumerate(self.tensor_vars):
            if i < len(result_tensors):
                result_tensor = result_tensors[i]
                if isinstance(tensor_var, TensorVariable) and isinstance(
                    result_tensor, TensorVariable
                ):
                    tensor_var.proxy = result_tensor.proxy

        # also update the original mutable input variables' proxies
        # after some collective (e.g., _all_reduce), mutable_var.proxy == tensor_var.proxy (same value),
        # but when we update tensor_var.proxy above, mutable_var.proxy still points
        # to the old value. we need to update mutable_var.proxy to the wait result
        # so that when the function returns the original inputs, they use the waited
        # tensors in the graph.
        for result_idx, mutable_var in enumerate(self.mutable_vars):
            if result_idx < len(result_tensors):
                result_tensor = result_tensors[result_idx]
                if isinstance(mutable_var, TensorVariable) and isinstance(
                    result_tensor, TensorVariable
                ):
                    logger.debug(
                        "Updating mutable_var proxy after wait: %s -> %s",
                        mutable_var.proxy,
                        result_tensor.proxy,
                    )
                    mutable_var.proxy = result_tensor.proxy
                    # mark as mutated so dynamo regenerates from new proxy
                    if (
                        hasattr(mutable_var, "mutation_type")
                        and mutable_var.mutation_type is not None
                    ):
                        tx.output.side_effects.mutation(mutable_var)

                    # also try to update symbolic_locals if this variable came from a local
                    # This ensures that when the function returns the variable by name,
                    # dynamo uses the updated tensor with the waited proxy
                    if (
                        hasattr(mutable_var, "source")
                        and mutable_var.source is not None
                    ):
                        source = mutable_var.source
                        logger.debug(
                            "mutable_var source type: %s, source: %s",
                            type(source).__name__,
                            source,
                        )
                        local_name = None
                        # LocalSource (function parameters)
                        if hasattr(source, "local_name"):
                            local_name = source.local_name  # pyre-ignore[16]
                        # other source types that might have name()
                        elif hasattr(source, "name") and callable(source.name):
                            local_name = source.name()

                        if local_name and local_name in tx.symbolic_locals:
                            logger.debug(
                                "Replacing symbolic_locals[%s] with result_tensor",
                                local_name,
                            )
                            tx.symbolic_locals[local_name] = result_tensor
                        elif local_name:
                            logger.debug(
                                "local_name=%s not in symbolic_locals. Keys: %s",
                                local_name,
                                list(tx.symbolic_locals.keys()),
                            )
                    else:
                        logger.debug("mutable_var has no source or source is None")

        return ConstantVariable.create(None)


class AsyncWorkWaitMethod(VariableTracker):
    """Variable for the wait() method of AsyncWorkVariable."""

    def __init__(self, work_var: AsyncWorkVariable, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.work_var = work_var

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: Sequence[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        return self.work_var._do_wait(tx)


def _create_op_wrapper_with_defaults(
    fake_self: Any,
    torch_op: Any,
    op_info: dict[str, Any],
) -> Callable[..., Any]:
    """Create a wrapper function that applies default values before calling the torch op.

    This is used when FakeScriptObject.__getattribute__ returns a callable for collective
    methods. The wrapper ensures that default values from param specs are applied when
    the caller omits optional arguments.

    Uses in-place ops (which now return mutated tensors) for proper mutation tracking.
    Functionalization will convert to functional ops and maintain data flow.
    For async ops (async_op=True), returns a FakeWork object for wait() support.

    Args:
        fake_self: The FakeScriptObject instance (passed as first arg to op)
        torch_op: The torch op to call (in-place version for mutating ops)
        op_info: Dict containing param_schema for parsing args

    Returns:
        A callable that applies defaults and calls the torch op
    """
    from torch.comms.functional.async_tensor import FakeWork

    schema = op_info["param_schema"]
    has_mutable_inputs = len(schema.mutable_params) > 0

    def op_wrapper(*args, **kwargs):
        # Use ParsedArgs to handle defaults and argument parsing
        parsed = schema.parse_method_args(fake_self, args, kwargs)

        # Call the op with fake_self as first argument
        # In-place ops now return the mutated tensors
        result = torch_op(*parsed.to_values())

        # Check if async_op is True
        is_async = parsed.get_value("async_op") or False

        if has_mutable_inputs:
            # For async ops, return a FakeWork object that tracks the result tensors
            if is_async:
                return FakeWork(result)

            # Sync ops - return None (caller will use the original tensor which
            # has in-place semantics from user's perspective)
            return None

        # Non-mutating ops just return the result
        return result

    return op_wrapper


def _patch_dynamo_for_opaque_methods(
    registered_collectives: dict[str, dict[str, Any]],
    type_name_to_class: dict[str, type],
) -> None:
    """Patch TorchScriptObjectVariable.var_getattr to allow registered methods on opaque types.

    This enables using the original API (comm.all_reduce(...)) while still treating
    the objects as opaque. When a registered method is accessed, we return a custom
    variable that generates the torch op call directly (bypassing method inlining).

    Args:
        registered_collectives: Dict mapping op_name to collective info from registry.
        type_name_to_class: Dict mapping opaque type names to classes.
    """
    # Update our global type name to class mapping
    _TYPE_NAME_TO_CLASS.update(type_name_to_class)

    # Build mapping of (class, method_name) -> op_info
    for op_name, info in registered_collectives.items():
        target_class = info["target_class"]
        method_name = info["name"]
        _METHOD_TO_OP[(target_class, method_name)] = {
            "op_name": op_name,
            **info,
        }
        logger.info(
            "Registered method mapping: (%s, %s) -> %s",
            target_class.__name__,
            method_name,
            op_name,
        )

    if not _METHOD_TO_OP:
        return

    logger.info("_TYPE_NAME_TO_CLASS = %s", _TYPE_NAME_TO_CLASS)

    from torch._library.fake_class_registry import FakeScriptObject

    # Capture the original before patching (closure variable, not global)
    _original_fake_getattr = FakeScriptObject.__getattribute__

    def patched_fake_getattr(self, name):
        # Check if this is a collective method - return a wrapper that invokes the torch op
        # with default values applied from the param specs
        # Note: Constant methods (get_rank, get_size, etc.) are now handled by
        # register_opaque_type with members=MemberType.USE_REAL
        try:
            script_class_name = object.__getattribute__(self, "script_class_name")
            target_class = _TYPE_NAME_TO_CLASS.get(script_class_name)
            if target_class is not None:
                key = (target_class, name)
                if key in _METHOD_TO_OP:
                    op_info = _METHOD_TO_OP[key]
                    op_name = op_info["op_name"]
                    schema = op_info["param_schema"]

                    # Use in-place ops for mutating collectives - they now return tensors
                    # like native PyTorch in-place ops. Functionalization will convert to
                    # functional ops and update data flow.
                    if len(schema.mutable_params) > 0:
                        inplace_op_name = f"{op_name}_"
                        torch_op = getattr(torch.ops.torchcomms, inplace_op_name)
                    else:
                        torch_op = getattr(torch.ops.torchcomms, op_name)

                    # Return a wrapper that applies defaults before calling the op
                    return _create_op_wrapper_with_defaults(self, torch_op, op_info)
        except AttributeError:
            pass

        # Fall back to original FakeScriptObject behavior
        return _original_fake_getattr(self, name)

    FakeScriptObject.__getattribute__ = patched_fake_getattr  # type: ignore[method-assign]
    logger.info(
        "Patched FakeScriptObject.__getattribute__ to handle collective methods"
    )

    # Note: We intentionally do NOT register FakeScriptObject as a pytree node.
    # Unregistered types are automatically treated as leaves by default, which is
    # the correct behavior for opaque objects like FakeScriptObject.

    # Note: The actual var_getattr patching is done in _patch_var_getattr
    # which handles collective methods (constant methods are now handled by
    # register_opaque_type with members=MemberType.USE_REAL)
    logger.info(
        "Registered %s collective methods for dynamo patching", len(_METHOD_TO_OP)
    )


def _patch_var_getattr() -> None:
    """Patch TorchScriptObjectVariable.var_getattr for collective methods.

    Constant methods (get_rank, get_size, etc.) are now handled by
    register_opaque_type with members=MemberType.USE_REAL.
    """
    from torch._dynamo.source import AttrSource
    from torch._dynamo.variables.script_object import TorchScriptObjectVariable

    # Store the original method
    original_var_getattr = TorchScriptObjectVariable.var_getattr

    def patched_var_getattr(
        self: TorchScriptObjectVariable,
        tx: "InstructionTranslator",
        name: str,
    ) -> VariableTracker:
        # Check if this is one of our registered opaque types
        value = self.value
        target_class = None

        # Handle FakeScriptObject wrapping - get the class from script_class_name
        if hasattr(value, "script_class_name"):
            script_class_name = object.__getattribute__(value, "script_class_name")
            target_class = _TYPE_NAME_TO_CLASS.get(script_class_name)
        else:
            # Not a FakeScriptObject - check if it's a known class directly
            for cls in _TYPE_NAME_TO_CLASS.values():
                if isinstance(value, cls):
                    target_class = cls
                    break

        if target_class is not None:
            key = (target_class, name)

            # Check for collective methods (all_reduce, broadcast, etc.)
            if key in _METHOD_TO_OP:
                source = AttrSource(self.source, name) if self.source else None  # type: ignore[call-arg]
                op_info = _METHOD_TO_OP[key]
                logger.info(
                    "Returning TorchCommMethodVariable for %s.%s",
                    target_class.__name__,
                    name,
                )
                return TorchCommMethodVariable(
                    obj_var=self,
                    method_name=name,
                    target_class=target_class,
                    op_info=op_info,
                    source=source,
                )

        return original_var_getattr(self, tx, name)

    # Apply the patch
    TorchScriptObjectVariable.var_getattr = patched_var_getattr  # type: ignore[method-assign]

    logger.info(
        "Patched TorchScriptObjectVariable.var_getattr for %s collective methods",
        len(_METHOD_TO_OP),
    )


def register_with_dynamo() -> None:
    """
    Register TorchComm method handlers with dynamo.

    This patches TorchScriptObjectVariable.var_getattr to intercept TorchComm
    collective method calls and generate torch op calls directly.

    Constant methods (get_rank, get_size, etc.) are handled by register_opaque_type
    with members=MemberType.USE_REAL in collectives.py.
    """
    # Import _TYPE_NAME_TO_CLASS from param_parsing to merge registrations
    try:
        from torch.comms.functional.param_parsing import (
            _TYPE_NAME_TO_CLASS as REGISTRY_TYPE_NAME_TO_CLASS,
        )

        # Merge registry's type mappings into ours
        _TYPE_NAME_TO_CLASS.update(REGISTRY_TYPE_NAME_TO_CLASS)
    except ImportError:
        pass

    # Apply the var_getattr patch
    _patch_var_getattr()
