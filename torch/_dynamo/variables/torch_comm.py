"""
Dynamo variable trackers for torchcomms objects.

This module provides specialized variable tracking for torchcomms opaque objects
during Dynamo tracing. TorchCommsObjectVariable extends TorchScriptObjectVariable
to handle collective method calls (e.g., all_reduce, broadcast) by looking up
ops registered via register_opaque_custom_method in opaque_object.py.
"""

import logging
from collections.abc import Sequence
from typing import Any, TYPE_CHECKING

import torch
from torch._library.opaque_object import (
    get_opaque_type_name,
    is_opaque_type,
)

from .misc import LambdaVariable
from .base import VariableTracker
from .constant import ConstantVariable
from .script_object import TorchScriptObjectVariable

if TYPE_CHECKING:
    from torch._dynamo.symbolic_convert import InstructionTranslator

logger = logging.getLogger(__name__)


class TorchCommsObjectVariable(TorchScriptObjectVariable):
    """Subclass of TorchScriptObjectVariable for torchcomms opaque objects.

    For registered collective methods, routes through TorchCommMethodVariable
    to generate torch op calls directly.
    """

    @classmethod
    def is_matching_value(cls, value: Any) -> bool:
        """Check if a value (FakeScriptObject or real object) is a torchcomms object."""
        if hasattr(value, "script_class_name"):
            return value.script_class_name.startswith("torchcomms.")
        return cls._is_torchcomms_type(type(value))

    @classmethod
    def _is_torchcomms_type(cls, user_cls: type) -> bool:
        if not is_opaque_type(user_cls):
            return False
        try:
            name = get_opaque_type_name(user_cls)
        except ValueError:
            return False
        return name is not None and name.startswith("torchcomms.")

    def var_getattr(self, tx: "InstructionTranslator", name: str) -> VariableTracker:
        if self._get_registered_method_metadata(name) is not None:
            # Always route registered collective methods through call_method,
            # regardless of whether the object has a source. This avoids
            # tracing through the eager wrapper which contains untraceable
            # calls like _get_dispatch_mode.
            return LambdaVariable(
                lambda *args, **kwargs: self.call_method(tx, name, args, kwargs)
            )
        return super().var_getattr(tx, name)

    def call_method(
        self,
        tx: "InstructionTranslator",
        name: str,
        args: Sequence[Any],
        kwargs: dict[str, Any],
    ) -> VariableTracker:
        op_info = self._get_registered_method_metadata(name)
        if op_info is not None:
            real_obj = self.value.real_obj  # pyrefly: ignore[missing-attribute]
            method_var = TorchCommMethodVariable(
                obj_var=self,
                method_name=name,
                target_class=type(real_obj),
                op_info=op_info,
            )
            return method_var.call_function(tx, args, kwargs)

        return super().call_method(tx, name, args, kwargs)

    def _get_registered_method_metadata(
        self, method_name: str
    ) -> dict[str, Any] | None:
        """Look up registered collective method metadata from torchcomms."""
        real_obj = self.value.real_obj  # pyrefly: ignore[missing-attribute]
        try:
            from torchcomms.functional.registry import get_registered_method_metadata  # type: ignore[import-not-found]
        except ImportError:
            return None
        return get_registered_method_metadata(type(real_obj), method_name)


class TorchCommMethodVariable(VariableTracker):
    """Variable that directly generates torch op calls for collective methods."""

    def __init__(
        self,
        obj_var: TorchScriptObjectVariable,
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
        from torch._dynamo.variables import TensorVariable
        from torch._dynamo.variables.builder import wrap_fx_proxy
        from torch._dynamo.variables.lists import BaseListVariable, ListVariable
        from torch._dynamo.variables.script_object import TorchScriptObjectVariable

        op_name = self.op_info["op_name"]
        schema = self.op_info["param_schema"]
        input_params = schema.input_params
        extra_params = schema.extra_params
        output_params = schema.output_params

        has_mutable_inputs = len(schema.mutable_params) > 0

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
        mutable_tensor_vars: list[VariableTracker] = []
        for i, arg_var in enumerate(arg_list):
            if isinstance(arg_var, ConstantVariable):
                proxy_args.append(arg_var.value)
            elif isinstance(arg_var, TensorVariable):
                proxy_args.append(arg_var.as_proxy())
                if async_op and i < len(input_params) and input_params[i].mutable:
                    mutable_tensor_vars.append(arg_var)
            elif isinstance(arg_var, TorchScriptObjectVariable):
                proxy_args.append(arg_var.as_proxy())
            elif isinstance(arg_var, ListVariable):
                proxy_args.append([t.as_proxy() for t in arg_var.items])
                if async_op and i < len(input_params) and input_params[i].mutable:
                    for item in arg_var.items:
                        if isinstance(item, TensorVariable):
                            mutable_tensor_vars.append(item)
            elif hasattr(arg_var, "as_proxy"):
                proxy_args.append(arg_var.as_proxy())
            else:
                if hasattr(arg_var, "value"):
                    proxy_args.append(arg_var.value)  # type: ignore[attr-defined]
                else:
                    raise ValueError(
                        f"Cannot convert argument {i} of type {type(arg_var)} to proxy"
                    )

        proxy = tx.output.create_proxy(
            "call_function",
            torch_op,
            tuple(proxy_args),
            {},
        )

        needs_async_dummy = schema.needs_async_dummy_return

        if has_mutable_inputs:
            result_var = wrap_fx_proxy(tx=tx, proxy=proxy)

            mutable_arg_indices = [idx - 1 for idx in schema.mutable_indices]

            if isinstance(result_var, BaseListVariable):
                result_tensors = list(result_var.items)
            else:
                result_tensors = [result_var]

            logger.debug(
                "Functional op %s: mutable_arg_indices=%s, len(args)=%d, len(result_tensors)=%d",
                op_name,
                mutable_arg_indices,
                len(args),
                len(result_tensors),
            )

            def unwrap_lazy(var: VariableTracker) -> VariableTracker:
                if hasattr(var, "realize") and callable(var.realize):
                    return var.realize()
                return var

            def find_parent_list_and_index(
                tensor_var: TensorVariable, tx: "InstructionTranslator"
            ) -> tuple[ListVariable | None, int | None]:
                from torch._dynamo.source import GetItemSource

                source = tensor_var.source
                if source is None:
                    return None, None

                if isinstance(source, GetItemSource):
                    base_source = source.base
                    index = source.index

                    if hasattr(tx.output, "input_source_to_var"):
                        if base_source in tx.output.input_source_to_var:
                            parent = tx.output.input_source_to_var[base_source]
                            if isinstance(parent, ListVariable):
                                return parent, index

                    if hasattr(base_source, "local_name"):
                        local_name = base_source.local_name  # pyre-ignore[16]
                        if local_name in tx.symbolic_locals:
                            parent = tx.symbolic_locals[local_name]
                            parent = unwrap_lazy(parent)
                            if isinstance(parent, ListVariable):
                                return parent, index

                return None, None

            collected_mutable_vars: list[VariableTracker] = []
            for i, mutable_idx in enumerate(mutable_arg_indices):
                if mutable_idx < len(args):
                    mutable_var = args[mutable_idx]
                    mutable_var = unwrap_lazy(mutable_var)
                    if isinstance(mutable_var, TensorVariable):
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

                                parent_list, index = find_parent_list_and_index(
                                    mutable_var, tx
                                )
                                if parent_list is not None and index is not None:
                                    logger.debug(
                                        "Found parent list, updating item at index %s",
                                        index,
                                    )
                                    if index < len(parent_list.items):
                                        new_items = list(parent_list.items)
                                        new_items[index] = result_tensor
                                        parent_list.items = new_items
                                        tx.output.side_effects.mutation(parent_list)
                    elif isinstance(mutable_var, ListVariable):
                        for item in mutable_var.items:
                            if isinstance(item, TensorVariable):
                                collected_mutable_vars.append(item)
                        new_items = list(mutable_var.items)
                        for j, _ in enumerate(mutable_var.items):
                            if j < len(result_tensors):
                                result_tensor = result_tensors[j]
                                if isinstance(result_tensor, TensorVariable):
                                    new_items[j] = result_tensor
                        mutable_var.items = new_items
                        tx.output.side_effects.mutation(mutable_var)

            if async_op:
                return AsyncWorkVariable(
                    result_tensors, mutable_vars=collected_mutable_vars
                )
            else:
                return ConstantVariable.create(None)

        elif output_params:
            return wrap_fx_proxy(tx=tx, proxy=proxy)
        elif async_op:
            if needs_async_dummy:
                dummy_var = wrap_fx_proxy(tx=tx, proxy=proxy)
                mutable_tensor_vars = [dummy_var]
            return AsyncWorkVariable(mutable_tensor_vars)
        else:
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
        self.mutable_vars = mutable_vars or []

    def as_proxy(self) -> None:
        return None

    def python_type(self) -> type:
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

    def _do_wait(self, tx: "InstructionTranslator") -> VariableTracker:
        from torch._dynamo.variables import TensorVariable
        from torch._dynamo.variables.builder import wrap_fx_proxy
        from torch._dynamo.variables.lists import BaseListVariable

        logger.info(
            "_do_wait called: tensor_vars=%d, mutable_vars=%d",
            len(self.tensor_vars),
            len(self.mutable_vars),
        )

        tensor_proxies = [tv.as_proxy() for tv in self.tensor_vars]
        proxy = tx.output.create_proxy(
            "call_function",
            torch.ops.torchcomms.torchcomm_wait_tensors_.default,
            (tensor_proxies,),
            {},
        )

        result_var = wrap_fx_proxy(tx=tx, proxy=proxy)

        if isinstance(result_var, BaseListVariable):
            result_tensors = list(result_var.items)
        else:
            result_tensors = [result_var]

        # Update tensor var proxies to point to waited results
        for i, tensor_var in enumerate(self.tensor_vars):
            if i < len(result_tensors):
                result_tensor = result_tensors[i]
                if isinstance(tensor_var, TensorVariable) and isinstance(
                    result_tensor, TensorVariable
                ):
                    tensor_var.proxy = result_tensor.proxy

        # Update original mutable input proxies after wait
        for i, mutable_var in enumerate(self.mutable_vars):
            if i < len(result_tensors):
                result_tensor = result_tensors[i]
                if isinstance(mutable_var, TensorVariable) and isinstance(
                    result_tensor, TensorVariable
                ):
                    mutable_var.proxy = result_tensor.proxy
                    if (
                        hasattr(mutable_var, "mutation_type")
                        and mutable_var.mutation_type is not None
                    ):
                        tx.output.side_effects.mutation(mutable_var)

                    if (
                        hasattr(mutable_var, "source")
                        and mutable_var.source is not None
                    ):
                        source = mutable_var.source
                        local_name = None
                        if hasattr(source, "local_name"):
                            local_name = source.local_name
                        elif hasattr(source, "name") and callable(source.name):
                            local_name = source.name()

                        if local_name and local_name in tx.symbolic_locals:
                            tx.symbolic_locals[local_name] = result_tensor  # type: ignore[assignment]

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
