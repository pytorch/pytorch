from collections.abc import Callable
from typing import Any, Optional

import torch

from .. import graph_bytecode_inputs
from ..bytecode_transformation import create_call_function
from .base import VariableTracker
from .constant import ConstantVariable
from .ctx_manager import ContextWrappingVariable
from .dicts import ConstDictVariable
from .lists import TupleVariable


class CUDAMemPoolVariable(VariableTracker):
    """
    Represents a torch.cuda.MemPool object.
    Modeled after StreamVariable to handle the opaque C++ object lifecycle.
    """

    def __init__(
        self,
        proxy,
        value: torch.cuda.MemPool,
        user_object_index: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.proxy = proxy
        self.value = value
        self.user_object_index = user_object_index

    def var_getattr(self, tx, name):
        if name == "id":
            return ConstantVariable.create(self.value.id)
        return super().var_getattr(tx, name)

    def reconstruct(self, codegen):
        if self.source:
            return super().reconstruct(codegen)

        if self.user_object_index is not None:
            codegen.add_push_null(
                lambda: codegen.load_import_from(
                    graph_bytecode_inputs.__name__,
                    "get_external_object_by_index",
                )
            )
            codegen.append_output(codegen.create_load_const(self.user_object_index))
            codegen.extend_output(create_call_function(1, False))
        else:
            name = codegen.tx.output.install_global_by_id("mempool", self.value)
            codegen.append_output(codegen.create_load_global(name, add=True))

    @staticmethod
    def make_construct_in_graph_mempool_fn(
        args: TupleVariable, kwargs: ConstDictVariable
    ) -> Callable[[int, Any], None]:
        def fn(index: int, codegen: Any) -> None:
            codegen.add_push_null(
                lambda: codegen.load_import_from(
                    torch._dynamo.utils.__name__, "build_mempool"
                )
            )
            codegen(args)
            codegen(kwargs)
            codegen.extend_output(create_call_function(2, False))
            codegen.extend_output(create_call_function(1, False))

        return fn

    def module_name(self):
        return "torch.cuda"

    def fn_name(self):
        return "MemPool"


class CUDAMemPoolContextVariable(ContextWrappingVariable):
    """
    Represents the torch.cuda.use_mem_pool context manager.
    Handles the low-level C calls required to switch the allocator.
    """

    @staticmethod
    def create(tx, pool_var, device_var=None, **kwargs):
        if (
            device_var is None
            or isinstance(device_var, ConstantVariable)
            and device_var.value is None
        ):
            device_proxy = tx.output.create_proxy(
                "call_function", torch.cuda.current_device, (), {}
            )
        else:
            if isinstance(device_var, ConstantVariable):
                idx = torch.cuda._utils._get_device_index(device_var.value)
                device_proxy = tx.output.create_proxy("call_function", int, (idx,), {})
            else:
                device_proxy = tx.output.create_proxy(
                    "call_function",
                    torch.cuda._utils._get_device_index,
                    (device_var.as_proxy(),),
                    {},
                )

        return CUDAMemPoolContextVariable(
            target_values=[pool_var, device_proxy], initial_values=[], **kwargs
        )

    def enter(self, tx):
        pool_var, device_proxy = self.target_values

        if isinstance(pool_var, CUDAMemPoolVariable):
            pool_id = pool_var.value.id
        else:
            pool_id = pool_var.as_proxy().node.args[0].id

        tx.output.create_proxy(
            "call_function",
            torch.cuda.memory._cuda_beginAllocateCurrentThreadToPool,
            (device_proxy, pool_id),
            {},
        )

    def exit(self, tx, *args):
        pool_var, device_proxy = self.target_values

        if isinstance(pool_var, CUDAMemPoolVariable):
            pool_id = pool_var.value.id
        else:
            pool_id = pool_var.as_proxy().node.args[0].id

        tx.output.create_proxy(
            "call_function",
            torch.cuda.memory._cuda_endAllocateToPool,
            (device_proxy, pool_id),
            {},
        )

        tx.output.create_proxy(
            "call_function",
            torch.cuda.memory._cuda_releasePool,
            (device_proxy, pool_id),
            {},
        )
