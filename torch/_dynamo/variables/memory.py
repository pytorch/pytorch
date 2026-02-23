import collections
from collections.abc import Callable
from typing import Any, Optional, TYPE_CHECKING

import torch

from .. import graph_bytecode_inputs
from ..bytecode_transformation import create_call_function
from .base import VariableTracker
from .constant import ConstantVariable
from .ctx_manager import FxTracebackAnnotateVariable
from .dicts import ConstDictVariable
from .lists import TupleVariable


if TYPE_CHECKING:
    from torch._dynamo.symbolic_convert import InstructionTranslator

    from ..codegen import PyCodegen


class SymbolicMempoolState:
    """Track the currently entered mempool if any.
    Similar to SymbolicStreamState but for memory pools.
    The mempool id is a tuple of two ints (MempoolId_t in C++).
    """

    def __init__(self) -> None:
        # Stack of CUDAMemPoolVariable objects currently active
        self.cur_mempool_stack: collections.deque[CUDAMemPoolVariable] = (
            collections.deque()
        )

    def enter_mempool(self, mempool: "CUDAMemPoolVariable") -> None:
        self.cur_mempool_stack.append(mempool)

    def exit_mempool(self) -> None:
        self.cur_mempool_stack.pop()

    def cur_mempool(self) -> Optional["CUDAMemPoolVariable"]:
        if len(self.cur_mempool_stack) > 0:
            return self.cur_mempool_stack[-1]
        return None

    def in_mempool_context(self) -> bool:
        return len(self.cur_mempool_stack) > 0


class CUDAMemPoolVariable(VariableTracker):
    """
    Represents a torch.cuda.MemPool object.
    Modeled after StreamVariable to handle the opaque C++ object lifecycle.
    The mempool id is a tuple of two ints (MempoolId_t in C++).
    """

    def __init__(
        self,
        proxy: Any,
        value: torch.cuda.MemPool,
        user_object_index: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.proxy = proxy
        self.value = value
        self.user_object_index = user_object_index

    def var_getattr(self, tx: "InstructionTranslator", name: str) -> VariableTracker:
        if name == "id":
            return ConstantVariable.create(self.value.id)
        return super().var_getattr(tx, name)

    def reconstruct(self, codegen: "PyCodegen") -> None:
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
    ) -> Callable[[int, "PyCodegen"], None]:
        def fn(index: int, codegen: "PyCodegen") -> None:
            codegen.add_push_null(
                lambda: codegen.load_import_from(
                    graph_bytecode_inputs.__name__,
                    "stash_graph_created_object",
                )
            )
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

    def module_name(self) -> str:
        return "torch.cuda"

    def fn_name(self) -> str:
        return "MemPool"

    def as_proxy(self) -> Any:
        return self.proxy


class CUDAMemPoolContextVariable(FxTracebackAnnotateVariable):
    """
    Represents the torch.cuda.use_mem_pool context manager.
    Handles the low-level C calls required to switch the allocator.
    Uses FxTracebackAnnotateVariable to annotate FX nodes with mempool information.
    """

    @staticmethod
    def create(
        tx: "InstructionTranslator",
        pool_var: "CUDAMemPoolVariable",
        device_var: Optional[VariableTracker] = None,
        **kwargs: Any,
    ) -> "CUDAMemPoolContextVariable":
        return CUDAMemPoolContextVariable(pool_var=pool_var, **kwargs)

    def __init__(
        self,
        pool_var: "CUDAMemPoolVariable",
        # device_proxy: Any,
        **kwargs: Any,
    ) -> None:
        self.pool_var = pool_var
        # self.device_proxy = device_proxy
        # The mempool id is a tuple of two ints (MempoolId_t)
        # We use the user_object_index to identify the pool in annotations
        super().__init__(
            target_values={"mempool": self.get_mempool().user_object_index},
            initial_values=None,
            **kwargs,
        )

    def enter(
        self, tx: "InstructionTranslator", *args: VariableTracker
    ) -> VariableTracker:
        # Track the mempool context
        tx.symbolic_mempool_state.enter_mempool(self.get_mempool())

        return super().enter(tx)

    def exit(
        self, tx: "InstructionTranslator", *args: VariableTracker
    ) -> VariableTracker:
        # Exit the mempool context
        tx.symbolic_mempool_state.exit_mempool()

        return super().exit(tx, *args)

    def supports_graph_breaks(self) -> bool:
        return True

    def get_mempool(self) -> "CUDAMemPoolVariable":
        return self.pool_var

    def module_name(self) -> str:
        return "torch.cuda"

    def fn_name(self) -> str:
        return "use_mem_pool"
