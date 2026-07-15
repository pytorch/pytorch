from contextlib import ExitStack
from typing import Any, TYPE_CHECKING

import torch
from torch.fx import has_side_effect, Proxy

from ..bytecode_transformation import create_call_function
from ..exc import raise_type_error, unimplemented
from ..graph_bytecode_inputs import get_external_object_by_index
from ..guards import GuardBuilder, install_guard
from ..source import AttrSource, CallFunctionNoArgsSource, ImportSource
from .base import VariableTracker
from .constant import ConstantVariable
from .ctx_manager import ContextWrappingVariable


if TYPE_CHECKING:
    from torch._dynamo.symbolic_convert import InstructionTranslatorBase

    from ..codegen import PyCodegen

from torch._library.custom_ops import custom_op


def _get_mempool_by_index(index: int) -> torch.cuda.MemPool:
    mempool = get_external_object_by_index(index)
    if not isinstance(mempool, torch.cuda.MemPool):
        raise RuntimeError(
            f"use_mem_pool expected a torch.cuda.MemPool object at index {index}"
        )
    return mempool


def _current_cuda_device_source() -> CallFunctionNoArgsSource:
    return CallFunctionNoArgsSource(
        AttrSource(AttrSource(ImportSource("torch"), "cuda"), "current_device")
    )


# These marker ops are intentional even though Inductor also annotates enclosed
# FX nodes. The annotations tell Inductor which scheduler nodes allocate in a
# user pool, while begin/end keep the pool lifetime visible as ordered FX
# side-effects for non-Inductor backends.
@custom_op("mempool::begin", mutates_args=())
def begin_mempool(device_index: int, mempool_index: int) -> None:
    mempool = _get_mempool_by_index(mempool_index)
    torch.cuda.memory._cuda_beginAllocateCurrentThreadToPool(device_index, mempool.id)


@begin_mempool.register_fake
def _(device_index: int, mempool_index: int) -> None:
    pass


has_side_effect(torch.ops.mempool.begin.default)


@custom_op("mempool::end", mutates_args=())
def end_mempool(device_index: int, mempool_index: int) -> None:
    mempool = _get_mempool_by_index(mempool_index)
    torch.cuda.memory._cuda_endAllocateToPool(device_index, mempool.id)
    torch.cuda.memory._cuda_releasePool(device_index, mempool.id)


@end_mempool.register_fake
def _(device_index: int, mempool_index: int) -> None:
    pass


has_side_effect(torch.ops.mempool.end.default)


class CUDAMemPoolVariable(VariableTracker):
    """Represents a torch.cuda.MemPool object."""

    def __init__(
        self,
        proxy: Proxy,
        value: torch.cuda.MemPool,
        user_object_index: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.proxy = proxy
        self.value = value
        self.user_object_index = user_object_index

    def python_type(self) -> type:
        return torch.cuda.MemPool

    def get_real_python_backed_value(self) -> object:
        return self.value

    def getattro_impl(
        self, tx: "InstructionTranslatorBase", name: str
    ) -> VariableTracker:
        if name == "id":
            if self.source:
                install_guard(self.source.make_guard(GuardBuilder.ID_MATCH))
            return ConstantVariable.create(self.value.id)
        unimplemented(
            gb_type="unsupported torch.cuda.MemPool attribute",
            context=f"torch.cuda.MemPool.{name}",
            explanation="Dynamo only supports reading torch.cuda.MemPool.id.",
            hints=[],
        )

    def as_proxy(self) -> Proxy:
        return self.proxy

    def reconstruct(self, codegen: "PyCodegen") -> None:
        if self.source:
            return super().reconstruct(codegen)

        codegen.add_push_null(
            lambda: codegen.load_import_from(
                torch._dynamo.graph_bytecode_inputs.__name__,
                "get_external_object_by_index",
            )
        )
        codegen.append_output(codegen.create_load_const(self.user_object_index))
        codegen.extend_output(create_call_function(1, False))


class CUDAMemPoolContextVariable(ContextWrappingVariable):
    """Represents the torch.cuda.use_mem_pool context manager."""

    def __init__(
        self,
        mempool: CUDAMemPoolVariable,
        device_index: int,
        **kwargs: Any,
    ) -> None:
        self.mempool = mempool
        self.device_index = device_index
        super().__init__(target_values=(), initial_values=None, **kwargs)

    @staticmethod
    def create(
        tx: "InstructionTranslatorBase",
        mempool: VariableTracker,
        device: VariableTracker | None = None,
        **kwargs: Any,
    ) -> "CUDAMemPoolContextVariable":
        if not isinstance(mempool, CUDAMemPoolVariable):
            raise_type_error(
                tx,
                "torch.cuda.use_mem_pool() expected a torch.cuda.MemPool argument",
            )

        if device is None or (
            isinstance(device, ConstantVariable) and device.value is None
        ):
            install_guard(
                _current_cuda_device_source().make_guard(GuardBuilder.EQUALS_MATCH)
            )
            device_index = torch.cuda.current_device()
        elif device.is_python_constant():
            device_index = torch.cuda._utils._get_device_index(
                device.as_python_constant()
            )
        else:
            unimplemented(
                gb_type="torch.cuda.use_mem_pool with non-constant device",
                context=f"device={device}",
                explanation="Dynamo requires the device argument to torch.cuda.use_mem_pool to be a Python constant.",
                hints=[],
            )

        return CUDAMemPoolContextVariable(mempool, device_index, **kwargs)

    def enter(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        stack = ExitStack()
        stack.enter_context(
            torch.fx.traceback.annotate(
                {
                    "mempool": self.mempool.user_object_index,
                    "mempool_device": self.device_index,
                }
            )
        )
        stack.enter_context(torch.fx.traceback.preserve_node_meta())
        self.set_cleanup_hook(tx, lambda: stack.close())
        tx.output.create_proxy(
            "call_function",
            torch.ops.mempool.begin,
            (self.device_index, self.mempool.user_object_index),
            {},
        )
        return ConstantVariable.create(None)

    def exit(
        self, tx: "InstructionTranslatorBase", *args: VariableTracker
    ) -> VariableTracker:
        tx.output.create_proxy(
            "call_function",
            torch.ops.mempool.end,
            (self.device_index, self.mempool.user_object_index),
            {},
        )
        self.cleanup_assert()
        return ConstantVariable.create(None)

    def module_name(self) -> str:
        return "torch.cuda"

    def fn_name(self) -> str:
        return "use_mem_pool"

    def reconstruct_type(self, codegen: "PyCodegen") -> None:
        # Supporting graph breaks safely requires reconstructing the eager
        # torch.cuda.use_mem_pool context on the resume path while keeping the
        # compiled begin/end marker ops balanced around each compiled segment.
        unimplemented(
            gb_type="torch.cuda.use_mem_pool graph break",
            context=str(self),
            explanation="Dynamo doesn't support graph breaks inside torch.cuda.use_mem_pool.",
            hints=[],
        )
