from contextlib import ExitStack
from typing import Any, TYPE_CHECKING

import torch
from torch.fx import has_side_effect, Proxy

from ..bytecode_transformation import create_call_function
from ..exc import raise_type_error, unimplemented
from ..graph_bytecode_inputs import get_external_object_by_index
from ..guards import GuardBuilder, install_guard
from ..source import AttrSource, CallFunctionNoArgsSource, ImportSource
from .base import Member, readonly_setter, VariableTracker
from .constant import ConstantVariable
from .ctx_manager import ContextWrappingVariable


if TYPE_CHECKING:
    from torch._dynamo.device_interface import DeviceInterface
    from torch._dynamo.symbolic_convert import InstructionTranslatorBase

    from ..codegen import PyCodegen

from torch._library.custom_ops import custom_op


def _current_device_source(device_type: str) -> CallFunctionNoArgsSource:
    return CallFunctionNoArgsSource(
        AttrSource(AttrSource(ImportSource("torch"), device_type), "current_device")
    )


def _get_device_index_for(device_type: str, device: Any) -> int:
    if device_type == "cuda":
        return torch.cuda._utils._get_device_index(device)
    if device_type == "xpu":
        return torch.xpu._get_device_index(device)
    raise RuntimeError(f"mempool ops do not support device type {device_type}")


def pool_device_type(value: Any) -> str:
    """Map a MemPool instance to the device type of its registered interface."""
    from torch._dynamo.device_interface import (
        DeviceInterface,
        get_registered_device_interfaces,
    )

    for device_type, candidate in get_registered_device_interfaces():
        # Skip interfaces without the mempool contract.
        if not candidate.supports_mempool():
            continue
        if isinstance(value, candidate.get_mempool_type()):
            return device_type
    raise RuntimeError(f"expected a MemPool object, got {type(value).__qualname__}")


def _get_mempool_and_iface(
    mempool_index: int, device_type: str
) -> tuple[type["DeviceInterface"], Any]:
    """Fetch the mempool at ``mempool_index`` and its device interface.

    ``device_type`` is known at trace time (``MemPoolVariable`` carries it) and
    travels inside the marker ops, so this needs neither a pool-type cache nor
    a registry scan.
    """
    from torch._dynamo.device_interface import get_interface_for_device

    iface = get_interface_for_device(device_type)
    if not iface.supports_mempool():
        raise RuntimeError(f"mempool ops do not support device type {device_type}")
    mempool = get_external_object_by_index(mempool_index)
    if not isinstance(mempool, iface.get_mempool_type()):
        raise RuntimeError(
            f"use_mem_pool expected a "
            f"{iface.get_mempool_type().__qualname__} object at "
            f"index {mempool_index}, got {type(mempool).__qualname__}"
        )
    return iface, mempool


# These marker ops are intentional even though Inductor also annotates enclosed
# FX nodes. The annotations tell Inductor which scheduler nodes allocate in a
# user pool, while begin/end keep the pool lifetime visible as ordered FX
# side-effects for non-Inductor backends.
@custom_op("mempool::begin", mutates_args=())
def begin_mempool(device_index: int, mempool_index: int, device_type: str) -> None:
    iface, mempool = _get_mempool_and_iface(mempool_index, device_type)
    iface.check_mem_pool_device(mempool, device_index)
    iface.begin_allocate_current_thread_to_pool(device_index, mempool.id)


@begin_mempool.register_fake
def _(device_index: int, mempool_index: int, device_type: str) -> None:
    pass


has_side_effect(torch.ops.mempool.begin.default)


@custom_op("mempool::end", mutates_args=())
def end_mempool(device_index: int, mempool_index: int, device_type: str) -> None:
    iface, mempool = _get_mempool_and_iface(mempool_index, device_type)
    iface.end_allocate_current_thread_to_pool(device_index, mempool.id)
    iface.release_pool(device_index, mempool.id)


@end_mempool.register_fake
def _(device_index: int, mempool_index: int, device_type: str) -> None:
    pass


has_side_effect(torch.ops.mempool.end.default)


class MemPoolVariable(VariableTracker):
    """Represents a torch.cuda.MemPool or torch.xpu.MemPool object."""

    def __init__(
        self,
        proxy: Proxy,
        value: Any,
        user_object_index: int,
        device_type: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.proxy = proxy
        self.value = value
        self.user_object_index = user_object_index
        self.device_type = device_type

    def python_type(self) -> type:
        from torch._dynamo.device_interface import get_interface_for_device

        return get_interface_for_device(self.device_type).get_mempool_type()

    def get_real_python_backed_value(self) -> object:
        return self.value

    def _id(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        if self.source:
            install_guard(self.source.make_guard(GuardBuilder.ID_MATCH))
        return ConstantVariable.create(self.value.id)

    tp_members = {
        "id": Member(_id, readonly_setter),
    }

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


class MemPoolContextVariable(ContextWrappingVariable):
    """Represents a torch.cuda.use_mem_pool / torch.xpu.use_mem_pool context manager."""

    def __init__(
        self,
        mempool: MemPoolVariable,
        device_index: int,
        device_type: str,
        **kwargs: Any,
    ) -> None:
        self.mempool = mempool
        self.device_index = device_index
        self.device_type = device_type
        super().__init__(target_values=(), initial_values=None, **kwargs)

    @staticmethod
    def create(
        tx: "InstructionTranslatorBase",
        mempool: VariableTracker,
        device: VariableTracker | None = None,
        *,
        device_type: str,
        **kwargs: Any,
    ) -> "MemPoolContextVariable":
        if not isinstance(mempool, MemPoolVariable):
            raise_type_error(
                tx,
                f"torch.{device_type}.use_mem_pool() expected a "
                f"torch.{device_type}.MemPool argument",
            )
        if mempool.device_type != device_type:
            raise_type_error(
                tx,
                f"torch.{device_type}.use_mem_pool() got a "
                f"torch.{mempool.device_type}.MemPool argument",
            )

        if device is None or (
            isinstance(device, ConstantVariable) and device.value is None
        ):
            install_guard(
                _current_device_source(device_type).make_guard(
                    GuardBuilder.EQUALS_MATCH
                )
            )
            device_index = getattr(torch, device_type).current_device()
        elif device.is_python_constant():
            device_index = _get_device_index_for(
                device_type, device.as_python_constant()
            )
        else:
            unimplemented(
                gb_type="torch.cuda.use_mem_pool with non-constant device",
                context=f"device={device}",
                explanation=(
                    "Dynamo requires the device argument to "
                    "torch.cuda.use_mem_pool to be a Python constant."
                ),
                hints=[],
            )

        return MemPoolContextVariable(mempool, device_index, device_type, **kwargs)

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
            (self.device_index, self.mempool.user_object_index, self.device_type),
            {},
        )
        return ConstantVariable.create(None)

    def exit(
        self, tx: "InstructionTranslatorBase", *args: VariableTracker
    ) -> VariableTracker:
        tx.output.create_proxy(
            "call_function",
            torch.ops.mempool.end,
            (self.device_index, self.mempool.user_object_index, self.device_type),
            {},
        )
        self.cleanup_assert()
        return ConstantVariable.create(None)

    def module_name(self) -> str:
        return f"torch.{self.device_type}"

    def fn_name(self) -> str:
        return "use_mem_pool"

    def reconstruct_type(self, codegen: "PyCodegen") -> None:
        # Supporting graph breaks safely requires reconstructing the eager
        # torch.<device>.use_mem_pool context on the resume path while keeping
        # the compiled begin/end marker ops balanced around each compiled
        # segment.
        unimplemented(
            gb_type="torch.cuda.use_mem_pool graph break",
            context=str(self),
            explanation=(
                "Dynamo doesn't support graph breaks inside torch.cuda.use_mem_pool."
            ),
            hints=[],
        )
