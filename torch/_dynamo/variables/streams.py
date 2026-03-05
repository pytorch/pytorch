import collections
from collections.abc import Callable
from typing import Any, Optional

import torch
from torch._dynamo.variables.dicts import ConstDictVariable
from torch._dynamo.variables.lists import TupleVariable
from torch.fx import has_side_effect, Proxy

from .. import graph_break_hints
from ..bytecode_transformation import create_call_function
from ..exc import TYPE_CHECKING, unimplemented
from ..graph_bytecode_inputs import (
    get_external_object_by_index,
    register_graph_created_object,
)
from ..source import CurrentStreamSource
from .base import VariableTracker
from .constant import CONSTANT_VARIABLE_NONE, ConstantVariable
from .ctx_manager import FxTracebackAnnotateVariable
from .lazy import LazyVariableTracker


if TYPE_CHECKING:
    from torch._dynamo.symbolic_convert import InstructionTranslator

    from ..codegen import PyCodegen

from torch._library.custom_ops import custom_op


Tensor = torch.Tensor


def new_event(*args: Any, **kwargs: Any) -> int:
    event = torch.Event(*args, **kwargs)
    return register_graph_created_object(
        event,
        EventVariable.make_construct_in_graph_event_fn(
            TupleVariable([]), ConstDictVariable({})
        ),
    )


def new_stream(*args: tuple[Any], **kwargs: Any) -> int:
    stream = torch.Stream(*args, **kwargs)  # type: ignore[no-matching-overload,call-overload]
    return register_graph_created_object(
        stream,
        StreamVariable.make_construct_in_graph_stream_fn(
            TupleVariable([]), ConstDictVariable({})
        ),
    )


def _codegen_current_stream(device: torch.device, cg: "PyCodegen") -> None:
    cg.add_push_null(
        lambda: cg.load_import_from(
            torch._dynamo.graph_bytecode_inputs.__name__,  # type: ignore[implicit-imports]
            "stash_graph_created_object",
        )
    )
    cg(CurrentStreamSource(device))
    cg.extend_output(create_call_function(1, False))


def get_current_stream(device: torch.device) -> int:
    stream = torch.accelerator.current_stream(device)
    return register_graph_created_object(
        stream, lambda _, cg: _codegen_current_stream(device, cg)
    )


def _get_stream_by_index(index: int) -> torch.Stream:
    stream = get_external_object_by_index(index)
    assert isinstance(stream, torch.Stream), (
        f"Fork/join stream expected a stream object at index {index}"
    )
    return stream


def _get_event_by_index(index: int) -> torch.Event:
    event = get_external_object_by_index(index)
    assert isinstance(event, torch.Event), (
        f"Record/wait event expected an event object at index {index}"
    )
    return event


@custom_op("streams::fork", mutates_args=())
def fork_stream(
    from_index: int,  # kept to make stream transitions clearer
    to_index: int,
) -> None:
    torch.accelerator.set_stream(_get_stream_by_index(to_index))


@fork_stream.register_fake
def _(
    from_index: int,  # kept to make stream transitions clearer
    to_index: int,
) -> None:
    pass


has_side_effect(torch.ops.streams.fork.default)


@custom_op("streams::join", mutates_args=())
def join_stream(from_index: int, to_index: int) -> None:
    torch.accelerator.set_stream(_get_stream_by_index(to_index))


@join_stream.register_fake
def _(
    from_index: int,
    to_index: int,
) -> None:
    pass


has_side_effect(torch.ops.streams.join.default)


@custom_op("streams::record_event", mutates_args=())
def record_event(event_index: int, stream_index: int) -> None:
    event = _get_event_by_index(event_index)
    stream = _get_stream_by_index(stream_index)
    stream.record_event(event)


@record_event.register_fake
def _(
    event_index: int,
    stream_index: int,
) -> None:
    pass


has_side_effect(torch.ops.streams.record_event.default)


@custom_op("streams::wait_event", mutates_args=())
def wait_event(event_index: int, stream_index: int) -> None:
    event = _get_event_by_index(event_index)
    stream = _get_stream_by_index(stream_index)
    stream.wait_event(event)


@wait_event.register_fake
def _(
    event_index: int,
    stream_index: int,
) -> None:
    pass


has_side_effect(torch.ops.streams.wait_event.default)


@custom_op("streams::wait_stream", mutates_args=())
def wait_stream(waiting_stream_index: int, waited_on_stream_index: int) -> None:
    waiting = _get_stream_by_index(waiting_stream_index)
    waited_on = _get_stream_by_index(waited_on_stream_index)
    waiting.wait_stream(waited_on)


@wait_stream.register_fake
def _(
    event_index: int,
    stream_index: int,
) -> None:
    pass


has_side_effect(torch.ops.streams.wait_stream.default)


@custom_op("streams::sync_dealloc", mutates_args=())
def sync_dealloc(
    wait_event_index: int, src_stream_index: int, to_dealloc: torch.Tensor
) -> None:
    """An op which waits on an event and moves the last usage of to_dealloc
    after the wait, so that after the sync occurs, the deallocation or
    subsequent reuse of the tensor's memory will be guaranteed to happen
    after a side stream is finished using it.
    See https://docs.pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html#torch.Tensor.record_stream
    for more details"""
    torch.ops.streams.wait_event.default(wait_event_index, src_stream_index)


has_side_effect(torch.ops.streams.sync_dealloc.default)


@custom_op("streams::record_stream", mutates_args=())
def record_stream(tensor: torch.Tensor, stream_index: int) -> None:
    tensor.record_stream(_get_stream_by_index(stream_index))


@record_stream.register_fake
def _(
    src_stream_index: int,
    wait_event_index: int,
    to_dealloc: torch.Tensor,
) -> None:
    pass


class SymbolicStreamState:
    """Track the currently entered stream if any"""

    def __init__(self) -> None:
        from ..source import CurrentStreamSource

        cur_stack: list[StreamVariable] = []
        if torch.accelerator.is_available():
            stream_var = LazyVariableTracker.create(
                torch.accelerator.current_stream(),
                source=CurrentStreamSource(torch.accelerator.current_stream().device),
            )
            cur_stack = [stream_var]  # type: ignore[list-item]

        self.cur_stream_stack: collections.deque[StreamVariable] = collections.deque(
            cur_stack
        )

    def enter_stream(self, stream: "StreamVariable") -> None:
        self.cur_stream_stack.append(stream)

    def exit_stream(self) -> None:
        self.cur_stream_stack.pop()

    def cur_stream(self, device: torch.device | None = None) -> "StreamVariable":
        if device is not None:
            for stream in reversed(self.cur_stream_stack):
                if stream.device == device:
                    return stream

        return self.cur_stream_stack[-1]

    def in_stream_context(self) -> bool:
        return len(self.cur_stream_stack) > 0


class StreamContextVariable(FxTracebackAnnotateVariable):
    """This represents torch.cuda.StreamContext"""

    @staticmethod
    def create(
        tx: "InstructionTranslator",
        stream_to_enter: "StreamVariable",
        **kwargs: dict[str, Any],
    ) -> "StreamContextVariable":
        return StreamContextVariable(
            stream_to_enter,
            **kwargs,
        )

    def __init__(self, stream: Optional["StreamVariable"], **kwargs: Any) -> None:
        self.stream = stream
        super().__init__(
            target_values={"stream": self.get_stream().user_object_index},
            initial_values=None,
            **kwargs,
        )

    def enter(
        self, tx: "InstructionTranslator", *args: VariableTracker
    ) -> VariableTracker:
        # to stream, from stream is the order of the arguments
        # we are entering the target, and leaving the initial stream
        tx.symbolic_stream_state.enter_stream(self.get_stream())
        return super().enter(tx)

    def exit(
        self, tx: "InstructionTranslator", *args: VariableTracker
    ) -> VariableTracker:
        # to stream, from stream is the order of the arguments
        # we are leaving the target, and entering the initial stream
        tx.symbolic_stream_state.exit_stream()
        return super().exit(tx, *args)

    def supports_graph_breaks(self) -> bool:
        return True

    def get_stream(self) -> "StreamVariable":
        assert self.stream, "Stream context should have a separate stream"
        return self.stream


class StreamVariable(StreamContextVariable):
    """Represents the device-agnostic torch.Stream class"""

    def __init__(
        self,
        proxy: Proxy,
        value: torch.Stream,
        user_object_index: int | None = None,
        **kwargs: Any,
    ) -> None:
        # Index into the user object table
        # used to pass arbitrary objects to the graph
        if proxy is not None and "example_value" in proxy.node.meta:
            assert proxy.node.meta["example_value"] == value

        self.proxy = proxy
        self.value = value
        self.device = value.device

        self.user_object_index = user_object_index
        super().__init__(None, **kwargs)

    def python_type(self) -> type:
        return torch.Stream

    def call_method(
        self,
        tx: "InstructionTranslator",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        assert hasattr(self.value, name), f"no stream method found named {name}"

        from ..utils import cmp_name_to_op_mapping, proxy_args_kwargs
        from .builder import wrap_fx_proxy_cls

        if name in ("wait_stream", "synchronize", "wait_event"):
            tx.output.create_proxy(
                "call_method", name, *proxy_args_kwargs([self] + args, kwargs)
            )
            return CONSTANT_VARIABLE_NONE
        elif name == "query":
            return wrap_fx_proxy_cls(
                target_cls=ConstantVariable,
                tx=tx,
                proxy=tx.output.create_proxy(
                    "call_method", name, *proxy_args_kwargs([self] + args, kwargs)
                ),
            )
        elif name == "record_event":
            return wrap_fx_proxy_cls(
                target_cls=EventVariable,
                tx=tx,
                proxy=tx.output.create_proxy(
                    "call_method", name, *proxy_args_kwargs([self] + args, kwargs)
                ),
            )
        elif name in cmp_name_to_op_mapping and len(args) == 1 and not kwargs:
            from ..guards import GuardBuilder, install_guard

            if self.source:
                install_guard(self.source.make_guard(GuardBuilder.EQUALS_MATCH))

            # NB : Checking for mutation is necessary because we compare
            # constant values
            other = args[0]
            if not isinstance(other, StreamVariable):
                return VariableTracker.build(tx, NotImplemented)

            if other.source:
                assert self.source is not None
                install_guard(self.source.make_guard(GuardBuilder.EQUALS_MATCH))
            return VariableTracker.build(
                tx,
                cmp_name_to_op_mapping[name](self.value, other.value),  # type: ignore[arg-type]
            )

        return super().call_method(tx, name, args, kwargs)

    def as_proxy(self) -> Proxy:
        return self.proxy

    def module_name(self) -> str:
        return "torch._C"

    def fn_name(self) -> str:
        return "Stream"

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # If we got here, this stream is fully subsumed by the graph - this means it is
        # not an input or global
        assert not self.source
        if self.user_object_index is not None:
            codegen.add_push_null(
                lambda: codegen.load_import_from(
                    torch._dynamo.graph_bytecode_inputs.__name__,
                    "get_external_object_by_index",
                )
            )
            codegen.append_output(codegen.create_load_const(self.user_object_index))
            codegen.extend_output(create_call_function(1, False))
        else:
            # This will support the legacy behavior
            prefix = f"_stream_{self.device}"
            name = codegen.tx.output.install_global_by_id(prefix, self.value)
            codegen.append_output(codegen.create_load_global(name, add=True))

    def get_stream(self) -> "StreamVariable":
        return self

    @staticmethod
    def make_construct_in_graph_stream_fn(
        args: TupleVariable, kwargs: ConstDictVariable
    ) -> Callable[[int, "PyCodegen"], None]:
        def fn(index: int, codegen: "PyCodegen") -> None:
            codegen.add_push_null(
                lambda: codegen.load_import_from(
                    torch._dynamo.graph_bytecode_inputs.__name__,  # type: ignore[implicit-imports]
                    "stash_graph_created_object",
                )
            )
            codegen.add_push_null(
                lambda: codegen.load_import_from(
                    torch._dynamo.utils.__name__, "build_stream"
                )
            )
            codegen(args)
            codegen(kwargs)
            codegen.extend_output(create_call_function(2, False))
            codegen.extend_output(create_call_function(1, False))

        return fn


class EventVariable(VariableTracker):
    def __init__(
        self,
        proxy: Proxy,
        value: torch.Event,
        user_object_index: int | None,
        **kwargs: Any,
    ) -> None:
        if proxy is not None and "example_value" in proxy.node.meta:
            assert proxy.node.meta["example_value"] == value
        super().__init__(**kwargs)
        self.proxy = proxy
        self.value = value
        self.user_object_index = user_object_index

    def call_method(
        self,
        tx: "InstructionTranslator",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        from ..utils import proxy_args_kwargs
        from .builder import wrap_fx_proxy_cls

        if name == "wait":
            tx.output.create_proxy(
                "call_function",
                torch.ops.streams.wait_event,
                (
                    self.user_object_index,
                    EventVariable._get_stream_arg(tx, args, kwargs).user_object_index,
                ),
                {},
            )
            return CONSTANT_VARIABLE_NONE
        elif name == "record":
            tx.output.create_proxy(
                "call_function",
                torch.ops.streams.record_event,
                (
                    self.user_object_index,
                    EventVariable._get_stream_arg(tx, args, kwargs).user_object_index,
                ),
                {},
            )
            return CONSTANT_VARIABLE_NONE
        elif name == "synchronize":
            tx.output.create_proxy(
                "call_method", name, *proxy_args_kwargs([self] + args, kwargs)
            )
            return CONSTANT_VARIABLE_NONE
        elif name == "query":
            return wrap_fx_proxy_cls(
                target_cls=ConstantVariable,
                tx=tx,
                proxy=tx.output.create_proxy(
                    "call_method", name, *proxy_args_kwargs([self] + args, kwargs)
                ),
            )
        else:
            method_name = (
                f"{type(self.value).__module__}.{type(self.value).__qualname__}.{name}"
            )
            unimplemented(
                gb_type="Unsupported event method",
                context=str(name),
                explanation=f"Dynamo doesn't support tracing the {method_name} method. "
                f"We currently support wait, record, synchronize, and query.",
                hints=[
                    *graph_break_hints.SUPPORTABLE,
                ],
            )

    def as_proxy(self) -> Proxy:
        return self.proxy

    @staticmethod
    def _get_stream_arg(
        tx: "InstructionTranslator",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> "StreamVariable":
        stream_arg = None
        if args:
            stream_arg = args[0]
        elif kwargs:
            stream_arg = kwargs.get("stream")

        if not stream_arg:
            stream_arg = tx.symbolic_stream_state.cur_stream()

        return stream_arg  # type: ignore[return-value]

    @staticmethod
    def make_construct_in_graph_event_fn(
        args: TupleVariable, kwargs: ConstDictVariable
    ) -> Callable[[int, "PyCodegen"], None]:
        def fn(index: int, codegen: "PyCodegen") -> None:
            codegen.add_push_null(
                lambda: codegen.load_import_from(
                    torch._dynamo.graph_bytecode_inputs.__name__,  # type: ignore[implicit-imports]
                    "stash_graph_created_object",
                )
            )
            codegen.add_push_null(
                lambda: codegen.load_import_from(
                    torch._dynamo.utils.__name__, "build_event"
                )
            )
            codegen(args)
            codegen(kwargs)
            codegen.extend_output(create_call_function(2, False))
            codegen.extend_output(create_call_function(1, False))

        return fn

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # If we got here, this event is fully subsumed by the graph - this means it is
        # not an input or global
        assert not self.source
        # Similar to stream handling, we lift the event into a global and then codegen bytecode to load it from there.
        prefix = "_event"
        name = codegen.tx.output.install_global_by_id(prefix, self.value)
        codegen.append_output(codegen.create_load_global(name, add=True))
