from collections import deque
from dataclasses import dataclass
from typing import Any

import torch
from torch.fx import Node, Proxy
from torch.utils._ordered_set import OrderedSet

from .. import graph_break_hints
from ..exc import TYPE_CHECKING, unimplemented_v2
from .base import VariableTracker
from .constant import ConstantVariable


if TYPE_CHECKING:
    from torch._dynamo.symbolic_convert import InstructionTranslator
    from ..codegen import PyCodegen

from torch._library.custom_ops import custom_op


# Avoid circular dependency for the dataclass
TensorVariable = Any
Tensor = torch.Tensor


@custom_op("streams::fork", mutates_args={"args"})
def fork_stream_(
    index: int, device: torch.device, device_index: int, args: list[Tensor]
) -> None:
    pass


@fork_stream_.register_fake
def _(index: int, device: torch.device, device_index: int, args: list[Tensor]) -> None:
    pass


@custom_op("streams::join", mutates_args={"args"})
def join_stream_(
    index: int, device: torch.device, device_index: int, args: list[Tensor]
) -> None:
    pass


@join_stream_.register_fake
def _(index: int, device: torch.device, device_index: int, args: list[Tensor]) -> None:
    pass


# Stream state consists of the fork stream node
# and the external to the stream that are accessed from within the
# stream
@dataclass
class StreamState:
    # the fork node that initiated the creation of this stream state
    # we will finalize it once the stream state is popped
    fork_node: Node
    # Nodes not created within the stream
    external_nodes: OrderedSet[Node]
    # Nodes created within the stream
    internal_nodes: OrderedSet[Node]


class StreamStateManager:
    """
    Class used to track the current stream context we are in and identify
    any used tensors as external (created outside the stream context) or
    internal (created within the stream context). We use this information to
    ensure the fork op is dependent on any external tensors, so that it will not
    be reordered before them or after ops which use the externally created tensors.
    Analagously, we use the internal tensors to ensure that the join op is not
    reordered before any internally created tensors or after ops which use the
    internally created tensors.

    To actually implement this, we have a stack of stream states which track any external tensors that
    have not yet been seen within the stream context and any tensors created within the stream context.
    Once we exit the stream context we populate the args of fork with all external tensors which have been used,
    and join with any internal tensors that were created.
    """

    def __init__(self) -> None:
        self.state_stack: deque[StreamState] = deque()

    def in_stream_context(self) -> bool:
        return bool(self.state_stack)

    def track_internal_node(self, node: Node) -> None:
        # if we are in a stream context, all created nodes are internal
        if self.in_stream_context():
            # if we have seen the node before, it is an internal
            self._cur_state().internal_nodes.add(node)

    def track_node(self, node: Node) -> None:
        # If we are in a stream context, args of ops may be external
        if self.in_stream_context() and node not in self._internal_nodes():
            self._external_nodes().add(node)

    def push_stream_state(self, node: Node) -> None:
        self.state_stack.append(StreamState(node, OrderedSet(), OrderedSet()))

    def pop_stream_state(self) -> StreamState:
        assert self.state_stack, "No stream state to pop"
        return self.state_stack.pop()

    def _cur_state(self) -> StreamState:
        assert self.state_stack, "No stream state to pop"
        return self.state_stack[-1]

    def _internal_nodes(self) -> OrderedSet[Node]:
        return self._cur_state().internal_nodes

    def _external_nodes(self) -> OrderedSet[Node]:
        return self._cur_state().external_nodes


stream_state_mgr = StreamStateManager()


class StreamVariable(VariableTracker):
    def __init__(
        self,
        proxy: Proxy,
        value: torch.Stream,
        device: torch.device,
        **kwargs: Any,
    ) -> None:
        if proxy is not None and "example_value" in proxy.node.meta:
            assert proxy.node.meta["example_value"] == value
        assert value.device.type == device.type, (
            "stream value is not equal to the passed device"
        )
        super().__init__(**kwargs)
        self.proxy = proxy
        self.value = value
        self.device = device

    def python_type(self) -> type:
        return torch.Stream

    def call_method(
        self,
        tx: "InstructionTranslator",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> "VariableTracker":
        assert hasattr(self.value, name), f"no stream method found named {name}"

        from ..utils import cmp_name_to_op_mapping, proxy_args_kwargs
        from .builder import wrap_fx_proxy_cls

        if name in ("wait_stream", "synchronize", "wait_event"):
            tx.output.create_proxy(
                "call_method", name, *proxy_args_kwargs([self] + args, kwargs)
            )
            return ConstantVariable(None)
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
                return ConstantVariable.create(NotImplemented)

            if other.source:
                install_guard(self.source.make_guard(GuardBuilder.EQUALS_MATCH))
            return ConstantVariable.create(
                cmp_name_to_op_mapping[name](self.value, other.value)  # type: ignore[arg-type]
            )

        return super().call_method(tx, name, args, kwargs)

    def as_proxy(self) -> Proxy:
        return self.proxy

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # If we got here, this stream is fully subsumed by the graph - this means it is
        # not an input or global
        assert not self.source
        # Since we just proved that - for other such structures, like lists and dicts, reconstruction
        # is fine and sound according to dynamo principles of treating collectives. However,
        # streams are special in that we want to preserve the identity of the stream as the same as in the graph
        # Normally, we would do this via codegen for the proxy mapping to an output - we cannot do this yet, as we do not
        # yet have a plan for how we want to handle the case where the stream is used as an input or an output. Pending
        # design, to unblock current work, we lift the stream into a global and then codegen bytecode to load it from there.
        prefix = f"_stream_{self.device}"
        name = codegen.tx.output.install_global_by_id(prefix, self.value)
        codegen.append_output(codegen.create_load_global(name, add=True))


class EventVariable(VariableTracker):
    def __init__(self, proxy: Proxy, value: torch.Event, **kwargs: Any) -> None:
        if proxy is not None and "example_value" in proxy.node.meta:
            assert proxy.node.meta["example_value"] == value
        super().__init__(**kwargs)
        self.proxy = proxy
        self.value = value

    def call_method(
        self,
        tx: "InstructionTranslator",
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        from ..utils import proxy_args_kwargs
        from .builder import wrap_fx_proxy_cls

        if name in ("wait", "record", "synchronize"):
            tx.output.create_proxy(
                "call_method", name, *proxy_args_kwargs([self] + args, kwargs)
            )
            return ConstantVariable(None)
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
            unimplemented_v2(
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

    def reconstruct(self, codegen: "PyCodegen") -> None:
        # If we got here, this event is fully subsumed by the graph - this means it is
        # not an input or global
        assert not self.source
        # Similar to stream handling, we lift the event into a global and then codegen bytecode to load it from there.
        prefix = "_event"
        name = codegen.tx.output.install_global_by_id(prefix, self.value)
        codegen.append_output(codegen.create_load_global(name, add=True))
