from typing import Any, Optional

import torch
from torch.fx import Proxy

from .. import graph_break_hints
from ..device_interface import get_interface_for_device
from ..exc import TYPE_CHECKING, unimplemented_v2
from .base import VariableTracker
from .constant import ConstantVariable
from .ctx_manager import ContextWrappingVariable
from .misc import GetAttrVariable


if TYPE_CHECKING:
    from torch._dynamo.symbolic_convert import InstructionTranslator

    from ..codegen import PyCodegen

from torch._library.custom_ops import custom_op


Tensor = torch.Tensor


@custom_op("streams::fork", mutates_args=())
def fork_stream(
    from_index: int,
    from_device: torch.device,
    to_index: int,
    to_device: torch.device,
) -> None:
    pass


@fork_stream.register_fake
def _(
    from_index: int,
    from_device: torch.device,
    to_index: int,
    to_device: torch.device,
) -> None:
    pass


@custom_op("streams::join", mutates_args=())
def join_stream(
    from_index: int,
    from_device: torch.device,
    to_index: int,
    to_device: torch.device,
) -> None:
    pass


@join_stream.register_fake
def _(
    from_index: int,
    from_device: torch.device,
    to_index: int,
    to_device: torch.device,
) -> None:
    pass


class StreamContextVariable(ContextWrappingVariable):
    """This represents torch.cuda.StreamContext"""

    @staticmethod
    def create(
        tx: "InstructionTranslator",
        target_value: "StreamVariable",
        **kwargs: dict[str, Any],
    ) -> "StreamContextVariable":
        return StreamContextVariable(
            target_values=[target_value],
            initial_values=[
                StreamContextVariable._get_current_stream(target_value.device, tx)
            ],
            device=target_value.device,
            **kwargs,
        )

    def __init__(
        self,
        target_values: list["StreamVariable"],
        device: torch.device,
        initial_values: Optional[list["StreamVariable"]] = None,
        **kwargs: dict[str, Any],
    ) -> None:
        super().__init__(
            target_values=target_values, initial_values=initial_values, **kwargs
        )
        self.device = device
        self.set_stream_id = get_interface_for_device(self.device)._set_stream_by_id

    def enter(self, tx: "InstructionTranslator") -> "VariableTracker":
        # to stream, from stream is the order of the arguments
        # we are entering the target, and leaving the initial stream
        tx.output.create_proxy(
            "call_function",
            torch.ops.streams.fork.default,
            self._target_stream_proxies() + self._initial_stream_proxies(),
            {},
        )
        return ConstantVariable.create(None)

    def exit(self, tx: "InstructionTranslator", *args: tuple[Any]) -> "VariableTracker":
        # to stream, from stream is the order of the arguments
        # we are leaving the target, and entering the initial stream
        tx.output.create_proxy(
            "call_function",
            torch.ops.streams.join.default,
            self._initial_stream_proxies() + self._target_stream_proxies(),
            {},
        )
        return ConstantVariable.create(None)

    def _initial_stream_proxies(self) -> tuple[Proxy, Proxy]:
        assert self.initial_values, "No initial stream to move from"
        return StreamContextVariable._extract_stream_properties(
            self.initial_values[0].as_proxy()
        )

    def _target_stream_proxies(self) -> tuple[Proxy, Proxy]:
        return StreamContextVariable._extract_stream_properties(
            self.target_values[0].as_proxy()
        )

    @staticmethod
    def _extract_stream_properties(stream_proxy: Proxy) -> tuple[Proxy, Proxy]:
        stream_index = GetAttrVariable.create_getattr_proxy(stream_proxy, "stream_id")
        stream_device = GetAttrVariable.create_getattr_proxy(stream_proxy, "device")
        return stream_index, stream_device

    @staticmethod
    def _get_current_stream(
        device: torch.device, tx: "InstructionTranslator"
    ) -> "StreamVariable":
        from .builder import wrap_fx_proxy_cls

        current_stream_method = get_interface_for_device(device).current_stream
        current_stream = wrap_fx_proxy_cls(
            StreamVariable,
            tx,
            tx.output.create_proxy(
                "call_function",
                current_stream_method,
                (None,),
                {},
            ),
        )
        return current_stream


class StreamVariable(VariableTracker):
    """Represents the device-agnostic torch.Stream class"""

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
