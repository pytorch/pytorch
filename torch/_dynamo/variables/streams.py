from typing import Any

import torch
from torch.fx import Proxy

from .. import graph_break_hints
from ..codegen import PyCodegen
from ..exc import TYPE_CHECKING, unimplemented_v2
from .base import VariableTracker
from .constant import ConstantVariable


if TYPE_CHECKING:
    from torch._dynamo.symbolic_convert import InstructionTranslator


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
            # NB : Checking for mutation is necessary because we compare
            # constant values
            other = args[0]
            if not isinstance(other, StreamVariable):
                return ConstantVariable.create(NotImplemented)
            return ConstantVariable.create(
                cmp_name_to_op_mapping[name](self.value, other.value)  # type: ignore[arg-type]
            )

        return super().call_method(tx, name, args, kwargs)

    def as_proxy(self) -> Proxy:
        return self.proxy

    def reconstruct(self, codegen: PyCodegen) -> None:
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

    def reconstruct(self, codegen: PyCodegen) -> None:
        # If we got here, this event is fully subsumed by the graph - this means it is
        # not an input or global
        assert not self.source
        # Similar to stream handling, we lift the event into a global and then codegen bytecode to load it from there.
        prefix = "_event"
        name = codegen.tx.output.install_global_by_id(prefix, self.value)
        codegen.append_output(codegen.create_load_global(name, add=True))
