import operator
import traceback
import typing
from contextlib import nullcontext
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from functorch.experimental import control_flow
from functorch.experimental import _map
from functorch.experimental._map import _unstack_pytree
from torch import fx
from torch._dispatch.python import enable_python_dispatcher
from torch._export.pass_infra.node_metadata import NodeMetadata
from torch._export.pass_infra.proxy_value import ProxyValue
from torch._subclasses import FakeTensor, UnsupportedFakeTensorException
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx import traceback as fx_traceback
from torch.fx.experimental.proxy_tensor import PythonKeyTracer
from torch.fx.graph import CodeGen
from torch.fx.passes.infra.pass_base import PassBase, PassResult
from torch.fx.passes.shape_prop import _extract_tensor_metadata, TensorMetadata
from torch.utils import _pytree as pytree


__all__ = ["_ExportPassBase"]


Argument = Any
Value = Any
Fn = Callable[..., Any]
PassType = Callable[[torch.fx.GraphModule], Optional[PassResult]]


class ExportPassBaseError(RuntimeError):
    pass


class _ExportPassBase(PassBase):
    """
    Interpreter-based pass class to help users maintain the IR spec while writing
    transformations.
    """

    @staticmethod
    def _create_dummy_node_metadata():
        return NodeMetadata({"stack_trace": traceback.format_exc(-1)})


    class ExportTracer(PythonKeyTracer):
        """
        Tracer used to create nodes during the retracing part of the Expo_ExportPassBasertPassBase
        """
        def __init__(self, callback: "_ExportPassBase", codegen: CodeGen) -> None:
            super().__init__()
            self.callback = callback
            self.root = torch.nn.Module()
            self.graph = torch.fx.Graph()
            self.graph.set_codegen(codegen)
            self.tensor_attrs: Dict[str, torch.Tensor] = {}  # type: ignore[assignment]
            self.fake_tensor_mode: Optional[FakeTensorMode] = None
            self.submodules: Dict[torch.nn.Module, str] = {}

        def trace(self) -> None:
            raise ExportPassBaseError("ExportTracer doesn't support trace().")

        def create_arg(self, a: Argument) -> torch.fx.Node:
            if isinstance(a, torch.nn.Module):
                if a not in self.submodules:
                    name_submodule = f"submodule_{len(self.submodules)}"
                    self.root.add_module(name_submodule, a)
                    self.submodules[a] = name_submodule
            elif isinstance(a, FakeTensor):
                if not hasattr(a, "constant") or a.constant is None:
                    raise ExportPassBaseError(f"Cannot add {a} to graph.")
                a = a.constant
            node = super().create_arg(a)
            if (
                isinstance(a, torch.Tensor)
                and isinstance(node, torch.fx.Node)
                and node.op == "get_attr"
            ):
                self.set_metadata(node, a)
                self.callback.on_attr(ProxyValue(a, node))
            return node

        def set_metadata(
            self, node: torch.fx.Node, value: Argument,
        ) -> None:
            # propagate the fake tensor or sym nodes
            def make_val(
                x: Argument,
            ) -> Union[FakeTensor, torch.SymInt, torch.SymFloat, torch.SymBool, int, None]:
                if isinstance(x, FakeTensor):
                    return x
                elif isinstance(x, torch.Tensor):
                    if x.is_quantized:
                        # TODO (tmanlaibaatar) properly support Quantized FakeTensor
                        x = torch.dequantize(x)

                    try:
                        assert self.fake_tensor_mode is not None
                        # TODO we should allocate static shapes
                        # for param/buffer values
                        if isinstance(x, torch.nn.Parameter):
                            fake_tensor = self.fake_tensor_mode.from_tensor(
                                x, static_shapes=True
                            )
                        else:
                            fake_tensor = self.fake_tensor_mode.from_tensor(x)
                    except UnsupportedFakeTensorException:
                        # TODO: This is just a workaround to get over the
                        # x.as_subclass error
                        print(
                            "Fakeifying a Tensor subclass is not supported \
                            right now. Instead a TensorMetadata is used."
                        )
                        fake_tensor = None
                    return fake_tensor
                elif isinstance(x, (torch.SymInt, torch.SymFloat, torch.SymBool, int, float, bool, str)):
                    return x
                else:
                    return None

            node.meta["val"] = pytree.tree_map(make_val, value)

            # Set the tensor_metadata for values that do not have a corresponding FakeTensor
            def make_tensor_meta(x: Argument) -> Optional[TensorMetadata]:
                if not isinstance(x, FakeTensor) and isinstance(x, torch.Tensor):
                    if x.is_quantized:
                        # TODO (tmanlaibaatar) properly support Quantized FakeTensor
                        x = torch.dequantize(x)

                    try:
                        assert self.fake_tensor_mode is not None
                        _ = self.fake_tensor_mode.from_tensor(x)
                        tensor_meta = None
                    except UnsupportedFakeTensorException:
                        # TODO: This is just a workaround to get over the
                        # x.as_subclass error
                        tensor_meta = _extract_tensor_metadata(x)
                    return tensor_meta
                else:
                    return None

            node.meta["tensor_meta"] = pytree.tree_map(make_tensor_meta, value)

    class ExportInterpreter(fx.Interpreter):
        """
        Interpreter to callback on any _ExportPassBase functions
        """
        def __init__(self, callback: "_ExportPassBase", gm: fx.GraphModule) -> None:
            super().__init__(gm)
            self.callback = callback
            self.node: torch.fx.Node = next(iter(gm.graph.nodes))

        def placeholder(
            self,
            target: str,
            args: Tuple[Argument, ...],
            kwargs: Dict[str, Argument],
        ) -> ProxyValue:
            arg = super().placeholder(target, args, kwargs)
            return self.callback.placeholder(target, arg, NodeMetadata(self.node.meta))

        def output(
            self,
            target: torch.fx.node.Target,
            args: Tuple[Argument, ...],
            kwargs: Dict[str, Argument],
        ) -> ProxyValue:
            return self.callback.output(args[0], NodeMetadata(self.node.meta)).data

        def call_function(
            self,
            target: torch.fx.node.Target,
            args: Tuple[Argument, ...],
            kwargs: Dict[str, Argument],
        ) -> ProxyValue:
            meta = NodeMetadata(self.node.meta)

            if target == operator.getitem:
                value, key = args
                return self.callback.call_getitem(value, key, meta)
            elif getattr(target, "__module__", None) == "_operator":
                assert callable(target)
                return self.callback.call_sym(target, args, meta)
            elif isinstance(target, (torch._ops.OpOverload, torch._ops.OpOverloadPacket)):
                return self.callback.call_operator(
                    target,
                    args,
                    kwargs,
                    meta,
                )
            elif target == control_flow.cond:
                pred, true_fn, false_fn, inputs = args
                return self.callback.call_cond(pred, true_fn, false_fn, inputs, meta)
            elif target == _map.map_impl:
                f, num_args, *rest = args  # type: ignore[assignment]
                return self.callback.call_map(f, num_args, list(rest), meta)
            # For other unregistered HigherOrderOps, just interpret them blindly
            elif isinstance(target, torch._ops.HigherOrderOperator):
                return self.callback._fx(
                    "call_function",
                    target,
                    args,
                    kwargs,
                    meta,
                )
            else:
                raise ExportPassBaseError(f"Unsupported target type: {target}")

        def get_attr(
            self, target: str, args: Tuple[Argument, ...], kwargs: Dict[str, Argument]
        ) -> Argument:
            return super().get_attr(target, args, kwargs)

        def call_module(
            self,
            target: torch.fx.node.Target,
            args: Tuple[Argument, ...],
            kwargs: Dict[str, Argument],
        ) -> None:
            raise ExportPassBaseError("call_module is not supported.")

        def call_method(
            self, target: str, args: Tuple[Argument, ...], kwargs: Dict[str, Argument]
        ) -> None:
            raise ExportPassBaseError("call_method is not supported.")

        def run_node(self, n: torch.fx.Node) -> Argument:
            self.node = n
            self.callback.node_debug_str = n.format_node()
            return super().run_node(n)

    def __init__(self) -> None:
        self.interpreter = torch.fx.Interpreter(
            torch.fx.GraphModule(torch.nn.Module(), torch.fx.Graph())
        )
        self.tracer = self.ExportTracer(self, CodeGen())
        self.fake_tensor_mode: Optional[FakeTensorMode] = None
        self._initialized = True
        self.node_debug_str: typing.Optional[str] = None

    def _fx(
        self,
        kind: str,
        target: torch.fx.node.Target,
        args: Tuple[Argument, ...],
        kwargs: Dict[str, Argument],
        meta: NodeMetadata,
    ) -> ProxyValue:
        args_data, kwargs_data = pytree.tree_map_only(
            ProxyValue, lambda x: x.data, (args, kwargs)
        )
        res_data = getattr(self.interpreter, kind)(target, args_data, kwargs_data)
        args_proxy, kwargs_proxy = pytree.tree_map_only(
            ProxyValue, lambda x: x.proxy, (args, kwargs)
        )

        name = None
        if isinstance(target, torch._ops.OpOverload):
            name = self.tracer.graph._target_to_str(target.overloadpacket.__name__)

        res_proxy = self.tracer.create_proxy(kind, target, args_proxy, kwargs_proxy, name=name)
        res_proxy.node.meta.update(meta.data)
        self.tracer.set_metadata(res_proxy.node, res_data)
        return ProxyValue(res_data, res_proxy)

    def inputs(self, graph_module: torch.fx.GraphModule) -> List[Argument]:
        # TODO(angelayi): Update this with what we decide to do for metadata in
        # the exported graph module
        if (args := graph_module.meta.get("args", None)) is not None:
            return list(args)

        def extract_input(node: torch.fx.Node) -> Optional[FakeTensor]:
            if "val" in node.meta:
                return node.meta["val"]
            elif tensor_meta := node.meta.get("tensor_meta"):
                assert self.fake_tensor_mode is not None
                return FakeTensor(
                    self.fake_tensor_mode,
                    torch.empty(
                        tensor_meta.shape,
                        dtype=tensor_meta.dtype,
                        device="meta",
                        requires_grad=tensor_meta.requires_grad,
                        memory_format=tensor_meta.memory_format,
                    ),
                    torch.device("cpu"),
                )
            elif len(node.users) == 0:
                return None
            raise ExportPassBaseError(
                f"Cannot construct an input for graph module: {graph_module}.",
            )

        return [
            extract_input(node)
            for node in graph_module.graph.nodes
            if node.op == "placeholder"
        ]

    def on_attr(self, attr: ProxyValue) -> None:
        pass

    def placeholder(self, name: str, arg: Argument, meta: NodeMetadata) -> ProxyValue:
        arg_proxy = self.tracer.create_proxy("placeholder", name, (), {})
        arg_proxy.node.meta = meta.data
        self.tracer.set_metadata(arg_proxy.node, arg)
        return ProxyValue(arg, arg_proxy)

    def call_operator(
        self,
        op,
        args: Tuple[Argument, ...],
        kwargs: Dict[str, Argument],
        meta: NodeMetadata,
    ) -> ProxyValue:
        return self._fx("call_function", op, args, kwargs, meta)

    def call_sym(
        self,
        target: Fn,
        args: Tuple[Argument, ...],
        meta: NodeMetadata,
    ) -> ProxyValue:
        return self._fx("call_function", target, args, {}, meta)

    def call_cond(
        self,
        pred: ProxyValue,
        true_fn: torch.fx.GraphModule,
        false_fn: torch.fx.GraphModule,
        inputs: List[Argument],
        meta: NodeMetadata,
    ) -> ProxyValue:
        true_branch = self.call_submodule(true_fn, tuple(inputs))
        false_branch = self.call_submodule(false_fn, tuple(inputs))
        assert true_branch is not None
        assert false_branch is not None
        return self._fx(
            "call_function",
            control_flow.cond,
            (pred, true_branch.graph_module, false_branch.graph_module, inputs),
            {},
            meta,
        )

    def call_map(
        self,
        f: torch.fx.GraphModule,
        num_args: int,
        args: List[ProxyValue],
        meta: NodeMetadata,
    ) -> ProxyValue:
        xs = _unstack_pytree([arg.data for arg in args[:num_args]])[0]
        pos_args = args[num_args:]
        f_branch = self.call_submodule(f, tuple(xs + [arg.data for arg in pos_args]))
        assert f_branch is not None
        return self._fx(
            "call_function",
            _map.map_impl,
            (f_branch.graph_module, num_args, *args),
            {},
            meta,
        )

    def call_getitem(
        self, value: ProxyValue, key: int, meta: NodeMetadata
    ) -> ProxyValue:
        return self._fx("call_function", operator.getitem, (value, key), {}, meta)

    def output(self, results: List[Argument], meta: NodeMetadata) -> ProxyValue:
        return self._fx("output", "output", (results,), {}, meta)

    def call_submodule(
        self, graph_module: fx.GraphModule, inputs: Tuple[Argument, ...]
    ) -> PassResult:
        prev_tracer, self.tracer = self.tracer, self.ExportTracer(
            self, graph_module.graph._codegen
        )
        self.tracer.fake_tensor_mode = prev_tracer.fake_tensor_mode
        interpreter = self.ExportInterpreter(self, graph_module)
        prev_interpreter, self.interpreter = self.interpreter, torch.fx.Interpreter(
            torch.fx.GraphModule(torch.nn.Module(), torch.fx.Graph())
        )
        inputs_data = pytree.tree_map_only(ProxyValue, lambda x: x.data, inputs)
        with fx_traceback.preserve_node_meta():
            interpreter.run(*inputs_data)

        new_graph_module = torch.fx.GraphModule(self.tracer.root, self.tracer.graph)

        self.tracer = prev_tracer
        self.interpreter = prev_interpreter
        return PassResult(
            new_graph_module,
            True,
        )

    def call(self, graph_module: fx.GraphModule) -> PassResult:
        if not getattr(self, "_initialized", False):
            raise ExportPassBaseError(
                "ExportPass is not initialized with __init__().",
            )

        inputs = self.inputs(graph_module)

        fake_tensor_mode = None
        for i in inputs:
            if isinstance(i, FakeTensor):
                assert (
                    fake_tensor_mode is None or fake_tensor_mode is i.fake_mode
                ), "Multiple fake tensor mode detected."
                fake_tensor_mode = i.fake_mode
        if fake_tensor_mode is None:
            self.tracer.fake_tensor_mode = FakeTensorMode(allow_non_fake_inputs=True)
            fake_tensor_mode = nullcontext()  # type: ignore[assignment]
            dispatcher_mode = nullcontext()  # type: ignore[assignment]
        else:
            self.tracer.fake_tensor_mode = fake_tensor_mode
            dispatcher_mode = enable_python_dispatcher()  # type: ignore[assignment]
        self.fake_tensor_mode = self.tracer.fake_tensor_mode

        with fake_tensor_mode, dispatcher_mode:  # type: ignore[assignment, union-attr]
            result = self.call_submodule(graph_module, tuple(inputs))

        return result
