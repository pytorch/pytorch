# mypy: allow-untyped-defs
import contextlib
import functools
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Union

import torch
from torch._dynamo.external_utils import (
    call_backward,
    call_hook,
    FakeCompiledAutogradEngine,
)
from torch._dynamo.source import GetItemSource, LocalSource
from torch._dynamo.utils import counters, lazy_format_graph_code, set_locals_to_steal
from torch._logging import getArtifactLogger, trace_structured
from torch._prims_common import clone_preserve_strides
from torch._subclasses import FakeTensorMode
from torch.fx import GraphModule
from torch.fx.experimental._backward_state import BackwardState
from torch.fx.experimental.proxy_tensor import (
    decompose,
    disable_autocast_cache,
    disable_proxy_modes_tracing,
    fetch_object_proxy,
    ProxyTorchDispatchMode,
    PythonKeyTracer,
    track_tensor_tree,
)
from torch.fx.experimental.symbolic_shapes import DimDynamic, ShapeEnv
from torch.fx.traceback import preserve_node_meta, set_stack_trace
from torch.utils._traceback import CapturedTraceback


if TYPE_CHECKING:
    from torch.fx.proxy import Proxy


compiled_autograd_log = getArtifactLogger(__name__, "compiled_autograd")
verbose_log = getArtifactLogger(__name__, "compiled_autograd_verbose")


def snapshot_verbose_logging_enabled():
    return torch._logging._internal.log_state.is_artifact_enabled(
        "compiled_autograd_verbose"
    )


def cpp_verbose_log_fn(msg: str) -> None:
    verbose_log.debug(msg)


def snapshot_cudagraph_enabled():
    return torch._inductor.config.triton.cudagraphs


def maybe_clone(x):
    if x is not None:
        return clone_preserve_strides(x)
    return x


class AutogradCompilerInstance:
    def __init__(self, compiler_fn) -> None:
        self.compiler_fn = compiler_fn
        self.stack = contextlib.ExitStack()
        self.close = self.stack.close
        self.shape_env = ShapeEnv()
        self.fake_tensor_mode = FakeTensorMode(
            allow_fallback_kernels=True,
            allow_non_fake_inputs=True,
            shape_env=self.shape_env,
        )
        self.fx_tracer = PythonKeyTracer()
        self.proxy_mode = ProxyTorchDispatchMode(self.fx_tracer, "symbolic")
        self.hooks_proxy: Optional[Proxy] = None
        self.graph_placeholders = ["inputs", "sizes", "scalars", "hooks"]

    def wrap_fake(self, x, source):
        assert isinstance(x, torch.Tensor)
        return self.fake_tensor_mode.from_tensor(x, source=source)

    @staticmethod
    def source(name, idx) -> GetItemSource:
        return GetItemSource(LocalSource(name), idx)

    def begin_capture(
        self,
        inputs: List[torch.Tensor],
        sizes: List[int],
        scalars: List[Union[int, float]],
    ):
        counters["compiled_autograd"]["captures"] += 1
        self.aot_graph_cls_name: Optional[str] = None
        self.aot_graph_infos: Dict[int, Dict[str, Any]] = {}
        self.fx_tracer.root = torch.nn.Module()
        self.fx_tracer.graph = torch.fx.Graph(tracer_cls=PythonKeyTracer)
        self.fx_tracer.tensor_attrs = {}
        args_proxy, sizes_proxy, scalars_proxy, self.hooks_proxy = (
            self.fx_tracer.create_proxy("placeholder", name, (), {})
            for name in self.graph_placeholders
        )

        # tensor inputs to fake tensors
        inputs = [
            self.wrap_fake(x, self.source("inputs", idx))
            for idx, x in enumerate(inputs)
        ]
        self.bind_tensors_to_proxies(inputs, args_proxy)

        # size inputs to symints
        sizes = [
            self.shape_env.create_unspecified_symint_and_symbol(
                val,
                self.source("sizes", idx),
                DimDynamic.DYNAMIC,
            )
            for idx, val in enumerate(sizes)
        ]
        self.bind_tensors_to_proxies(sizes, sizes_proxy)

        for idx, val in enumerate(scalars):
            source = self.source("scalars", idx)
            if isinstance(val, int):
                scalars[idx] = self.shape_env.create_unspecified_symint_and_symbol(
                    val,
                    source,
                    DimDynamic.DYNAMIC,
                )
            elif isinstance(val, float):
                scalars[idx] = self.shape_env.create_symfloatnode(
                    self.shape_env.create_unspecified_symbol(
                        val,
                        source=source,
                        dynamic_dim=DimDynamic.DYNAMIC,
                    ),
                    hint=val,
                    source=source,
                )
            else:
                raise AssertionError("Unexpected scalar type: ", type(val))
        self.bind_tensors_to_proxies(scalars, scalars_proxy)

        # TODO(jansel): are all these modes needed?
        self.stack.enter_context(decompose({}))
        self.stack.enter_context(self.fake_tensor_mode)
        self.stack.enter_context(self.proxy_mode)
        self.stack.enter_context(disable_autocast_cache())
        self.stack.enter_context(preserve_node_meta())
        return inputs, sizes, scalars

    def proxy_call_backward(
        self,
        inputs,
        output_metadatas,
        saved_tensors,
        backward_idx: int,
    ):
        assert self.hooks_proxy is not None
        backward_c_function = self.hooks_proxy[backward_idx]  # type: ignore[index]
        proxies = self.fx_tracer.create_proxy(
            kind="call_function",
            target=call_backward,
            args=(
                backward_c_function,
                self.to_proxy(saved_tensors),
                *self.to_proxy(inputs),
            ),
            kwargs={},
        )

        with disable_proxy_modes_tracing():
            # create fake Tensors
            grad_ins: List[Optional[torch.Tensor]] = []
            for output_metadata in output_metadatas:
                if output_metadata is None:
                    grad_ins.append(None)
                    continue

                layout, device, dtype, size = output_metadata
                grad_ins.append(
                    torch.empty(size=size, dtype=dtype, layout=layout, device=device)
                )
            self.bind_tensors_to_proxies(grad_ins, proxies)
        return tuple(grad_ins)

    def proxy_call_hook(self, hook, *args, **kwargs):
        return self.fx_tracer.create_proxy(
            "call_function",
            call_hook,
            (
                hook,
                *[self.to_proxy(x) for x in args],
            ),
            kwargs,
        )

    def tensor_pre_hook(self, inputs, hook_id, i: int):
        assert self.hooks_proxy is not None
        hook = self.hooks_proxy[hook_id]  # type: ignore[index]
        proxy = self.proxy_call_hook(
            hook,
            inputs[i],
            hook_type="tensor_pre_hook",
        )
        with disable_proxy_modes_tracing():
            inputs[i] = maybe_clone(inputs[i])
            self.bind_tensors_to_proxies([inputs[i]], [proxy])
        return inputs

    def pre_hook(self, inputs, hook_id):
        assert self.hooks_proxy is not None
        hook = self.hooks_proxy[hook_id]  # type: ignore[index]
        proxies = self.proxy_call_hook(
            hook,
            inputs,
            hook_type="pre_hook",
        )
        with disable_proxy_modes_tracing():
            inputs = [maybe_clone(x) for x in inputs]
            self.bind_tensors_to_proxies(inputs, proxies)
        return inputs

    def post_hook(self, outputs, inputs, hook_id):
        assert self.hooks_proxy is not None
        hook = self.hooks_proxy[hook_id]  # type: ignore[index]
        proxies = self.proxy_call_hook(
            hook,
            outputs,
            inputs,
            hook_type="post_hook",
        )
        with disable_proxy_modes_tracing():
            outputs = [maybe_clone(x) for x in outputs]
            self.bind_tensors_to_proxies(outputs, proxies)
        return outputs

    def post_acc_grad_hook(self, input, hook_id):
        assert isinstance(input, torch.Tensor)
        assert self.hooks_proxy is not None
        hook = self.hooks_proxy[hook_id]  # type: ignore[index]
        proxies = self.proxy_call_hook(
            hook,
            input,
            hook_type="post_acc_grad_hook",
        )
        with disable_proxy_modes_tracing():
            input = [maybe_clone(input)]
            self.bind_tensors_to_proxies(input, proxies)
        return input

    # Note: [Compiled autograd and cudagraphs]
    # Eager autograd backward implements scalars as 0-dim tensors, see DivBackward0::other_.
    # When compiled autograd traces those nodes, it lifts the scalar tensors, resulting in a graph
    # with some cpu 0-dim tensor inputs. To prevent the entire graph from skipping cudagraph, we move the
    # scalars tensors to cuda. This works because ATen/prims ops will accept cuda 0-dim tensors too.
    def move_graph_nodes_to_cuda(self, graph) -> List[int]:
        to_move: Dict[int, torch.fx.Node] = {}
        has_cuda_inputs = False
        nodes = list(graph.nodes)
        assert nodes[0].target == "inputs"
        inputs = nodes[0]
        inputs_users = list(inputs.users.keys())
        # input access nodes should immediately follow placeholder nodes
        first_getitem_idx = len(self.graph_placeholders)
        assert nodes[first_getitem_idx] == inputs_users[0]
        last_getitem_idx = first_getitem_idx + len(inputs_users) - 1
        assert nodes[last_getitem_idx] == inputs_users[-1]
        for i, node in enumerate(inputs_users):
            if not has_cuda_inputs and node.meta["val"].device.type == "cuda":
                has_cuda_inputs = True
                continue

            is_cpu = node.meta["val"].device.type == "cpu"
            is_scalar = len(node.meta["val"].size()) == 0
            if is_cpu and is_scalar:
                node_users = list(node.users.keys())
                if all(
                    isinstance(user.target, torch._ops.OpOverload)
                    and user.target.namespace in ("prims", "aten")
                    for user in node_users
                ):
                    # all users are prims/aten, can move safely
                    to_move[i] = node

        # only move cpu scalars to cuda if there were cuda activations in this graph,
        # this is to handle the case where cudagraphs is enabled on a cpu-only graph
        if has_cuda_inputs:
            for node in to_move.values():
                node.meta["val"] = node.meta["val"].cuda()

            # return runtime indices we need to move to cuda
            return list(to_move.keys())

        return []

    def end_capture(self, outputs):
        self.fx_tracer.create_proxy(
            "call_function",
            FakeCompiledAutogradEngine._exec_final_callbacks_stub,
            (),
            {},
        )
        self.stack.close()
        self.fx_tracer.create_node(
            "output",
            "output",
            (self.fx_tracer.create_arg(self.to_proxy(outputs)),),
            {},
        )
        self.rename_aot_dispatcher_nodes()
        self.reorder_accumulate_grad_nodes()
        runtime_inputs_to_move: List[int] = []
        if snapshot_cudagraph_enabled():
            runtime_inputs_to_move = self.move_graph_nodes_to_cuda(self.fx_tracer.graph)

        graph = GraphModule(
            self.fx_tracer.root, self.fx_tracer.graph, "CompiledAutograd"
        )
        set_locals_to_steal(graph, ["inputs"])
        lazy_graph_code = lazy_format_graph_code(
            "Compiled autograd graph",
            graph,
            include_device=True,
            include_stride=True,
            colored=True,
        )
        compiled_autograd_log.info("%s", lazy_graph_code)
        verbose_log.debug("%s", lazy_graph_code)
        trace_structured(
            "compiled_autograd_graph",
            payload_fn=lambda: graph.print_readable(print_output=False),
        )

        def runtime_wrapper(compiled_fn, inputs, sizes, scalars, hooks):
            global in_compiled_autograd_region
            try:
                in_compiled_autograd_region = True
                for i in runtime_inputs_to_move:
                    inputs[i] = inputs[i].pin_memory().cuda(non_blocking=True)

                return compiled_fn(inputs, sizes, scalars, hooks)
            finally:
                in_compiled_autograd_region = False

        return runtime_wrapper, self.compiler_fn(graph)

    def rename_aot_dispatcher_nodes(self):
        """
        Renames nodes as they appear in the AOTDispatcher backward graphs, prefixed by AOT id
        e.g. AOTDispatcher backward graph X's `sin_Y` -> `aotX_sin_Y`
        """
        if self.aot_graph_cls_name is None:
            return

        def is_similar(a: torch.fx.node.Node, b: torch.fx.node.Node):
            target_match = a.target == b.target
            if not target_match:
                target_match = (
                    hasattr(a.target, "__name__")
                    and hasattr(b.target, "__name__")
                    and a.target.__name__ == b.target.__name__
                )
            return (
                target_match
                and a.op == b.op
                and a.type == b.type
                and len(a.all_input_nodes) == len(b.all_input_nodes)
            )

        for nodecall_index, info in self.aot_graph_infos.items():
            ca_node_start_idx = info["ca_node_start_idx"]
            aot_id = info["aot_id"]
            aot_graph = info["aot_gm"].graph

            # 1. Find the first op from user code in the AOT graph
            aot_it = iter(aot_graph.nodes)
            aot_node = next(aot_it)
            assert aot_node is not None
            try:
                while aot_node.op != "call_function":
                    aot_node = next(aot_it)
            except StopIteration:
                continue

            try:
                # 2. Find the first op in the compiled autograd graph segment
                ca_it = iter(self.fx_tracer.graph.nodes)
                for _ in range(ca_node_start_idx):
                    next(ca_it)
                ca_node = next(ca_it)

                # Graphs should all end with output node
                while ca_node.op != "output" and not is_similar(ca_node, aot_node):
                    # The compiled autograd graph may contain lazily inserted ops
                    # We skip those when aligning nodes
                    ca_node = next(ca_it)

                # 3. Keep alligned and rename nodes
                while aot_node.op != "output" and ca_node.op != "output":
                    if not ca_node.users:
                        # TODO: DCE for compiled autograd graph
                        ca_node = next(ca_it)
                        continue

                    if not is_similar(aot_node, ca_node):
                        # There should be no lazily inserted ops in the middle of a match
                        # So any deviation is an error
                        raise StopIteration

                    ca_node.name = f"aot{aot_id}_{aot_node.name}"
                    for i, inp in enumerate(aot_node.all_input_nodes):
                        ca_node.all_input_nodes[i].name = f"aot{aot_id}_{inp.name}"

                    aot_node = next(aot_it)
                    ca_node = next(ca_it)
            except StopIteration:
                verbose_log.debug(
                    "Failed to match %s%s (NodeCall %s) nodes with AOT backward graph %s nodes",
                    self.aot_graph_cls_name,
                    aot_id,
                    nodecall_index,
                    aot_id,
                )

    def reorder_accumulate_grad_nodes(self):
        """
        Usage of AOTAutograd causes all the accumulate_grad_ nodes to get pushed to the end of
        the graph.  This differs from eager mode, which schedules them as soon as possible. This
        pass attempts to reorder the graph to mimic eager behavior.
        """
        for node in self.fx_tracer.graph.find_nodes(
            op="call_function", target=torch.ops.inductor.accumulate_grad_.default
        ):
            arg = max(node.args)  # last arg
            if arg is not node.prev and arg.op != "placeholder":
                arg.append(node)

    def to_proxy(self, t):
        if t is None:
            return None
        if isinstance(t, list):
            return [self.to_proxy(x) for x in t]
        if isinstance(t, tuple):
            return tuple(self.to_proxy(x) for x in t)
        # can it be torch.SymInt as the code used to imply?
        assert isinstance(t, torch.Tensor)
        proxy_tensor = fetch_object_proxy(self.fx_tracer, t)
        assert isinstance(proxy_tensor, torch.fx.experimental.proxy_tensor._ProxyTensor)
        return proxy_tensor.proxy

    def bind_tensors_to_proxies(self, tensors, proxies):
        if isinstance(proxies, torch.fx.Proxy):
            proxies = [proxies[i] for i in range(len(tensors))]  # type: ignore[index]
        assert len(tensors) == len(proxies)
        track_tensor_tree(tensors, proxies, constant=None, tracer=self.fx_tracer)

    def bind_backward_state(self, index: int):
        assert self.hooks_proxy is not None
        proxy = self.hooks_proxy[index]  # type: ignore[index]
        bw_state = BackwardState()
        track_tensor_tree(bw_state, proxy, constant=None, tracer=self.fx_tracer)
        return bw_state

    def set_node_origin(
        self,
        node_name: str,
        nodecall_index: int,
        pyobj: Optional[torch.autograd.Function],
    ):
        maybe_aot_id = ""
        if pyobj is not None:
            forward_cls = pyobj._forward_cls  # type: ignore[attr-defined]
            if hasattr(forward_cls, "_aot_id"):
                # backward was created by AOT Dispatcher
                self.aot_graph_cls_name = node_name
                maybe_aot_id = forward_cls._aot_id
                self.aot_graph_infos[nodecall_index] = {
                    "ca_node_start_idx": len(self.fx_tracer.graph.nodes),
                    "aot_id": maybe_aot_id,
                    "aot_gm": forward_cls._lazy_backward_info.bw_module,
                }

        new_code = f"{node_name}{maybe_aot_id} (NodeCall {nodecall_index})"
        raw_stack_trace = CapturedTraceback.extract().format()[-1]
        new_stack_trace = raw_stack_trace.replace(
            "raw_stack_trace = CapturedTraceback.extract().format()[-1]", new_code
        )
        set_stack_trace(new_stack_trace)


# state of the autograd engine dispatch, kept in sync by enable/disable context managers
compiled_autograd_enabled = False

# global flag to check if we are processing graphs produced from a compiled autograd graph
in_compiled_autograd_region = False


@contextlib.contextmanager
def enable(compiler_fn):
    prior = torch._C._dynamo.compiled_autograd.set_autograd_compiler(
        functools.partial(AutogradCompilerInstance, compiler_fn)
    )
    if snapshot_verbose_logging_enabled():
        torch._C._dynamo.compiled_autograd.set_verbose_logger(cpp_verbose_log_fn)
    global compiled_autograd_enabled
    compiled_autograd_enabled = True
    try:
        with torch.autograd.set_multithreading_enabled(False):
            yield
    finally:
        if not prior:
            compiled_autograd_enabled = False
        torch._C._dynamo.compiled_autograd.set_autograd_compiler(prior)


@contextlib.contextmanager
def disable():
    prior = torch._C._dynamo.compiled_autograd.set_autograd_compiler(None)
    global compiled_autograd_enabled
    compiled_autograd_enabled = False
    try:
        yield
    finally:
        if prior:
            compiled_autograd_enabled = True
        torch._C._dynamo.compiled_autograd.set_autograd_compiler(prior)


# return to starting state of a new process
def reset() -> None:
    global compiled_autograd_enabled
    compiled_autograd_enabled = False

    assert not in_compiled_autograd_region
    torch._C._dynamo.compiled_autograd.set_autograd_compiler(None)
    torch._C._dynamo.compiled_autograd.set_verbose_logger(None)
