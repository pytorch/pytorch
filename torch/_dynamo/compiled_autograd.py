# mypy: allow-untyped-defs
import contextlib
import functools
import threading
from dataclasses import dataclass
from logging import Logger
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING, Union

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


@dataclass
class CompiledAutogradTLS:
    next_ctx_id: int = 0
    in_compiled_autograd_region: bool = False
    compiler: Optional["AutogradCompilerInstance"] = None
    vlogger: Optional[Logger] = None


class TLSWrapper:
    tls_key = "compiled_autograd_state"

    def __init__(self):
        self._local = threading.local()

    def _get_tls(self) -> CompiledAutogradTLS:
        if hasattr(self._local, self.tls_key):
            # first look in python
            state = getattr(self._local, self.tls_key)
        if torch._C._is_key_in_tls(self.tls_key):
            # then look in cpp
            state = torch._C._get_obj_in_tls(self.tls_key)
        else:
            # init new thread created outside of autograd
            # TODO: what if context manager wrapped outside of thread?
            setattr(self._local, self.tls_key, CompiledAutogradTLS())
            state = getattr(self._local, self.tls_key)
            torch._C._stash_obj_in_tls(self.tls_key, state)
        return state

    # queries on the object stored in TLS
    def get(self, name):
        return getattr(self._get_tls(), name)

    def set_tls(self, **kwargs) -> Callable[[], None]:
        priors: Dict[str, Any] = {}
        for k, v in kwargs.items():
            state = self._get_tls()
            priors[k] = getattr(state, k)
            setattr(state, k, v)

        torch._C._dynamo.compiled_autograd.notify_autograd_engine()

        def revert():
            self.set_tls(**priors)

        return revert

    def enter_ctx(self) -> Callable[[], None]:
        state = self._get_tls()
        state.next_ctx_id += 1
        id = state.next_ctx_id

        def exit():
            assert (
                state is self._get_tls()
            ), "Runtime must begin and end on the same thread"
            assert state.next_ctx_id == id, (
                "Error nesting compiled autograd context managers: "
                "inner context managers must have shorter lifetime than the outer context manager"
            )
            state.next_ctx_id -= 1

        return exit

    def enter_compiled_region(self) -> Callable[[], None]:
        state = self._get_tls()
        prior = state.in_compiled_autograd_region
        state.in_compiled_autograd_region = True
        assert prior is False, "Nested compiled autograd regions are not supported"

        def exit():
            assert (
                state is self._get_tls()
            ), "Runtime must begin and end on the same thread"
            assert state.in_compiled_autograd_region is True
            state.in_compiled_autograd_region = prior

        return exit


local = TLSWrapper()


def enabled() -> bool:
    return local.get("compiler") is not None


def in_compiled_autograd_region() -> bool:
    return local.get("in_compiled_autograd_region")


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
        origins: List[List[Tuple[int, str]]],
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

        self.stack.enter_context(preserve_node_meta())
        inputs_origins, sizes_origins, scalars_origins = origins
        # tensor inputs to fake tensors
        inputs = [
            self.wrap_fake(x, self.source("inputs", idx))
            for idx, x in enumerate(inputs)
        ]
        self.bind_tensors_to_proxies(inputs, args_proxy, inputs_origins)

        # size inputs to symints
        sizes = [
            self.shape_env.create_unspecified_symint_and_symbol(
                val,
                self.source("sizes", idx),
                DimDynamic.DYNAMIC,
            )
            for idx, val in enumerate(sizes)
        ]
        self.bind_tensors_to_proxies(sizes, sizes_proxy, sizes_origins)

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
        self.bind_tensors_to_proxies(scalars, scalars_proxy, scalars_origins)

        # TODO(jansel): are all these modes needed?
        self.stack.enter_context(decompose({}))
        self.stack.enter_context(self.fake_tensor_mode)
        self.stack.enter_context(self.proxy_mode)
        self.stack.enter_context(disable_autocast_cache())
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
        proxy = self.proxy_call_hook(
            hook,
            input,
            hook_type="post_acc_grad_hook",
        )
        with disable_proxy_modes_tracing():
            input = [maybe_clone(input)]
            self.bind_tensors_to_proxies(input, [proxy])
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
        if torch._inductor.config.triton.cudagraphs:
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
            try:
                exit_compiled_region = local.enter_compiled_region()
                for i in runtime_inputs_to_move:
                    inputs[i] = inputs[i].pin_memory().cuda(non_blocking=True)

                with disable():
                    return compiled_fn(inputs, sizes, scalars, hooks)
            finally:
                exit_compiled_region()

        return runtime_wrapper, self.compiler_fn(graph)

    def rename_aot_dispatcher_nodes(self):
        """
        Renames nodes as they appear in the AOTDispatcher backward graphs, prefixed by AOT id
        e.g. AOTDispatcher backward graph X's `sin_Y` -> `aotX_sin_Y`
        """
        if self.aot_graph_cls_name is None:
            return

        def is_similar(ca: torch.fx.node.Node, aot: torch.fx.node.Node):
            # 1. comparing using target (for aten ops)
            target_match = ca.target == aot.target
            if not target_match:
                # 2. comparing using name (for HOPs)
                target_match = (
                    hasattr(ca.target, "__name__")
                    and hasattr(aot.target, "__name__")
                    and ca.target.__name__ == aot.target.__name__
                )
            if (
                not target_match
                and hasattr(ca.target, "name")
                and hasattr(aot.target, "name")
                and aot.target.name() == "aten::reshape"
                and hasattr(aot.meta.get("original_aten"), "name")
            ):
                # 3. undo view_to_reshape post grad pass
                target_match = ca.target.name() == aot.meta["original_aten"].name()

            return (
                target_match
                and ca.op == aot.op
                and ca.type == aot.type
                and len(ca.all_input_nodes) == len(aot.all_input_nodes)
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

                    if not is_similar(ca_node, aot_node):
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

    def bind_tensors_to_proxies(
        self, tensors, proxies, origins: Optional[List[Tuple[int, str]]] = None
    ):
        if isinstance(proxies, torch.fx.Proxy):
            if origins:
                assert len(origins) == len(tensors)
                bound_proxies = []
                for i in range(len(tensors)):
                    nodecall_index, node_name = origins[i]
                    self.set_node_origin(node_name, nodecall_index, None)
                    bound_proxies.append(proxies[i])  # type: ignore[index]
                proxies = bound_proxies
            else:
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


# global flag to check if compiled autograd is enabled but Dynamo stance is "force_eager"
compiled_autograd_enabled_force_eager = False


@contextlib.contextmanager
def enable(compiler_fn):
    from torch._dynamo import eval_frame

    if eval_frame._stance.stance == "force_eager":
        # If user explicitly sets Dynamo stance to "force_eager", we want Compiled Autograd
        # to fall back to eager as well.
        global compiled_autograd_enabled_force_eager
        compiled_autograd_enabled_force_eager = True
        try:
            yield
        finally:
            compiled_autograd_enabled_force_eager = False
    else:
        # we need to import this, because user might not have imported it if they directly use this context manager
        # we need to lazily import it, because of circular dependencies
        import torch._inductor.cudagraph_trees

        exit_ctx = local.enter_ctx()
        revert_tls = local.set_tls(
            compiler=functools.partial(AutogradCompilerInstance, compiler_fn),
            vlogger=verbose_log
            if torch._logging._internal.log_state.is_artifact_enabled(
                "compiled_autograd_verbose"
            )
            else None,
        )
        try:
            with torch.autograd.set_multithreading_enabled(False):
                yield
        finally:
            revert_tls()
            exit_ctx()


@contextlib.contextmanager
def disable():
    exit_ctx = local.enter_ctx()
    revert_tls = local.set_tls(
        compiler=None,
        vlogger=None,
    )
    try:
        yield
    finally:
        revert_tls()
        exit_ctx()


# return to starting state of a new process
def reset() -> None:
    assert local.get("next_ctx_id") == 0
    assert local.get("in_compiled_autograd_region") is False
    local.set_tls(
        compiler=None,
        vlogger=None,
    )
