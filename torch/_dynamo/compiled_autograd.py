# mypy: allow-untyped-defs
import contextlib
import functools
import itertools
import operator
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING, Union

import torch
from torch._dynamo.external_utils import (
    call_backward,
    call_hook,
    FakeCompiledAutogradEngine,
)
from torch._dynamo.source import GetItemSource, LocalSource
from torch._dynamo.utils import (
    counters,
    get_chromium_event_logger,
    lazy_format_graph_code,
    set_locals_to_steal,
)
from torch._guards import compile_context, CompileContext, CompileId
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
from torch.utils._ordered_set import OrderedSet
from torch.utils._traceback import CapturedTraceback


if TYPE_CHECKING:
    from torch.fx.proxy import Proxy


compiled_autograd_log = getArtifactLogger(__name__, "compiled_autograd")
verbose_log = getArtifactLogger(__name__, "compiled_autograd_verbose")


def snapshot_verbose_logging_enabled():
    return torch._logging._internal.log_state.is_artifact_enabled(
        "compiled_autograd_verbose"
    )


def snapshot_cudagraph_enabled():
    return torch._inductor.config.triton.cudagraphs


def maybe_clone(x):
    if x is not None:
        return clone_preserve_strides(x)
    return x


_graph_placeholders = ["inputs", "sizes", "scalars", "hooks"]
_impure_targets = OrderedSet(
    [
        call_hook,
        call_backward,
        FakeCompiledAutogradEngine._exec_final_callbacks_stub,
        torch.ops.inductor.accumulate_grad_.default,
    ]
)

COMPILE_COUNTER = itertools.count()


def make_compile_context(compiled_autograd_id):
    return compile_context(
        CompileContext(
            CompileId(
                compiled_autograd_id=compiled_autograd_id,
                frame_id=None,
                frame_compile_id=None,
            )
        )
    )


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
        self.id = next(COMPILE_COUNTER)
        self.compile_context = make_compile_context(self.id)
        self.compile_context.__enter__()
        self.start_time_ns = time.time_ns()
        get_chromium_event_logger().log_event_start(
            "compiled_autograd",
            self.start_time_ns,
            {"graph_id": self.id},
            log_pt2_compile_event=True,
        )

        self.aot_graph_cls_name: Optional[str] = None
        self.aot_graph_infos: Dict[int, Dict[str, Any]] = {}
        self.fx_tracer.root = torch.nn.Module()
        self.fx_tracer.graph = torch.fx.Graph(tracer_cls=PythonKeyTracer)
        self.fx_tracer.tensor_attrs = {}
        args_proxy, sizes_proxy, scalars_proxy, self.hooks_proxy = (
            self.fx_tracer.create_proxy("placeholder", name, (), {})
            for name in _graph_placeholders
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
        # Needed to make sure we don't accidentally specialize any symbols
        assert self.fake_tensor_mode.shape_env is not None
        env = self.fake_tensor_mode.shape_env
        self.stack.enter_context(
            torch.fx.experimental.symbolic_shapes._suppress_guards(env)
        )
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
        first_getitem_idx = len(_graph_placeholders)
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

    def is_sym_node(self, node):
        return (
            isinstance(node, torch.fx.Node)
            and node.op == "call_function"
            and node.target
            in [torch.ops.aten.sym_size.int, torch.ops.aten.sym_numel.default]
        )

    def dce(self):
        # Most of these removed nodes would have been removed during Dynamo and AOTDispatch
        # Remove some of these nodes earlier to improve compilation speed

        # Dynamo guards will error instead of creating aliasing guards unless we unpack them in the graph
        unpack_nodes: OrderedSet[torch.fx.Node] = OrderedSet()
        for i, node in enumerate(self.fx_tracer.graph.find_nodes(op="placeholder")):
            unpack_nodes.update(node.users.keys())
        assert i == len(_graph_placeholders) - 1

        def is_impure(node):
            return (
                node in unpack_nodes
                or node.op == "placeholder"
                or node.op == "output"
                or (node.op == "call_function" and node.target in _impure_targets)
            )

        self.fx_tracer.graph.eliminate_dead_code(is_impure)

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
        self.reorder_tensor_pre_hook_nodes()
        self.reorder_pre_hook_nodes_to_schedule_asap()
        self.reorder_accumulate_grad_nodes()
        self.reorder_pre_hook_nodes_to_mimic_eager()
        self.reorder_post_acc_grad_hook_nodes()
        self.reorder_post_hook_nodes()
        # TODO(yf225): work around: remove dead codes like `sym_size` and `sym_numel` which are not used downstream. e.g.
        # ```
        # sym_numel_default = torch.ops.aten.sym_numel.default(sum_109);  sum_109 = None
        # eq_115 = 16 == sym_numel_default;  sym_numel_default = eq_115 = None
        # sym_size_int_39 = torch.ops.aten.sym_size.int(getitem_112, 1);  getitem_112 = None
        # eq_116 = 16 == sym_size_int_39;  eq_116 = None
        # eq_117 = 16 == sym_size_int_39;  sym_size_int_39 = eq_117 = None
        # ```
        # Proper fix is Richard's Python compiled autograd effort which will avoid calling make_fx and
        # should prevent these ops from going into the CA graph.
        self.dce()
        runtime_inputs_to_move: List[int] = []
        if snapshot_cudagraph_enabled():
            runtime_inputs_to_move = self.move_graph_nodes_to_cuda(self.fx_tracer.graph)

        graph = GraphModule(
            self.fx_tracer.root, self.fx_tracer.graph, f"CompiledAutograd{self.id}"
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

                with _disable(), make_compile_context(self.id):
                    return compiled_fn(inputs, sizes, scalars, hooks)
            finally:
                in_compiled_autograd_region = False

        get_chromium_event_logger().log_event_end(
            "compiled_autograd",
            time.time_ns(),
            {"graph_id": self.id},
            self.start_time_ns,
            log_pt2_compile_event=True,
        )
        self.compile_context.__exit__(None, None, None)
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

        # number of times we saw this AOT backward graph, used to dedup reused graphs
        aot_id_counter: Dict[int, int] = defaultdict(int)
        for nodecall_index, info in self.aot_graph_infos.items():
            ca_node_start_idx = info["ca_node_start_idx"]
            aot_id = info["aot_id"]
            aot_id_postfix = ""
            aot_graph = info["aot_gm"].graph
            if aot_id_counter[aot_id]:
                aot_id_postfix = f"_{aot_id_counter[aot_id]}"
            aot_id_counter[aot_id] += 1

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

                    ca_node.name = f"aot{aot_id}{aot_id_postfix}_{aot_node.name}"
                    for i, inp in enumerate(aot_node.all_input_nodes):
                        ca_node.all_input_nodes[
                            i
                        ].name = f"aot{aot_id}{aot_id_postfix}_{inp.name}"

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

    @staticmethod
    def get_all_nodes(args):
        # filter out non-Node args, like None
        nodes = [n for n in args if type(n) is torch.fx.Node]
        return nodes

    @staticmethod
    def is_placeholder(node):
        if node.op == "placeholder" or (
            node.op == "call_function"
            and node.target == operator.getitem
            and node.args[0].op == "placeholder"
        ):
            return True
        return False

    def reorder_accumulate_grad_nodes(self):
        """
        Usage of AOTAutograd causes all the accumulate_grad_ nodes to get pushed to the end of
        the graph.  This differs from eager mode, which schedules them as soon as possible. This
        pass attempts to reorder the graph to mimic eager behavior.
        """
        for node in self.fx_tracer.graph.find_nodes(
            op="call_function", target=torch.ops.inductor.accumulate_grad_.default
        ):
            param_node, grad_node = node.args[0], node.args[1]
            getitem_node = None
            if grad_node.target == operator.getitem:
                getitem_node = grad_node
                grad_node = getitem_node.args[0]

            arg = max([param_node, grad_node])  # last arg
            if arg is not node.prev and not self.is_placeholder(arg):
                arg.append(node)
                if getitem_node is not None:
                    arg.append(getitem_node)

    def reorder_tensor_pre_hook_nodes(self):
        """
        Usage of AOTAutograd causes all the tensor_pre_hook nodes to get pushed
        to the end of the graph. This differs from eager mode, which schedules
        them as soon as possible. This pass attempts to reorder the graph to
        mimic eager behavior.
        """
        for node in self.fx_tracer.graph.find_nodes(
            op="call_function", target=call_hook
        ):
            if node.kwargs.get("hook_type", None) != "tensor_pre_hook":
                continue

            getitem_node = node.args[0]
            input_node = node.args[1]  # tensor_pre_hook handle only one grad tensor

            if input_node is not node.prev and not self.is_placeholder(input_node):
                input_node.append(getitem_node)
                getitem_node.append(node)

    def reorder_pre_hook_nodes_to_schedule_asap(self):
        """
        In this function, we schedule the pre hooks as soon as possible. This
        does not match eager behavior (schedule pre hook right before its
        registered node), but it can make acc grad be scheduled properly when
        the pre hooks are registered to them. After reordering acc grad node, we
        will reorder the pre hooks again to mimic eager behavior.
        """
        for node in self.fx_tracer.graph.find_nodes(
            op="call_function", target=call_hook
        ):
            if node.kwargs.get("hook_type", None) != "pre_hook":
                continue

            getitem_node = node.args[0]
            # pre_hook handle a tuple of grad tensors
            input_nodes = self.get_all_nodes(node.args[1])

            to_remove = []
            to_append = []
            hook_block = [node]  # contain the hook and hook args getitem
            for n in input_nodes:
                if n.op == "call_function" and n.target == operator.getitem:
                    to_append.append(n.args[0])
                    to_remove.append(n)
                    hook_block.append(n)
            for a, b in zip(to_remove, to_append):
                input_nodes.remove(a)
                input_nodes.append(b)

            arg = max(input_nodes)  # last input
            if arg is not node.prev and not self.is_placeholder(arg):
                arg.append(getitem_node)
                for n in hook_block:
                    getitem_node.append(n)

    def reorder_pre_hook_nodes_to_mimic_eager(self):
        """
        Usage of AOTAutograd causes all the pre_hook nodes to get pushed to the
        end of the graph. This differs from eager mode, which schedules them
        right before their registered node execution. This pass attempts to
        reorder the graph to mimic eager behavior.
        """
        pre_hooks = []
        for node in self.fx_tracer.graph.find_nodes(
            op="call_function", target=call_hook
        ):
            if node.kwargs.get("hook_type", None) != "pre_hook":
                continue
            pre_hooks.append(node)

        for node in reversed(pre_hooks):
            hook_getitem_node = node.args[0]

            users = list(node.users.keys())
            if len(users) == 0:
                continue

            # users are all getitem ops and they are used by same registered node
            assert all(
                user.op == "call_function" and user.target == operator.getitem
                for user in users
            )
            registered_node = next(iter(users[0].users.keys()))

            if registered_node is not node.next:
                registered_node.prepend(hook_getitem_node)
                registered_node.prepend(node)
                for getitem in users:
                    registered_node.prepend(getitem)

    def reorder_post_acc_grad_hook_nodes(self):
        """
        Usage of AOTAutograd causes all the post_acc_grad_hook nodes to get
        pushed to the end of the graph. This differs from eager mode, which
        schedules them as soon as possible. This pass attempts to reorder the
        graph to mimic eager behavior.
        """
        post_acc_grad_hooks = []
        for node in self.fx_tracer.graph.find_nodes(
            op="call_function", target=call_hook
        ):
            if node.kwargs.get("hook_type", None) != "post_acc_grad_hook":
                continue
            post_acc_grad_hooks.append(node)

        # nodes in post_acc_grad_hooks are in topo order. For hooks registered
        # to same node, we should keep their relative order
        for node in reversed(post_acc_grad_hooks):
            getitem_node = node.args[0]
            param_node = node.args[1]  # post_acc_grad_hook handle one param

            # find the corresponding acc_grad node
            acc_grad_node = None
            for n in list(param_node.users.keys()):
                if (
                    n.op == "call_function"
                    and n.target == torch.ops.inductor.accumulate_grad_.default
                ):
                    acc_grad_node = n
                    break

            assert (
                acc_grad_node is not None
            ), "post_acc_grad_hook must have corresponding acc grad node"

            # append post_acc_grad_hook after acc_grad node
            acc_grad_node.append(getitem_node)
            getitem_node.append(node)

    def reorder_post_hook_nodes(self):
        """
        Usage of AOTAutograd causes all the post_hook nodes to get pushed to the
        end of the graph. This differs from eager mode, which schedules them as
        soon as possible. This pass attempts to reorder the graph to mimic eager
        behavior.
        """
        post_hooks = []
        for node in self.fx_tracer.graph.find_nodes(
            op="call_function", target=call_hook
        ):
            if node.kwargs.get("hook_type", None) != "post_hook":
                continue
            post_hooks.append(node)

        for node in reversed(post_hooks):
            getitem_node = node.args[0]
            output_nodes = node.args[1]
            input_nodes = node.args[2]

            if len(output_nodes) > 0:
                continue

            input_nodes_and_users = []
            input_nodes_and_users.extend(list(input_nodes))
            for input_node in input_nodes:
                input_nodes_and_users.extend(
                    user
                    for user in list(input_node.users.keys())
                    if not (
                        user.op == "call_function"
                        and user.target == call_hook
                        and node.kwargs.get("hook_type", None) == "post_hook"
                    )
                )

            arg = max(input_nodes_and_users)  # last input users
            if (
                arg.op == "call_function"
                and arg.target == torch.ops.inductor.accumulate_grad_.default
            ):
                param_node = arg.args[0]
                post_acc_grad_hook_node = None
                for n in list(param_node.users.keys()):
                    if (
                        n.op == "call_function"
                        and n.target == call_hook
                        and n.kwargs.get("hook_type", None) == "post_acc_grad_hook"
                    ):
                        post_acc_grad_hook_node = n

                if post_acc_grad_hook_node is not None:
                    post_acc_grad_hook_node.append(getitem_node)
                    getitem_node.append(node)
                    continue

            if arg is not node.prev and not self.is_placeholder(arg):
                arg.append(getitem_node)
                getitem_node.append(node)

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


# state of the autograd engine dispatch, kept in sync by enable/disable context managers
compiled_autograd_enabled = False

# global flag to check if compiled autograd is enabled but Dynamo stance is "force_eager"
compiled_autograd_enabled_force_eager = False

# global flag to check if we are processing graphs produced from a compiled autograd graph
in_compiled_autograd_region = False


@contextlib.contextmanager
def _enable(compiler_fn, dynamic=False):
    if dynamic:
        assert type(dynamic) is bool

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

        (
            prior_compiler,
            prior_dynamic,
        ) = torch._C._dynamo.compiled_autograd.set_autograd_compiler(
            functools.partial(AutogradCompilerInstance, compiler_fn), dynamic
        )
        if snapshot_verbose_logging_enabled():
            torch._C._dynamo.compiled_autograd.set_verbose_logger(verbose_log)
        global compiled_autograd_enabled
        compiled_autograd_enabled = True
        try:
            with torch.autograd.set_multithreading_enabled(False):
                yield
        finally:
            if not prior_compiler:
                compiled_autograd_enabled = False
            torch._C._dynamo.compiled_autograd.set_autograd_compiler(
                prior_compiler, prior_dynamic
            )


@contextlib.contextmanager
def _disable():
    (
        prior_compiler,
        prior_dynamic,
    ) = torch._C._dynamo.compiled_autograd.set_autograd_compiler(None, False)
    global compiled_autograd_enabled
    compiled_autograd_enabled = False
    try:
        yield
    finally:
        if prior_compiler:
            compiled_autograd_enabled = True
        torch._C._dynamo.compiled_autograd.set_autograd_compiler(
            prior_compiler, prior_dynamic
        )


# return to starting state of a new process
def reset() -> None:
    global compiled_autograd_enabled
    compiled_autograd_enabled = False
    assert not in_compiled_autograd_region
    torch._C._dynamo.compiled_autograd.set_autograd_compiler(None, False)
    torch._C._dynamo.compiled_autograd.set_verbose_logger(None)
    torch._C._dynamo.compiled_autograd.clear_cache()
    global COMPILE_COUNTER
    COMPILE_COUNTER = itertools.count()
