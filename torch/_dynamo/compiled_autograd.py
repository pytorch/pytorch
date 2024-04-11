import contextlib
import functools
from typing import List, Optional

import torch
from torch._dynamo.external_utils import call_backward, call_hook, exec_final_callbacks
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
from torch.fx.proxy import Proxy

compiled_autograd_log = getArtifactLogger(__name__, "compiled_autograd")


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

    def wrap_fake(self, x, source):
        assert isinstance(x, torch.Tensor)
        return self.fake_tensor_mode.from_tensor(x, source=source)

    @staticmethod
    def source(name, idx) -> GetItemSource:
        return GetItemSource(LocalSource(name), idx)

    def begin_capture(self, inputs: List[torch.Tensor], sizes: List[int]):
        counters["compiled_autograd"]["captures"] += 1
        self.fx_tracer.root = torch.nn.Module()
        self.fx_tracer.graph = torch.fx.Graph(tracer_cls=PythonKeyTracer)
        self.fx_tracer.tensor_attrs = {}
        args_proxy = self.fx_tracer.create_proxy("placeholder", "inputs", (), {})
        sizes_proxy = self.fx_tracer.create_proxy("placeholder", "sizes", (), {})
        self.hooks_proxy = self.fx_tracer.create_proxy("placeholder", "hooks", (), {})

        # tensor inputs to fake tensors
        inputs = [
            self.wrap_fake(x, self.source("inputs", idx))
            for idx, x in enumerate(inputs)
        ]
        proxies = [args_proxy[i] for i in range(len(inputs))]
        self.bind_tensors_to_proxies(inputs, proxies)

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

        # TODO(jansel): are all these modes needed?
        self.stack.enter_context(decompose({}))
        self.stack.enter_context(self.fake_tensor_mode)
        self.stack.enter_context(self.proxy_mode.sym_mode)
        self.stack.enter_context(self.proxy_mode)
        self.stack.enter_context(disable_autocast_cache())
        return inputs, sizes

    def proxy_call_backward(
        self,
        inputs,
        output_metadatas,
        saved_tensors,
        backward_idx: int,
    ):
        assert self.hooks_proxy is not None
        backward_fn = self.hooks_proxy[backward_idx]  # type: ignore[index]
        proxies = self.fx_tracer.create_proxy(
            kind="call_function",
            target=call_backward,
            args=(
                backward_fn,
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

    def proxy_call_hook(self, hook, *args):
        return self.fx_tracer.create_proxy(
            "call_function",
            call_hook,
            (
                hook,
                *[self.to_proxy(x) for x in args],
            ),
            {},
        )

    def tensor_pre_hook(self, inputs, hook_id, i: int):
        assert self.hooks_proxy is not None
        hook = self.hooks_proxy[hook_id]  # type: ignore[index]
        print(f"tensor_pre_hook: {hook}")
        proxy = self.proxy_call_hook(
            hook,
            inputs[i],
        )
        with disable_proxy_modes_tracing():
            inputs[i] = maybe_clone(inputs[i])
            self.bind_tensors_to_proxies([inputs[i]], [proxy])
        return inputs

    def pre_hook(self, inputs, hook_id):
        assert self.hooks_proxy is not None
        hook = self.hooks_proxy[hook_id]  # type: ignore[index]
        print(f"pre_hook: {hook}")
        proxies = self.proxy_call_hook(
            hook,
            inputs,
        )
        with disable_proxy_modes_tracing():
            inputs = [maybe_clone(x) for x in inputs]
            self.bind_tensors_to_proxies(inputs, proxies)
        return inputs

    def post_hook(self, outputs, inputs, hook_id):
        assert self.hooks_proxy is not None
        hook = self.hooks_proxy[hook_id]  # type: ignore[index]
        print(f"post_hook: {hook}")
        proxies = self.proxy_call_hook(
            hook,
            outputs,
            inputs,
        )
        with disable_proxy_modes_tracing():
            outputs = [maybe_clone(x) for x in outputs]
            self.bind_tensors_to_proxies(outputs, proxies)
        return outputs

    def post_acc_grad_hook(self, input, hook_id):
        assert isinstance(input, torch.Tensor)
        assert self.hooks_proxy is not None
        hook = self.hooks_proxy[hook_id]  # type: ignore[index]
        print(f"post_acc_grad_hook: {hook}")
        proxies = self.proxy_call_hook(
            hook,
            input,
        )
        with disable_proxy_modes_tracing():
            input = [maybe_clone(input)]
            self.bind_tensors_to_proxies(input, proxies)
        return input

    def end_capture(self, outputs):
        self.fx_tracer.create_proxy(
            "call_function",
            exec_final_callbacks,
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
        self.reorder_accumulate_grad_nodes()
        graph = GraphModule(
            self.fx_tracer.root, self.fx_tracer.graph, "CompiledAutograd"
        )
        set_locals_to_steal(graph, ["inputs"])
        compiled_autograd_log.info(
            "%s", lazy_format_graph_code("Compiled autograd graph", graph)
        )
        trace_structured(
            "compiled_autograd_graph",
            payload_fn=lambda: graph.print_readable(print_output=False),
        )
        return self.compiler_fn(graph)

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
        assert isinstance(t, (torch.Tensor, torch.SymInt))
        return fetch_object_proxy(self.fx_tracer)(t).proxy

    def bind_tensors_to_proxies(self, tensors, proxies):
        if isinstance(proxies, torch.fx.Proxy):
            proxies = [proxies[i] for i in range(len(tensors))]
        assert len(tensors) == len(proxies)
        track_tensor_tree(tensors, proxies, constant=None, tracer=self.fx_tracer)

    def bind_backward_state(self, index: int):
        assert self.hooks_proxy is not None
        proxy = self.hooks_proxy[index]  # type: ignore[index]
        bw_state = BackwardState()
        track_tensor_tree(bw_state, proxy, constant=None, tracer=self.fx_tracer)
        return bw_state


compiled_autograd_enabled = False

# We may have code like:
# with enable(compiler_fn):
#   ...
#   with disable():
#     ...
#   ...
# The disable() call just want to disable compiled autograd temporarily.
# But overall the feature is enabled.
#
# The code covered by the disable context manager has no way to know if
# compiled autograd is overall eanbled. Use another variable
# compiled_autograd_enabled_count to indicate how many times compiled
# autograd has been enabled in the call stack for this purpose.
compiled_autograd_enabled_count = 0


@contextlib.contextmanager
def enable(compiler_fn):
    prior = torch._C._dynamo.compiled_autograd.set_autograd_compiler(
        functools.partial(AutogradCompilerInstance, compiler_fn)
    )
    global compiled_autograd_enabled, compiled_autograd_enabled_count
    compiled_autograd_enabled = True
    compiled_autograd_enabled_count += 1
    try:
        with torch.autograd.set_multithreading_enabled(False):
            yield
    finally:
        compiled_autograd_enabled_count -= 1
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
