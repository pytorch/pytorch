import contextlib
import functools
from typing import List

import torch
from torch._dynamo.external_utils import call_hook
from torch._dynamo.source import GetItemSource, LocalSource
from torch._dynamo.utils import counters
from torch._prims_common import clone_preserve_strides
from torch._subclasses import FakeTensorMode
from torch.fx import GraphModule
from torch.fx.experimental.proxy_tensor import (
    decompose,
    disable_autocast_cache,
    disable_proxy_modes_tracing,
    fetch_tensor_proxy,
    ProxyTorchDispatchMode,
    PythonKeyTracer,
    track_tensor_tree,
)
from torch.fx.experimental.symbolic_shapes import DimDynamic, ShapeEnv


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
        self.hooks_proxy = None

    def wrap_fake(self, x, source):
        assert isinstance(x, torch.Tensor)
        return self.fake_tensor_mode.from_tensor(x, source=source)

    @staticmethod
    def source(name, idx):
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
        self.stack.enter_context(disable_proxy_modes_tracing(enable_current=True))
        return inputs, sizes

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
        hook = self.hooks_proxy[hook_id]
        proxy = self.proxy_call_hook(
            hook,
            inputs[i],
        )
        with disable_proxy_modes_tracing():
            inputs[i] = maybe_clone(inputs[i])
            self.bind_tensors_to_proxies([inputs[i]], [proxy])
        return inputs

    def pre_hook(self, inputs, hook_id):
        hook = self.hooks_proxy[hook_id]
        proxies = self.proxy_call_hook(
            hook,
            inputs,
        )
        with disable_proxy_modes_tracing():
            inputs = [maybe_clone(x) for x in inputs]
            self.bind_tensors_to_proxies(inputs, proxies)
        return inputs

    def post_hook(self, outputs, inputs, hook_id):
        hook = self.hooks_proxy[hook_id]
        proxies = self.proxy_call_hook(
            hook,
            outputs,
            inputs,
        )
        with disable_proxy_modes_tracing():
            outputs = [maybe_clone(x) for x in outputs]
            self.bind_tensors_to_proxies(outputs, proxies)
        return outputs

    def end_capture(self, outputs):
        self.stack.close()
        self.fx_tracer.create_node(
            "output",
            "output",
            (self.fx_tracer.create_arg(self.to_proxy(outputs)),),
            {},
        )
        return self.compiler_fn(
            GraphModule(self.fx_tracer.root, self.fx_tracer.graph, "CompiledAutograd")
        )

    def to_proxy(self, t):
        if t is None:
            return None
        if isinstance(t, list):
            return [self.to_proxy(x) for x in t]
        if isinstance(t, tuple):
            return tuple(self.to_proxy(x) for x in t)
        assert isinstance(t, (torch.Tensor, torch.SymInt))
        return fetch_tensor_proxy(self.fx_tracer)(t).proxy

    def bind_tensors_to_proxies(self, tensors, proxies):
        if isinstance(proxies, torch.fx.Proxy):
            proxies = [proxies[i] for i in range(len(tensors))]
        assert len(tensors) == len(proxies)
        track_tensor_tree(tensors, proxies, constant=None, tracer=self.fx_tracer)


@contextlib.contextmanager
def enable(compiler_fn):
    prior = torch._C._dynamo.compiled_autograd.set_autograd_compiler(
        functools.partial(AutogradCompilerInstance, compiler_fn)
    )
    yield
    torch._C._dynamo.compiled_autograd.set_autograd_compiler(prior)


@contextlib.contextmanager
def disable():
    prior = torch._C._dynamo.compiled_autograd.set_autograd_compiler(None)
    yield
    torch._C._dynamo.compiled_autograd.set_autograd_compiler(prior)
