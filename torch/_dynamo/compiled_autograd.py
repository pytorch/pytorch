import contextlib
import functools

import torch
from torch._dynamo.source import ConstantSource
from torch._dynamo.utils import counters
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


class AutogradCompilerInstance:
    def __init__(self, compiler_fn) -> None:
        self.compiler_fn = compiler_fn
        self.stack = contextlib.ExitStack()
        self.fake_tensor_mode = FakeTensorMode(
            allow_fallback_kernels=True,
            allow_non_fake_inputs=True,
        )
        self.fx_tracer = PythonKeyTracer()
        self.proxy_mode = ProxyTorchDispatchMode(self.fx_tracer, "fake")
        self.gm = None

    def wrap_fake(self, x, name):
        assert isinstance(x, torch.Tensor)
        return self.fake_tensor_mode.from_tensor(x, source=ConstantSource(name))

    def begin_capture(self, inputs):
        counters["compiled_autograd"]["captures"] += 1
        print("BEGIN_CAPTURE", len(inputs))
        self.fx_tracer.root = torch.nn.Module()
        self.fx_tracer.graph = torch.fx.Graph(tracer_cls=PythonKeyTracer)
        self.fx_tracer.tensor_attrs = {}

        inputs = [self.wrap_fake(x, f"inputs[{i}]") for i, x in enumerate(inputs)]
        args_proxy = self.fx_tracer.create_proxy("placeholder", "inputs", (), {})
        proxies = [args_proxy[i] for i in range(len(inputs))]
        self.bind_tensors_to_proxies(inputs, proxies)

        # TODO(jansel): are all these modes needed?
        self.stack.enter_context(decompose({}))
        self.stack.enter_context(self.fake_tensor_mode)
        self.stack.enter_context(self.proxy_mode.sym_mode)
        self.stack.enter_context(self.proxy_mode)
        self.stack.enter_context(disable_autocast_cache())
        self.stack.enter_context(disable_proxy_modes_tracing(enable_current=True))
        return inputs

    def tensor_pre_hook(self, inputs, hook, i: int):
        proxy = self.fx_tracer.create_proxy(
            "call_function", hook, (self.to_proxy(inputs[i]),), {}
        )
        with disable_proxy_modes_tracing():
            inputs[i] = inputs[i].clone()
            self.bind_tensors_to_proxies([inputs[i]], [proxy])
        return inputs

    def pre_hook(self, inputs, hook):
        proxies = self.fx_tracer.create_proxy(
            "call_function", hook, (self.to_proxy(inputs),), {}
        )
        with disable_proxy_modes_tracing():
            inputs = [x.clone() for x in inputs]
            self.bind_tensors_to_proxies(inputs, proxies)
        return inputs

    def post_hook(self, outputs, inputs, hook):
        proxies = self.fx_tracer.create_proxy(
            "call_function",
            hook,
            (
                self.to_proxy(outputs),
                self.to_proxy(inputs),
            ),
            {},
        )
        with disable_proxy_modes_tracing():
            outputs = [x.clone() for x in outputs]
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
        print(self.fx_tracer.graph)
        print("END_CAPTURE", len(outputs))
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
        return fetch_tensor_proxy(self.fx_tracer)(t).proxy

    def bind_tensors_to_proxies(self, tensors, proxies):
        if isinstance(proxies, torch.fx.Proxy):
            proxies = [proxies[i] for i in range(len(tensors))]
        assert len(tensors) == len(proxies)
        assert all(x is not None for x in tensors)
        assert all(x is not None for x in proxies)
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
