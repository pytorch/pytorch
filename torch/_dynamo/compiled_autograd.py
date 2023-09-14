import contextlib
import functools
from typing import List

import torch
from torch._dynamo.external_utils import call_hook
from torch._dynamo.source import GetItemSource, LocalSource
from torch._dynamo.utils import counters, lazy_format_graph_code
from torch._logging import getArtifactLogger
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

compiled_autograd_log = getArtifactLogger(__name__, "compiled_autograd")

import torch.utils._pytree as pytree
from torch._C import _ExcludeDispatchKeyGuard, DispatchKey, DispatchKeySet
from torch._higher_order_ops.utils import autograd_not_implemented

from torch._ops import HigherOrderOperator
from torch.utils._python_dispatch import _get_current_dispatch_mode


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
        graph = GraphModule(
            self.fx_tracer.root, self.fx_tracer.graph, "CompiledAutograd"
        )
        compiled_autograd_log.info(
            "%s", lazy_format_graph_code("Compiled autograd graph", graph)
        )
        return self.compiler_fn(graph)

    def to_proxy(self, t):
        if t is None:
            return None
        if isinstance(t, list):
            return [self.to_proxy(x) for x in t]
        if isinstance(t, tuple):
            return tuple(self.to_proxy(x) for x in t)
        assert isinstance(t, (torch.Tensor, torch.SymInt))
        fetched_tensor = fetch_tensor_proxy(self.fx_tracer)(t)
        if hasattr(fetched_tensor, "proxy"):
            return fetched_tensor.proxy
        else:
            return self.fx_tracer.unwrap_proxy(
                torch._functorch.aot_autograd.from_fun(fetched_tensor)
            )

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
    with torch.autograd.set_multithreading_enabled(False):
        yield
    torch._C._dynamo.compiled_autograd.set_autograd_compiler(prior)


@contextlib.contextmanager
def disable():
    prior = torch._C._dynamo.compiled_autograd.set_autograd_compiler(None)
    yield
    torch._C._dynamo.compiled_autograd.set_autograd_compiler(prior)


# Invoke backward hook is a higher order op meant for both invoking a bound hook,
# and for registering it as a call_function in the backward graph.
# This allows us to re-enter dynamo during compiled autograd to trace (or graph break)
# the hook as needed. This, in turn, means we can support hooks in backward with complex python
# state mutation. If we were to not do this, the hooks would get inlined into their composing aten ops,
# and we would lose the python state mutation.
def _invoke_in_backward(*args, fn, reenter):
    return _invoke_in_backward_op(*args, fn=fn, reenter=reenter)


_invoke_in_backward_op = HigherOrderOperator("_invoke_in_backward")


def dynamo_interceding_fn_wrapper(*args, fn):
    # This wrapper intercepts calls to fn, and calls the real fn via _invoke_in_backward
    # However, as reenter is set to false, the call_function created during trace
    # will point to the actual fn, and not to this function.
    return _invoke_in_backward_op(*args, fn=fn, reenter=False)


@_invoke_in_backward_op.py_impl(ProxyTorchDispatchMode)
def inner_trace(*args, fn, reenter):
    mode = _get_current_dispatch_mode()
    if isinstance(fn, functools.partial):
        fn.__name__ = fn.func.__name__  # type: ignore[attr-defined]
    original_fn = fn
    if reenter:
        # If the reenter flag is set, we wrap the original fn in dynamo_interceding_fn_wrapper
        # and write that to the graph. This produces an aot_autograd graph during backwards that
        # points to dynamo_interceding_fn_wrapper. Then, during compiled autograd, we use
        # dynamo_interceding_fn_wrapper to _invoke_in_backward the original fn under dynamo. The actual
        # dynamo part of dynamo_interceding_fn_wrapper happens during compiled autograd.
        fn = functools.partial(dynamo_interceding_fn_wrapper, fn=fn)
        fn.__name__ = fn.func.__name__

    args = pytree.tree_map(torch._functorch.aot_autograd.from_fun, args)
    proxy_args = pytree.tree_map(mode.tracer.unwrap_proxy, args)
    out_proxy = mode.tracer.create_proxy(
        "call_function", fn, proxy_args, {}, name="invocation"
    )
    args = original_fn(*args)
    args = track_tensor_tree(args, out_proxy, constant=None, tracer=mode.tracer)
    return pytree.tree_map(torch._functorch.aot_autograd.to_fun, args)


@_invoke_in_backward_op.py_impl(FakeTensorMode)
def inner_fake(*args, fn, reenter):
    raise RuntimeError("This op should never be invoked here")


@_invoke_in_backward_op.py_impl(DispatchKey.CompositeExplicitAutograd)
def _invoke_in_backward_op_dense(*args, fn, reenter):
    mode = _get_current_dispatch_mode()
    assert mode is None, "Mode should never be enabled for CPU/CUDA key"
    return fn(*args)


_invoke_in_backward_op.py_impl(DispatchKey.Autograd)(
    autograd_not_implemented(_invoke_in_backward_op, deferred_error=True)
)


@_invoke_in_backward_op.py_impl(DispatchKey.Functionalize)
def _invoke_in_backward_functionalized(*args, fn, reenter):
    mode = _get_current_dispatch_mode()
    with _ExcludeDispatchKeyGuard(DispatchKeySet(DispatchKey.Functionalize)):
        return _invoke_in_backward_op(*args, fn=fn, reenter=reenter)


# TODO(voz): Make this automatic for keys, this is very ugly atm
_invoke_in_backward_op.fallthrough(DispatchKey.PythonDispatcher)  # type: ignore[attr-defined]
_invoke_in_backward_op.fallthrough(DispatchKey.PythonTLSSnapshot)  # type: ignore[attr-defined]
_invoke_in_backward_op.fallthrough(DispatchKey.ADInplaceOrView)
_invoke_in_backward_op.fallthrough(DispatchKey.BackendSelect)
_invoke_in_backward_op.fallthrough(DispatchKey.AutocastCPU)  # type: ignore[attr-defined]
_invoke_in_backward_op.fallthrough(DispatchKey.AutocastCUDA)  # type: ignore[attr-defined]
