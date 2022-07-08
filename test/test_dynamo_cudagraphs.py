# Owner(s): ["module: cuda graphs"]

import torch
from unittest.mock import patch
from collections import defaultdict
from typing import Set
from torch.fx import GraphModule
from torch.nn import Module
from torch.utils._pytree import tree_map
from torch._subclasses import FakeTensorMode
from torch.fx.passes.backends.cudagraphs import partition_cudagraphs
from torch.multiprocessing.reductions import StorageWeakRef
from torch.fx.experimental.proxy_tensor import (
    ProxyTensor,
    ProxyTorchDispatchMode,
    wrap_output,
    unwrap_proxy,
    PythonKeyTracer,
)
from torch.utils._python_dispatch import enable_torch_dispatch_mode
from torch.testing._internal.common_utils import (
    TestCase,
    run_tests,
)

try:
    import torchdynamo

    TEST_DYNAMO = True
except ImportError:
    TEST_DYNAMO = False

TEST_CUDA = torch.cuda.is_available()

if not TEST_CUDA or not TEST_DYNAMO:
    print("CUDA or dynamo not available, skipping tests", file=sys.stderr)
    TestCase = object  # noqa: F811


def cloner(t):
    if isinstance(t, torch.Tensor):
        return t.clone()
    else:
        return t


class CudaGraphModule(Module):
    gm: GraphModule
    mutated_inputs: Set[int]

    def __init__(self, gm, mutated_inputs):
        super().__init__()
        self.gm = gm
        self.mutated_inputs = mutated_inputs

    warmed_up = False

    # these are all None or all filled
    graph = None
    static_inputs = None
    static_outputs = None

    # NB: we override __call__ as we don't need any nn.Module machinery
    # and to reduce overhead
    def __call__(self, *args):
        # TODO: once we've recorded here, we'd like to replace the __call__
        # implementation with compiled bytecode that copies into static, replays
        # the cuda graph, then copies out.  First condition is the hotpath,
        # needs optimizing
        if self.graph is not None:
            assert len(args) == len(self.static_inputs)
            for dst, src in zip(self.static_inputs, args):
                dst.copy_(src)
            self.graph.replay()
            for i in self.mutated_inputs:
                args[i].copy_(self.static_inputs[i])
            return tree_map(cloner, self.static_outputs)

        elif self.warmed_up:
            # record
            self.static_inputs = [x.clone() for x in args]
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.static_outputs = self.gm(*self.static_inputs)
            # NB: recording doesn't actually run the operations, so
            # now we immediately replay the graph to serve up the result
            self.graph.replay()
            for i in self.mutated_inputs:
                args[i].copy_(self.static_inputs[i])
            return tree_map(cloner, self.static_outputs)

        else:
            # warmup
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                r = self.gm(*args)
            torch.cuda.current_stream().wait_stream(stream)
            self.warmed_up = True
            return r


class FindInputMutations(torch.fx.Interpreter):
    def __init__(self, gm):
        super().__init__(gm)
        self.inputs = defaultdict(set)
        self.input_idx = 0
        self.mutated_inputs = set()

    def placeholder(self, target, args, kwargs):
        r = super().placeholder(target, args, kwargs)
        # NB: inputs could be aliased
        self.inputs[StorageWeakRef(r.storage())].add(self.input_idx)
        self.input_idx += 1
        return r

    def call_function(self, target, args, kwargs):
        schema = target._schema
        for i, arg in enumerate(schema.arguments):
            if i < len(args):
                argument = args[i]
            else:
                if arg.name not in kwargs:
                    continue
                argument = kwargs[arg.name]
            mut_arg = False
            if arg.alias_info:
                if arg.alias_info.is_write:
                    mut_arg = True
            if mut_arg:
                self.mutated_inputs |= self.inputs[StorageWeakRef(argument.storage())]
        return super().call_function(target, args, kwargs)

    def __call__(self, *args):
        super().run(*args)
        return self.mutated_inputs


class ProxyTensorInterpreter(torch.fx.Interpreter):
    def __init__(self, module: torch.fx.GraphModule, **kwargs):
        super().__init__(module, **kwargs)
        self.new_graph = torch.fx.Graph()
        self.new_module = torch.fx.GraphModule(module, self.new_graph)
        self.tracer = torch.fx.proxy.GraphAppendingTracer(self.new_graph)

    def placeholder(self, target, args, kwargs):
        out = super().placeholder(target, args, kwargs)
        return ProxyTensor(
            out, torch.fx.Proxy(self.new_graph.placeholder(target), self.tracer)
        )

    def get_attr(self, target, args, kwargs):
        out = super().get_attr(target, args, kwargs)
        self.new_module.register_buffer(target, self.module.get_buffer(target))
        return ProxyTensor(
            out, torch.fx.Proxy(self.new_graph.get_attr(target), self.tracer)
        )

    # Use the mode in case the function call doesn't have any tensor arguments
    def call_function(self, target, args, kwargs):
        with ProxyTorchDispatchMode(self.tracer):
            return super().call_function(target, args, kwargs)

    def call_method(self, target, args, kwargs):
        with ProxyTorchDispatchMode(self.tracer):
            return super().call_method(target, args, kwargs)

    # Can't do call_module because the interpreter not reentrant

    def output(self, target, args, kwargs):
        out = super().output(target, args, kwargs)

        def unwrap(e):
            return e.proxy.node if isinstance(e, ProxyTensor) else e

        self.new_graph.output(tree_map(unwrap, out))
        return out


def unwrap_elem(e):
    return e.elem if isinstance(e, ProxyTensor) else e


def unwrap_proxy_node(e):
    return e.proxy.node if isinstance(e, ProxyTensor) else e


class ApplyCudaGraphs(torch.fx.Interpreter):
    # All module calls are assumed to be fusion groups, since
    # this is post AOTAutograd which would have squashed all the modules.
    # Module assumed to be called only once.
    def call_module(self, target, args, kwargs):
        if hasattr(self, 'proxy_mode'):
            proxy_mode = self.proxy_mode
        else:
            from torch._C import _get_torch_dispatch_mode
            proxy_mode = _get_torch_dispatch_mode()
            assert isinstance(proxy_mode, ProxyTorchDispatchMode)
        with enable_torch_dispatch_mode(proxy_mode.inner, replace=proxy_mode):
            assert not kwargs
            # Don't trace the module, but do run the module to get the correct
            # out result
            out = super().call_module(target, tree_map(unwrap_elem, args), kwargs)
            submod = self.module.get_submodule(target)
            mutated_inputs = FindInputMutations(submod)(*map(unwrap_elem, args))
            proxy_mode.tracer.root.add_module(target, CudaGraphModule(submod, mutated_inputs))
            return wrap_output(
                out,
                proxy_mode.tracer.create_proxy(
                    "call_module",
                    target,
                    tree_map(unwrap_proxy, args),
                    tree_map(unwrap_proxy, kwargs)
                )
            )

def trace_interp(interp, inputs):
    new_graph = torch.fx.Graph()
    new_module = torch.fx.GraphModule(interp.module, new_graph)
    tracer = PythonKeyTracer()
    tracer.graph = new_graph
    tracer.root = new_module
    tracer.tensor_attrs = {}
    fake_mode = FakeTensorMode()
    args = [
        ProxyTensor(fake_mode.from_tensor(i), tracer.create_proxy("placeholder", n.target, n.args, n.kwargs))
        for i, n in zip(inputs, filter(lambda n: n.op == "placeholder", interp.module.graph.nodes))
    ]
    proxy_mode = ProxyTorchDispatchMode(tracer)
    interp.proxy_mode = proxy_mode
    with fake_mode, proxy_mode:
        outs = interp.run(*args)
    new_graph.output(tree_map(unwrap_proxy_node, outs))
    new_module.recompile()
    return new_module

def fake_signature(fn, nargs):
    """FX gets confused by varargs, de-confuse it"""
    argnames = ",".join(f"arg{i}" for i in range(nargs))
    return eval(f"lambda {argnames}: fn({argnames})", {"fn": fn})

def trace_interp2(interp, inputs):
    # this looks cool but it mutates the original module
    return torch.fx.experimental.proxy_tensor.make_fx(fake_signature(interp.run, len(inputs)), use_fake=True)(*inputs)


def cudagraphs(model, inputs):
    model = partition_cudagraphs(model, inputs)
    model = trace_interp2(ApplyCudaGraphs(model), inputs)
    return model


def aot_autograd_cudagraphs(model, inputs):
    kwargs = {
        # these are taken from memory_efficient_fusion()
        "fw_compiler": cudagraphs,
        "bw_compiler": cudagraphs,
        "hasher_type": "StaticShapeHasher",
    }

    def _wrapped_bw_compiler(*args, **kwargs):
        # stop TorchDynamo from trying to compile our generated backwards pass
        return torchdynamo.disable(bw_compiler(*args, **kwargs))

    bw_compiler = kwargs.get("bw_compiler") or kwargs["fw_compiler"]
    kwargs["bw_compiler"] = _wrapped_bw_compiler

    from functorch.compile import aot_module_simplified

    return aot_module_simplified(model, **kwargs)


class TestDynamoCudaGraphs(TestCase):
    @patch("torchdynamo.config.verify_correctness", True)
    def test_basic(self):
        def model(x, y):
            return (x + y) * y

        with torchdynamo.optimize(aot_autograd_cudagraphs):
            for i in range(5):
                x = torch.randn(3, device="cuda", requires_grad=True)
                y = torch.randn(3, device="cuda")
                loss = model(x, y).sum()
                loss.backward()

    @patch("torchdynamo.config.verify_correctness", True)
    def test_dtoh(self):
        def model(x, y):
            a = x + y
            b = a.cpu() * 3
            return b

        with torchdynamo.optimize(aot_autograd_cudagraphs):
            for i in range(5):
                x = torch.randn(3, device="cuda", requires_grad=True)
                y = torch.randn(3, device="cuda")
                loss = model(x, y).sum()
                loss.backward()

    @patch("torchdynamo.config.verify_correctness", True)
    def test_htod(self):
        def model(x, y):
            a = x + y
            return a * 3

        with torchdynamo.optimize(aot_autograd_cudagraphs):
            for i in range(5):
                x = torch.randn(3, device="cuda", requires_grad=True)
                y = torch.randn((), device="cpu")
                loss = model(x, y).sum()
                loss.backward()

    @patch("torchdynamo.config.verify_correctness", True)
    def test_mutate_input(self):
        def model(x, y):
            y.add_(3)
            return x * y

        with torchdynamo.optimize(aot_autograd_cudagraphs):
            for i in range(5):
                with self.subTest(i):
                    x = torch.randn(3, device="cuda", requires_grad=True)
                    y = torch.randn(3, device="cuda")
                    y_orig = y.clone()
                    loss = model(x, y).sum()
                    self.assertEqual(y, y_orig + 3)
                    loss.backward()

    def test_constant_proxy_tensor(self):
        from torch.fx.experimental.proxy_tensor import make_fx

        def f():
            val = torch.tensor(float('inf'))
            return torch.full((100, 100), val)

        make_fx(f)()

    def test_constant_proxy_tensor_mut(self):
        from torch.fx.experimental.proxy_tensor import make_fx

        def f():
            val = torch.tensor(float(1))
            val.add_(2)
            return torch.full((100, 100), val)

        make_fx(f)()

    @patch("torchdynamo.config.verify_correctness", True)
    def test_mutate_constant(self):
        def model(x, y):
            c = torch.tensor(1)
            c.add_(2)
            return x * y * 0 + c

        with torchdynamo.optimize(aot_autograd_cudagraphs):
            for i in range(5):
                with self.subTest(i):
                    x = torch.randn(1, device="cuda", requires_grad=True)
                    y = torch.randn(1, device="cuda")
                    loss = model(x, y).sum()
                    self.assertEqual(loss, torch.tensor(3.0, device="cuda"))
                    loss.backward()

    @patch("torchdynamo.config.verify_correctness", True)
    def test_factory(self):
        def model(y):
            x = torch.zeros(3, device="cuda:0")
            x.add_(3)
            return x * y

        with torchdynamo.optimize(aot_autograd_cudagraphs):
            for i in range(5):
                with self.subTest(i):
                    y = torch.randn(3, device="cuda:0", requires_grad=True)
                    loss = model(y).sum()
                    loss.backward()


if __name__ == "__main__":
    run_tests()
