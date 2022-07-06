import torch
import contextlib
import dataclasses
import inspect
from unittest.mock import patch
from collections import defaultdict
from typing import Set
from torch.fx import GraphModule
from torch.nn import Module
from torch.utils._pytree import tree_map
from torch._subclasses import FakeTensorMode
from torch.fx.passes.backends.cudagraphs import partition_cudagraphs
from torch.multiprocessing.reductions import StorageWeakRef
from torch.fx.experimental.proxy_tensor import make_fx, ProxyTensor, wrap_output
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
    print('CUDA or dynamo not available, skipping tests', file=sys.stderr)
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
        return ProxyTensor(out, torch.fx.Proxy(self.new_graph.placeholder(target), self.tracer))

    def get_attr(self, target, args, kwargs):
        out = super().get_attr(target, args, kwargs)
        return ProxyTensor(out, torch.fx.Proxy(self.new_graph.get_attr(target), self.tracer))

    # call_function, call_method, call_module get traced automatically by the ProxyTensors.
    # NB: methods and modules will get inlined if you don't override explicitly

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

class ApplyCudaGraphs(ProxyTensorInterpreter):
    # All module calls are assumed to be fusion groups, since
    # this is post AOTAutograd which would have squashed all the modules.
    # Module assumed to be called only once.
    def call_module(self, target, args, kwargs):
        assert not kwargs
        # Don't trace the module, but do run the module to get the correct
        # out result
        out = super().call_module(target, tree_map(unwrap_elem, args), kwargs)
        submod = self.module.get_submodule(target)
        mutated_inputs = FindInputMutations(submod)(*map(unwrap_elem, args))
        # smh the module didn't get transferred wut
        self.new_module.add_submodule(target, CudaGraphModule(submod, mutated_inputs))
        return wrap_output(out, torch.fx.Proxy(self.new_graph.call_module(target, tree_map(unwrap_proxy_node, args), tree_map(unwrap_proxy_node, kwargs)), self.tracer))

def cudagraphs(model, inputs):
    model = partition_cudagraphs(model, inputs)

    # Your interpreter
    t = ApplyCudaGraphs(model)
    with FakeTensorMode.push() as mode:
        t.run(*map(mode.from_tensor, inputs))
    model = t.new_module
    model.recompile()

    return model

def aot_autograd_cudagraphs(model, inputs):
    from functorch.compile import default_decompositions
    from functorch.compile import min_cut_rematerialization_partition
    from functorch.compile import ts_compile

    kwargs = {
        # these are taken from memory_efficient_fusion()
        "fw_compiler": cudagraphs,
        "bw_compiler": cudagraphs,
        # "partition_fn": min_cut_rematerialization_partition,
        "hasher_type": "StaticShapeHasher",
        "decompositions": default_decompositions,
    }

    def _wrapped_bw_compiler(*args, **kwargs):
        # stop TorchDynamo from trying to compile our generated backwards pass
        return torchdynamo.disable(bw_compiler(*args, **kwargs))

    bw_compiler = kwargs.get("bw_compiler") or kwargs["fw_compiler"]
    kwargs["bw_compiler"] = _wrapped_bw_compiler

    from functorch.compile import aot_module_simplified

    return aot_module_simplified(model, **kwargs)


class TestDynamoCudaGraphs(TestCase):
    @patch('torchdynamo.config.verify_correctness', True)
    def test_basic(self):
        def model(x, y):
            return (x + y) * y

        with torchdynamo.optimize(aot_autograd_cudagraphs):
            for i in range(5):
                x = torch.randn(3, device='cuda', requires_grad=True)
                y = torch.randn(3, device='cuda')
                loss = model(x, y).sum()
                loss.backward()

    @patch('torchdynamo.config.verify_correctness', True)
    def test_dtoh(self):
        def model(x, y):
            a = x + y
            b = a.cpu() * 3
            return b

        with torchdynamo.optimize(aot_autograd_cudagraphs):
            for i in range(5):
                x = torch.randn(3, device='cuda', requires_grad=True)
                y = torch.randn(3, device='cuda')
                loss = model(x, y).sum()
                loss.backward()

    @patch('torchdynamo.config.verify_correctness', True)
    def test_htod(self):
        def model(x, y):
            a = x + y
            return a * 3

        with torchdynamo.optimize(aot_autograd_cudagraphs):
            for i in range(5):
                x = torch.randn(3, device='cuda', requires_grad=True)
                y = torch.randn((), device='cpu')
                loss = model(x, y).sum()
                loss.backward()

    @patch('torchdynamo.config.verify_correctness', True)
    def test_mutate_input(self):
        def model(x, y):
            y.add_(3)
            return x * y

        with torchdynamo.optimize(aot_autograd_cudagraphs):
            for i in range(5):
                with self.subTest(i):
                    x = torch.randn(3, device='cuda', requires_grad=True)
                    y = torch.randn(3, device='cuda')
                    y_orig = y.clone()
                    loss = model(x, y).sum()
                    self.assertEqual(y, y_orig + 3)
                    loss.backward()


if __name__ == "__main__":
    run_tests()
