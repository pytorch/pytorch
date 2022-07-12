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
    PythonKeyTracer,
)
from torch.utils._python_dispatch import enable_torch_dispatch_mode
from torch.testing._internal.common_utils import (
    TestCase,
    run_tests,
)
import unittest

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


def find_input_mutations(g):
    FK = 'fake_result'
    inputs = defaultdict(set)
    input_idx = 0
    mutated_inputs = set()
    for n in g.nodes:
        if n.op == 'placeholder':
            inputs[StorageWeakRef(n.meta[FK].storage())].add(input_idx)
            input_idx += 1
        elif n.op == 'call_function':
            schema = n.target._schema
            for i, arg in enumerate(schema.arguments):
                if i < len(n.args):
                    argument = n.args[i]
                else:
                    if arg.name not in n.kwargs:
                        continue
                    argument = n.kwargs[arg.name]
                mut_arg = False
                if arg.alias_info:
                    if arg.alias_info.is_write:
                        mut_arg = True
                if mut_arg:
                    # TODO: not correct for args that contain tensors in a struct
                    # like list
                    mutated_inputs |= inputs[StorageWeakRef(argument.meta[FK].storage())]
        # TODO: error on unrecognized nodes
    return mutated_inputs


class FindInputMutations(torch.fx.Interpreter):
    def __init__(self, gm, **kwargs):
        super().__init__(gm, **kwargs)
        self.inputs = defaultdict(set)
        self.input_idx = 0
        self.mutated_inputs = set()

    def placeholder(self, target, args, kwargs):
        r = super().placeholder(target, args, kwargs)
        self.inputs[StorageWeakRef(r.storage())].add(self.input_idx)
        self.input_idx += 1
        return r

    def call_function(self, target, args, kwargs):
        r = super().call_function(target, args, kwargs)
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
                # TODO: not correct for args that contain tensors in a struct
                # like list
                mutated_inputs |= self.inputs[StorageWeakRef(r.storage())]
        # TODO: error on unrecognized nodes
        return r

    def __call__(self, *args):
        self.run(*args)
        return self.mutated_inputs


class ApplyCudaGraphs(torch.fx.Interpreter):
    def call_module(self, target, args, kwargs):
        assert not kwargs
        submod = self.module.get_submodule(target)
        self.module.delete_submodule(target)
        mutated_inputs = FindInputMutations(submod)(*args)
        self.module.add_submodule(target, CudaGraphModule(submod, mutated_inputs))
        r = super().call_module(target, args, kwargs)
        return r

    def run(self, *args):
        with FakeTensorMode.push() as mode:
            return super().run(*map(mode.from_tensor, args))


# Mutates input graph
def apply_cuda_graphs(gm):
    for n in gm.graph.nodes:
        if n.op == 'call_module':
            assert not n.kwargs
            submod = gm.get_submodule(n.target)
            gm.delete_submodule(n.target)
            mutated_inputs = find_input_mutations(submod.graph)
            gm.add_submodule(n.target, CudaGraphModule(submod, mutated_inputs))
    # NB: we didn't actually change the graph, no need for recompile


def cudagraphs(model, inputs):
    model = partition_cudagraphs(model, inputs)
    #apply_cuda_graphs(model)
    ApplyCudaGraphs(model).run(*inputs)
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


def composed(*decs):
    def deco(f):
        for dec in reversed(decs):
            f = dec(f)
        return f
    return deco


patch_all = composed(
    patch("torchdynamo.config.verify_correctness", True),
    patch("functorch._src.config.use_functionalize", True),
)


N_ITERS = 5


class TestDynamoCudaGraphs(TestCase):
    @patch_all
    def test_basic(self):
        def model(x, y):
            return (x + y) * y

        with torchdynamo.optimize(aot_autograd_cudagraphs):
            for i in range(N_ITERS):
                x = torch.randn(3, device="cuda", requires_grad=True)
                y = torch.randn(3, device="cuda")
                loss = model(x, y).sum()
                loss.backward()

    @patch_all
    def test_dtoh(self):
        def model(x, y):
            a = x + y
            b = a.cpu() * 3
            return b

        with torchdynamo.optimize(aot_autograd_cudagraphs):
            for i in range(N_ITERS):
                x = torch.randn(3, device="cuda", requires_grad=True)
                y = torch.randn(3, device="cuda")
                loss = model(x, y).sum()
                loss.backward()

    @patch_all
    def test_htod(self):
        def model(x, y):
            a = x + y
            return a * 3

        with torchdynamo.optimize(aot_autograd_cudagraphs):
            for i in range(N_ITERS):
                x = torch.randn(3, device="cuda", requires_grad=True)
                y = torch.randn((), device="cpu")
                loss = model(x, y).sum()
                loss.backward()

    @unittest.expectedFailure
    @patch_all
    def test_mutate_input(self):
        def model(x, y):
            y.add_(3)
            return x * y

        with torchdynamo.optimize(aot_autograd_cudagraphs):
            for i in range(N_ITERS):
                with self.subTest(i):
                    x = torch.randn(3, device="cuda", requires_grad=True)
                    y = torch.randn(3, device="cuda")
                    y_orig = y.clone()
                    loss = model(x, y).sum()
                    self.assertEqual(y, y_orig + 3)
                    loss.backward()

    @patch_all
    def test_mutate_constant(self):
        def model(x, y):
            c = torch.tensor(1)
            c.add_(2)
            return x * y * 0 + c

        with torchdynamo.optimize(aot_autograd_cudagraphs):
            for i in range(N_ITERS):
                with self.subTest(i):
                    x = torch.randn(1, device="cuda", requires_grad=True)
                    y = torch.randn(1, device="cuda")
                    loss = model(x, y).sum()
                    self.assertEqual(loss, torch.tensor(3.0, device="cuda"))
                    #loss.backward()

    @patch_all
    def test_factory(self):
        def model(y):
            x = torch.zeros(3, device="cuda:0")
            x.add_(3)
            return x * y

        with torchdynamo.optimize(aot_autograd_cudagraphs):
            for i in range(N_ITERS):
                with self.subTest(i):
                    y = torch.randn(3, device="cuda:0", requires_grad=True)
                    loss = model(y).sum()
                    loss.backward()

    """
    @patch_all
    def test_mutated_metadata(self):
        def model(x):
            x = x.clone()
            y = x.view(0)
            x.resize_(20)
            x.fill_(2)
            y.fill_(3)
            return x, y

        with torchdynamo.optimize(aot_autograd_cudagraphs):
            for i in range(1):
                with self.subTest(i):
                    x = torch.empty(0, device="cuda:0")
                    rx, ry = model(x)
                    self.assertEqual(rx, torch.full((20,), 2., device="cuda:0"))
                    self.assertEqual(ry, torch.empty(0, device="cuda:0"))
    """

    @patch_all
    def test_dead_fill(self):
        def model(x):
            x = x.clone()
            y = x[0:0]
            x.fill_(2)
            y.fill_(3)
            return x, y

        with torchdynamo.optimize(aot_autograd_cudagraphs):
            for i in range(N_ITERS):
                with self.subTest(i):
                    x = torch.empty(20, device="cuda:0")
                    rx, ry = model(x)
                    self.assertEqual(rx, torch.full((20,), 2., device="cuda:0"))
                    self.assertEqual(ry, torch.empty(0, device="cuda:0"))


if __name__ == "__main__":
    run_tests()
