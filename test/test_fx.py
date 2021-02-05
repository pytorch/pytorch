import builtins
import contextlib
import copy
import functools
import math
import numbers
import operator
import os
import pickle
import sys
import torch
import unittest
from math import sqrt
from pathlib import Path
from torch.multiprocessing import Process
from torch.testing import FileCheck
from torch.fx import symbolic_trace, Proxy, Node, GraphModule, Interpreter, Tracer, Transformer, Graph, wrap
from torch.fx.node import Target
from torch.fx.passes import shape_prop
from torch.fx.immutable_collections import immutable_dict, immutable_list
from copy import deepcopy

from torch.fx.proxy import TraceError

from fx.quantization import Quantizer
from fx.test_subgraph_rewriter import TestSubgraphRewriter  # noqa: F401

from typing import Any, Callable, Dict, NamedTuple, List, Optional, Tuple, Union
from torch.testing._internal.common_utils import run_tests, TEST_WITH_ROCM, IS_WINDOWS, IS_SANDCASTLE, IS_MACOS
from torch.testing._internal.jit_utils import JitTestCase

from fx.named_tup import MyNamedTup

try:
    from torchvision.models import resnet18
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
skipIfNoTorchVision = unittest.skipIf(not HAS_TORCHVISION, "no torchvision")

class SimpleTest(torch.nn.Module):
    def forward(self, x):
        return torch.relu(x + 3.0)

def a_non_torch_leaf(a, b):
    return a + b

# Test wrap() passing both a function name as well as a function
# directly
def a_lifted_leaf(a, b):
    return a[0] + a[1] + b

wrap('a_lifted_leaf')
# Test wrapping twice doesn't break anything
wrap('a_lifted_leaf')

def a_lifted_leaf2(a, b):
    return a[0] + a[1] + b

wrap(a_lifted_leaf2)

wrap('len')

@wrap
def wrapped_via_decorator(a):
    return a + 1


real_wrapped_via_decorator = wrapped_via_decorator
real_a_lifed_leaf = a_lifted_leaf
real_a_lifed_leaf2 = a_lifted_leaf2
_sqrt = sqrt

wrap('wrapper_fn')

def wrapper_fn(x):
    return torch.foo(x)

class Pair(NamedTuple):
    x : torch.Tensor
    y : torch.Tensor

class TestFX(JitTestCase):
    def checkGraphModule(self, m: torch.nn.Module, args, kwargs=None):
        """Check that an nn.Module's results match the GraphModule version
        for a given set of args/kwargs.
        """
        kwargs = kwargs if kwargs else {}
        ref_outs = m(*args, **kwargs)
        gm = symbolic_trace(m)
        gm.graph.lint(gm)
        test_outs = gm(*args, **kwargs)
        self.assertEqual(ref_outs, test_outs)

    def test_graph_module(self):
        class MySub(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = torch.nn.Parameter(torch.rand(4, 3))

            def forward(self, x):
                return self.w + x

        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = torch.nn.Linear(4, 3)
                self.sub_mod = MySub()
                self.w = torch.nn.Parameter(torch.rand(3))

            def forward(self, A, B, c):
                t = torch.sigmoid(A) + self.lin(c)
                return self.sub_mod(t.data + self.w + t + 1 - A + B // A + -A + A.add(B, alpha=3))

        m = MyModule()
        gm = symbolic_trace(m)

        ms = torch.jit.script(gm)

        class M2(torch.nn.Module):
            def forward(self, A):
                m, idx = torch.max(A, 0)
                return m + 1, idx + 1

        m2 = M2()
        gm2 = symbolic_trace(m2)

        class T(torch.nn.Module):

            def forward(self, A, b=4, *args, c=5, **kwargs):
                x = A + 1 + args[0] + kwargs['3']
                return x

        t = T()
        symbolic_trace(t)

    def test_custom_import(self):
        graph = torch.fx.Graph()
        a = graph.placeholder('x')
        b = graph.placeholder('y')
        c = graph.call_function(a_non_torch_leaf, (a, b))
        d = graph.call_function(torch.sin, (c,))
        graph.output(d)
        gm = GraphModule(torch.nn.Module(), graph)
        x, y = torch.rand(1), torch.rand(1)
        self.assertEqual(torch.sin(x + y), gm(x, y))

    def test_args_kwargs(self):
        class T(torch.nn.Module):
            def forward(self, *args, **kwargs):
                x = args[0] + kwargs['foo']
                return x

        t = T()
        self.checkGraphModule(t, (torch.rand(1), torch.rand(1)), {'foo': torch.rand(1)})

    def test_args_kwargs_no_self(self):
        class T(torch.nn.Module):
            def forward(*args, **kwargs):  # noqa: B902
                self = args[0]
                return torch.relu(args[1])

        t = T()
        with self.assertRaisesRegex(RuntimeError, r'cannot be part of \*args expansion'):
            self.checkGraphModule(t, (torch.rand(1), torch.rand(1)), {'foo': torch.rand(1)})

    def test_fx_shifts(self):
        class MyModule(torch.nn.Module):
            def forward(self, x):
                return x << 3, x >> 3

        input = torch.LongTensor(10).random_(0, 1024)

        m = MyModule()
        self.checkGraphModule(m, (input,))

    def test_dict(self):
        class MyDictMod(torch.nn.Module):
            def forward(self, d):
                return d['3'].relu(), {'4' : d['3'].neg()}

        input_dict = {'3': torch.rand(3, 4)}
        m = MyDictMod()

        self.checkGraphModule(m, (input_dict,))

    def test_disallow_override(self):
        # Custom delegate to disallow in-place tensor operations
        class NoMutableCallTracer(Tracer):
            def create_node(self, kind : str, target : Union[str, Callable],
                            args : Tuple[Any], kwargs : Dict[str, Any], name : Optional[str] = None,
                            type_expr : Optional[Any] = None) -> Node:
                name = target if isinstance(target, str) else torch.typename(target)
                if name[-1] == '_':
                    raise RuntimeError('In-place operations are not supported')
                return super().create_node(kind, target, args, kwargs, name)

        # Test method
        class MyInplaceMod(torch.nn.Module):
            def forward(self, x):
                x.add_(3.0)
                return x

        m = MyInplaceMod()

        with self.assertRaisesRegex(RuntimeError, 'In-place operations'):
            NoMutableCallTracer().trace(m)

        # Test free function
        class MyInplaceMod2(torch.nn.Module):
            def forward(self, x):
                torch.log_(x)
                return x
        m2 = MyInplaceMod2()
        with self.assertRaisesRegex(RuntimeError, 'In-place operations'):
            NoMutableCallTracer().trace(m2)

        # Test symbolic node as an arg
        class MyInplaceMod3(torch.nn.Module):
            def forward(self, x):
                y = torch.ones(3, 4)
                y.add_(x)
                return x
        m3 = MyInplaceMod3()
        with self.assertRaisesRegex(RuntimeError, 'In-place operations'):
            NoMutableCallTracer().trace(m3)

    def test_leaf_module(self):
        # Custom delegate to make it so that there are no leaf modules, everything
        # should get traced through
        class NoLeafModulesTracer(Tracer):
            def is_leaf_module(self, m, qualname):
                return False

        class MyReluMod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                return self.relu(x)

        mrm = MyReluMod()
        sym = NoLeafModulesTracer().trace(mrm)
        for node in sym.nodes:
            self.assertNotEqual(node.op, 'call_module')
        sym.lint(sym)

    def test_wrap(self):
        self.assertEqual(3 + 4 + 5, a_lifted_leaf((3, 4), 5))

        def to_trace(y):
            return a_lifted_leaf((4, y), 3) + a_lifted_leaf((3, 4), 5) + a_lifted_leaf((y, y), y)

        m = symbolic_trace(to_trace)
        self.assertIn('a_lifted_leaf', m.code)
        self.assertEqual(27, m(2))
        self.assertIs(a_lifted_leaf, real_a_lifed_leaf)

    def test_wrap_fn_directly(self):
        self.assertEqual(3 + 4 + 5, a_lifted_leaf2((3, 4), 5))

        def to_trace(y):
            return a_lifted_leaf2((4, y), 3) + a_lifted_leaf2((3, 4), 5) + a_lifted_leaf2((y, y), y)

        m = symbolic_trace(to_trace)
        self.assertIn('a_lifted_leaf2', m.code)
        self.assertEqual(27, m(2))
        self.assertIs(a_lifted_leaf2, real_a_lifed_leaf2)

    def test_wrapped_via_decorator(self):
        self.assertEqual(wrapped_via_decorator(0), 1)

        def to_trace(y):
            return wrapped_via_decorator(y)

        m = symbolic_trace(to_trace)
        self.assertIn('wrapped_via_decorator', m.code)
        self.assertEqual(m(0), 1)
        self.assertIs(wrapped_via_decorator, real_wrapped_via_decorator)
        self.assertFalse(hasattr(wrapped_via_decorator, "__fx_already_patched"))

    def test_graph_edit_with_proxy(self):
        class M(torch.nn.Module):
            def forward(self, a, b):
                return a + b
        m = M()
        g = symbolic_trace(m).graph
        new_g = torch.fx.Graph()
        val_map : Dict[Node, Node] = {}
        output_val = new_g.graph_copy(g, val_map)
        t = Proxy(output_val)
        # test that we can use proxy objects to generate more graph code later for things that do not need to work with modules.
        new_g.output((t + t).node)
        gm = GraphModule(m, new_g)
        gm.graph.lint(gm)
        self.assertEqual(gm(3, 4), 14)

    def test_graph_unique_names(self):
        class M(torch.nn.Module):
            def forward(self, a, b):
                return a + b
        m = M()
        g = symbolic_trace(m).graph
        new_g = torch.fx.Graph()
        val_map : Dict[Node, Node] = {}
        output_val = new_g.graph_copy(g, val_map)
        t = Proxy(output_val)
        # test that we can use proxy objects to generate more graph code later for things that do not need to work with modules.
        new_g.output((t + t).node)
        gm = GraphModule(m, new_g)
        seen_names : Set[str] = set()
        for node in gm.graph.nodes:
            assert node.name not in seen_names
            seen_names.add(node.name)

    def test_graph_unique_names_manual(self):
        graph : torch.fx.Graph = torch.fx.Graph()
        a : torch.fx.Node = graph.create_node('placeholder', 'x')
        b : torch.fx.Node = graph.create_node('call_module', 'linear_mod', args=(a,), name='foo_1_1')
        c : torch.fx.Node = graph.create_node('get_attr', 'y_attr', name='foo_1')
        d : torch.fx.Node = graph.create_node('call_function', operator.add, args=(b, c))
        graph.output(d)
        graph2 = torch.fx.Graph()
        val_map : Dict[Node, Node] = {}
        graph2.graph_copy(graph, val_map)
        seen_names : Set[str] = set()
        for node in graph2.nodes:
            assert node.name not in seen_names
            seen_names.add(node.name)

    @skipIfNoTorchVision
    def test_resnet(self):
        resnet = resnet18()
        resnet.train()

        res_graph = symbolic_trace(resnet)
        res_script = torch.jit.script(res_graph)

        ip = torch.rand(1, 3, 224, 224)

        a = resnet(ip)
        b = res_graph(ip)
        c = res_script(ip)
        self.assertEqual(a, b)
        self.assertEqual(a, c)

        quantizer = Quantizer(res_graph)

        for i in range(10):
            quantizer.observe((torch.rand(1, 3, 224, 224),))

        qgraph = quantizer.quantize()
        qgraph.graph.lint(qgraph)
        qgraph_script = torch.jit.script(qgraph)

        d = qgraph(ip)
        e = qgraph_script(ip)

        assert (a - d).abs().max() < 2
        self.assertEqual(d, e)

    def test_unpack(self):
        class M(torch.nn.Module):
            def forward(self, a, b):
                c, d = a
                return c + d + b

        a = (torch.rand(1), torch.rand(1))
        b = torch.rand(1)
        m = M()
        self.checkGraphModule(m, (a, b))

    def test_native_callable(self):
        if TEST_WITH_ROCM or IS_SANDCASTLE or IS_WINDOWS or IS_MACOS:
            raise unittest.SkipTest("non-portable load_library call used in test")
        torch_root = Path(__file__).resolve().parent.parent
        p = torch_root / 'build' / 'lib' / 'libtorchbind_test.so'
        torch.ops.load_library(str(p))
        # This test exercises the case where we use FX to translate from Python
        # code to some native callable object
        #
        # For the purposes of testing, we use ElementwiseInterpreter defined
        # in test_custom_class.cpp.
        #
        # We test that we can
        # 1) Construct a native callable from FX IR
        # 2) Construct a drop-in replacement module that delegates to the
        #    native callable rather than the original code
        # 3) Run both the original code and native callable wrapper with
        #    equivalent results
        # 4) TorchScript compile the native callable wrapper and confirm
        #    equivalent results with the reference
        # 5) TorchScript serialize and deserialize the native callable
        #    and confirm equivalent results with the reference

        # We use this simple Module as a reference computation
        class MySimpleMod(torch.nn.Module):
            def forward(self, x):
                return 3.0 * x + x

        msm = MySimpleMod()

        # This is what a lowering pass might look like: a function that takes
        # a valid nn.Module, symbolically traces it, lowers the Module to some
        # representation, and wraps that representation up into another
        # nn.Module instance that handles dispatch to the compiled/lowered code.
        def lower_to_elementwise_interpreter(orig_mod : torch.nn.Module) -> torch.nn.Module:
            # ===== Stage 1: Symbolic trace the module =====
            mod = symbolic_trace(orig_mod)

            # ===== Stage 2: Lower GraphModule representation to the C++
            #       interpreter's instruction format ======
            instructions = []
            constant_idx = 0
            constants = {}
            fn_input_names = []

            target_to_name = {
                operator.add : "add",
                operator.mul : "mul"
            }

            output_node : Optional[Node] = None
            # For each instruction, create a triple
            # (instruction_name : str, inputs : List[str], output : str)
            # to feed into the C++ interpreter
            for n in mod.graph.nodes:
                target, args, out_name = n.target, n.args, n.name
                assert len(n.kwargs) == 0, "kwargs currently not supported"

                if n.op == 'placeholder':
                    # Placeholders specify function argument names. Save these
                    # for later when we generate the wrapper GraphModule
                    fn_input_names.append(target)
                elif n.op == 'call_function':
                    assert target in target_to_name, "Unsupported call target " + target
                    arg_names = []
                    for arg in args:
                        if not isinstance(arg, Node):
                            # Pull out constants. These constants will later be
                            # fed to the interpreter C++ object via add_constant()
                            arg_name = f'constant_{constant_idx}'
                            constants[arg_name] = torch.Tensor(
                                [arg] if isinstance(arg, numbers.Number) else arg)
                            arg_names.append(arg_name)
                            constant_idx += 1
                        else:
                            arg_names.append(arg.name)
                    instructions.append((target_to_name[target], arg_names, out_name))
                elif n.op == 'output':
                    if output_node is not None:
                        raise RuntimeError('Multiple output nodes!')
                    output_node = n
                else:
                    raise RuntimeError('Unsupported opcode ' + n.op)

            interpreter = torch.classes._TorchScriptTesting._ElementwiseInterpreter()
            # Load constants
            for k, v in constants.items():
                interpreter.add_constant(k, v)
            # Specify names for positional input arguments
            interpreter.set_input_names(fn_input_names)
            # Load instructions
            interpreter.set_instructions(instructions)
            # Specify name for single output
            assert isinstance(output_node.args[0], torch.fx.Node)
            interpreter.set_output_name(output_node.args[0].name)

            # ===== Stage 3: Create a wrapper GraphModule around the interpreter =====
            class WrapperModule(torch.nn.Module):
                def __init__(self, interpreter):
                    super().__init__()
                    self.interpreter = interpreter

            wrapper = WrapperModule(interpreter)

            # Create a graph that: 1) Takes function arguments 2) Invokes the interpreter
            # 3) Returns the speficied return value

            # FIXME: The following code could be greatly simplified by symbolic_trace'ing
            # the wrapper with a Tracer that considers the Wrapper instance a root
            # module, however, I can't get `__call__` exposed on TorchBind classes
            # without it messing up Python `hasattr` for some reason. More digging
            # into CPython's implementation of hasattr is probably in order...

            graph = torch.fx.Graph()
            # Add placeholders for fn inputs
            placeholder_nodes = []
            for name in fn_input_names:
                placeholder_nodes.append(graph.create_node('placeholder', name))

            # Get the interpreter object
            interpreter_node = graph.create_node('get_attr', 'interpreter')

            # Add a node to call the interpreter instance
            output_node = graph.create_node(
                op='call_method', target='__call__', args=(interpreter_node, placeholder_nodes))

            # Register output
            graph.output(output_node)

            graph.lint(wrapper)

            # Return final GraphModule!!!
            return GraphModule(wrapper, graph)


        # Lower GraphModule to C++ interpreter
        lowered = lower_to_elementwise_interpreter(msm)

        # Compare correctness with original module
        x = torch.rand(3, 4)
        ref_out = msm(x)
        test_out = lowered(x)
        torch.testing.assert_allclose(test_out, ref_out)

        # Test TorchScript compilation
        scripted_lowered = torch.jit.script(lowered)
        script_out = scripted_lowered(x)
        torch.testing.assert_allclose(script_out, ref_out)

        # Test TorchScript ser/de
        import_copy = self.getExportImportCopy(scripted_lowered)
        imported_out = import_copy(x)
        torch.testing.assert_allclose(imported_out, ref_out)

    def test_reserved_getattr(self):
        """Ensure that we do not name any nodes with a reserved builtin like `getattr`"""
        class M(torch.nn.Module):
            def forward(self, a):
                return a.foo.bar.baz

        m = M()
        m_g = symbolic_trace(m)
        m_g.graph.lint(m_g)
        for node in m_g.graph.nodes:
            self.assertTrue(node.name != "getattr")

    def test_node_tagging(self):
        class TaggingTracer(Tracer):
            def create_node(self, kind : str, target : Union[str, Callable],
                            args : Tuple[Any], kwargs : Dict[str, Any], name : Optional[str] = None,
                            type_expr : Optional[Any] = None) -> Node:
                n = super().create_node(kind, target, args, kwargs, name)
                n.tag = 'foo'
                return n

        class M(torch.nn.Module):
            def forward(self, a, b):
                return a + b

        m = M()
        g = TaggingTracer().trace(m)
        g.lint(m)
        for n in g.nodes:
            self.assertTrue(hasattr(n, 'tag'))
            self.assertEqual(n.tag, 'foo')

    def test_tensor_attribute(self):
        class TensorAttribute(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.tensor = torch.rand(3, 4)

            def forward(self, x):
                return torch.nn.functional.linear(x, self.tensor)

        ta = TensorAttribute()
        traced = symbolic_trace(ta)
        traced(torch.rand(4, 4))

        class WrapperForQualname(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.ta = TensorAttribute()

            def forward(self, x):
                return torch.nn.functional.linear(x, self.ta.tensor)

        wfq = WrapperForQualname()
        traced2 = symbolic_trace(wfq)
        traced2.graph.lint(traced2)
        traced2(torch.rand(4, 4))

    def test_symbolic_trace_sequential(self):
        class Simple(torch.nn.Module):
            def forward(self, x):
                return torch.neg(x)

        seq = torch.nn.Sequential(
            Simple(),
            Simple(),
            Simple()
        )
        traced = symbolic_trace(seq)
        traced.graph.lint(traced)
        x = torch.rand(3, 4)
        self.assertEqual(traced(x), seq(x))

    def test_tensor_constant(self):
        class ConstTensor(torch.nn.Module):
            def forward(self, x):
                return torch.nn.functional.linear(x, torch.zeros(3, 4))

        ct = ConstTensor()
        traced = symbolic_trace(ct)
        traced.graph.lint(traced)
        traced(torch.rand(4, 4))

    def test_pickle_graphmodule(self):
        class Nested(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.st = torch.nn.Linear(4, 4)

            def forward(self, x):
                return self.st(x)

        n = Nested()
        traced = symbolic_trace(n)
        traced.graph.lint(traced)
        pickled = pickle.dumps(traced)
        loaded = pickle.loads(pickled)
        loaded.graph.lint(loaded)
        x = torch.rand(3, 4)
        self.assertEqual(loaded(x), traced(x))

    def test_all_input_nodes(self):
        graph : torch.fx.Graph = torch.fx.Graph()
        a : torch.fx.Node = graph.placeholder('x')
        b : torch.fx.Node = graph.call_module('linear_mod', args=(a,))
        c : torch.fx.Node = graph.get_attr('y_attr')
        d : torch.fx.Node = graph.call_function(operator.add, args=(b, c))
        e : torch.fx.Node = graph.call_function(torch.unsqueeze, args=(d, 0))
        graph.output(e)
        graph.lint()

        self.assertEqual(b.all_input_nodes, [a])
        self.assertEqual(c.all_input_nodes, [])
        self.assertEqual(d.all_input_nodes, [b, c])
        self.assertEqual(e.all_input_nodes, [d])

    def test_deepcopy_graphmodule_with_transform(self):
        st = SimpleTest()
        traced = symbolic_trace(st)
        traced.graph.lint(traced)

        def transform(traced):
            new_graph = torch.fx.Graph()
            val_map : Dict[Node, Node] = {}
            output_value = new_graph.graph_copy(traced.graph, val_map)
            relu_out = new_graph.create_node(
                op='call_method', target='neg', args=(output_value,), kwargs={})
            new_graph.output(relu_out)
            return GraphModule(traced, new_graph)
        transformed = transform(traced)
        transformed.graph.lint(transformed)
        copied = copy.deepcopy(transformed)
        self.assertNotEqual(id(type(transformed)), id(type(copied)))
        x = torch.randn(3, 4)
        self.assertEqual(copied(x), transformed(x))

    def test_deepcopy_with_submods_params(self):
        class Bar(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.param = torch.nn.Parameter(torch.rand(3, 4))

            def forward(self, x):
                return torch.relu(x) + self.param

        class Baz(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.param = torch.nn.Parameter(torch.rand(3, 4))
                self.bar = Bar()

            def forward(self, x):
                return self.bar(x) - self.param

        baz = Baz()
        traced = symbolic_trace(baz)
        traced.graph.lint(traced)
        copied = copy.deepcopy(traced)
        copied.graph.lint(copied)

    def test_unpack_list_better_error(self):
        class SomeArgs(torch.nn.Module):
            def forward(self, a, b):
                return torch.rand(3, 4)

        class UnpacksList(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.sa = SomeArgs()

            def forward(self, x : list):
                return self.sa(*x)

        ul = UnpacksList()
        with self.assertRaisesRegex(TraceError, 'Proxy object cannot be iterated.'):
            symbolic_trace(ul)

    def test_unpack_dict_better_error(self):
        class SomeKwargs(torch.nn.Module):
            def forward(self, x=3, y=4):
                return torch.rand(3, 4)

        class UnpacksDict(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.sk = SomeKwargs()

            def forward(self, x : dict):
                return self.sk(**x)

        ud = UnpacksDict()
        with self.assertRaisesRegex(TraceError, 'Proxy object cannot be iterated.'):
            symbolic_trace(ud)

    def test_pretty_print_targets(self):
        # Test that Graph pretty-print prints friendly name for targets
        # in `operator` and `builtins`

        class SomeMod(torch.nn.Module):
            def forward(self, x):
                return torch.add(x.foo + x.bar, 3.0)

        traced = symbolic_trace(SomeMod())
        graph_str = str(traced.graph)
        self.assertIn('builtins.getattr', graph_str)
        self.assertIn('operator.add', graph_str)
        self.assertIn('torch.add', graph_str)

    def test_pretty_print_node(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.param: torch.nn.Parameter = torch.nn.Parameter(
                    torch.rand(3, 4))
                self.linear = torch.nn.Linear(4, 5)

            def forward(self, x: torch.Tensor, y: int = 2):
                return self.linear(x[y] + self.param).clamp(min=0.0, max=1.0)

        traced = symbolic_trace(M())

        all_formatted = "\n".join([n.format_node() for n in traced.graph.nodes])

        FileCheck().check("x").check("placeholder") \
            .check("y").check("placeholder") \
            .check("getitem").check("call_function") \
            .check("param").check("get_attr") \
            .check("add").check("call_function") \
            .check("linear").check("call_module") \
            .check("clamp").check("call_method") \
            .run(all_formatted)

    def test_script_tensor_constant(self):
        # TorchScript seems to ignore attributes that start with `__`.
        # We used to call anonymous Tensor values `__tensor_constant*`, but
        # they were getting ignored by script. Now they're called
        # `_tensor_constant*`
        class IHaveATensorConstant(torch.nn.Module):
            def forward(self, x):
                return x + torch.rand(3, 4)

        traced = torch.fx.symbolic_trace(IHaveATensorConstant())
        torch.jit.script(traced)

    def test_torch_fx_len(self):
        class FXLenTest(torch.nn.Module):
            def forward(self, x):
                return len(x)

        traced = symbolic_trace(FXLenTest())
        self.assertEqual(traced(torch.rand(3, 4)), 3)

        # Test scriptability
        scripted = torch.jit.script(FXLenTest())
        self.assertEqual(scripted(torch.rand(3)), 3)

        traced_scripted = torch.jit.script(traced)
        self.assertEqual(traced_scripted(torch.rand(3)), 3)

        # Test non-proxy len
        class FXLenTest2(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.l = [3, 4, 5]

            def forward(self, x):
                return x + len(self.l)

        traced2 = symbolic_trace(FXLenTest2())
        inp = torch.rand(3, 4)
        self.assertEqual(traced2(inp), inp + 3.0)
        self.assertIs(len, builtins.len)

    def test_sqrt(self):
        class Sqrt1(torch.nn.Module):
            def forward(self, x):
                return sqrt(x.size(0))

        class Sqrt2(torch.nn.Module):
            def forward(self, x):
                return math.sqrt(x.size(0))

        class Sqrt3(torch.nn.Module):
            def forward(self, x):
                return x + math.sqrt(2) + sqrt(2)

        self.checkGraphModule(Sqrt1(), [torch.zeros(8)])
        self.checkGraphModule(Sqrt2(), [torch.zeros(8)])
        self.checkGraphModule(Sqrt3(), [torch.zeros(8)])
        self.assertIs(sqrt, _sqrt)
        self.assertIs(math.sqrt, _sqrt)

    def test_torch_custom_ops(self):
        class M(torch.nn.Module):
            def forward(self, a):
                b = torch.ops.aten.sigmoid(a)
                c = torch.ops.aten.cat([a, b])
                return torch.ops.aten.cat((c, c))
        m = M()
        input = torch.randn(3)
        ref_out = m(input)
        gm = symbolic_trace(m)
        gm.graph.lint(gm)
        out = gm(input)
        self.assertEqual(out, ref_out)

    def test_pretty_print(self):
        st = SimpleTest()
        traced = symbolic_trace(st)
        traced.graph.lint(traced)
        printed = str(traced)
        assert 'SimpleTest()' in printed
        assert 'torch.relu' in printed

    def test_pretty_print_graph(self):
        class KwargPrintTest(torch.nn.Module):
            def forward(self, x):
                return torch.squeeze(x + 3.0, dim=2)
        st = KwargPrintTest()
        traced = symbolic_trace(st)
        traced.graph.lint(traced)
        stringed = str(traced.graph)
        for s in ['args', 'kwargs', '#users']:
            assert s in stringed

    def test_graph_fns(self):
        g = Graph()
        a = g.placeholder('a')
        b = g.call_module('linear', (a,))
        c = g.get_attr('bias')
        d = g.call_method('add', (b, c))
        e = g.call_function(torch.sin, (d,))
        g.output(e)
        mod = torch.nn.Module()
        mod.linear = torch.nn.Linear(3, 4)
        mod.bias = torch.rand(4)
        gm = GraphModule(mod, g)
        gm.graph.lint(gm)
        input = torch.rand(3)
        r = gm(input)
        ref = torch.sin(mod.linear(input) + mod.bias)
        self.assertEqual(r, ref)

    def test_remove_uses(self):
        g : torch.fx.Graph = Graph()
        x : torch.fx.Node = g.placeholder('x')
        relu : torch.fx.Node = g.call_function(torch.relu, (x,))
        neg : torch.fx.Node = g.call_function(torch.neg, (relu,))
        g.output(neg)

        neg.replace_all_uses_with(relu)
        g.erase_node(neg)

        self.assertTrue(neg not in relu.users)

    def test_nonetype_annotation(self):
        eb = torch.nn.EmbeddingBag(3, 4)
        symbolic_trace(eb)

    def test_construct_root_dict(self):
        graph : torch.fx.Graph = torch.fx.Graph()
        a : torch.fx.Node = graph.create_node('placeholder', 'x')
        b : torch.fx.Node = graph.create_node('call_module', 'foo.bar.baz', args=(a,))
        c : torch.fx.Node = graph.create_node('get_attr', 'zip.zap.zam')
        d : torch.fx.Node = graph.create_node('call_function', operator.add, args=(b, c))
        graph.output(d)

        linear_mod : torch.nn.Module = torch.nn.Linear(3, 4)
        add_param : torch.Tensor = torch.rand(3, 4)
        gm : torch.fx.GraphModule = torch.fx.GraphModule(
            {'foo.bar.baz': linear_mod, 'zip.zap.zam' : add_param}, graph)
        gm.graph.lint(gm)

        assert 'self.foo.bar.baz' in gm.code

        x : torch.Tensor = torch.rand(3, 3)
        out : torch.Tensor = gm(x)
        ref_out : torch.Tensor = linear_mod(x) + add_param
        self.assertEqual(out, ref_out)

    def test_symbolic_trace_assert(self):

        class AssertsTensorShape(torch.nn.Module):
            def forward(self, x):
                torch._assert(x.shape[1] > 4, "assert_foobar")
                return x

        m = AssertsTensorShape()
        # verify traceability
        traced = symbolic_trace(m)
        # verify assertion on traced model works correctly at runtime
        traced(torch.rand(4, 5))
        with self.assertRaisesRegex(AssertionError, "assert_foobar"):
            traced(torch.rand(4, 3))
        # verify the symbolically traced module is scriptable
        ms = torch.jit.script(m)
        with self.assertRaisesRegex(torch.jit.Error, "assert_foobar"):
            ms(torch.rand(4, 3))

    def test_trace_fn_constant(self):
        some_constant = torch.rand(3, 4)

        def add_const(x):
            return some_constant + x

        traced = symbolic_trace(add_const)

        input = torch.rand(3, 4)
        self.assertEqual(traced(input), add_const(input))

    def test_copy_no_remap(self):
        traced = symbolic_trace(SimpleTest())
        g = traced.graph
        copied = torch.fx.Graph()
        for node in g.nodes:
            copied.node_copy(node)
        with self.assertRaisesRegex(RuntimeError, 'does not belong to this Graph'):
            copied.lint()

    def test_wrong_topo(self):
        graph : torch.fx.Graph = torch.fx.Graph()
        a : torch.fx.Node = graph.create_node('placeholder', 'x')
        b : torch.fx.Node = graph.create_node('call_module', 'foo.bar.baz', args=(a,))
        c : torch.fx.Node = graph.create_node('get_attr', 'zip.zap.zam')
        d : torch.fx.Node = graph.create_node('call_function', operator.add, args=(b, c))
        graph.output(d)
        nodes = list(graph.nodes)
        nodes[3].append(nodes[2])
        with self.assertRaisesRegex(RuntimeError, 'was used before it has been defined'):
            graph.lint()

    def test_example_shape_prop(self):
        class TestCase(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attr = torch.randn(3, 4)
                self.submod = torch.nn.Linear(4, 4)

            def forward(self, x):
                return torch.neg(self.submod(x.relu() + self.attr))
        tc = TestCase()
        tc_traced = symbolic_trace(tc)
        ref_out = tc_traced(torch.rand(3, 4))
        shape_prop.ShapeProp(tc_traced).propagate(torch.rand(3, 4))

        # Make sure we're testing all opcodes
        opcodes = set()
        output_shape : Optional[torch.Shape] = None
        for node in tc_traced.graph.nodes:
            opcodes.add(node.op)
            if node.op == 'output':
                output_shape = node.args[0].shape
        self.assertEqual(opcodes, set(['placeholder', 'get_attr', 'call_function', 'call_method',
                                       'call_module', 'output']))

        # Test shape propogation and make sure results match actual
        self.assertEqual(output_shape, ref_out.shape)

    def test_interpreter(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.param = torch.nn.Parameter(torch.rand(3, 4))
                self.linear = torch.nn.Linear(4, 5)

            def forward(self, x):
                return self.linear(x + self.param).clamp(min=0.0, max=1.0)

        m = MyModule()
        gm = torch.fx.symbolic_trace(m)

        interpreter = Interpreter(gm)
        input = torch.randn(3, 4)
        self.assertEqual(interpreter.run(input), gm(input))
        self.assertEqual(interpreter.run(input), m(input))

    def test_interpreter_run_node_override(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.param = torch.nn.Parameter(torch.rand(3, 4))
                self.linear = torch.nn.Linear(4, 5)

            def forward(self, x):
                return self.linear(x + self.param).clamp(min=0.0, max=1.0)

        m = MyModule()
        gm = torch.fx.symbolic_trace(m)

        class RunNodeInterpreter(Interpreter):
            def __init__(self, module):
                super().__init__(module)

            def run_node(self, n : Node) -> Any:
                result = super().run_node(n)
                n.cached_value = result
                return result

        input = torch.randn(3, 4)
        RunNodeInterpreter(gm).run(input)
        for node in gm.graph.nodes:
            assert hasattr(node, 'cached_value')

    def test_interpreter_onthefly_swap(self):

        def fn(x):
            return torch.sigmoid(x).neg()

        gm = torch.fx.symbolic_trace(fn)

        class NegSigmSwapInterpreter(Interpreter):
            def call_function(self, target : Target, args : Tuple, kwargs : Dict) -> Any:
                if target == torch.sigmoid:
                    return torch.neg(*args, **kwargs)
                return super().call_function(n)

            def call_method(self, target : Target, args : Tuple, kwargs : Dict) -> Any:
                if target == 'neg':
                    call_self, *args_tail = args
                    return call_self.sigmoid(*args_tail, **kwargs)
                return super().call_method(n)

        input = torch.randn(3, 4)
        result = NegSigmSwapInterpreter(gm).run(input)
        self.assertEqual(result, torch.neg(input).sigmoid())

    def test_interpreter_partial_eval(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.param = torch.nn.Parameter(torch.rand(3, 4))
                self.linear = torch.nn.Linear(4, 5)

            def forward(self, x):
                return self.linear(x + self.param).clamp(min=0.0, max=1.0)

        gm = torch.fx.symbolic_trace(MyModule())
        interp = Interpreter(gm)
        env = {}
        for node in gm.graph.nodes:
            if node.op == 'call_module' and node.target == 'linear':
                env[node] = torch.arange(0, 12, 1).reshape(3, 4) - 6.0
                break
        assert len(env) == 1
        x = torch.randn(3, 4)
        result = interp.run(x, initial_env=env)
        self.assertEqual(result, (torch.arange(0, 12, 1).reshape(3, 4) - 6.0).clamp(0.0, 1.0))

    def test_interpreter_star_args(self):
        def with_star_args(x, *args):
            return x + args[0]

        gm = torch.fx.symbolic_trace(with_star_args)
        interp = Interpreter(gm)
        result = interp.run(torch.ones(3, 4), torch.ones(3, 4), torch.rand(3, 4))
        self.assertEqual(result, torch.ones(3, 4) * 2.0)

    def test_transformer_noop(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.param = torch.nn.Parameter(torch.rand(3, 4))
                self.linear = torch.nn.Linear(4, 5)

            def forward(self, x):
                return self.linear(x + self.param).clamp(min=0.0, max=1.0)

        m = MyModule()
        gm = torch.fx.symbolic_trace(m)

        new_gm = Transformer(gm).transform()

        input = torch.randn(3, 4)
        self.assertEqual(new_gm(input), gm(input))

    def test_transformer_op_swap(self):

        def fn(x):
            return torch.sigmoid(x).neg()

        gm = torch.fx.symbolic_trace(fn)

        class NegSigmSwapXformer(Transformer):
            def call_function(self, target : Target, args : Tuple, kwargs : Dict) -> Any:
                if target == torch.sigmoid:
                    return torch.neg(*args, **kwargs)
                return super().call_function(n)

            def call_method(self, target : Target, args : Tuple, kwargs : Dict) -> Any:
                if target == 'neg':
                    call_self, *args_tail = args
                    return call_self.sigmoid(*args_tail, **kwargs)
                return super().call_method(n)

        transformed = NegSigmSwapXformer(gm).transform()
        input = torch.randn(3, 4)
        self.assertEqual(transformed(input), torch.neg(input).sigmoid())

    def test_fn_type_annotations(self):
        class Foo(torch.nn.Module):
            def forward(self, p : Pair, z : torch.Tensor, i : int) -> Dict[str, torch.Tensor]:
                return {'a': p.x + p.y + z + i}

        foo_scripted = torch.jit.script(Foo())
        foo_scripted(Pair(torch.rand(5), torch.rand(5)), torch.rand(5), 3)

        fxed = symbolic_trace(Foo())
        fxed_scripted = torch.jit.script(fxed)
        fxed_scripted(Pair(torch.rand(5), torch.rand(5)), torch.rand(5), 3)

    def test_fn_type_annotation_empty(self):
        def forward(a : List[torch.Tensor]):
            return a[0]
        torch.jit.script(symbolic_trace(forward))

    def test_wrapped_method(self):
        def wrap_with_relu(fn):
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                return torch.relu(fn(*args, **kwargs))
            return wrapper

        class Foo(torch.nn.Module):
            @wrap_with_relu
            def forward(self, x, w):
                return torch.matmul(x, w)

        f = Foo()
        traced = symbolic_trace(f)
        x, w = torch.rand(3, 4), torch.rand(4, 4)
        self.assertTrue(any(n.target == torch.relu for n in traced.graph.nodes))

    def test_empty_graph_codegen(self):
        graph = torch.fx.Graph()
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)
        self.assertEqual(gm(), None)

    def test_sequential(self):
        m = torch.nn.Sequential(torch.nn.Conv2d(1, 1, 1))
        gm = torch.fx.symbolic_trace(m)
        gm_copy = copy.deepcopy(gm)

    def test_ctx_mgr(self):
        @contextlib.contextmanager
        def do_nothing():
            yield

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()

            @do_nothing()
            def forward(self, x):
                return torch.relu(x)

        m = M()
        self.checkGraphModule(m, (torch.rand(3, 4),))

    def test_typename_print(self):
        graph : torch.fx.Graph = torch.fx.Graph()
        x : torch.fx.Node = graph.create_node('placeholder', 'x')
        b : torch.fx.Node = graph.create_node('call_function', target=torch.relu, args=(x,),
                                              type_expr=List[float])
        output : torch.fx.Node = graph.output(b)
        self.assertTrue('typing.List[float]' in str(graph))

    def test_ellipsis(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                return x + y[:, 1:10, ...]

        traced = symbolic_trace(M())
        x, y = torch.rand(5, 9, 3, 4), torch.rand(5, 15, 3, 4)
        self.assertEqual(traced(x, y), x + y[:, 1:10, ...])

    def test_inf_nan(self):
        class FooMod(torch.nn.Module):
            def forward(self, x):
                return x + float('inf'), x + float('-inf'), x + float('nan')

        fm = FooMod()
        self.checkGraphModule(fm, (torch.rand(3, 4),))

    def test_inf_nan_kwds(self):
        graph : torch.fx.Graph = torch.fx.Graph()
        x : torch.fx.Node = graph.create_node('placeholder', 'x')
        b : torch.fx.Node = graph.create_node('call_function', operator.add, (x, float('inf')), {}, name='inf')
        c : torch.fx.Node = graph.create_node('call_function', operator.add, (x, float('nan')), {}, name='nan')
        graph.output((b, c))

        gm = torch.fx.GraphModule(torch.nn.Module(), graph)
        x = torch.rand(3, 4)
        self.assertEqual(gm(x), (x + float('inf'), x + float('nan')))

    def test_deepcopy_recursion_depth(self):
        depth = sys.getrecursionlimit() + 20

        g = torch.fx.Graph()
        x = g.placeholder('x')
        for i in range(depth):
            x = g.call_function(torch.relu, (x,))
        g.output(x)

        copied_graph = copy.deepcopy(g)

        val_map = {}
        for orig_node, new_node in zip(g.nodes, copied_graph.nodes):
            val_map[orig_node] = new_node

        for orig_node, new_node in zip(g.nodes, copied_graph.nodes):
            orig_users = set(orig_node.users.keys())
            orig_users_equiv = set(val_map[u] for u in orig_users)
            new_users = set(new_node.users.keys())
            self.assertEqual(orig_users_equiv, new_users)

    @skipIfNoTorchVision
    def test_replace_uses(self):
        rn18 = resnet18()

        class LowerReluTracer(torch.fx.Tracer):
            def is_leaf_module(self, m : torch.nn.Module, qualname : str):
                if isinstance(m, torch.nn.ReLU):
                    return False
                return super().is_leaf_module(m, qualname)

        rn18_traced = GraphModule(rn18, LowerReluTracer().trace(rn18))

        to_erase = []
        for node in rn18_traced.graph.nodes:
            if node.op == 'call_function' and node.target in [torch.relu, torch.nn.functional.relu]:
                kwargs = node.kwargs.copy()
                # Neg doesn't have in-place
                kwargs.pop('inplace')
                with rn18_traced.graph.inserting_before(node):
                    new_node = rn18_traced.graph.call_function(
                        the_function=torch.neg, args=node.args, kwargs=node.kwargs)
                node.replace_all_uses_with(replace_with=new_node)
                to_erase.append(node)

        for node in to_erase:
            rn18_traced.graph.erase_node(node)

    def test_insertion_point(self):
        graph : torch.fx.Graph = torch.fx.Graph()
        x : torch.fx.Node = graph.create_node('placeholder', 'x')
        b : torch.fx.Node = graph.create_node('call_function', target=torch.relu, args=(x,))
        output : torch.fx.Node = graph.output(b)

        with graph.inserting_before(b):
            neg : torch.fx.Node = graph.call_function(the_function=torch.neg, args=(x,))
            _, *relu_args = b.args
            b.args = (neg, *relu_args)

        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        input = torch.randn(33, 44)
        self.assertEqual(gm(input), torch.relu(torch.neg(input)))


    def test_move_before(self):
        graph : torch.fx.Graph = torch.fx.Graph()
        x : torch.fx.Node = graph.create_node('placeholder', 'x')
        b : torch.fx.Node = graph.create_node('call_function', target=torch.relu, args=(x,))
        output : torch.fx.Node = graph.output(b)

        neg : torch.fx.Node = graph.call_function(the_function=torch.neg, args=(x,))
        _, *relu_args = b.args
        b.args = (neg, *relu_args)
        b.prepend(neg)

        gm = torch.fx.GraphModule(torch.nn.Module(), graph)

        input = torch.randn(33, 44)
        self.assertEqual(gm(input), torch.relu(torch.neg(input)))

    def test_erase_node_error(self):
        st = SimpleTest()
        traced = symbolic_trace(st)

        for node in traced.graph.nodes:
            # Test deleting with uses both in another Node and at the output
            if node.target in [operator.add, torch.relu]:
                with self.assertRaisesRegex(RuntimeError, 'but it still had .* users in the graph'):
                    traced.graph.erase_node(node)

    def test_copy_it(self):
        d = immutable_dict([(3, 4), (5, 6)])
        l = immutable_list([(3, 4), (5, 6)])

        self.assertEqual(d, deepcopy(d))
        self.assertEqual(l, deepcopy(l))

    def test_find_uses(self):
        graph = torch.fx.Graph()
        x = torch.fx.Proxy(graph.placeholder('x'))

        y = torch.relu(x)
        z = x + x
        u = torch.neg(x)
        graph.output((y + z + u).node)
        graph.lint()

        users_of_x = x.node.users
        self.assertEqual(len(users_of_x), 3)
        expected_ops = set(['relu', 'add', 'neg'])
        for use in users_of_x:
            assert any(use.name.startswith(prefix) for prefix in expected_ops)

    def test_inline_graph(self):
        class InlineInto(torch.nn.Module):
            def forward(self, x):
                return torch.relu(x)

        class ToInline(torch.nn.Module):
            def forward(self, x):
                return torch.neg(x)

        inline_into = symbolic_trace(InlineInto())
        to_inline = symbolic_trace(ToInline())

        combined_graph = torch.fx.Graph()
        output_node = combined_graph.graph_copy(inline_into.graph, {})

        input_node = list(to_inline.graph.nodes)[0]
        assert input_node and input_node.op == 'placeholder'

        val_map = {input_node : output_node}
        output = combined_graph.graph_copy(to_inline.graph, val_map)
        combined_graph.output(output)

        combined_module = torch.fx.GraphModule(torch.nn.Module(), combined_graph)

        input = torch.rand(3, 4)
        self.assertEqual(combined_module(input), input.relu().neg())

    def test_multi_insert_point(self):
        graph = torch.fx.Graph()
        x = torch.fx.Proxy(graph.placeholder('x'))
        relu = torch.relu(x)

        with graph.inserting_before(relu.node):
            y = torch.neg(x)
            z = torch.tanh(y)

        graph.output((relu.node, z.node))
        graph.lint()

        expected_ops = ['x', 'neg', 'tanh', 'relu']
        for node, expected in zip(graph.nodes, expected_ops):
            assert expected in node.name

    def test_reassign_args_kwargs_uses(self):
        graph = torch.fx.Graph()
        x, y = Proxy(graph.placeholder('x')), Proxy(graph.placeholder('y'))
        z = x + y
        zed = z + z + z
        graph.output(zed.node)
        graph.lint()

        # zed = z + z + z -> zed = z + z + x
        zed.node.args = (zed.node.args[0], x.node)
        self.assertEqual(x.node.users.keys(), [z.node, zed.node])

        # z = x + y -> z = y + y
        z.node.args = (y.node, y.node)
        self.assertEqual(x.node.users.keys(), [zed.node])

    def test_trace_function(self):
        def foo(x, y):
            return torch.relu(x) + y

        x, y = torch.randn(3, 4), torch.randn(3, 4)
        self.checkGraphModule(foo, (x, y))

    def test_direct_param_use(self):
        class TransposeTest(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.b = torch.nn.Parameter(torch.rand(4, 3))

            def forward(self, x):
                return self.b

        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = TransposeTest()

            def forward(self, x):
                return self.a.b, self.a.b.t(), self.a.b.view(12)

        traced = torch.fx.symbolic_trace(Foo())
        assert(all('constant' not in node.target for node in traced.graph.nodes))

    def test_single_default_arg(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, y=1):
                return y

        m = M()
        self.checkGraphModule(m, ())
        self.checkGraphModule(m, (3,))

    def test_multiple_default_args(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, y=1, z=2):
                return y + z

        m = M()
        self.checkGraphModule(m, ())
        self.checkGraphModule(m, (3,))
        self.checkGraphModule(m, (3, 4))

    def test_regular_and_default_args(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y=1):
                return x + y

        m = M()
        self.checkGraphModule(m, (2,))
        self.checkGraphModule(m, (2, 3))

    def test_string_literal_return(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self):
                return "foo"

        m = M()
        self.checkGraphModule(m, ())

    def test_namedtuple_return_qualname(self):
        class NamedTupReturn(torch.nn.Module):
            def forward(self, x):
                return MyNamedTup(x, x)

        traced = symbolic_trace(NamedTupReturn())
        input = torch.rand(3, 4)
        self.assertEqual(traced(input), MyNamedTup(input, input))

    def test_update_args_kwargs_yells_at_you(self):
        symtraced = symbolic_trace(SimpleTest())
        node = next(iter(symtraced.graph.nodes))
        with self.assertRaisesRegex(AttributeError, '__update_args_kwargs'):
            node.__update_args_kwargs((), {})

    def test_torchbind_class_attribute_in_fx(self):
        if TEST_WITH_ROCM or IS_SANDCASTLE or IS_WINDOWS or IS_MACOS:
            self.skipTest("torch.classes._TorchScriptTesting._StackString is registered, skipping")

        class FooBar1234(torch.nn.Module):
            def __init__(self):
                super(FooBar1234, self).__init__()
                self.f = torch.classes._TorchScriptTesting._StackString(["3", "4"])

            def forward(self):
                return self.f.top()

        m = FooBar1234()
        self.checkGraphModule(m, ())

    def test_namedtuple_return_trace(self):
        class NamedTupReturn(torch.nn.Module):
            def forward(self, x):
                return Pair(x, x)

        traced = symbolic_trace(NamedTupReturn())
        input = torch.rand(3, 4)
        self.assertEqual(traced(input), Pair(input, input))

    def test_return_type_exists(self):
        class ReturnTypeModule(torch.nn.Module):
            def other(self, x: List[str]) -> List[str]:
                return x

            def forward(self, x: List[str]) -> List[str]:
                return self.other(x)

        traced = symbolic_trace(ReturnTypeModule())
        self.assertIn("-> typing.List[str]", traced._code)
        scripted = torch.jit.script(traced)
        self.assertIn("-> List[str]", scripted.code)

    def getitem_inner(self):
        class GetItemBase(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer('pe', torch.randn(8, 8))

        class GetItem1(GetItemBase):
            def forward(self, x):
                return self.pe[:, :x.size(0)]

        class GetItem2(GetItemBase):
            def forward(self, x):
                return self.pe[x.size(0)]

        class GetItem3(GetItemBase):
            def forward(self, x):
                return self.pe[4]  # fx creates `self._tensor_constant0` here

        self.checkGraphModule(GetItem1(), [torch.zeros(4)])
        self.checkGraphModule(GetItem2(), [torch.zeros(4)])
        self.checkGraphModule(GetItem3(), [torch.zeros(4)])

    @unittest.skipUnless(os.environ.get("FX_PATCH_GETITEM") == "1",
                         "Will be checked in test_getitem_subproc")
    def test_getitem(self):
        self.getitem_inner()

    def test_getitem_subproc(self):
        # need to run this test in a subproc to work around:
        #   https://github.com/pytorch/pytorch/issues/50710
        proc = Process(target=run_getitem_target)
        proc.start()
        proc.join()
        self.assertEqual(proc.exitcode, 0)


    def test_user_friendly_call_provenance_with_function(self):
        def fn(x):
            return wrapper_fn(x)

        traced = torch.fx.symbolic_trace(fn)

        with self.assertRaisesRegex(RuntimeError, "'wrapper_fn' is "
                                    "being compiled since it was called"
                                    " from 'fn.forward'"):
            scripted = torch.jit.script(traced)

    def test_user_friendly_call_provenance_with_module(self):
        class M(torch.nn.Module):
            def forward(self, x):
                return wrapper_fn(x)

        traced = torch.fx.symbolic_trace(M())

        with self.assertRaisesRegex(RuntimeError, "'wrapper_fn' is "
                                    "being compiled since it was called"
                                    " from 'M.forward'"):
            scripted = torch.jit.script(traced)

    def test_snake_case(self):
        class M(torch.nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.activations = torch.nn.ModuleDict([
                    ["snake_case", torch.nn.ReLU()],
                    ["PascalCase", torch.nn.LeakyReLU()],
                    ["ALL_CAPS", torch.nn.PReLU()]
                ])

            def forward(self, x):
                a = self.activations["snake_case"](x)
                b = self.activations["PascalCase"](x)
                c = self.activations["ALL_CAPS"](x)
                return a, b, c

        traced = symbolic_trace(M())

        check = [
            ("activations_snake_case", "activations.snake_case"),
            ("activations_pascal_case", "activations.PascalCase"),
            ("activations_all_caps", "activations.ALL_CAPS")
        ]

        i = 0
        for node in traced.graph.nodes:
            if node.op == "placeholder" or node.op == "output":
                continue
            name = check[i][0]
            target = check[i][1]
            self.assertEqual(name, node.name)
            self.assertEqual(target, node.target)
            i += 1
        self.assertEqual(i, 3)

    def test_no_mutation(self):
        from torch.fx.immutable_collections import immutable_list
        x = immutable_list([3, 4])
        with self.assertRaisesRegex(NotImplementedError, "new_args"):
            x[0] = 4


def run_getitem_target():
    from torch.fx.symbolic_trace import _wrapped_methods_to_patch
    _wrapped_methods_to_patch.append((torch.Tensor, "__getitem__"))
    try:
        TestFX().getitem_inner()
    finally:
        _wrapped_methods_to_patch.pop()


if __name__ == '__main__':
    run_tests()
