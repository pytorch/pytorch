import torch
import unittest
import operator
import numbers
import pickle
import copy
from torch.fx import symbolic_trace, Proxy, Node, GraphModule, DefaultDelegate
from torch.fx.proxy import TraceError

from fx.quantization import Quantizer

from typing import Any, Callable, Dict, Optional, Tuple, Union
from torch.testing._internal.common_utils import run_tests, skipIfRocm
from torch.testing._internal.jit_utils import JitTestCase

try:
    from torchvision.models import resnet18
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
skipIfNoTorchVision = unittest.skipIf(not HAS_TORCHVISION, "no torchvision")

class SimpleTest(torch.nn.Module):
    def forward(self, x):
        return torch.relu(x + 3.0)

class TestFX(JitTestCase):
    def checkGraphModule(self, m: torch.nn.Module, args, kwargs=None):
        """Check that an nn.Module's results match the GraphModule version
        for a given set of args/kwargs.
        """
        kwargs = kwargs if kwargs else {}
        ref_outs = m(*args, **kwargs)
        gm = symbolic_trace(m)
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

    def test_args_kwargs(self):
        class T(torch.nn.Module):
            def forward(self, *args, **kwargs):
                x = args[0] + kwargs['foo']
                return x

        t = T()
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
        class NoMutableCallDelegate(DefaultDelegate):
            def create_node(self, kind : str, target : Union[str, Callable],
                            args : Tuple[Any], kwargs : Dict[str, Any], name : Optional[str] = None) -> Node:
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
            symbolic_trace(m, delegate_class=NoMutableCallDelegate)

        # Test free function
        class MyInplaceMod2(torch.nn.Module):
            def forward(self, x):
                torch.log_(x)
                return x
        m2 = MyInplaceMod2()
        with self.assertRaisesRegex(RuntimeError, 'In-place operations'):
            symbolic_trace(m2, delegate_class=NoMutableCallDelegate)

        # Test symbolic node as an arg
        class MyInplaceMod3(torch.nn.Module):
            def forward(self, x):
                y = torch.ones(3, 4)
                y.add_(x)
                return x
        m3 = MyInplaceMod3()
        with self.assertRaisesRegex(RuntimeError, 'In-place operations'):
            symbolic_trace(m3, delegate_class=NoMutableCallDelegate)

    def test_leaf_module(self):
        # Custom delegate to make it so that there are no leaf modules, everything
        # should get traced through
        class NoLeafModulesDelegate(DefaultDelegate):
            def is_leaf_module(self, m):
                return False

        class MyReluMod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                return self.relu(x)

        mrm = MyReluMod()
        sym = symbolic_trace(mrm, delegate_class=NoLeafModulesDelegate)
        for node in sym.graph.nodes:
            self.assertNotEqual(node.op, 'call_module')

    def test_graph_edit_with_proxy(self):
        class M(torch.nn.Module):
            def forward(self, a, b):
                return a + b
        m = M()
        g = symbolic_trace(m).graph
        t = Proxy(g.result)
        # test that we can use proxy objects to generate more graph code later for things that do not need to work with modules.
        g.output((t + t).node)
        gm = GraphModule(m, g)
        self.assertEqual(gm(3, 4), 14)

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
        assert torch.allclose(a, b)
        assert torch.allclose(a, c)

        quantizer = Quantizer(res_graph)

        for i in range(10):
            quantizer.observe((torch.rand(1, 3, 224, 224),))

        qgraph = quantizer.quantize()
        qgraph_script = torch.jit.script(qgraph)

        d = qgraph(ip)
        e = qgraph_script(ip)

        assert (a - d).abs().max() < 2
        assert torch.allclose(d, e)

    def test_unpack(self):
        class M(torch.nn.Module):
            def forward(self, a, b):
                c, d = a
                return c + d + b

        a = (torch.rand(1), torch.rand(1))
        b = torch.rand(1)
        m = M()
        self.checkGraphModule(m, (a, b))

    @skipIfRocm
    def test_native_callable(self):
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

                else:
                    raise RuntimeError('Unsupported opcode' + n.op)

            interpreter = torch.classes._TorchScriptTesting._ElementwiseInterpreter()
            # Load constants
            for k, v in constants.items():
                interpreter.add_constant(k, v)
            # Specify names for positional input arguments
            interpreter.set_input_names(fn_input_names)
            # Load instructions
            interpreter.set_instructions(instructions)
            # Specify name for single output
            interpreter.set_output_name(mod.graph.result.name)

            # ===== Stage 3: Create a wrapper GraphModule around the interpreter =====
            class WrapperModule(torch.nn.Module):
                def __init__(self, interpreter):
                    super().__init__()
                    self.interpreter = interpreter

            wrapper = WrapperModule(interpreter)

            # Create a graph that: 1) Takes function arguments 2) Invokes the interpreter
            # 3) Returns the speficied return value

            # FIXME: The following code could be greatly simplified by symbolic_trace'ing
            # the wrapper with a Delegate that considers the Wrapper instance a root
            # module, however, I can't get `__call__` exposed on TorchBind classes
            # without it messing up Python `hasattr` for some reason. More digging
            # into CPython's implementation of hasattr is probably in order...

            graph = torch.fx.Graph()
            # Add placeholders for fn inputs
            placeholder_nodes = []
            for name in fn_input_names:
                placeholder_nodes.append(graph.create_node('placeholder', name))

            # Get the interpreter object
            interpreter_node = graph.create_node('get_param', 'interpreter')

            # Add a node to call the interpreter instance
            output_node = graph.create_node(
                op='call_method', target='__call__', args=(interpreter_node, placeholder_nodes))

            # Register output
            graph.output(output_node)

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
        for node in m_g.graph.nodes:
            self.assertTrue(node.name != "getattr")

    def test_node_tagging(self):
        class TaggingDelegate(DefaultDelegate):
            def create_node(self, kind : str, target : Union[str, Callable],
                            args : Tuple[Any], kwargs : Dict[str, Any], name : Optional[str] = None) -> Node:
                n = super().create_node(kind, target, args, kwargs, name)
                n.tag = 'foo'
                return n

        class M(torch.nn.Module):
            def forward(self, a, b):
                return a + b

        m = M()
        g = symbolic_trace(m, TaggingDelegate).graph
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
        traced2(torch.rand(4, 4))

    def test_tensor_constant(self):
        class ConstTensor(torch.nn.Module):
            def forward(self, x):
                return torch.nn.functional.linear(x, torch.zeros(3, 4))

        ct = ConstTensor()
        traced = symbolic_trace(ct)
        traced(torch.rand(4, 4))

    def test_pickle_graphmodule(self):
        st = SimpleTest()
        traced = symbolic_trace(st)
        pickled = pickle.dumps(traced)
        loaded = pickle.loads(pickled)
        x = torch.rand(3, 4)
        self.assertEqual(loaded(x), traced(x))

    def test_deepcopy_graphmodule_with_transform(self):
        st = SimpleTest()
        traced = symbolic_trace(st)

        def transform(traced):
            new_graph = copy.deepcopy(traced.graph)
            delegate = torch.fx.DefaultDelegate(traced.root, new_graph)
            relu_out = delegate.create_node(
                kind='call_method', target='neg', args=(new_graph.result,), kwargs={})
            new_graph.output(relu_out)
            return GraphModule(traced.root, new_graph)
        transformed = transform(traced)
        copied = copy.deepcopy(transformed)
        x = torch.randn(3, 4)
        self.assertEqual(copied(x), transformed(x))

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
        with self.assertRaisesRegex(TraceError, 'Proxy object cannot be unpacked as function argument'):
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
        with self.assertRaisesRegex(TraceError, 'Proxy object cannot be unpacked as function argument'):
            symbolic_trace(ud)

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
        out = gm(input)
        self.assertEqual(out, ref_out)


if __name__ == '__main__':
    run_tests()
