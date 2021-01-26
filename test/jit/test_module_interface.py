# flake8: noqa
# TODO: enable linting check for this file

from typing import List, Any
import torch
import torch.nn as nn
import os
import sys
from torch import Tensor
from torch.testing._internal.jit_utils import JitTestCase, make_global

# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.jit_utils import JitTestCase, execWrapper

if __name__ == '__main__':
    raise RuntimeError("This test file is not meant to be run directly, use:\n\n"
                       "\tpython test/test_jit.py TESTNAME\n\n"
                       "instead.")

class OrigModule(nn.Module):
    def __init__(self):
        super(OrigModule, self).__init__()

    def one(self, inp1: Tensor, inp2: Tensor) -> Tensor:
        return inp1 + inp2 + 1

    def two(self, input: Tensor) -> Tensor:
        return input + 2

    def forward(self, input: Tensor) -> Tensor:
        return input + self.one(input, input) + 1

class NewModule(nn.Module):
    def __init__(self):
        super(NewModule, self).__init__()

    def one(self, inp1: Tensor, inp2: Tensor) -> Tensor:
        return inp1 * inp2 + 1

    def forward(self, input: Tensor) -> Tensor:
        return self.one(input, input + 1)

class TestModuleInterface(JitTestCase):
    def test_not_submodule_interface_call(self):
        @torch.jit.interface
        class ModuleInterface(nn.Module):
            def one(self, inp1: Tensor, inp2: Tensor) -> Tensor:
                pass

        class TestNotModuleInterfaceCall(nn.Module):
            proxy_mod : ModuleInterface

            def __init__(self):
                super(TestNotModuleInterfaceCall, self).__init__()
                self.proxy_mod = OrigModule()

            def forward(self, input: Tensor) -> Tensor:
                return self.proxy_mod.two(input)

        with self.assertRaisesRegex(RuntimeError, "Tried to access nonexistent attribute"):
            torch.jit.script(TestNotModuleInterfaceCall())

    def test_module_interface(self):
        @torch.jit.interface
        class OneTwoModule(nn.Module):
            def one(self, x: Tensor, y: Tensor) -> Tensor:
                pass

            def two(self, x: Tensor) -> Tensor:
                pass

            def forward(self, x: Tensor) -> Tensor:
                pass

        @torch.jit.interface
        class OneTwoClass(object):
            def one(self, x: Tensor, y: Tensor) -> Tensor:
                pass

            def two(self, x: Tensor) -> Tensor:
                pass

        class FooMod(nn.Module):
            def one(self, x: Tensor, y: Tensor) -> Tensor:
                return x + y

            def two(self, x: Tensor) -> Tensor:
                return 2 * x

            def forward(self, x: Tensor) -> Tensor:
                return self.one(self.two(x), x)

        class BarMod(nn.Module):
            def one(self, x: Tensor, y: Tensor) -> Tensor:
                return x * y

            def two(self, x: Tensor) -> Tensor:
                return 2 / x

            def forward(self, x: Tensor) -> Tensor:
                return self.two(self.one(x, x))

            @torch.jit.export
            def forward2(self, x: Tensor) -> Tensor:
                return self.two(self.one(x, x)) + 1

        make_global(OneTwoModule, OneTwoClass)
        def use_module_interface(mod_list: List[OneTwoModule], x: torch.Tensor):
            return mod_list[0].forward(x) + mod_list[1].forward(x)

        def use_class_interface(mod_list: List[OneTwoClass], x: Tensor) -> Tensor:
            return mod_list[0].two(x) + mod_list[1].one(x, x)

        scripted_foo_mod = torch.jit.script(FooMod())
        scripted_bar_mod = torch.jit.script(BarMod())
        self.checkScript(use_module_interface,
                         ([scripted_foo_mod, scripted_bar_mod], torch.rand(3, 4),))
        self.checkScript(use_class_interface,
                         ([scripted_foo_mod, scripted_bar_mod], torch.rand(3, 4),))

        def call_module_interface_on_other_method(mod_interface: OneTwoModule, x: Tensor) -> Tensor:
            return mod_interface.forward2(x)

        # ensure error out when we call the module on the method other than the interface specified.
        with self.assertRaisesRegex(RuntimeError, "Tried to access nonexistent attribute or method"):
            self.checkScript(call_module_interface_on_other_method, (scripted_bar_mod, torch.rand(3, 4),))

    def test_module_doc_string(self):
        @torch.jit.interface
        class TestInterface(nn.Module):
            def one(self, inp1, inp2):
                # type: (Tensor, Tensor) -> Tensor
                pass
            def forward(self, input):
                # type: (Tensor) -> Tensor
                r"""stuff 1"""
                r"""stuff 2"""
                pass
                r"""stuff 3"""

        class TestModule(nn.Module):
            proxy_mod : TestInterface

            def __init__(self):
                super(TestModule, self).__init__()
                self.proxy_mod = OrigModule()

            def forward(self, input):
                # type: (Tensor) -> Tensor
                return self.proxy_mod.forward(input)

        input = torch.randn(3, 4)
        self.checkModule(TestModule(), (input,))

    def test_module_interface_subtype(self):
        @torch.jit.interface
        class OneTwoModule(nn.Module):
            def one(self, x: Tensor, y: Tensor) -> Tensor:
                pass

            def two(self, x: Tensor) -> Tensor:
                pass

            def forward(self, x: Tensor) -> Tensor:
                pass

        make_global(OneTwoModule)
        @torch.jit.script
        def as_module_interface(x: OneTwoModule) -> OneTwoModule:
            return x

        @torch.jit.script
        class Foo(object):
            def one(self, x: Tensor, y: Tensor) -> Tensor:
                return x + y

            def two(self, x: Tensor) -> Tensor:
                return 2 * x

            def forward(self, x: Tensor) -> Tensor:
                return self.one(self.two(x), x)

        # check class object is not a subtype of module interface
        with self.assertRaisesRegex(RuntimeError, "ScriptModule class can be subtype of module interface"):
            as_module_interface(Foo())

        class WrongMod(nn.Module):
            def two(self, x: int) -> int:
                return 2 * x

            def forward(self, x: Tensor) -> Tensor:
                return x + torch.randn(3, self.two(3))

        scripted_wrong_mod = torch.jit.script(WrongMod())

        # wrong module that is not compatible with module interface
        with self.assertRaisesRegex(RuntimeError, "is not compatible with interface"):
            as_module_interface(scripted_wrong_mod)

        # Check that interface implementations can be contravariant in argument types and covariant in return type.
        @torch.jit.interface
        class TensorToAny(nn.Module):
            def forward(self, input: torch.Tensor) -> Any:
                pass

        make_global(TensorToAny)
        @torch.jit.script
        def as_tensor_to_any(x: TensorToAny) -> TensorToAny:
            return x

        @torch.jit.interface
        class AnyToAny(nn.Module):
            def forward(self, input: Any) -> Any:
                pass

        make_global(AnyToAny)
        @torch.jit.script
        def as_any_to_any(x: AnyToAny) -> AnyToAny:
            return x

        class TensorToAnyImplA(nn.Module):
            def forward(self, input: Any) -> Any:
                return input

        class TensorToAnyImplB(nn.Module):
            def forward(self, input: Any) -> torch.Tensor:
                return torch.tensor([1])

        class AnyToAnyImpl(nn.Module):
            def forward(self, input: Any) -> torch.Tensor:
                return torch.tensor([1])

        as_tensor_to_any(torch.jit.script(TensorToAnyImplA()))
        as_tensor_to_any(torch.jit.script(TensorToAnyImplB()))
        as_any_to_any(torch.jit.script(AnyToAnyImpl()))


    def test_module_interface_inheritance(self):
        with self.assertRaisesRegex(RuntimeError, "does not support inheritance yet. Please directly"):
            @torch.jit.interface
            class InheritMod(nn.ReLU):
                def three(self, x: Tensor) -> Tensor:
                    return 3 * x

    def test_module_swap(self):
        @torch.jit.interface
        class ModuleInterface(nn.Module):
            def one(self, inp1: Tensor, inp2: Tensor) -> Tensor:
                pass

            def forward(self, input: Tensor) -> Tensor:
                pass

        class TestModule(nn.Module):
            proxy_mod : ModuleInterface

            def __init__(self):
                super(TestModule, self).__init__()
                self.proxy_mod = OrigModule()

            def forward(self, input: Tensor) -> Tensor:
                return self.proxy_mod.forward(input)

        scripted_mod = torch.jit.script(TestModule())
        input = torch.randn(3, 4)
        self.assertEqual(scripted_mod(input), 3 * input + 2)

        # module swap with module that have the same interface
        scripted_mod.proxy_mod = torch.jit.script(NewModule())
        self.assertEqual(scripted_mod(input), input * (input + 1) + 1)

        # module swap with non-scripted module should throw error
        with self.assertRaisesRegex(RuntimeError, "a ScriptModule with non-scripted module"):
            scripted_mod.proxy_mod = NewModule()

    def test_module_swap_wrong_module(self):
        @torch.jit.interface
        class ModuleInterface(nn.Module):
            def one(self, inp1: Tensor, inp2: Tensor) -> Tensor:
                pass

            def forward(self, input: Tensor) -> Tensor:
                pass

        class NewModuleWrong(nn.Module):
            def __init__(self):
                super(NewModuleWrong, self).__init__()

            def forward(self, input: int) -> int:
                return input + 1

        class TestModule(nn.Module):
            proxy_mod : ModuleInterface

            def __init__(self):
                super(TestModule, self).__init__()
                self.proxy_mod = OrigModule()

            def forward(self, input: Tensor) -> Tensor:
                return self.proxy_mod.forward(input)

        scripted_mod = torch.jit.script(TestModule())
        # module swap with in-compatible interface
        with self.assertRaisesRegex(RuntimeError, "is not compatible with interface"):
            scripted_mod.proxy_mod = torch.jit.script(NewModuleWrong())

    def test_module_swap_no_lazy_compile(self):
        @torch.jit.interface
        class ModuleInterface(nn.Module):
            def one(self, inp1: Tensor, inp2: Tensor) -> Tensor:
                pass

            def forward(self, input: Tensor) -> Tensor:
                pass

        class TestModule(nn.Module):
            proxy_mod : ModuleInterface

            def __init__(self):
                super(TestModule, self).__init__()
                self.proxy_mod = OrigModule()

            def forward(self, input: Tensor) -> Tensor:
                return self.proxy_mod.forward(input)

        class NewModuleMethodNotLazyCompile(nn.Module):
            def __init__(self):
                super(NewModuleMethodNotLazyCompile, self).__init__()

            def one(self, inp1: Tensor, inp2: Tensor) -> Tensor:
                return inp1 * inp2 + 1

            def forward(self, input: Tensor) -> Tensor:
                return input + 1

        scripted_mod = torch.jit.script(TestModule())
        # module swap with module that have the same interface, but the method not get
        # lazily compiled from forward, user need to export it explicitly for swap to work
        with self.assertRaisesRegex(RuntimeError, "is not compatible with interface"):
            scripted_mod.proxy_mod = torch.jit.script(NewModuleMethodNotLazyCompile())

        class NewModuleMethodManualExport(nn.Module):
            def __init__(self):
                super(NewModuleMethodManualExport, self).__init__()

            @torch.jit.export
            def one(self, inp1: Tensor, inp2: Tensor) -> Tensor:
                return inp1 * inp2 + 1

            def forward(self, input: Tensor) -> Tensor:
                return input + 1

        scripted_mod.proxy_mod = torch.jit.script(NewModuleMethodManualExport())
        input = torch.randn(3, 4)
        self.assertEqual(scripted_mod(input), input + 1)

    def test_module_swap_no_module_interface(self):
        # test module swapping with no module interface
        class TestNoModuleInterface(nn.Module):
            def __init__(self):
                super(TestNoModuleInterface, self).__init__()
                self.proxy_mod = OrigModule()

            def forward(self, input: Tensor) -> Tensor:
                return self.proxy_mod(input)

        scripted_no_module_interface = torch.jit.script(TestNoModuleInterface())
        # proxy mod is swapped with the new ScriptModule that share the same JIT type, should succeed.
        scripted_no_module_interface.proxy_mod = torch.jit.script(OrigModule())
        # proxy_mod is neither a module interface or have the same JIT type, should fail
        with self.assertRaisesRegex(RuntimeError,
                                    "Expected a value of type '__torch__.jit.test_module_interface.OrigModule \(.*\)' " +
                                    "for field 'proxy_mod', but found '__torch__.jit.test_module_interface.NewModule \(.*\)'"):
            scripted_no_module_interface.proxy_mod = torch.jit.script(NewModule())

    def test_script_module_as_interface_swap(self):
        @torch.jit.interface
        class ModuleInterface(nn.Module):
            def one(self, inp1: Tensor, inp2: Tensor) -> Tensor:
                pass

            def forward(self, input: Tensor) -> Tensor:
                pass

        class OrigScriptModule(torch.jit.ScriptModule):
            def __init__(self):
                super(OrigScriptModule, self).__init__()

            @torch.jit.script_method
            def one(self, inp1: Tensor, inp2: Tensor) -> Tensor:
                return inp1 + inp2 + 1

            @torch.jit.script_method
            def forward(self, input: Tensor) -> Tensor:
                return input + self.one(input, input) + 1

        class NewScriptModule(torch.jit.ScriptModule):
            def __init__(self):
                super(NewScriptModule, self).__init__()

            @torch.jit.script_method
            def one(self, inp1: Tensor, inp2: Tensor) -> Tensor:
                return inp1 * inp2 + 1

            @torch.jit.script_method
            def forward(self, input: Tensor) -> Tensor:
                return self.one(input, input + 1)

        class TestNNModuleWithScriptModule(nn.Module):
            proxy_mod : ModuleInterface

            def __init__(self):
                super(TestNNModuleWithScriptModule, self).__init__()
                self.proxy_mod = OrigScriptModule()

            def forward(self, input: Tensor) -> Tensor:
                return self.proxy_mod.forward(input)

        input = torch.randn(3, 4)
        scripted_mod = torch.jit.script(TestNNModuleWithScriptModule())
        self.assertEqual(scripted_mod(input), 3 * input + 2)

        scripted_mod.proxy_mod = NewScriptModule()
        self.assertEqual(scripted_mod(input), input * (input + 1) + 1)

    # The call to forward of proxy_mod cannot be inlined. Making sure
    # Freezing is throwing an error for now.
    def test_freeze_module_with_interface(self):
        class SubModule(torch.nn.Module):
            def __init__(self):
                super(SubModule, self).__init__()
                self.b = 20

            def forward(self, x):
                return self.b

        class OrigMod(torch.nn.Module):
            def __init__(self):
                super(OrigMod, self).__init__()
                self.a = 0

            def forward(self, x):
                return self.a

        @torch.jit.interface
        class ModInterface(torch.nn.Module):
            def forward(self, x: Tensor) -> int:
                pass

        class TestModule(torch.nn.Module):
            proxy_mod : ModInterface

            def __init__(self):
                super(TestModule, self).__init__()
                self.proxy_mod = OrigMod()
                self.sub = SubModule()  # folded

            def forward(self, x):
                return self.proxy_mod(x) + self.sub(x)

        m = torch.jit.script(TestModule())
        m.eval()
        mf = torch._C._freeze_module(m._c)
        # Assume interface has no aliasing
        mf = torch._C._freeze_module(m._c, freezeInterfaces = True)
        input = torch.tensor([1])
        out_s = m.forward(input)
        out_f = mf.forward(input)
        self.assertEqual(out_s, out_f)

    def test_freeze_module_with_setattr_in_interface(self):
        class SubModule(torch.nn.Module):
            def __init__(self):
                super(SubModule, self).__init__()
                self.b = 20

            def forward(self, x):
                self.b += 2;
                return self.b
            @torch.jit.export
            def getb(self, x):
                return self.b

        class OrigMod(torch.nn.Module):
            def __init__(self):
                super(OrigMod, self).__init__()
                self.a = 0

            def forward(self, x):
                return self.a

        @torch.jit.interface
        class ModInterface(torch.nn.Module):
            def forward(self, x: Tensor) -> int:
                pass

        class TestModule(torch.nn.Module):
            proxy_mod : ModInterface

            def __init__(self):
                super(TestModule, self).__init__()
                self.proxy_mod = OrigMod()
                self.sub = SubModule()

            def forward(self, x):
                return self.proxy_mod(x) + self.sub.getb(x)

        m = torch.jit.script(TestModule())
        m.proxy_mod = m.sub
        m.eval()
        with self.assertRaisesRegex(RuntimeError, "failed to freeze interface attribute 'proxy_mod'"):
            mf = torch._C._freeze_module(m._c, freezeInterfaces = True)

    def test_freeze_module_with_inplace_mutation_in_interface(self):
        class SubModule(torch.nn.Module):
            def __init__(self):
                super(SubModule, self).__init__()
                self.b = torch.tensor([1.5])

            def forward(self, x):
                self.b[0] += 2;
                return self.b
            @torch.jit.export
            def getb(self, x):
                return self.b

        class OrigMod(torch.nn.Module):
            def __init__(self):
                super(OrigMod, self).__init__()
                self.a = torch.tensor([0.5])

            def forward(self, x):
                return self.a

        @torch.jit.interface
        class ModInterface(torch.nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                pass

        class TestModule(torch.nn.Module):
            proxy_mod : ModInterface

            def __init__(self):
                super(TestModule, self).__init__()
                self.proxy_mod = OrigMod()
                self.sub = SubModule()

            def forward(self, x):
                y = self.proxy_mod(x);
                z= self.sub.getb(x)
                return y[0] + z[0]

        m = torch.jit.script(TestModule())
        m.proxy_mod = m.sub
        m.sub.b = m.proxy_mod.b
        m.eval()
        with self.assertRaisesRegex(RuntimeError, "failed to freeze interface attribute 'proxy_mod'"):
            mf = torch._C._freeze_module(m._c, freezeInterfaces = True)

    def test_freeze_module_with_mutated_interface(self):
        class SubModule(torch.nn.Module):
            def __init__(self):
                super(SubModule, self).__init__()
                self.b = torch.tensor([1.5])

            def forward(self, x):
                return self.b
            @torch.jit.export
            def getb(self, x):
                return self.b

        class OrigMod(torch.nn.Module):
            def __init__(self):
                super(OrigMod, self).__init__()
                self.a = torch.tensor([0.5])

            def forward(self, x):
                return self.a

        @torch.jit.interface
        class ModInterface(torch.nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                pass

        class TestModule(torch.nn.Module):
            proxy_mod : ModInterface

            def __init__(self):
                super(TestModule, self).__init__()
                self.proxy_mod = OrigMod()
                self.sub = SubModule()

            def forward(self, x):
                self.proxy_mod = self.sub
                y = self.proxy_mod(x);
                z= self.sub.getb(x)
                return y[0] + z[0]

        m = torch.jit.script(TestModule())
        m.eval()
        with self.assertRaisesRegex(RuntimeError, "failed to freeze interface attribute 'proxy_mod'"):
            mf = torch._C._freeze_module(m._c, freezeInterfaces = True)

    def test_freeze_module_with_interface_and_fork(self):
        class SubModule(torch.nn.Module):
            def __init__(self):
                super(SubModule, self).__init__()
                self.b = torch.tensor([1.5])

            def forward(self, x):
                self.b[0] += 3.2
                return self.b

        class OrigMod(torch.nn.Module):
            def __init__(self):
                super(OrigMod, self).__init__()
                self.a = torch.tensor([0.5])

            def forward(self, x):
                return self.a

        @torch.jit.interface
        class ModInterface(torch.nn.Module):
            def forward(self, x: Tensor) -> Tensor:
                pass

        class TestModule(torch.nn.Module):
            proxy_mod : ModInterface

            def __init__(self):
                super(TestModule, self).__init__()
                self.proxy_mod = OrigMod()
                self.sub = SubModule()

            def forward(self, x):
                y = self.proxy_mod(x);
                z= self.sub(x)
                return y + z

        class MainModule(torch.nn.Module):
            def __init__(self):
                super(MainModule, self).__init__()
                self.test= TestModule();

            def forward(self, x):
                fut = torch.jit._fork(self.test.forward, x)
                y = self.test(x)
                z = torch.jit._wait(fut)
                return y + z

        m = torch.jit.script(MainModule())
        m.eval()
        mf = torch._C._freeze_module(m._c, freezeInterfaces = True)

    def test_module_apis_interface(self):
        @torch.jit.interface
        class ModuleInterface(nn.Module):
            def one(self, inp1: Tensor, inp2: Tensor) -> Tensor:
                pass

        class TestModule(nn.Module):
            proxy_mod : ModuleInterface

            def __init__(self):
                super(TestModule, self).__init__()
                self.proxy_mod = OrigModule()

            def forward(self, input):
                return input * 2

            @torch.jit.export
            def method(self, input):
                for module in self.modules():
                    input = module(input)
                return input

        with self.assertRaisesRegex(Exception, "Could not compile"):
            scripted_mod = torch.jit.script(TestModule())
