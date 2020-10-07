import os
import sys

from typing import List
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.testing._internal.jit_utils import JitTestCase

# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)

if __name__ == '__main__':
    raise RuntimeError("This test file is not meant to be run directly, use:\n\n"
                       "\tpython test/test_jit.py TESTNAME\n\n"
                       "instead.")

class TestModuleContainers(JitTestCase):
    def test_sequential_intermediary_types(self):
        class A(torch.nn.Module):
            def __init__(self):
                super(A, self).__init__()

            def forward(self, x):
                return x + 3

        class B(torch.nn.Module):
            def __init__(self):
                super(B, self).__init__()

            def forward(self, x):
                return {"1": x}

        class C(torch.nn.Module):
            def __init__(self):
                super(C, self).__init__()
                self.foo = torch.nn.Sequential(A(), B())

            def forward(self, x):
                return self.foo(x)

        self.checkModule(C(), (torch.tensor(1),))

    def test_moduledict(self):
        class Inner(torch.nn.Module):
            def forward(self, x):
                return x + 10

        class Inner2(torch.nn.Module):
            def forward(self, x):
                return x * 2

        class Inner3(torch.nn.Module):
            def forward(self, x):
                return (x - 4) * 3

        class M(torch.nn.Module):
            def __init__(self):
                super(M, self).__init__()
                modules = OrderedDict([
                    ('one', Inner()),
                    ('two', Inner2()),
                    ('three', Inner3()),
                ])
                self.moduledict = nn.ModuleDict(modules)

            def forward(self, x, skip_name):
                # type: (Tensor, str)
                names = torch.jit.annotate(List[str], [])
                values = []
                for name in self.moduledict:
                    names.append(name)

                for name, mod in self.moduledict.items():
                    if name != skip_name:
                        names.append(name)
                        x = mod(x)
                        values.append(x)

                for mod in self.moduledict.values():
                    x = mod(x)
                    values.append(x)

                for key in self.moduledict.keys():
                    names.append(key)

                return x, names

        class M2(M):
            def __init__(self):
                super(M2, self).__init__()

            def forward(self, x, skip_name):
                # type: (Tensor, str)
                names = torch.jit.annotate(List[str], [])
                values = []
                x2 = x
                iter = 0
                for name in self.moduledict:
                    names.append(name)

                for i, (name, mod) in enumerate(self.moduledict.items()):
                    iter += i
                    if name != skip_name:
                        names.append(name)
                        x = mod(x)
                        values.append(x)

                for i, mod in enumerate(self.moduledict.values()):
                    iter += i
                    x = mod(x)
                    values.append(x)

                for i, key in enumerate(self.moduledict.keys()):
                    iter += i
                    names.append(key)

                for mod, mod in zip(self.moduledict.values(), self.moduledict.values()):
                    iter += i
                    x2 = mod(mod(x2))

                return x, x2, names, iter


        for name in ["", "one", "two", "three"]:
            inp = torch.tensor(1)
            self.checkModule(M(), (inp, name))
            self.checkModule(M2(), (inp, name))

    def test_custom_container_forward(self):
        class Inner(torch.nn.Module):
            def forward(self, x):
                return x + 10

        class CustomSequential(nn.Sequential):
            def __init__(self):
                super(CustomSequential, self).__init__(
                    nn.ReLU(), Inner())

            def forward(self, x):
                x = x + 3
                for mod in self:
                    x = mod(x)
                return x - 5

        self.checkModule(CustomSequential(), (torch.tensor(.5),))

        class CustomModuleList(nn.ModuleList):
            def __init__(self):
                super(CustomModuleList, self).__init__(
                    [nn.ReLU(), Inner()])

            def forward(self, x):
                x = x + 3
                for mod in self:
                    x = mod(x)
                return x - 5

        self.checkModule(CustomModuleList(), (torch.tensor(.5),))

        class CustomModuleDict(nn.ModuleDict):
            def __init__(self):
                super(CustomModuleDict, self).__init__(
                    OrderedDict([
                        ('one', Inner()),
                        ('two', nn.ReLU()),
                        ('three', Inner()),
                    ]))

            def forward(self, x):
                x = x + 3
                names = torch.jit.annotate(List[str], [])
                for name, mod in self.items():
                    x = mod(x)
                    names.append(name)
                return names, x - 5

        self.checkModule(CustomModuleDict(), (torch.tensor(.5),))

    def test_script_module_list_sequential(self):
        class M(torch.jit.ScriptModule):
            def __init__(self, mod_list):
                super(M, self).__init__()
                self.mods = mod_list

            @torch.jit.script_method
            def forward(self, v):
                for m in self.mods:
                    v = m(v)
                return v

        with torch.jit.optimized_execution(False):
            m = M(nn.Sequential(nn.ReLU()))
            self.assertExportImportModule(m, (torch.randn(2, 2),))

    def test_script_modulelist_index(self):
        class Sub(torch.nn.Module):
            def __init__(self, i):
                super(Sub, self).__init__()
                self.i = i

            def forward(self, thing):
                return thing - self.i

        class M(torch.nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.mods = nn.ModuleList([Sub(i) for i in range(10)])

            def forward(self, v):
                v = self.mods[4].forward(v)
                v = self.mods[-1].forward(v)
                v = self.mods[-9].forward(v)
                return v

        x = torch.tensor(1)
        self.checkModule(M(), (x,))

        class MForward(torch.nn.Module):
            def __init__(self):
                super(MForward, self).__init__()
                self.mods = nn.ModuleList([Sub(i) for i in range(10)])

            def forward(self, v):
                v = self.mods[4](v)
                v = self.mods[-1](v)
                v = self.mods[-9](v)
                return v

        self.checkModule(MForward(), (torch.tensor(1),))

        class M2(M):
            def __init__(self):
                super(M2, self).__init__()

            def forward(self, v):
                return self.mods[-11].forward(v)

        with self.assertRaisesRegex(Exception, "Index -11 out of range"):
            torch.jit.script(M2())


        class M2(M):
            def __init__(self):
                super(M2, self).__init__()

            def forward(self, v):
                return self.mods[-11].forward(v)

        with self.assertRaisesRegex(Exception, "Index -11 out of range"):
            torch.jit.script(M2())

        class M3(M):
            def __init__(self):
                super(M3, self).__init__()

            def forward(self, v):
                i = 3
                return self.mods[i].forward(v)

        with self.assertRaisesRegex(Exception, "Enumeration is supported"):
            torch.jit.script(M3())

    def test_module_interface_special_methods(self):
        class CustomModuleInterface(torch.nn.Module):
            def __init__(self):
                super(CustomModuleInterface, self).__init__()

        class CustomModuleList(CustomModuleInterface, torch.nn.ModuleList):
            def __init__(self, modules=None):
                CustomModuleInterface.__init__(self)
                torch.nn.ModuleList.__init__(self, modules)

        class CustomSequential(CustomModuleInterface, torch.nn.Sequential):
            def __init__(self, modules=None):
                CustomModuleInterface.__init__(self)
                torch.nn.Sequential.__init__(self, modules)

        class CustomModuleDict(CustomModuleInterface, torch.nn.ModuleDict):
            def __init__(self, modules=None):
                CustomModuleInterface.__init__(self)
                torch.nn.ModuleDict.__init__(self, modules)

        class MyModule(torch.nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()
                # work around aliasing issue for 'is' operator by scripting ReLU up front
                self.submod = torch.jit.script(torch.nn.ReLU())
                self.modulelist = CustomModuleList([self.submod])
                self.sequential = CustomSequential(self.submod)
                self.moduledict = CustomModuleDict({"submod": self.submod})

            def forward(self, inputs):
                assert self.modulelist[0] is self.submod, "__getitem__ failing for ModuleList"
                assert len(self.modulelist) == 1, "__len__ failing for ModuleList"
                for module in self.modulelist:
                    assert module is self.submod, "__iter__ failing for ModuleList"

                assert self.sequential[0] is self.submod, "__getitem__ failing for Sequential"
                assert len(self.sequential) == 1, "__len__ failing for Sequential"
                for module in self.sequential:
                    assert module is self.submod, "__iter__ failing for Sequential"

                assert self.moduledict["submod"] is self.submod, "__getitem__ failing for ModuleDict"
                assert len(self.moduledict) == 1, "__len__ failing for ModuleDict"

                # note: unable to index moduledict with a string variable currently
                i = 0
                for key in self.moduledict:
                    i += 1
                assert i == len(self.moduledict), "iteration failing for ModuleDict"

                assert "submod" in self.moduledict, "__contains__ fails for ModuleDict"

                for key in self.moduledict.keys():
                    assert key == "submod", "keys() fails for ModuleDict"

                for item in self.moduledict.items():
                    assert item[0] == "submod", "items() fails for ModuleDict"
                    assert item[1] is self.submod, "items() fails for ModuleDict"

                for value in self.moduledict.values():
                    assert value is self.submod, "values() fails for ModuleDict"

                return inputs

        m = MyModule()
        self.checkModule(m, [torch.randn(2, 2)])

    def test_special_method_with_override(self):
        class CustomModuleInterface(torch.nn.Module):
            def __init__(self):
                super(CustomModuleInterface, self).__init__()

        class CustomModuleList(CustomModuleInterface, torch.nn.ModuleList):
            def __init__(self, modules=None):
                CustomModuleInterface.__init__(self)
                torch.nn.ModuleList.__init__(self, modules)

            def __len__(self):
                # this is arbitrary, just to check that the overridden py __len__ from
                # CustomModuleList takes precedence over the automatically generated
                # __len__ added by the jit compiler
                return 2

        class MyModule(torch.nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()
                # work around aliasing issue for 'is' operator by scripting ReLU up front
                self.submod = torch.jit.script(torch.nn.ReLU())
                self.modulelist = CustomModuleList([self.submod])

            def forward(self, inputs):
                assert len(self.modulelist) == 2, "__len__ failing for ModuleList"
                return inputs

        m = MyModule()
        self.checkModule(m, [torch.randn(2, 2)])
        mm = torch.jit.script(m)

    def test_moduledict_getitem(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()
                self.relu = torch.jit.script(torch.nn.ReLU())
                self.tanh = torch.jit.script(torch.nn.Tanh())
                self.moduledict = torch.nn.ModuleDict({"relu": self.relu,
                                                       "tanh": self.tanh})

            def forward(self, input):
                assert self.moduledict['relu'] is self.relu
                assert self.moduledict['tanh'] is self.tanh
                return input

        m = MyModule()
        self.checkModule(m, [torch.randn(2, 2)])

    def test_moduledict_keyerror(self):
        class BadModule(torch.nn.Module):
            def __init__(self):
                super(BadModule, self).__init__()
                self.moduledict = torch.nn.ModuleDict({"foo": None,
                                                       "bar": None})

            def forward(self, input):
                assert self.moduledict['blah'] == "blah", "this is a keyerror"

        with self.assertRaisesRegex(RuntimeError, "Key Error, blah"):
            b = BadModule()
            torch.jit.script(b)

        class AnotherBadModule(torch.nn.Module):
            def __init__(self):
                super(AnotherBadModule, self).__init__()
                self.moduledict = torch.nn.ModuleDict({"foo": None,
                                                       "bar": None})

            def forward(self, input):
                idx = 'blah'
                assert self.moduledict[idx] == "blah", "this is a string literal error"

        with self.assertRaisesRegex(RuntimeError, "Unable to extract string literal index. "
                                                  "ModuleDict indexing is only supported with string literals."):
            b = AnotherBadModule()
            torch.jit.script(b)

    def test_empty_dict_override_contains(self):
        class CustomModuleInterface(torch.nn.Module):
            def __init__(self):
                super(CustomModuleInterface, self).__init__()

        class CustomModuleDict(CustomModuleInterface, torch.nn.ModuleDict):
            def __init__(self, modules=None):
                CustomModuleInterface.__init__(self)
                torch.nn.ModuleDict.__init__(self, modules)

        class MyModule(torch.nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()
                # work around aliasing issue for 'is' operator by scripting ReLU up front
                self.submod = torch.jit.script(torch.nn.ReLU())
                self.moduledict = CustomModuleDict()

            def forward(self, inputs):
                assert "submod" not in self.moduledict, "__contains__ fails for ModuleDict"
                return inputs

        m = MyModule()
        self.checkModule(m, [torch.randn(2, 2)])
