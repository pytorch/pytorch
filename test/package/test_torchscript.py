from unittest import skipIf

import torch
from torch.package import PackageExporter, PackageImporter
from torch.testing._internal.common_utils import run_tests

try:
    from torchvision.models import resnet18

    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
skipIfNoTorchVision = skipIf(not HAS_TORCHVISION, "no torchvision")


try:
    from .common import PackageTestCase
except ImportError:
    # Support the case where we run this file directly.
    from common import PackageTestCase  # type: ignore

from pathlib import Path

packaging_directory = Path(__file__).parent


class PackagingTsSerTest(PackageTestCase):
    """Torchscript saving and loading in torch.Package tests."""

    @skipIfNoTorchVision
    def test_save_ts(self):
        # Test basic saving of TS module
        class ModB(torch.nn.Module):
            def __init__(self, name: str):
                super().__init__()
                self.name = name
                self.tvmod = resnet18()

            def forward(self, input: str):
                input = input + "_modB:" + self.name
                return input

        class ModA(torch.nn.Module):
            def __init__(self, name: str, submodule_name: str):
                super().__init__()
                self.name = name
                self.modB = ModB(submodule_name)
                self.tvmod = resnet18()

            def forward(self, input: str):
                input = input + "_modA:" + self.name
                return self.modB(input)

        scripted_mod = torch.jit.script(ModA("a", "b"))

        filename = self.temp()
        with PackageExporter(filename, verbose=False) as e:
            e.save_pickle("res", "mod.pkl", scripted_mod)

        importer = PackageImporter(filename)
        loaded_mod = importer.load_pickle("res", "mod.pkl")
        self.assertEqual(loaded_mod("input"), scripted_mod("input"))

    @skipIfNoTorchVision
    def test_save_ts_modules_shared_code(self):
        #  Test to verify saving multiple modules with same top module
        #  but different submodules works
        class ModD(torch.nn.Module):
            def __init__(self, name: str):
                super().__init__()
                self.name = name
                self.tvmod = resnet18()

            def forward(self, input: str):
                input = input + "_modB:" + self.name
                return input

        class ModC(torch.nn.Module):
            def __init__(self, name: str, submodule_name: str):
                super().__init__()
                self.name = name
                self.modB = ModD(submodule_name)
                self.tvmod = resnet18()

            def forward(self, input: str):
                input = input + "_modA:" + self.name
                return self.modB(input)

        scripted_mod_0 = torch.jit.script(ModC("a", "b"))

        # redefinition is intentional
        class ModD(torch.nn.Module):  # noqa: F811
            def __init__(self, name: str):
                super().__init__()
                self.name = name
                self.tvmod = resnet18()

            def forward(self, input: str):
                input = input + "_modB(changed):" + self.name
                return input

        scripted_mod_1 = torch.jit.script(ModC("a", "b"))

        filename = self.temp()
        with PackageExporter(filename, verbose=False) as e:
            e.save_pickle("res", "mod1.pkl", scripted_mod_0)
            e.save_pickle("res", "mod2.pkl", scripted_mod_1)

        importer = PackageImporter(filename)
        loaded_mod_0 = importer.load_pickle("res", "mod1.pkl")
        loaded_mod_1 = importer.load_pickle("res", "mod2.pkl")
        self.assertEqual(loaded_mod_0("input"), scripted_mod_0("input"))
        self.assertEqual(loaded_mod_1("input"), scripted_mod_1("input"))
        self.assertNotEqual(loaded_mod_0("input"), loaded_mod_1("input"))

    @skipIfNoTorchVision
    def test_save_ts_independent_modules(self):
        # Test to verify saving multiple modules works
        class ModD(torch.nn.Module):
            def __init__(self, name: str):
                super().__init__()
                self.name = name
                self.tvmod = resnet18()

            def forward(self, input: str):
                input = input + "_modB:" + self.name
                return input

        class ModC(torch.nn.Module):
            def __init__(self, name: str, submodule_name: str):
                super().__init__()
                self.name = name
                self.modB = ModD(submodule_name)
                self.tvmod = resnet18()

            def forward(self, input: str):
                input = input + "_modA:" + self.name
                return self.modB(input)

        scripted_mod = torch.jit.script(ModC("a", "b"))

        class ModFoo(torch.nn.Module):
            def __init__(self, name: str):
                super().__init__()
                self.name = name
                self.tvmod = resnet18()

            def forward(self, input: str):
                input = input + "_modFoo:" + self.name
                return input

        scripted_mod_foo = torch.jit.script(ModFoo("foo"))

        filename = self.temp()
        with PackageExporter(filename, verbose=False) as e:
            e.save_pickle("res", "mod1.pkl", scripted_mod)
            e.save_pickle("res", "mod2.pkl", scripted_mod_foo)

        importer = PackageImporter(filename)
        loaded_mod_0 = importer.load_pickle("res", "mod1.pkl")
        loaded_mod_1 = importer.load_pickle("res", "mod2.pkl")
        self.assertEqual(loaded_mod_0("input"), scripted_mod("input"))
        self.assertEqual(loaded_mod_1("input"), scripted_mod_foo("input"))

    @skipIfNoTorchVision
    def test_save_ts_repeat_saving_mod(self):
        #  Test to verify saving multiple different modules and
        #  repeats of modules in package works
        class ModD(torch.nn.Module):
            def __init__(self, name: str):
                super().__init__()
                self.name = name
                self.tvmod = resnet18()

            def forward(self, input: str):
                input = input + "_modB:" + self.name
                return input

        class ModC(torch.nn.Module):
            def __init__(self, name: str, submodule_name: str):
                super().__init__()
                self.name = name
                self.modB = ModD(submodule_name)
                self.tvmod = resnet18()

            def forward(self, input: str):
                input = input + "_modA:" + self.name
                return self.modB(input)

        scripted_mod_c = torch.jit.script(ModC("a", "b"))
        scripted_mod_d = torch.jit.script(ModD("b"))

        class ModFoo(torch.nn.Module):
            def __init__(self, name: str):
                super().__init__()
                self.name = name
                self.tvmod = resnet18()

            def forward(self, input: str):
                input = input + "_modFoo:" + self.name
                return input

        scripted_mod_foo = torch.jit.script(ModFoo("foo"))

        filename = self.temp()
        with PackageExporter(filename, verbose=False) as e:
            e.save_pickle("res", "mod1.pkl", scripted_mod_c)
            e.save_pickle("res", "mod2.pkl", scripted_mod_foo)
            e.save_pickle("res", "mod3.pkl", scripted_mod_d)
            e.save_pickle("res", "mod4.pkl", scripted_mod_foo)
            e.save_pickle("res", "mod5.pkl", scripted_mod_d)
            e.save_pickle("res", "mod6.pkl", scripted_mod_foo)

        importer = PackageImporter(filename)
        loaded_mod_0 = importer.load_pickle("res", "mod1.pkl")
        loaded_mod_1 = importer.load_pickle("res", "mod3.pkl")
        loaded_mod_2 = importer.load_pickle("res", "mod6.pkl")
        self.assertEqual(loaded_mod_0("input"), scripted_mod_c("input"))
        self.assertEqual(loaded_mod_1("input"), scripted_mod_d("input"))
        self.assertEqual(loaded_mod_2("input"), scripted_mod_foo("input"))

    @skipIfNoTorchVision
    def test_save_ts_repeat_save(self):
        # Test to verify saving and loading same TS object works
        class ModFoo(torch.nn.Module):
            def __init__(self, name: str):
                super().__init__()
                self.name = name
                self.tvmod = resnet18()

            def forward(self, input: str):
                input = input + "_modFoo:" + self.name
                return input

        class ModBar(torch.nn.Module):
            def __init__(self, name: str):
                super().__init__()
                self.name = name

            def forward(self, input: str):
                input = input + "_modBar:" + self.name
                return input

        scripted_mod_foo = torch.jit.script(ModFoo("foo"))
        scripted_mod_bar = torch.jit.script(ModBar("bar"))

        filename_0 = self.temp()
        with PackageExporter(filename_0, verbose=False) as e:
            e.save_pickle("res", "mod1.pkl", scripted_mod_foo)

        importer_0 = PackageImporter(filename_0)
        loaded_module_0 = importer_0.load_pickle("res", "mod1.pkl")

        filename_1 = self.temp()
        with PackageExporter(filename_1, verbose=False) as e:
            e.save_pickle("res", "mod1.pkl", scripted_mod_bar)
            e.save_pickle("res", "mod2.pkl", loaded_module_0)

        importer_1 = PackageImporter(filename_1)
        loaded_module_1 = importer_1.load_pickle("res", "mod1.pkl")
        reloaded_module_0 = importer_1.load_pickle("res", "mod2.pkl")

        self.assertEqual(loaded_module_0("input"), scripted_mod_foo("input"))
        self.assertEqual(loaded_module_0("input"), reloaded_module_0("input"))
        self.assertEqual(loaded_module_1("input"), scripted_mod_bar("input"))

    @skipIfNoTorchVision
    def test_save_ts_multiple_packages(self):
        # Test to verify when saving multiple packages with same CU
        # that packages don't include unnecessary ts code files

        class ModFoo(torch.nn.Module):
            def __init__(self, name: str):
                super().__init__()
                self.name = name
                self.tvmod = resnet18()

            def forward(self, input: str):
                input = input + "_modFoo:" + self.name
                return input

        class ModBar(torch.nn.Module):
            def __init__(self, name: str):
                super().__init__()
                self.name = name

            def forward(self, input: str):
                input = input + "_modBar:" + self.name
                return input

        scripted_mod_foo = torch.jit.script(ModFoo("foo"))
        scripted_mod_bar = torch.jit.script(ModBar("bar"))

        filename_0 = self.temp()
        with PackageExporter(filename_0, verbose=False) as e:
            e.save_pickle("res", "mod1.pkl", scripted_mod_foo)

        importer_0 = importer = PackageImporter(filename_0)

        filename_1 = self.temp()
        with PackageExporter(filename_1, verbose=False) as e:
            e.save_pickle("res", "mod1.pkl", scripted_mod_bar)

        importer_1 = PackageImporter(filename_1)

        self.assertTrue("torchvision" in str(importer_0.file_structure()))
        self.assertFalse("torchvision" in str(importer_1.file_structure()))

    @skipIfNoTorchVision
    def test_save_ts_in_container(self):
        # Test saving of TS modules inside of container
        class ModB(torch.nn.Module):
            def __init__(self, name: str):
                super().__init__()
                self.name = name
                self.tvmod = resnet18()

            def forward(self, input: str):
                input = input + "_modB:" + self.name
                return input

        class ModA(torch.nn.Module):
            def __init__(self, name: str):
                super().__init__()
                self.name = name
                self.tvmod = resnet18()

            def forward(self, input: str):
                input = input + "_modA:" + self.name
                return input

        scripted_mod_a = torch.jit.script(ModA("a"))
        scripted_mod_b = torch.jit.script(ModB("b"))
        script_mods_list = [scripted_mod_a, scripted_mod_b]

        filename = self.temp()
        with PackageExporter(filename, verbose=False) as e:
            e.save_pickle("res", "list.pkl", script_mods_list)

        importer = PackageImporter(filename)
        loaded_mod_list = importer.load_pickle("res", "list.pkl")
        self.assertEqual(loaded_mod_list[0]("input"), scripted_mod_a("input"))
        self.assertEqual(loaded_mod_list[1]("input"), scripted_mod_b("input"))

    @skipIfNoTorchVision
    def test_ts_scripting_packaged_mod(self):
        # Test scripting a module loaded from a package
        # and saving it in a new package as a script object
        from package_a.test_module import SimpleTest

        orig_mod = SimpleTest()

        filename_0 = self.temp()
        with PackageExporter(filename_0, verbose=False) as e:
            e.save_pickle("model", "model.pkl", orig_mod)

        importer_0 = PackageImporter(filename_0)
        loaded_mod = importer_0.load_pickle("model", "model.pkl")

        input = torch.rand(2, 3)
        self.assertTrue(torch.allclose(loaded_mod(input), orig_mod(input)))

        scripted_mod = torch.jit.script(loaded_mod)

        filename_1 = self.temp()
        with PackageExporter(filename_1, importer=importer_0, verbose=False) as e:
            e.save_pickle("res", "scripted_mod.pkl", scripted_mod)

        importer_1 = PackageImporter(filename_1)
        loaded_mod_scripted = importer_1.load_pickle("res", "scripted_mod.pkl")

        self.assertTrue(torch.allclose(loaded_mod_scripted(input), orig_mod(input)))


if __name__ == "__main__":
    run_tests()
