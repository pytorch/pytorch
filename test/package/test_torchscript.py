from unittest import skipIf
from io import BytesIO

import torch
from torch.package import PackageExporter, PackageImporter
from torch.testing._internal.common_utils import run_tests, IS_FBCODE, IS_SANDCASTLE

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

    def test_save_ts(self):
        # Test basic saving of TS module
        class ModB(torch.nn.Module):
            def __init__(self, name: str):
                super().__init__()
                self.name = name
                self.tensor = torch.rand(1, 1, 10)

            def forward(self, input: str):
                input = input + "_modB:" + self.name
                return input

        class ModA(torch.nn.Module):
            def __init__(self, name: str, submodule_name: str):
                super().__init__()
                self.name = name
                self.modB = ModB(submodule_name)
                self.tensor = torch.rand(2, 1, 10)

            def forward(self, input: str):
                input = input + "_modA:" + self.name
                self.tensor = self.tensor * 2
                return self.modB(input)

        scripted_mod = torch.jit.script(ModA("a", "b"))

        buffer = BytesIO()
        with PackageExporter(buffer, verbose=False) as e:
            e.save_pickle("res", "mod.pkl", scripted_mod)

        buffer.seek(0)
        importer = PackageImporter(buffer)
        loaded_mod = importer.load_pickle("res", "mod.pkl")
        self.assertEqual(loaded_mod("input"), scripted_mod("input"))

    @skipIf(
        IS_FBCODE or IS_SANDCASTLE,
        "Tests that use temporary files are disabled in fbcode",
    )
    def test_save_ts_file(self):
        # Test basic saving of TS module in file
        class ModB(torch.nn.Module):
            def __init__(self, name: str):
                super().__init__()
                self.name = name
                self.tensor = torch.rand(1, 1, 10)

            def forward(self, input: str):
                input = input + "_modB:" + self.name
                return input

        class ModA(torch.nn.Module):
            def __init__(self, name: str, submodule_name: str):
                super().__init__()
                self.name = name
                self.modB = ModB(submodule_name)
                self.tensor = torch.rand(1, 1, 10)

            def forward(self, input: str):
                input = input + "_modA:" + self.name
                self.tensor = self.tensor * 2
                return self.modB(input)

        scripted_mod = torch.jit.script(ModA("a", "b"))

        filename = self.temp()
        with PackageExporter(filename, verbose=False) as e:
            e.save_pickle("res", "mod.pkl", scripted_mod)

        importer = PackageImporter(filename)
        loaded_mod = importer.load_pickle("res", "mod.pkl")
        self.assertEqual(loaded_mod("input"), scripted_mod("input"))

    def test_save_ts_modules_shared_code(self):
        #  Test to verify saving multiple modules with same top module
        #  but different submodules works
        class ModD(torch.nn.Module):
            def __init__(self, name: str):
                super().__init__()
                self.name = name
                self.tensor = torch.rand(1, 1, 10)

            def forward(self, input: str):
                input = input + "_modB:" + self.name
                self.tensor = self.tensor * 6
                return input

        class ModC(torch.nn.Module):
            def __init__(self, name: str, submodule_name: str):
                super().__init__()
                self.name = name
                self.modB = ModD(submodule_name)
                self.tensor = torch.rand(1, 8, 10)

            def forward(self, input: str):
                input = input + "_modA:" + self.name
                self.tensor = self.tensor * 20
                self.modB(input)
                return self.tensor

        scripted_mod_0 = torch.jit.script(ModC("a", "b"))

        # redefinition is intentional
        class ModD(torch.nn.Module):  # noqa: F811
            def __init__(self, name: str):
                super().__init__()
                self.name = name

            def forward(self, input: str):
                input = input + "_modB(changed):" + self.name
                return input

        scripted_mod_1 = torch.jit.script(ModC("a", "b"))

        buffer = BytesIO()
        with PackageExporter(buffer, verbose=False) as e:
            e.save_pickle("res", "mod1.pkl", scripted_mod_0)
            e.save_pickle("res", "mod2.pkl", scripted_mod_1)

        buffer.seek(0)
        importer = PackageImporter(buffer)
        loaded_mod_0 = importer.load_pickle("res", "mod1.pkl")
        loaded_mod_1 = importer.load_pickle("res", "mod2.pkl")
        self.assertEqual(loaded_mod_0("input"), scripted_mod_0("input"))
        self.assertEqual(loaded_mod_1("input"), scripted_mod_1("input"))
        self.assertNotEqual(loaded_mod_0("input"), loaded_mod_1("input"))

    def test_save_ts_independent_modules(self):
        # Test to verify saving multiple modules works
        class ModD(torch.nn.Module):
            def __init__(self, name: str):
                super().__init__()
                self.name = name

            def forward(self, input: str):
                input = input + "_modB:" + self.name
                return input

        class ModC(torch.nn.Module):
            def __init__(self, name: str, submodule_name: str):
                super().__init__()
                self.name = name
                self.modB = ModD(submodule_name)

            def forward(self, input: str):
                input = input + "_modA:" + self.name
                return self.modB(input)

        scripted_mod = torch.jit.script(ModC("a", "b"))

        class ModFoo(torch.nn.Module):
            def __init__(self, name: str):
                super().__init__()
                self.name = name

            def forward(self, input: str):
                input = input + "_modFoo:" + self.name
                return input

        scripted_mod_foo = torch.jit.script(ModFoo("foo"))

        buffer = BytesIO()
        with PackageExporter(buffer, verbose=False) as e:
            e.save_pickle("res", "mod1.pkl", scripted_mod)
            e.save_pickle("res", "mod2.pkl", scripted_mod_foo)

        buffer.seek(0)
        importer = PackageImporter(buffer)
        loaded_mod_0 = importer.load_pickle("res", "mod1.pkl")
        loaded_mod_1 = importer.load_pickle("res", "mod2.pkl")
        self.assertEqual(loaded_mod_0("input"), scripted_mod("input"))
        self.assertEqual(loaded_mod_1("input"), scripted_mod_foo("input"))

    def test_save_ts_repeat_saving_mod(self):
        #  Test to verify saving multiple different modules and
        #  repeats of modules in package works. Also tests that
        #  PyTorchStreamReader isn't having code hidden from
        #  PyTorchStreamWriter writing TS code files multiple times.

        class ModD(torch.nn.Module):
            def __init__(self, name: str):
                super().__init__()
                self.name = name

            def forward(self, input: str):
                input = input + "_modB:" + self.name
                return input

        class ModC(torch.nn.Module):
            def __init__(self, name: str, submodule_name: str):
                super().__init__()
                self.name = name
                self.modB = ModD(submodule_name)

            def forward(self, input: str):
                input = input + "_modA:" + self.name
                return self.modB(input)

        scripted_mod_c = torch.jit.script(ModC("a", "b"))
        scripted_mod_d = torch.jit.script(ModD("b"))

        class ModFoo(torch.nn.Module):
            def __init__(self, name: str):
                super().__init__()
                self.name = name
                self.tensor = torch.rand(100, 2, 3)

            def forward(self, input: str):
                input = input + "_modFoo:" + self.name
                self.tensor = self.tensor * 4
                return (input, self.tensor)

        scripted_mod_foo = torch.jit.script(ModFoo("foo"))

        buffer = BytesIO()
        with PackageExporter(buffer, verbose=False) as e:
            e.save_pickle("res", "mod1.pkl", scripted_mod_c)
            e.save_pickle("res", "mod2.pkl", scripted_mod_d)
            e.save_pickle("res", "mod3.pkl", scripted_mod_c)
            e.save_pickle("res", "mod4.pkl", scripted_mod_d)
            e.save_pickle("res", "mod5.pkl", scripted_mod_foo)

        buffer.seek(0)
        importer = PackageImporter(buffer)
        loaded_mod_0 = importer.load_pickle("res", "mod1.pkl")
        loaded_mod_1 = importer.load_pickle("res", "mod4.pkl")
        loaded_mod_2 = importer.load_pickle("res", "mod5.pkl")
        self.assertEqual(loaded_mod_0("input"), scripted_mod_c("input"))
        self.assertEqual(loaded_mod_1("input"), scripted_mod_d("input"))
        self.assertEqual(loaded_mod_2("input"), scripted_mod_foo("input"))

    def test_save_ts_repeat_save(self):
        # Test to verify saving and loading same TS object works
        class ModFoo(torch.nn.Module):
            def __init__(self, name: str):
                super().__init__()
                self.name = name

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

        buffer_0 = BytesIO()
        with PackageExporter(buffer_0, verbose=False) as e:
            e.save_pickle("res", "mod1.pkl", scripted_mod_foo)

        buffer_0.seek(0)
        importer_0 = PackageImporter(buffer_0)
        loaded_module_0 = importer_0.load_pickle("res", "mod1.pkl")

        buffer_1 = BytesIO()
        with PackageExporter(buffer_1, verbose=False) as e:
            e.save_pickle("res", "mod1.pkl", scripted_mod_bar)
            e.save_pickle("res", "mod2.pkl", loaded_module_0)

        buffer_1.seek(0)
        importer_1 = PackageImporter(buffer_1)
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
                self.tensor = torch.rand(1, 2, 3)

            def forward(self, input: str):
                input = input + "_modBar:" + self.name
                return input, (self.tensor * 4)

        scripted_mod_foo = torch.jit.script(ModFoo("foo"))
        scripted_mod_bar = torch.jit.script(ModBar("bar"))

        buffer_0 = BytesIO()
        with PackageExporter(buffer_0, verbose=False) as e:
            e.save_pickle("res", "mod1.pkl", scripted_mod_foo)

        buffer_0.seek(0)
        importer_0 = importer = PackageImporter(buffer_0)

        buffer_1 = BytesIO()
        with PackageExporter(buffer_1, verbose=False) as e:
            e.save_pickle("res", "mod1.pkl", scripted_mod_bar)

        buffer_1.seek(0)
        importer_1 = PackageImporter(buffer_1)

        self.assertTrue("torchvision" in str(importer_0.file_structure()))
        self.assertFalse("torchvision" in str(importer_1.file_structure()))

    def test_save_ts_in_container(self):
        # Test saving of TS modules inside of container
        class ModB(torch.nn.Module):
            def __init__(self, name: str):
                super().__init__()
                self.name = name

            def forward(self, input: str):
                input = input + "_modB:" + self.name
                return input

        class ModA(torch.nn.Module):
            def __init__(self, name: str):
                super().__init__()
                self.name = name

            def forward(self, input: str):
                input = input + "_modA:" + self.name
                return input

        scripted_mod_a = torch.jit.script(ModA("a"))
        scripted_mod_b = torch.jit.script(ModB("b"))
        script_mods_list = [scripted_mod_a, scripted_mod_b]

        buffer = BytesIO()
        with PackageExporter(buffer, verbose=False) as e:
            e.save_pickle("res", "list.pkl", script_mods_list)

        buffer.seek(0)
        importer = PackageImporter(buffer)
        loaded_mod_list = importer.load_pickle("res", "list.pkl")
        self.assertEqual(loaded_mod_list[0]("input"), scripted_mod_a("input"))
        self.assertEqual(loaded_mod_list[1]("input"), scripted_mod_b("input"))

    def test_ts_scripting_packaged_mod(self):
        # Test scripting a module loaded from a package
        # and saving it in a new package as a script object
        from package_a.test_module import SimpleTest

        orig_mod = SimpleTest()

        buffer_0 = BytesIO()
        with PackageExporter(buffer_0, verbose=False) as e:
            e.save_pickle("model", "model.pkl", orig_mod)

        buffer_0.seek(0)
        importer_0 = PackageImporter(buffer_0)
        loaded_mod = importer_0.load_pickle("model", "model.pkl")

        input = torch.rand(2, 3)
        self.assertTrue(torch.allclose(loaded_mod(input), orig_mod(input)))

        scripted_mod = torch.jit.script(loaded_mod)

        buffer_1 = BytesIO()
        with PackageExporter(buffer_1, importer=importer_0, verbose=False) as e:
            e.save_pickle("res", "scripted_mod.pkl", scripted_mod)

        buffer_1.seek(0)
        importer_1 = PackageImporter(buffer_1)
        loaded_mod_scripted = importer_1.load_pickle("res", "scripted_mod.pkl")

        self.assertTrue(torch.allclose(loaded_mod_scripted(input), orig_mod(input)))


if __name__ == "__main__":
    run_tests()
