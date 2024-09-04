# Owner(s): ["oncall: package/deploy"]

from io import BytesIO
from textwrap import dedent
from unittest import skipIf

import torch
from torch.package import PackageExporter, PackageImporter
from torch.testing._internal.common_utils import IS_FBCODE, IS_SANDCASTLE, run_tests


try:
    from .common import PackageTestCase
except ImportError:
    # Support the case where we run this file directly.
    from common import PackageTestCase

try:
    from torchvision.models import resnet18

    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
skipIfNoTorchVision = skipIf(not HAS_TORCHVISION, "no torchvision")


class TestPackageScript(PackageTestCase):
    """Tests for compatibility with TorchScript."""

    def test_package_interface(self):
        """Packaging an interface class should work correctly."""

        import package_a.fake_interface as fake

        uses_interface = fake.UsesInterface()
        scripted = torch.jit.script(uses_interface)
        scripted.proxy_mod = torch.jit.script(fake.NewModule())

        buffer = BytesIO()
        with PackageExporter(buffer) as pe:
            pe.intern("**")
            pe.save_pickle("model", "model.pkl", uses_interface)
        buffer.seek(0)

        package_importer = PackageImporter(buffer)
        loaded = package_importer.load_pickle("model", "model.pkl")

        scripted_loaded = torch.jit.script(loaded)
        scripted_loaded.proxy_mod = torch.jit.script(fake.NewModule())

        input = torch.tensor(1)

        self.assertEqual(scripted(input), scripted_loaded(input))

    def test_different_package_interface(self):
        """Test a case where the interface defined in the package is
        different than the one defined in the loading environment, to make
        sure TorchScript can distinguish between the two.
        """
        # Import one version of the interface
        import package_a.fake_interface as fake

        # Simulate a package that contains a different version of the
        # interface, with the exact same name.
        buffer = BytesIO()
        with PackageExporter(buffer) as pe:
            pe.save_source_string(
                fake.__name__,
                dedent(
                    """\
                    import torch
                    from torch import Tensor

                    @torch.jit.interface
                    class ModuleInterface(torch.nn.Module):
                        def one(self, inp1: Tensor) -> Tensor:
                            pass

                    class ImplementsInterface(torch.nn.Module):
                        def one(self, inp1: Tensor) -> Tensor:
                            return inp1 + 1

                    class UsesInterface(torch.nn.Module):
                        proxy_mod: ModuleInterface

                        def __init__(self) -> None:
                            super().__init__()
                            self.proxy_mod = ImplementsInterface()

                        def forward(self, input: Tensor) -> Tensor:
                            return self.proxy_mod.one(input)
                    """
                ),
            )
        buffer.seek(0)

        package_importer = PackageImporter(buffer)
        diff_fake = package_importer.import_module(fake.__name__)
        # We should be able to script successfully.
        torch.jit.script(diff_fake.UsesInterface())

    def test_package_script_class(self):
        import package_a.fake_script_class as fake

        buffer = BytesIO()
        with PackageExporter(buffer) as pe:
            pe.save_module(fake.__name__)
        buffer.seek(0)

        package_importer = PackageImporter(buffer)
        loaded = package_importer.import_module(fake.__name__)

        input = torch.tensor(1)
        self.assertTrue(
            torch.allclose(
                fake.uses_script_class(input), loaded.uses_script_class(input)
            )
        )

    def test_package_script_class_referencing_self(self):
        import package_a.fake_script_class as fake

        obj = fake.UsesIdListFeature()
        # intentionally script here to fill the compilation cache, to make sure
        # there is no false sharing between scripted types coming from the
        # package vs. outside environment.
        torch.jit.script(obj)

        buffer = BytesIO()
        with PackageExporter(buffer) as exporter:
            exporter.intern("**")
            exporter.save_pickle("obj", "obj.pkl", obj)

        buffer.seek(0)
        importer = PackageImporter(buffer)
        obj_loaded = importer.load_pickle("obj", "obj.pkl")
        scripted_obj_loaded = torch.jit.script(obj_loaded)

        # Make sure the scripted object can be serialized without error.
        buffer2 = scripted_obj_loaded.save_to_buffer()
        torch.jit.load(BytesIO(buffer2))

    def test_different_package_script_class(self):
        """Test a case where the script class defined in the package is
        different than the one defined in the loading environment, to make
        sure TorchScript can distinguish between the two.
        """
        import package_a.fake_script_class as fake

        # Simulate a package that contains a different version of the
        # script class ,with the attribute `bar` instead of `foo`
        buffer = BytesIO()
        with PackageExporter(buffer) as pe2:
            pe2.save_source_string(
                fake.__name__,
                dedent(
                    """\
                    import torch

                    @torch.jit.script
                    class MyScriptClass:
                        def __init__(self, x):
                            self.bar = x
                    """
                ),
            )
        buffer.seek(0)

        package_importer = PackageImporter(buffer)
        diff_fake = package_importer.import_module(fake.__name__)
        input = torch.rand(2, 3)
        loaded_script_class = diff_fake.MyScriptClass(input)
        orig_script_class = fake.MyScriptClass(input)
        self.assertEqual(loaded_script_class.bar, orig_script_class.foo)

    def test_save_scriptmodule(self):
        """
        Test basic saving of ScriptModule.
        """
        from package_a.test_module import ModWithTensor

        scripted_mod = torch.jit.script(ModWithTensor(torch.rand(1, 2, 3)))

        buffer = BytesIO()
        with PackageExporter(buffer) as e:
            e.save_pickle("res", "mod.pkl", scripted_mod)

        buffer.seek(0)
        importer = PackageImporter(buffer)
        loaded_mod = importer.load_pickle("res", "mod.pkl", map_location="cpu")
        input = torch.rand(1, 2, 3)
        self.assertEqual(loaded_mod(input), scripted_mod(input))

    @skipIf(
        IS_FBCODE or IS_SANDCASTLE,
        "Tests that use temporary files are disabled in fbcode",
    )
    def test_save_scriptmodule_file(self):
        """
        Test basic saving of ScriptModule in file.
        """
        from package_a.test_module import ModWithTensor

        scripted_mod = torch.jit.script(ModWithTensor(torch.rand(1, 2, 3)))

        filename = self.temp()
        with PackageExporter(filename) as e:
            e.save_pickle("res", "mod.pkl", scripted_mod)

        importer = PackageImporter(filename)
        loaded_mod = importer.load_pickle("res", "mod.pkl")
        input = torch.rand(1, 2, 3)
        self.assertEqual(loaded_mod(input), scripted_mod(input))

    def test_save_scriptmodule_with_submods(self):
        """
        Test basic saving of ScriptModule with submodule.
        """
        from package_a.test_module import ModWithSubmod, ModWithTensor

        scripted_mod = torch.jit.script(
            ModWithSubmod(ModWithTensor(torch.rand(1, 2, 3)))
        )

        buffer = BytesIO()
        with PackageExporter(buffer) as e:
            e.save_pickle("res", "mod.pkl", scripted_mod)

        buffer.seek(0)
        importer = PackageImporter(buffer)
        loaded_mod = importer.load_pickle("res", "mod.pkl", map_location="cpu")
        input = torch.rand(1, 2, 3)
        self.assertEqual(loaded_mod(input), scripted_mod(input))

    def test_save_scriptmodules_submod_redefinition(self):
        """
        Test to verify saving multiple ScriptModules with same top module
        but different submodules works. Submodule is redefined to between
        the defintion of the top module to check that the different concrete
        types of the modules are thoroughly recognized by serializaiton code.
        """

        class Submod(torch.nn.Module):
            def forward(self, input: str):
                input = input + "_submod"
                return input

        class TopMod(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.modB = Submod()

            def forward(self, input: str):
                return self.modB(input)

        scripted_mod_0 = torch.jit.script(TopMod())

        # redefinition is intentional, change single inner string
        # string attribute, should trigger new module type
        class Submod(torch.nn.Module):  # noqa: F811
            def forward(self, input: str):
                input = input + "_submod(changed)"
                return input

        scripted_mod_1 = torch.jit.script(TopMod())

        buffer = BytesIO()
        with PackageExporter(buffer) as e:
            e.save_pickle("res", "mod1.pkl", scripted_mod_0)
            e.save_pickle("res", "mod2.pkl", scripted_mod_1)

        buffer.seek(0)
        importer = PackageImporter(buffer)
        loaded_mod_0 = importer.load_pickle("res", "mod1.pkl")
        loaded_mod_1 = importer.load_pickle("res", "mod2.pkl")
        self.assertEqual(loaded_mod_0("input"), scripted_mod_0("input"))
        self.assertEqual(loaded_mod_1("input"), scripted_mod_1("input"))
        self.assertNotEqual(loaded_mod_0("input"), loaded_mod_1("input"))

    def test_save_independent_scriptmodules(self):
        """
        Test to verify saving multiple ScriptModules with completely
        separate code works.
        """
        from package_a.test_module import ModWithTensor, SimpleTest

        scripted_mod_0 = torch.jit.script(SimpleTest())
        scripted_mod_1 = torch.jit.script(ModWithTensor(torch.rand(1, 2, 3)))

        buffer = BytesIO()
        with PackageExporter(buffer) as e:
            e.save_pickle("res", "mod1.pkl", scripted_mod_0)
            e.save_pickle("res", "mod2.pkl", scripted_mod_1)

        buffer.seek(0)
        importer = PackageImporter(buffer)
        loaded_mod_0 = importer.load_pickle("res", "mod1.pkl")
        loaded_mod_1 = importer.load_pickle("res", "mod2.pkl")
        input = torch.rand(1, 2, 3)
        self.assertEqual(loaded_mod_0(input), scripted_mod_0(input))
        self.assertEqual(loaded_mod_1(input), scripted_mod_1(input))

    def test_save_repeat_scriptmodules(self):
        """
        Test to verify saving multiple different modules and
        repeats of same scriptmodule in package works. Also tests that
        PyTorchStreamReader isn't having code hidden from
        PyTorchStreamWriter writing ScriptModule code files multiple times.
        """
        from package_a.test_module import (
            ModWithSubmodAndTensor,
            ModWithTensor,
            SimpleTest,
        )

        scripted_mod_0 = torch.jit.script(SimpleTest())
        scripted_mod_1 = torch.jit.script(ModWithTensor(torch.rand(1, 2, 3)))
        scripted_mod_2 = torch.jit.script(
            ModWithSubmodAndTensor(
                torch.rand(1, 2, 3), ModWithTensor(torch.rand(1, 2, 3))
            )
        )

        buffer = BytesIO()
        with PackageExporter(buffer) as e:
            e.save_pickle("res", "mod0.pkl", scripted_mod_0)
            e.save_pickle("res", "mod1.pkl", scripted_mod_1)
            e.save_pickle("res", "mod2.pkl", scripted_mod_0)
            e.save_pickle("res", "mod3.pkl", scripted_mod_1)
            e.save_pickle("res", "mod4.pkl", scripted_mod_2)

        buffer.seek(0)
        importer = PackageImporter(buffer)
        loaded_mod_0 = importer.load_pickle("res", "mod0.pkl")
        loaded_mod_1 = importer.load_pickle("res", "mod3.pkl")
        loaded_mod_2 = importer.load_pickle("res", "mod4.pkl")
        input = torch.rand(1, 2, 3)
        self.assertEqual(loaded_mod_0(input), scripted_mod_0(input))
        self.assertEqual(loaded_mod_1(input), scripted_mod_1(input))
        self.assertEqual(loaded_mod_2(input), scripted_mod_2(input))

    def test_scriptmodules_repeat_save(self):
        """
        Test to verify saving and loading same ScriptModule object works
        across multiple packages.
        """
        from package_a.test_module import ModWithSubmodAndTensor, ModWithTensor

        scripted_mod_0 = torch.jit.script(ModWithTensor(torch.rand(1, 2, 3)))
        scripted_mod_1 = torch.jit.script(
            ModWithSubmodAndTensor(
                torch.rand(1, 2, 3), ModWithTensor(torch.rand(1, 2, 3))
            )
        )

        buffer_0 = BytesIO()
        with PackageExporter(buffer_0) as e:
            e.save_pickle("res", "mod1.pkl", scripted_mod_0)

        buffer_0.seek(0)
        importer_0 = PackageImporter(buffer_0)
        loaded_module_0 = importer_0.load_pickle("res", "mod1.pkl")

        buffer_1 = BytesIO()
        with PackageExporter(buffer_1) as e:
            e.save_pickle("res", "mod1.pkl", scripted_mod_1)
            e.save_pickle("res", "mod2.pkl", loaded_module_0)

        buffer_1.seek(0)
        importer_1 = PackageImporter(buffer_1)
        loaded_module_1 = importer_1.load_pickle("res", "mod1.pkl")
        reloaded_module_0 = importer_1.load_pickle("res", "mod2.pkl")

        input = torch.rand(1, 2, 3)
        self.assertEqual(loaded_module_0(input), scripted_mod_0(input))
        self.assertEqual(loaded_module_0(input), reloaded_module_0(input))
        self.assertEqual(loaded_module_1(input), scripted_mod_1(input))

    @skipIfNoTorchVision
    def test_save_scriptmodule_only_necessary_code(self):
        """
        Test to verify when saving multiple packages with same CU
        that packages don't include unnecessary torchscript code files.
        The TorchVision code should only be saved in the package that
        relies on it.
        """
        from package_a.test_module import ModWithTensor

        class ModWithTorchVision(torch.nn.Module):
            def __init__(self, name: str):
                super().__init__()
                self.tvmod = resnet18()

            def forward(self, input):
                return input * 4

        scripted_mod_0 = torch.jit.script(ModWithTorchVision("foo"))
        scripted_mod_1 = torch.jit.script(ModWithTensor(torch.rand(1, 2, 3)))

        buffer_0 = BytesIO()
        with PackageExporter(buffer_0) as e:
            e.save_pickle("res", "mod1.pkl", scripted_mod_0)

        buffer_0.seek(0)
        importer_0 = importer = PackageImporter(buffer_0)

        buffer_1 = BytesIO()
        with PackageExporter(buffer_1) as e:
            e.save_pickle("res", "mod1.pkl", scripted_mod_1)

        buffer_1.seek(0)
        importer_1 = PackageImporter(buffer_1)

        self.assertTrue("torchvision" in str(importer_0.file_structure()))
        self.assertFalse("torchvision" in str(importer_1.file_structure()))

    def test_save_scriptmodules_in_container(self):
        """
        Test saving of ScriptModules inside of container. Checks that relations
        between shared modules are upheld.
        """
        from package_a.test_module import ModWithSubmodAndTensor, ModWithTensor

        scripted_mod_a = torch.jit.script(ModWithTensor(torch.rand(1, 2, 3)))
        scripted_mod_b = torch.jit.script(
            ModWithSubmodAndTensor(torch.rand(1, 2, 3), scripted_mod_a)
        )
        script_mods_list = [scripted_mod_a, scripted_mod_b]

        buffer = BytesIO()
        with PackageExporter(buffer) as e:
            e.save_pickle("res", "list.pkl", script_mods_list)

        buffer.seek(0)
        importer = PackageImporter(buffer)
        loaded_mod_list = importer.load_pickle("res", "list.pkl")
        input = torch.rand(1, 2, 3)
        self.assertEqual(loaded_mod_list[0](input), scripted_mod_a(input))
        self.assertEqual(loaded_mod_list[1](input), scripted_mod_b(input))

    def test_save_eager_mods_sharing_scriptmodule(self):
        """
        Test saving of single ScriptModule shared by multiple
        eager modules (ScriptModule should be saved just once
        even though is contained in multiple pickles).
        """
        from package_a.test_module import ModWithSubmod, SimpleTest

        scripted_mod = torch.jit.script(SimpleTest())

        mod1 = ModWithSubmod(scripted_mod)
        mod2 = ModWithSubmod(scripted_mod)

        buffer = BytesIO()
        with PackageExporter(buffer) as e:
            e.intern("**")
            e.save_pickle("res", "mod1.pkl", mod1)
            e.save_pickle("res", "mod2.pkl", mod2)

        buffer.seek(0)
        importer = PackageImporter(buffer)
        file_structure = importer.file_structure()
        self.assertTrue(file_structure.has_file(".data/ts_code/0"))
        self.assertFalse(file_structure.has_file(".data/ts_code/1"))

    def test_load_shared_scriptmodules(self):
        """
        Test loading of single ScriptModule shared by multiple eager
        modules in single pickle (ScriptModule objects should be the same).
        """
        from package_a.test_module import (
            ModWithMultipleSubmods,
            ModWithSubmod,
            SimpleTest,
        )

        scripted_mod = torch.jit.script(SimpleTest())

        mod1 = ModWithSubmod(scripted_mod)
        mod2 = ModWithSubmod(scripted_mod)

        mod_parent = ModWithMultipleSubmods(mod1, mod2)

        buffer = BytesIO()
        with PackageExporter(buffer) as e:
            e.intern("**")
            e.save_pickle("res", "mod.pkl", mod_parent)

        buffer.seek(0)
        importer = PackageImporter(buffer)

        loaded_mod = importer.load_pickle("res", "mod.pkl")
        self.assertTrue(
            id(loaded_mod.mod1.script_mod) == id(loaded_mod.mod2.script_mod)
        )

    def test_save_shared_tensors(self):
        """
        Test tensors shared across eager and ScriptModules are serialized once.
        """
        from package_a.test_module import ModWithSubmodAndTensor, ModWithTensor

        shared_tensor = torch.rand(2, 3, 4)
        scripted_mod = torch.jit.script(ModWithTensor(shared_tensor))

        mod1 = ModWithSubmodAndTensor(shared_tensor, scripted_mod)
        mod2 = ModWithSubmodAndTensor(shared_tensor, scripted_mod)

        buffer = BytesIO()
        with PackageExporter(buffer) as e:
            e.intern("**")
            e.save_pickle("res", "tensor", shared_tensor)
            e.save_pickle("res", "mod1.pkl", mod1)
            e.save_pickle("res", "mod2.pkl", mod2)

        buffer.seek(0)
        importer = PackageImporter(buffer)
        loaded_mod_1 = importer.load_pickle("res", "mod1.pkl")

        # assert that there is only one storage stored in package
        file_structure = importer.file_structure(include=".data/*.storage")
        self.assertTrue(len(file_structure.children[".data"].children) == 1)

        input = torch.rand(2, 3, 4)
        self.assertEqual(loaded_mod_1(input), mod1(input))

    def test_load_shared_tensors(self):
        """
        Test tensors shared across eager and ScriptModules on load
        are the same.
        """
        from package_a.test_module import ModWithTensor, ModWithTwoSubmodsAndTensor

        shared_tensor = torch.ones(3, 3)

        scripted_mod_0 = torch.jit.script(ModWithTensor(shared_tensor))
        scripted_mod_1 = torch.jit.script(ModWithTensor(shared_tensor))

        mod1 = ModWithTwoSubmodsAndTensor(shared_tensor, scripted_mod_0, scripted_mod_1)

        self.assertEqual(
            shared_tensor.storage()._cdata,
            scripted_mod_0.tensor.storage()._cdata,
        )
        self.assertEqual(
            shared_tensor.storage()._cdata,
            scripted_mod_1.tensor.storage()._cdata,
        )

        buffer = BytesIO()
        with PackageExporter(buffer) as e:
            e.intern("**")
            e.save_pickle("res", "mod1.pkl", mod1)

        buffer.seek(0)
        importer = PackageImporter(buffer)
        loaded_mod_1 = importer.load_pickle("res", "mod1.pkl")

        self.assertEqual(
            loaded_mod_1.tensor.storage()._cdata,
            loaded_mod_1.sub_mod_0.tensor.storage()._cdata,
        )
        self.assertEqual(
            loaded_mod_1.tensor.storage()._cdata,
            loaded_mod_1.sub_mod_1.tensor.storage()._cdata,
        )

        loaded_mod_1.tensor.add_(torch.ones(3, 3))

        self.assertTrue(
            torch.allclose(loaded_mod_1.tensor, loaded_mod_1.sub_mod_0.tensor)
        )
        self.assertTrue(
            torch.allclose(loaded_mod_1.tensor, loaded_mod_1.sub_mod_1.tensor)
        )

    def test_load_shared_tensors_repackaged(self):
        """
        Test tensors shared across eager and ScriptModules on load
        are the same across multiple package saves and loads. This is
        an important test because not all of the tensor information is restored
        in python between packages. The python identity is not maintained, but
        the backing cpp TensorImpl is. We load/save storages based off of this
        cpp TensorImpl and not the python identity.
        """
        from package_a.test_module import ModWithTensor, ModWithTwoSubmodsAndTensor

        shared_tensor = torch.ones(3, 3)

        scripted_mod_0 = torch.jit.script(ModWithTensor(shared_tensor))
        scripted_mod_1 = torch.jit.script(ModWithTensor(shared_tensor))

        mod1 = ModWithTwoSubmodsAndTensor(shared_tensor, scripted_mod_0, scripted_mod_1)

        buffer_0 = BytesIO()
        with PackageExporter(buffer_0) as e:
            e.intern("**")
            e.save_pickle("res", "mod1.pkl", mod1)

        buffer_0.seek(0)
        importer_0 = PackageImporter(buffer_0)
        loaded_mod_0 = importer_0.load_pickle("res", "mod1.pkl")

        buffer_1 = BytesIO()
        with PackageExporter(buffer_1, importer=importer_0) as e:
            e.intern("**")
            e.save_pickle("res", "mod1.pkl", loaded_mod_0)

        buffer_1.seek(0)
        importer = PackageImporter(buffer_1)
        loaded_mod_1 = importer.load_pickle("res", "mod1.pkl")

        self.assertEqual(
            loaded_mod_1.tensor.storage()._cdata,
            loaded_mod_1.sub_mod_0.tensor.storage()._cdata,
        )
        self.assertEqual(
            loaded_mod_1.tensor.storage()._cdata,
            loaded_mod_1.sub_mod_1.tensor.storage()._cdata,
        )

        loaded_mod_1.tensor.add_(
            torch.ones(3, 3)
        )  # all tensors should reflect this change

        self.assertTrue(
            torch.allclose(loaded_mod_1.tensor, loaded_mod_1.sub_mod_0.tensor)
        )
        self.assertTrue(
            torch.allclose(loaded_mod_1.tensor, loaded_mod_1.sub_mod_1.tensor)
        )

    def test_saving_and_scripting_packaged_mod(self):
        """
        Test scripting a module loaded from a package
        and saving it in a new package as a script object.
        """
        from package_a.test_module import SimpleTest

        orig_mod = SimpleTest()

        buffer_0 = BytesIO()
        with PackageExporter(buffer_0) as e:
            e.intern("**")
            e.save_pickle("model", "model.pkl", orig_mod)

        buffer_0.seek(0)
        importer_0 = PackageImporter(buffer_0)
        loaded_mod = importer_0.load_pickle("model", "model.pkl")

        input = torch.rand(2, 3)
        self.assertEqual(loaded_mod(input), orig_mod(input))

        scripted_mod = torch.jit.script(loaded_mod)

        buffer_1 = BytesIO()
        with PackageExporter(buffer_1, importer=importer_0) as e:
            e.intern("**")
            e.save_pickle("res", "scripted_mod.pkl", scripted_mod)

        buffer_1.seek(0)
        importer_1 = PackageImporter(buffer_1)
        loaded_mod_scripted = importer_1.load_pickle("res", "scripted_mod.pkl")

        self.assertEqual(loaded_mod_scripted(input), orig_mod(input))

    def test_mixing_packaged_and_inline_modules(self):
        """
        Test saving inline and imported modules in same package with
        independent code.
        """

        class InlineMod(torch.nn.Module):
            def __init__(self, name: str):
                super().__init__()
                self.name = name
                self.tensor = torch.rand(1, 2, 3)

            def forward(self, input: str):
                input = input + "_modInline:" + self.name
                return input, (self.tensor * 4)

        inline_mod = InlineMod("inline")
        scripted_inline = torch.jit.script(inline_mod)

        from package_a.test_module import SimpleTest

        imported_mod = SimpleTest()
        scripted_imported = torch.jit.script(imported_mod)

        buffer = BytesIO()
        with PackageExporter(buffer) as e:
            e.save_pickle("model", "inline.pkl", scripted_inline)
            e.save_pickle("model", "imported.pkl", scripted_imported)

        buffer.seek(0)
        importer = PackageImporter(buffer)
        loaded_inline = importer.load_pickle("model", "inline.pkl")
        loaded_imported = importer.load_pickle("model", "imported.pkl")

        input = torch.rand(2, 3)
        self.assertEqual(loaded_imported(input), imported_mod(input))
        self.assertEqual(loaded_inline("input"), inline_mod("input"))

    @skipIfNoTorchVision
    def test_mixing_packaged_and_inline_modules_shared_code(self):
        """
        Test saving inline and imported modules in same package that
        share code.
        """

        class TorchVisionTestInline(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.tvmod = resnet18()

            def forward(self, x):
                x = a_non_torch_leaf(x, x)
                return torch.relu(x + 3.0)

        def a_non_torch_leaf(a, b):
            return a + b

        inline_mod = TorchVisionTestInline()
        scripted_inline = torch.jit.script(inline_mod)

        from package_c.test_module import TorchVisionTest

        imported_mod = TorchVisionTest()
        scripted_imported = torch.jit.script(imported_mod)

        buffer = BytesIO()
        with PackageExporter(buffer) as e:
            e.save_pickle("model", "inline.pkl", scripted_inline)
            e.save_pickle("model", "imported.pkl", scripted_imported)

        buffer.seek(0)
        importer = PackageImporter(buffer)
        loaded_inline = importer.load_pickle("model", "inline.pkl")
        loaded_imported = importer.load_pickle("model", "imported.pkl")

        input = torch.rand(2, 3)
        self.assertEqual(loaded_imported(input), imported_mod(input))
        self.assertEqual(loaded_inline(input), inline_mod(input))

    def test_tensor_sharing_pickle(self):
        """Test that saving a ScriptModule and a separately saving a tensor
        object causes no issues.
        """

        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.foo = torch.ones(2, 3)

            def forward(self):
                return self.foo

        scripted_m = torch.jit.script(M())
        original_tensor = torch.ones(0)

        f = BytesIO()
        with torch.package.PackageExporter(f) as exporter:
            exporter.save_pickle("model", "model.pkl", scripted_m)
            exporter.save_pickle("model", "input.pkl", original_tensor)

        f.seek(0)
        # Should be able to load correctly
        importer = PackageImporter(f)
        loaded_m = importer.load_pickle("model", "model.pkl")
        loaded_tensor = importer.load_pickle("model", "input.pkl")

        self.assertEqual(scripted_m.foo, loaded_m.foo)
        self.assertEqual(original_tensor, loaded_tensor)


if __name__ == "__main__":
    run_tests()
