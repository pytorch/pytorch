from unittest import skipIf
import inspect
from torch.testing._internal.common_utils import TestCase, run_tests, IS_WINDOWS
from tempfile import NamedTemporaryFile
from torch.package import PackageExporter, PackageImporter
from torch.package._mangling import PackageMangler, demangle, is_mangled, get_mangle_prefix
from pathlib import Path
from tempfile import TemporaryDirectory
import torch
from sys import version_info
from io import StringIO, BytesIO
import pickle

try:
    from torchvision.models import resnet18
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
skipIfNoTorchVision = skipIf(not HAS_TORCHVISION, "no torchvision")



packaging_directory = Path(__file__).parent

class PackagingTest(TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._temporary_files = []

    def temp(self):
        t = NamedTemporaryFile()
        name = t.name
        if IS_WINDOWS:
            t.close()  # can't read an open file in windows
        else:
            self._temporary_files.append(t)
        return name

    def tearDown(self):
        for t in self._temporary_files:
            t.close()
        self._temporary_files = []

    def test_saving_source(self):
        filename = self.temp()
        with PackageExporter(filename, verbose=False) as he:
            he.save_source_file('foo', str(packaging_directory / 'module_a.py'))
            he.save_source_file('foodir', str(packaging_directory / 'package_a'))
        hi = PackageImporter(filename)
        foo = hi.import_module('foo')
        s = hi.import_module('foodir.subpackage')
        self.assertEqual(foo.result, 'module_a')
        self.assertEqual(s.result, 'package_a.subpackage')

    def test_saving_string(self):
        filename = self.temp()
        with PackageExporter(filename, verbose=False) as he:
            src = """\
import math
the_math = math
"""
            he.save_source_string('my_mod', src)
        hi = PackageImporter(filename)
        m = hi.import_module('math')
        import math
        self.assertIs(m, math)
        my_mod = hi.import_module('my_mod')
        self.assertIs(my_mod.math, math)

    def test_save_module(self):
        filename = self.temp()
        with PackageExporter(filename, verbose=False) as he:
            import module_a
            import package_a
            he.save_module(module_a.__name__)
            he.save_module(package_a.__name__)
        hi = PackageImporter(filename)
        module_a_i = hi.import_module('module_a')
        self.assertEqual(module_a_i.result, 'module_a')
        self.assertIsNot(module_a, module_a_i)
        package_a_i = hi.import_module('package_a')
        self.assertEqual(package_a_i.result, 'package_a')
        self.assertIsNot(package_a_i, package_a)

    def test_save_module_binary(self):
        f = BytesIO()
        with PackageExporter(f, verbose=False) as he:
            import module_a
            import package_a
            he.save_module(module_a.__name__)
            he.save_module(package_a.__name__)
        f.seek(0)
        hi = PackageImporter(f)
        module_a_i = hi.import_module('module_a')
        self.assertEqual(module_a_i.result, 'module_a')
        self.assertIsNot(module_a, module_a_i)
        package_a_i = hi.import_module('package_a')
        self.assertEqual(package_a_i.result, 'package_a')
        self.assertIsNot(package_a_i, package_a)

    def test_pickle(self):
        import package_a.subpackage
        obj = package_a.subpackage.PackageASubpackageObject()
        obj2 = package_a.PackageAObject(obj)

        filename = self.temp()
        with PackageExporter(filename, verbose=False) as he:
            he.save_pickle('obj', 'obj.pkl', obj2)
        hi = PackageImporter(filename)

        # check we got dependencies
        sp = hi.import_module('package_a.subpackage')
        # check we didn't get other stuff
        with self.assertRaises(ImportError):
            hi.import_module('module_a')

        obj_loaded = hi.load_pickle('obj', 'obj.pkl')
        self.assertIsNot(obj2, obj_loaded)
        self.assertIsInstance(obj_loaded.obj, sp.PackageASubpackageObject)
        self.assertIsNot(package_a.subpackage.PackageASubpackageObject, sp.PackageASubpackageObject)

    def test_resources(self):
        filename = self.temp()
        with PackageExporter(filename, verbose=False) as he:
            he.save_text('main', 'main', "my string")
            he.save_binary('main', 'main_binary', "my string".encode('utf-8'))
            src = """\
import resources
t = resources.load_text('main', 'main')
b = resources.load_binary('main', 'main_binary')
"""
            he.save_source_string('main', src, is_package=True)
        hi = PackageImporter(filename)
        m = hi.import_module('main')
        self.assertEqual(m.t, "my string")
        self.assertEqual(m.b, "my string".encode('utf-8'))

    def test_extern(self):
        filename = self.temp()
        with PackageExporter(filename, verbose=False) as he:
            he.extern(['package_a.subpackage', 'module_a'])
            he.require_module('package_a.subpackage')
            he.require_module('module_a')
            he.save_module('package_a')
        hi = PackageImporter(filename)
        import package_a.subpackage
        import module_a

        module_a_im = hi.import_module('module_a')
        hi.import_module('package_a.subpackage')
        package_a_im = hi.import_module('package_a')

        self.assertIs(module_a, module_a_im)
        self.assertIsNot(package_a, package_a_im)
        self.assertIs(package_a.subpackage, package_a_im.subpackage)

    def test_extern_glob(self):
        filename = self.temp()
        with PackageExporter(filename, verbose=False) as he:
            he.extern(['package_a.*', 'module_*'])
            he.save_module('package_a')
            he.save_source_string('test_module', """\
import package_a.subpackage
import module_a
""")
        hi = PackageImporter(filename)
        import package_a.subpackage
        import module_a

        module_a_im = hi.import_module('module_a')
        hi.import_module('package_a.subpackage')
        package_a_im = hi.import_module('package_a')

        self.assertIs(module_a, module_a_im)
        self.assertIsNot(package_a, package_a_im)
        self.assertIs(package_a.subpackage, package_a_im.subpackage)

    def test_save_imported_module_fails(self):
        """
        Directly saving/requiring an PackageImported module should raise a specific error message.
        """
        import package_a.subpackage
        obj = package_a.subpackage.PackageASubpackageObject()
        obj2 = package_a.PackageAObject(obj)
        f1 = self.temp()
        with PackageExporter(f1, verbose=False) as pe:
            pe.save_pickle("obj", "obj.pkl", obj)

        importer1 = PackageImporter(f1)
        loaded1 = importer1.load_pickle("obj", "obj.pkl")

        f2 = self.temp()
        pe = PackageExporter(f2, verbose=False)
        pe.importers.insert(0, importer1.import_module)
        with self.assertRaisesRegex(ModuleNotFoundError, 'torch.package'):
            pe.require_module(loaded1.__module__)
        with self.assertRaisesRegex(ModuleNotFoundError, 'torch.package'):
            pe.save_module(loaded1.__module__)

    def test_exporting_mismatched_code(self):
        """
        If an object with the same qualified name is loaded from different
        packages, the user should get an error if they try to re-save the
        object with the wrong package's source code.
        """
        import package_a.subpackage
        obj = package_a.subpackage.PackageASubpackageObject()
        obj2 = package_a.PackageAObject(obj)
        f1 = self.temp()
        with PackageExporter(f1, verbose=False) as pe:
            pe.save_pickle("obj", "obj.pkl", obj2)

        importer1 = PackageImporter(f1)
        loaded1 = importer1.load_pickle("obj", "obj.pkl")
        importer2 = PackageImporter(f1)
        loaded2 = importer2.load_pickle("obj", "obj.pkl")

        f2 = self.temp()

        def make_exporter():
            pe = PackageExporter(f2, verbose=False)
            # Ensure that the importer finds the 'PackageAObject' defined in 'importer1' first.
            pe.importers.insert(0, importer1.import_module)
            return pe

        # This should fail. The 'PackageAObject' type defined from 'importer1'
        # is not necessarily the same 'obj2's version of 'PackageAObject'.
        pe = make_exporter()
        with self.assertRaises(pickle.PicklingError):
            pe.save_pickle("obj", "obj.pkl", obj2)

        # This should also fail. The 'PackageAObject' type defined from 'importer1'
        # is not necessarily the same as the one defined from 'importer2'
        pe = make_exporter()
        with self.assertRaises(pickle.PicklingError):
            pe.save_pickle("obj", "obj.pkl", loaded2)

        # This should succeed. The 'PackageAObject' type defined from
        # 'importer1' is a match for the one used by loaded1.
        pe = make_exporter()
        pe.save_pickle("obj", "obj.pkl", loaded1)

    def test_unique_module_names(self):
        import package_a.subpackage
        obj = package_a.subpackage.PackageASubpackageObject()
        obj2 = package_a.PackageAObject(obj)
        f1 = self.temp()
        with PackageExporter(f1, verbose=False) as pe:
            pe.save_pickle("obj", "obj.pkl", obj2)

        importer1 = PackageImporter(f1)
        loaded1 = importer1.load_pickle("obj", "obj.pkl")
        importer2 = PackageImporter(f1)
        loaded2 = importer2.load_pickle("obj", "obj.pkl")

        # Modules from loaded packages should not shadow the names of modules.
        # See mangling.md for more info.
        self.assertNotEqual(type(obj2).__module__, type(loaded1).__module__)
        self.assertNotEqual(type(loaded1).__module__, type(loaded2).__module__)

    @skipIf(version_info < (3, 7), 'mock uses __getattr__ a 3.7 feature')
    def test_mock(self):
        filename = self.temp()
        with PackageExporter(filename, verbose=False) as he:
            he.mock(['package_a.subpackage', 'module_a'])
            he.save_module('package_a')
            he.require_module('package_a.subpackage')
            he.require_module('module_a')
        hi = PackageImporter(filename)
        import package_a.subpackage
        _ = package_a.subpackage
        import module_a
        _ = module_a

        m = hi.import_module('package_a.subpackage')
        r = m.result
        with self.assertRaisesRegex(NotImplementedError, 'was mocked out'):
            r()

    @skipIf(version_info < (3, 7), 'mock uses __getattr__ a 3.7 feature')
    def test_mock_glob(self):
        filename = self.temp()
        with PackageExporter(filename, verbose=False) as he:
            he.mock(['package_a.*', 'module*'])
            he.save_module('package_a')
            he.save_source_string('test_module', """\
import package_a.subpackage
import module_a
""")
        hi = PackageImporter(filename)
        import package_a.subpackage
        _ = package_a.subpackage
        import module_a
        _ = module_a

        m = hi.import_module('package_a.subpackage')
        r = m.result
        with self.assertRaisesRegex(NotImplementedError, 'was mocked out'):
            r()

    @skipIf(version_info < (3, 7), 'mock uses __getattr__ a 3.7 feature')
    def test_custom_requires(self):
        filename = self.temp()

        class Custom(PackageExporter):
            def require_module(self, name, dependencies):
                if name == 'module_a':
                    self.save_mock_module('module_a')
                elif name == 'package_a':
                    self.save_source_string('package_a', 'import module_a\nresult = 5\n')
                else:
                    raise NotImplementedError('wat')

        with Custom(filename, verbose=False) as he:
            he.save_source_string('main', 'import package_a\n')

        hi = PackageImporter(filename)
        hi.import_module('module_a').should_be_mocked
        bar = hi.import_module('package_a')
        self.assertEqual(bar.result, 5)

    @skipIfNoTorchVision
    def test_resnet(self):
        resnet = resnet18()

        f1 = self.temp()

        # create a package that will save it along with its code
        with PackageExporter(f1, verbose=False) as e:
            # put the pickled resnet in the package, by default
            # this will also save all the code files references by
            # the objects in the pickle
            e.save_pickle('model', 'model.pkl', resnet)

            # check th debug graph has something reasonable:
            buf = StringIO()
            debug_graph = e._write_dep_graph(failing_module='torch')
            self.assertIn('torchvision.models.resnet', debug_graph)

        # we can now load the saved model
        i = PackageImporter(f1)
        r2 = i.load_pickle('model', 'model.pkl')

        # test that it works
        input = torch.rand(1, 3, 224, 224)
        ref = resnet(input)
        self.assertTrue(torch.allclose(r2(input), ref))

        # functions exist also to get at the private modules in each package
        torchvision = i.import_module('torchvision')

        f2 = self.temp()
        # if we are doing transfer learning we might want to re-save
        # things that were loaded from a package
        with PackageExporter(f2, verbose=False) as e:
            # We need to tell the exporter about any modules that
            # came from imported packages so that it can resolve
            # class names like torchvision.models.resnet.ResNet
            # to their source code.

            e.importers.insert(0, i.import_module)

            # e.importers is a list of module importing functions
            # that by default contains importlib.import_module.
            # it is searched in order until the first success and
            # that module is taken to be what torchvision.models.resnet
            # should be in this code package. In the case of name collisions,
            # such as trying to save a ResNet from two different packages,
            # we take the first thing found in the path, so only ResNet objects from
            # one importer will work. This avoids a bunch of name mangling in
            # the source code. If you need to actually mix ResNet objects,
            # we suggest reconstructing the model objects using code from a single package
            # using functions like save_state_dict and load_state_dict to transfer state
            # to the correct code objects.
            e.save_pickle('model', 'model.pkl', r2)

        i2 = PackageImporter(f2)
        r3 = i2.load_pickle('model', 'model.pkl')
        self.assertTrue(torch.allclose(r3(input), ref))

        # test we can load from a directory
        import zipfile
        zf = zipfile.ZipFile(f1, 'r')

        with TemporaryDirectory() as td:
            zf.extractall(path=td)
            iz = PackageImporter(str(Path(td) / Path(f1).name))
            r4 = iz.load_pickle('model', 'model.pkl')
            self.assertTrue(torch.allclose(r4(input), ref))

    @skipIfNoTorchVision
    def test_model_save(self):

        # This example shows how you might package a model
        # so that the creator of the model has flexibility about
        # how they want to save it but the 'server' can always
        # use the same API to load the package.

        # The convension is for each model to provide a
        # 'model' package with a 'load' function that actual
        # reads the model out of the archive.

        # How the load function is implemented is up to the
        # the packager.

        # get our normal torchvision resnet
        resnet = resnet18()


        f1 = self.temp()
        # Option 1: save by pickling the whole model
        # + single-line, similar to torch.jit.save
        # - more difficult to edit the code after the model is created
        with PackageExporter(f1, verbose=False) as e:
            e.save_pickle('model', 'pickled', resnet)
            # note that this source is the same for all models in this approach
            # so it can be made part of an API that just takes the model and
            # packages it with this source.
            src = """\
import resources # gives you access to the importer from within the package

# server knows to call model.load() to get the model,
# maybe in the future it passes options as arguments by convension
def load():
    return resources.load_pickle('model', 'pickled')
        """
            e.save_source_string('model', src, is_package=True)

        f2 = self.temp()
        # Option 2: save with state dict
        # - more code to write to save/load the model
        # + but this code can be edited later to adjust adapt the model later
        with PackageExporter(f2, verbose=False) as e:
            e.save_pickle('model', 'state_dict', resnet.state_dict())
            src = """\
import resources # gives you access to the importer from within the package
from torchvision.models.resnet import resnet18
def load():
    # if you want, you can later edit how resnet is constructed here
    # to edit the model in the package, while still loading the original
    # state dict weights
    r = resnet18()
    state_dict = resources.load_pickle('model', 'state_dict')
    r.load_state_dict(state_dict)
    return r
        """
            e.save_source_string('model', src, is_package=True)



        # regardless of how we chose to package, we can now use the model in a server in the same way
        input = torch.rand(1, 3, 224, 224)
        results = []
        for m in [f1, f2]:
            importer = PackageImporter(m)
            the_model = importer.import_module('model').load()
            r = the_model(input)
            results.append(r)

        self.assertTrue(torch.allclose(*results))

    @skipIfNoTorchVision
    def test_script_resnet(self):
        resnet = resnet18()

        f1 = self.temp()
        # Option 1: save by pickling the whole model
        # + single-line, similar to torch.jit.save
        # - more difficult to edit the code after the model is created
        with PackageExporter(f1, verbose=False) as e:
            e.save_pickle('model', 'pickled', resnet)

        i = PackageImporter(f1)
        loaded = i.load_pickle('model', 'pickled')
        torch.jit.script(loaded)


    def test_module_glob(self):
        from torch.package.exporter import _GlobGroup

        def check(include, exclude, should_match, should_not_match):
            x = _GlobGroup(include, exclude)
            for e in should_match:
                self.assertTrue(x.matches(e))
            for e in should_not_match:
                self.assertFalse(x.matches(e))

        check('torch.*', [], ['torch.foo', 'torch.bar'], ['tor.foo', 'torch.foo.bar', 'torch'])
        check('torch.**', [], ['torch.foo', 'torch.bar', 'torch.foo.bar', 'torch'], ['what.torch', 'torchvision'])
        check('torch.*.foo', [], ['torch.w.foo'], ['torch.hi.bar.baz'])
        check('torch.**.foo', [], ['torch.w.foo', 'torch.hi.bar.foo'], ['torch.f.foo.z'])
        check('torch*', [], ['torch', 'torchvision'], ['torch.f'])
        check('torch.**', ['torch.**.foo'], ['torch', 'torch.bar', 'torch.barfoo'], ['torch.foo', 'torch.some.foo'])
        check('**.torch', [], ['torch', 'bar.torch'], ['visiontorch'])

    @skipIf(version_info < (3, 7), 'mock uses __getattr__ a 3.7 feature')
    def test_pickle_mocked(self):
        import package_a.subpackage
        obj = package_a.subpackage.PackageASubpackageObject()
        obj2 = package_a.PackageAObject(obj)

        filename = self.temp()
        with PackageExporter(filename, verbose=False) as he:
            he.mock(include='package_a.subpackage')
            he.save_pickle('obj', 'obj.pkl', obj2)

        hi = PackageImporter(filename)
        with self.assertRaises(NotImplementedError):
            hi.load_pickle('obj', 'obj.pkl')

    def test_inspect_class(self):
        """Should be able to retrieve source for a packaged class."""
        import package_a.subpackage
        buffer = BytesIO()
        obj = package_a.subpackage.PackageASubpackageObject()

        with PackageExporter(buffer, verbose=False) as pe:
            pe.save_pickle('obj', 'obj.pkl', obj)

        buffer.seek(0)
        pi = PackageImporter(buffer)
        packaged_class = pi.import_module('package_a.subpackage').PackageASubpackageObject
        regular_class = package_a.subpackage.PackageASubpackageObject

        packaged_src = inspect.getsourcelines(packaged_class)
        regular_src = inspect.getsourcelines(regular_class)
        self.assertEqual(packaged_src, regular_src)


class ManglingTest(TestCase):
    def test_unique_manglers(self):
        """
        Each mangler instance should generate a unique mangled name for a given input.
        """
        a = PackageMangler()
        b = PackageMangler()
        self.assertNotEqual(a.mangle("foo.bar"), b.mangle("foo.bar"))

    def test_mangler_is_consistent(self):
        """
        Mangling the same name twice should produce the same result.
        """
        a = PackageMangler()
        self.assertEqual(a.mangle("abc.def"), a.mangle("abc.def"))

    def test_roundtrip_mangling(self):
        a = PackageMangler()
        self.assertEqual("foo", demangle(a.mangle("foo")))

    def test_is_mangled(self):
        a = PackageMangler()
        b = PackageMangler()
        self.assertTrue(is_mangled(a.mangle("foo.bar")))
        self.assertTrue(is_mangled(b.mangle("foo.bar")))

        self.assertFalse(is_mangled("foo.bar"))
        self.assertFalse(is_mangled(demangle(a.mangle("foo.bar"))))

    def test_demangler_multiple_manglers(self):
        """
        PackageDemangler should be able to demangle name generated by any PackageMangler.
        """
        a = PackageMangler()
        b = PackageMangler()

        self.assertEqual("foo.bar", demangle(a.mangle("foo.bar")))
        self.assertEqual("bar.foo", demangle(b.mangle("bar.foo")))

    def test_mangle_empty_errors(self):
        a = PackageMangler()
        with self.assertRaises(AssertionError):
            a.mangle("")

    def test_demangle_base(self):
        """
        Demangling a mangle parent directly should currently return an empty string.
        """
        a = PackageMangler()
        mangled = a.mangle("foo")
        mangle_parent = mangled.partition(".")[0]
        self.assertEqual("", demangle(mangle_parent))

    def test_mangle_prefix(self):
        a = PackageMangler()
        mangled = a.mangle("foo.bar")
        mangle_prefix = get_mangle_prefix(mangled)
        self.assertEqual(mangle_prefix + "." + "foo.bar", mangled)


if __name__ == '__main__':
    run_tests()
