from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from textwrap import dedent
from unittest import skipIf

import torch
from torch.package import PackageExporter, PackageImporter, sys_importer
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


@skipIfNoTorchVision
class ModelTest(PackageTestCase):
    """End-to-end tests packaging an entire model."""

    def test_resnet(self):
        resnet = resnet18()

        f1 = self.temp()

        # create a package that will save it along with its code
        with PackageExporter(f1, verbose=False) as e:
            # put the pickled resnet in the package, by default
            # this will also save all the code files references by
            # the objects in the pickle
            e.save_pickle("model", "model.pkl", resnet)

            # check th debug graph has something reasonable:
            buf = StringIO()
            debug_graph = e._write_dep_graph(failing_module="torch")
            self.assertIn("torchvision.models.resnet", debug_graph)

        # we can now load the saved model
        i = PackageImporter(f1)
        r2 = i.load_pickle("model", "model.pkl")

        # test that it works
        input = torch.rand(1, 3, 224, 224)
        ref = resnet(input)
        self.assertTrue(torch.allclose(r2(input), ref))

        # functions exist also to get at the private modules in each package
        torchvision = i.import_module("torchvision")

        f2 = self.temp()
        # if we are doing transfer learning we might want to re-save
        # things that were loaded from a package.
        # We need to tell the exporter about any modules that
        # came from imported packages so that it can resolve
        # class names like torchvision.models.resnet.ResNet
        # to their source code.
        with PackageExporter(f2, verbose=False, importer=(i, sys_importer)) as e:
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
            e.save_pickle("model", "model.pkl", r2)

        i2 = PackageImporter(f2)
        r3 = i2.load_pickle("model", "model.pkl")
        self.assertTrue(torch.allclose(r3(input), ref))

        # test we can load from a directory
        import zipfile

        zf = zipfile.ZipFile(f1, "r")

        with TemporaryDirectory() as td:
            zf.extractall(path=td)
            iz = PackageImporter(str(Path(td) / Path(f1).name))
            r4 = iz.load_pickle("model", "model.pkl")
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
            e.save_pickle("model", "pickled", resnet)
            # note that this source is the same for all models in this approach
            # so it can be made part of an API that just takes the model and
            # packages it with this source.
            src = dedent(
                """\
                import importlib
                import torch_package_importer as resources

                # server knows to call model.load() to get the model,
                # maybe in the future it passes options as arguments by convension
                def load():
                    return resources.load_pickle('model', 'pickled')
                """
            )
            e.save_source_string("model", src, is_package=True)

        f2 = self.temp()
        # Option 2: save with state dict
        # - more code to write to save/load the model
        # + but this code can be edited later to adjust adapt the model later
        with PackageExporter(f2, verbose=False) as e:
            e.save_pickle("model", "state_dict", resnet.state_dict())
            src = dedent(
                """\
                import importlib
                import torch_package_importer as resources

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
            )
            e.save_source_string("model", src, is_package=True)

        # regardless of how we chose to package, we can now use the model in a server in the same way
        input = torch.rand(1, 3, 224, 224)
        results = []
        for m in [f1, f2]:
            importer = PackageImporter(m)
            the_model = importer.import_module("model").load()
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
            e.save_pickle("model", "pickled", resnet)

        i = PackageImporter(f1)
        loaded = i.load_pickle("model", "pickled")
        torch.jit.script(loaded)


if __name__ == "__main__":
    run_tests()
