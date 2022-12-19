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


class TestPackageCompile(PackageTestCase):
    """Tests for compatibility with TorchScript."""

    def test_compile_resnet_after_package(self):
        resnet = resnet18()
        f1 = self.temp()

        # create a package that will save it along with its code
        with PackageExporter(f1) as e:
            # put the pickled resnet in the package, by default
            # this will also save all the code files references by
            # the objects in the pickle
            e.extern(["sys", "numpy.**", "mkl.**", "PIL.**"])
            e.intern("**")
            e.save_pickle("model", "model.pkl", resnet)

        # we can now load the saved model
        i = PackageImporter(f1)
        r2 = i.load_pickle("model", "model.pkl")
        r2c = torch.compile(r2)
        # test that it works
        input = torch.rand(1, 3, 224, 224)
        ref = resnet(input)
        self.assertEqual(r2c(input), ref)

    def test_compile_resnet_before_package(self):
        resnet = resnet18()
        cresnet = torch.compile(resnet)
        f1 = self.temp()

        # create a package that will save it along with its code
        with PackageExporter(f1) as e:
            # put the pickled resnet in the package, by default
            # this will also save all the code files references by
            # the objects in the pickle
            e.extern(["sys", "numpy.**", "mkl.**", "PIL.**"])
            e.intern("**")
            e.save_pickle("model", "model.pkl", cresnet)

        # we can now load the saved model
        i = PackageImporter(f1)
        r2 = i.load_pickle("model", "model.pkl")
        # test that it works
        input = torch.rand(1, 3, 224, 224)
        ref = resnet(input)
        self.assertEqual(r2(input), ref)



if __name__ == "__main__":
    run_tests()
