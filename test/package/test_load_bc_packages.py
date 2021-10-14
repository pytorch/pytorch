import pickle
import torch
from package_a import PackageAObject
from torch.package import PackageExporter, PackageImporter, sys_importer
from torch.fx import symbolic_trace
from pathlib import Path
import time
from package_a.test_nn_module import TestNnModule
from torch.testing._internal.common_utils import run_tests, IS_FBCODE, IS_SANDCASTLE
from unittest import skipIf
import os

try:
    from .common import PackageTestCase
except ImportError:
    # Support the case where we run this file directly.
    from common import PackageTestCase

packaging_directory = f'{Path(__file__).parent}/package_bc'
# packaging_directory = f'/home/sahanp/pytorch/test/package/package_bc'
torch.package.package_exporter._gate_torchscript_serialization = False
# test nn module uses DCGANGenerator
class TestLoadBCPackages(PackageTestCase):
    def generate_bc_packages(self):

        test_nn_module = TestNnModule()
        test_torchscript_module = torch.jit.script(TestNnModule())
        test_fx_module : torch.fx.GraphModule = symbolic_trace(TestNnModule())
        with PackageExporter(f"{packaging_directory}/test_nn_module.pt") as pe1:
            pe1.intern("**")
            pe1.save_pickle("nn_module","nn_module.pkl",test_nn_module)
        with PackageExporter(f"{packaging_directory}/test_torchscript_module.pt") as pe2:
            pe2.intern("**")
            pe2.save_pickle("torchscript_module","torchscript_module.pkl",test_torchscript_module)
        with PackageExporter(f"{packaging_directory}/test_fx_module.pt") as pe3:
            pe3.intern("**")
            pe3.save_pickle("fx_module","fx_module.pkl",test_fx_module)

    def test_load_bc_packages(self):
        self.generate_bc_packages()
        importer1 = PackageImporter(f"{packaging_directory}/test_nn_module.pt")
        loaded1 = importer1.load_pickle("nn_module","nn_module.pkl")
        importer2 = PackageImporter(f"{packaging_directory}/test_torchscript_module.pt")
        loaded2 = importer2.load_pickle("torchscript_module","torchscript_module.pkl")
        importer3 = PackageImporter(f"{packaging_directory}/test_fx_module.pt")
        loaded3 = importer3.load_pickle("fx_module","fx_module.pkl")

        # delete files in directory for next time
        for f in os.listdir(packaging_directory):
            os.remove(os.path.join(packaging_directory, f))

if __name__ == "__main__":
    run_tests()
