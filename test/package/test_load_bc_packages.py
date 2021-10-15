from torch.package import PackageImporter
from torch.testing._internal.common_utils import run_tests
from pathlib import Path

try:
    from .common import PackageTestCase
except ImportError:
    # Support the case where we run this file directly.
    from common import PackageTestCase

packaging_directory = f'{Path(__file__).parent}/package_bc'

class TestLoadBCPackages(PackageTestCase):

    def test_load_bc_packages(self):
        importer1 = PackageImporter(f"{packaging_directory}/test_nn_module.pt")
        loaded1 = importer1.load_pickle("nn_module", "nn_module.pkl")
        importer2 = PackageImporter(f"{packaging_directory}/test_torchscript_module.pt")
        loaded2 = importer2.load_pickle("torchscript_module", "torchscript_module.pkl")
        importer3 = PackageImporter(f"{packaging_directory}/test_fx_module.pt")
        loaded3 = importer3.load_pickle("fx_module", "fx_module.pkl")

if __name__ == "__main__":
    run_tests()
