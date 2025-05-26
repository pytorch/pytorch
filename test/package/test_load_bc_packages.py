# Owner(s): ["oncall: package/deploy"]

from pathlib import Path
from unittest import skipIf

from torch.package import PackageImporter
from torch.testing._internal.common_utils import IS_FBCODE, IS_SANDCASTLE, run_tests


try:
    from .common import PackageTestCase
except ImportError:
    # Support the case where we run this file directly.
    from common import PackageTestCase

packaging_directory = f"{Path(__file__).parent}/package_bc"


class TestLoadBCPackages(PackageTestCase):
    """Tests for checking loading has backwards compatiblity"""

    @skipIf(
        IS_FBCODE or IS_SANDCASTLE,
        "Tests that use temporary files are disabled in fbcode",
    )
    def test_load_bc_packages_nn_module(self):
        """Tests for backwards compatible nn module"""
        importer1 = PackageImporter(f"{packaging_directory}/test_nn_module.pt")
        importer1.load_pickle("nn_module", "nn_module.pkl")

    @skipIf(
        IS_FBCODE or IS_SANDCASTLE,
        "Tests that use temporary files are disabled in fbcode",
    )
    def test_load_bc_packages_torchscript_module(self):
        """Tests for backwards compatible torchscript module"""
        importer2 = PackageImporter(f"{packaging_directory}/test_torchscript_module.pt")
        importer2.load_pickle("torchscript_module", "torchscript_module.pkl")

    @skipIf(
        IS_FBCODE or IS_SANDCASTLE,
        "Tests that use temporary files are disabled in fbcode",
    )
    def test_load_bc_packages_fx_module(self):
        """Tests for backwards compatible fx module"""
        importer3 = PackageImporter(f"{packaging_directory}/test_fx_module.pt")
        importer3.load_pickle("fx_module", "fx_module.pkl")


if __name__ == "__main__":
    run_tests()
