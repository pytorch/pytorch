# Owner(s): ["module: unknown"]

import glob
import os
import unittest

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


site_packages = os.path.dirname(os.path.dirname(torch.__file__))
distinfo = glob.glob(os.path.join(site_packages, "torch-*dist-info"))


class TestLicense(TestCase):
    @unittest.skipIf(len(distinfo) == 0, "no installation in site-package to test")
    def test_distinfo_license(self):
        """If pytorch is installed via a wheel, third-party licenses live at
        site-packages/torch-*.dist-info/licenses/third_party/<lib>/<file>
        per PEP 639. Verify at least one is shipped."""
        if len(distinfo) > 1:
            raise AssertionError(
                'Found too many "torch-*dist-info" directories '
                f'in "{site_packages}, expected only one'
            )
        third_party = os.path.join(distinfo[0], "licenses", "third_party")
        files = [
            f
            for f in glob.glob(os.path.join(third_party, "**"), recursive=True)
            if os.path.isfile(f)
        ]
        self.assertTrue(
            files,
            f"No third-party license files found under {third_party}",
        )


if __name__ == "__main__":
    run_tests()
