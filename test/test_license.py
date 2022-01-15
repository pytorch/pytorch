# Owner(s): ["module: unknown"]

import glob
import io
import os
import unittest

import torch
from torch.testing._internal.common_utils import TestCase, run_tests


try:
    from third_party.build_bundled import create_bundled
except ImportError:
    create_bundled = None

license_file = 'third_party/LICENSES_BUNDLED.txt'
starting_txt = 'The Pytorch repository and source distributions bundle'
site_packages = os.path.dirname(os.path.dirname(torch.__file__))
distinfo = glob.glob(os.path.join(site_packages, 'torch-*dist-info'))

class TestLicense(TestCase):

    @unittest.skipIf(not create_bundled, "can only be run in a source tree")
    def test_license_for_wheel(self):
        current = io.StringIO()
        create_bundled('third_party', current)
        with open(license_file) as fid:
            src_tree = fid.read()
        if not src_tree == current.getvalue():
            raise AssertionError(
                f'the contents of "{license_file}" do not '
                'match the current state of the third_party files. Use '
                '"python third_party/build_bundled.py" to regenerate it')

    @unittest.skipIf(len(distinfo) == 0, "no installation in site-package to test")
    def test_distinfo_license(self):
        """If run when pytorch is installed via a wheel, the license will be in
        site-package/torch-*dist-info/LICENSE. Make sure it contains the third
        party bundle of licenses"""

        if len(distinfo) > 1:
            raise AssertionError('Found too many "torch-*dist-info" directories '
                                 f'in "{site_packages}, expected only one')
        with open(os.path.join(os.path.join(distinfo[0], 'LICENSE'))) as fid:
            txt = fid.read()
            self.assertTrue(starting_txt in txt)

if __name__ == '__main__':
    run_tests()
