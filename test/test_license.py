import io
import unittest

from torch.testing._internal.common_utils import TestCase, run_tests


try:
    from third_party.build_bundled import create_bundled
except ImportError:
    create_bundled = None

license_file = 'third_party/LICENSES_BUNDLED.txt'

class TestLicense(TestCase):

    @unittest.skipIf(not create_bundled, "can only be run in a source tree")
    def test_license_in_wheel(self):
        current = io.StringIO()
        create_bundled('third_party', current)
        with open(license_file) as fid:
            src_tree = fid.read()
        if not src_tree == current.getvalue():
            raise AssertionError(
                f'the contents of "{license_file}" do not '
                'match the current state of the third_party files. Use '
                '"python third_party/build_bundled.py" to regenerate it')


if __name__ == '__main__':
    run_tests()
