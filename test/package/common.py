import io
import os
import sys
from tempfile import NamedTemporaryFile

from torch.testing._internal.common_utils import IS_WINDOWS, IS_FBCODE, IS_SANDCASTLE, TestCase


class PackageTestCase(TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._temporary_files = []

    def temp(self):
        # Use a buffer instead of a temporary file in fbcode.
        if IS_FBCODE or IS_SANDCASTLE:
            return io.BytesIO()

        t = NamedTemporaryFile()
        name = t.name
        if IS_WINDOWS:
            t.close()  # can't read an open file in windows
        else:
            self._temporary_files.append(t)
        return name

    def setUp(self):
        """Add test/package/ to module search path. This ensures that
        importing our fake packages via, e.g. `import package_a` will always
        work regardless of how we invoke the test.
        """
        super().setUp()
        self.package_test_dir = os.path.dirname(os.path.realpath(__file__))
        self.orig_sys_path = sys.path.copy()
        sys.path.append(self.package_test_dir)

    def tearDown(self):
        super().tearDown()
        sys.path = self.orig_sys_path

        # remove any temporary files
        for t in self._temporary_files:
            t.close()
        self._temporary_files = []
