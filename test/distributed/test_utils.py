# Owner(s): ["oncall: distributed"]

import io

import torch.distributed as dist
from torch.distributed.checkpoint.utils import _create_file_view


if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

from torch.testing._internal.common_utils import (
    TestCase,
    run_tests,
)

class TestReaderView(TestCase):
    def setUp(self):
        buffer = io.BytesIO(bytearray(range(ord('A'), ord('Z') + 1)))
        self.front_view = _create_file_view(buffer, 0, 5)

        buffer = io.BytesIO(bytearray(range(ord('A'), ord('Z') + 1)))
        self.middle_view = _create_file_view(buffer, 10, 5)

        buffer = io.BytesIO(bytearray(range(ord('A'), ord('Z') + 1)))
        self.back_view = _create_file_view(buffer, len(buffer.getbuffer()) - 5, 5)

    def testShortRead(self):
        self.assertEqual(self.front_view.read(3), b"ABC")
        self.assertEqual(self.middle_view.read(3), b"KLM")
        self.assertEqual(self.back_view.read(3), b"VWX")

    def testLongRead(self):
        self.assertEqual(self.front_view.read(10), b"ABCDE")
        self.assertEqual(self.middle_view.read(10), b"KLMNO")
        self.assertEqual(self.back_view.read(10), b"VWXYZ")

    def testAllRead(self):
        self.assertEqual(self.front_view.read(-1), b"ABCDE")
        self.assertEqual(self.middle_view.read(-1), b"KLMNO")
        self.assertEqual(self.back_view.read(-1), b"VWXYZ")

    def testShortReadinto(self):
        ba = bytearray(3)

        self.assertEqual(self.front_view.readinto(ba), 3)
        self.assertEqual(ba, b"ABC")

        self.assertEqual(self.middle_view.readinto(ba), 3)
        self.assertEqual(ba, b"KLM")

        self.assertEqual(self.back_view.readinto(ba), 3)
        self.assertEqual(ba, b"VWX")

    def testLongReadinto(self):
        ba = bytearray(8)
        self.assertEqual(self.front_view.readinto(ba), 5)
        self.assertEqual(ba, b"ABCDE\0\0\0")
        self.assertEqual(self.front_view.readinto(ba), 0)
        self.assertEqual(ba, b"ABCDE\0\0\0")

        self.assertEqual(self.middle_view.readinto(ba), 5)
        self.assertEqual(ba, b"KLMNO\0\0\0")
        self.assertEqual(self.middle_view.readinto(ba), 0)
        self.assertEqual(ba, b"KLMNO\0\0\0")

        self.assertEqual(self.back_view.readinto(ba), 5)
        self.assertEqual(ba, b"VWXYZ\0\0\0")
        self.assertEqual(self.back_view.readinto(ba), 0)
        self.assertEqual(ba, b"VWXYZ\0\0\0")


if __name__ == "__main__":
    run_tests()
