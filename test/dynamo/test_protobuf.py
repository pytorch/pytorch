# Owner(s): ["module: dynamo"]
import unittest

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


TEST_PROTOBUF = True
try:
    from .sample_pb2 import EgoFeatureIndex
except ImportError:
    TEST_PROTOBUF = False


@unittest.skipIf(not TEST_PROTOBUF, "Missing protobuf")
class ProtobufTests(TestCase):
    def test_enum_member(self):
        @torch.compile(backend="eager", fullgraph=True)
        def f():
            return EgoFeatureIndex.EGO_SPEED

        ret = f()
        self.assertEqual(ret, EgoFeatureIndex.EGO_SPEED)


if __name__ == "__main__":
    run_tests()
