# Owner(s): ["module: dynamo"]
import unittest

import torch
from torch._dynamo.test_case import run_tests, TestCase
from torch.testing._internal.common_utils import HardwareClassification


TEST_PROTOBUF = True
try:
    from google.protobuf import (
        descriptor_pb2 as _descriptor_pb2,
        descriptor_pool as _descriptor_pool,
    )
    from google.protobuf.internal import enum_type_wrapper
except ImportError:
    TEST_PROTOBUF = False


def _make_test_enum(speed=3):
    """Create a minimal protobuf enum in a private descriptor pool."""
    file = _descriptor_pb2.FileDescriptorProto(name="sample.proto", syntax="proto3")
    enum = file.enum_type.add(name="EgoFeatureIndex")
    enum.value.add(name="EGO_IS_VALID", number=0)
    enum.value.add(name="EGO_POSITION_X", number=1)
    enum.value.add(name="EGO_POSITION_Y", number=2)
    if speed is not None:
        enum.value.add(name="EGO_SPEED", number=speed)
    descriptor = _descriptor_pool.DescriptorPool().AddSerializedFile(
        file.SerializeToString()
    )
    enum_desc = descriptor.enum_types_by_name["EgoFeatureIndex"]
    return enum_type_wrapper.EnumTypeWrapper(enum_desc)


if TEST_PROTOBUF:
    EgoFeatureIndex = _make_test_enum()


@unittest.skipIf(not TEST_PROTOBUF, "Missing protobuf")
class ProtobufTests(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_enum_member_access(self):
        """Access a protobuf enum member inside torch.compile (fullgraph)."""

        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            idx = EgoFeatureIndex.EGO_SPEED
            return x[:, idx : idx + 1]

        x = torch.randn(4, 5)
        result = f(x)
        self.assertEqual(result, x[:, 3:4])

    def test_enum_member_as_constant(self):
        """Protobuf enum value should be treated as a constant int."""

        @torch.compile(backend="eager", fullgraph=True)
        def f():
            return EgoFeatureIndex.EGO_SPEED

        self.assertEqual(f(), 3)

    def test_enum_multiple_members(self):
        """Access multiple protobuf enum members in a single compiled function."""

        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            px = EgoFeatureIndex.EGO_POSITION_X
            py = EgoFeatureIndex.EGO_POSITION_Y
            return x[:, px] + x[:, py]

        x = torch.randn(4, 5)
        result = f(x)
        self.assertEqual(result, x[:, 1] + x[:, 2])

    def test_enum_invalid_member(self):
        """A compiled caller can handle a missing member as AttributeError."""

        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            try:
                return x + EgoFeatureIndex.NONEXISTENT
            except AttributeError:
                return x + 1

        x = torch.tensor([1, 2])
        self.assertEqual(f(x), x + 1)

    def test_enum_member_guard_after_mutation(self):
        @torch.compile(backend="eager", fullgraph=True, dynamic=False)
        def f(x, enum):
            try:
                return x + enum.EGO_SPEED
            except AttributeError:
                return x + 19

        x = torch.tensor([1, 2])
        enum = _make_test_enum()
        original = enum.DESCRIPTOR
        self.assertEqual(f(x, enum), x + 3)
        enum._enum_type = _make_test_enum(speed=7).DESCRIPTOR
        self.assertEqual(f(x, enum), x + 7)
        enum.EGO_SPEED = 11
        self.assertEqual(f(x, enum), x + 11)
        del enum.EGO_SPEED
        enum._enum_type = _make_test_enum(speed=None).DESCRIPTOR
        self.assertEqual(f(x, enum), x + 19)
        enum._enum_type = original
        self.assertEqual(f(x, enum), x + 3)

    def test_enum_member_in_arithmetic(self):
        """Use protobuf enum value in tensor arithmetic."""

        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            speed = EgoFeatureIndex.EGO_SPEED
            valid = EgoFeatureIndex.EGO_IS_VALID
            return x * speed + valid

        x = torch.tensor([1.0, 2.0, 3.0])
        result = f(x)
        expected = x * 3 + 0
        self.assertEqual(result, expected)


if __name__ == "__main__":
    run_tests()
