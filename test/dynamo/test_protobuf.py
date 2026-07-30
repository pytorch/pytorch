# Owner(s): ["module: dynamo"]
import unittest

import torch
import torch._dynamo
import torch._dynamo.testing
from torch.testing._internal.common_utils import run_tests, TestCase


TEST_PROTOBUF = True
try:
    from google.protobuf import descriptor_pool as _descriptor_pool
    from google.protobuf.internal import enum_type_wrapper
except ImportError:
    TEST_PROTOBUF = False


def _make_test_enum():
    """Create a minimal protobuf enum from a serialized descriptor."""
    descriptor = _descriptor_pool.Default().AddSerializedFile(
        b"\n\x0csample.proto*Z\n\x0f\x45goFeatureIndex"
        b"\x12\x10\n\x0c\x45GO_IS_VALID\x10\x00"
        b"\x12\x12\n\x0e\x45GO_POSITION_X\x10\x01"
        b"\x12\x12\n\x0e\x45GO_POSITION_Y\x10\x02"
        b"\x12\r\n\tEGO_SPEED\x10\x03\x62\x06proto3"
    )
    enum_desc = descriptor.enum_types_by_name["EgoFeatureIndex"]
    return enum_type_wrapper.EnumTypeWrapper(enum_desc)


if TEST_PROTOBUF:
    EgoFeatureIndex = _make_test_enum()


@unittest.skipIf(not TEST_PROTOBUF, "Missing protobuf")
class ProtobufTests(TestCase):
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

    def test_enum_no_graph_break(self):
        """Verify protobuf enum access does not cause a graph break."""
        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt)
        def f(x):
            idx = EgoFeatureIndex.EGO_SPEED
            return x[:, idx : idx + 1]

        f(torch.randn(4, 5))
        self.assertEqual(cnt.frame_count, 1)

    def test_enum_invalid_member(self):
        """Accessing a nonexistent member should raise an error at trace time."""

        @torch.compile(backend="eager")
        def f():
            return EgoFeatureIndex.NONEXISTENT

        with self.assertRaises(AttributeError):
            f()

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
