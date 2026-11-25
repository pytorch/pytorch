# Owner(s): ["module: dynamo"]
import unittest
from unittest import mock

import torch
from torch._dynamo.test_case import run_tests, TestCase
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
)


TEST_PROTOBUF = True
try:
    from google.protobuf import (
        descriptor_pb2 as _descriptor_pb2,
        descriptor_pool as _descriptor_pool,
    )
    from google.protobuf.internal import api_implementation, enum_type_wrapper
except ImportError:
    TEST_PROTOBUF = False


def _make_test_descriptors(speed=3):
    file = _descriptor_pb2.FileDescriptorProto(name="sample.proto", syntax="proto3")
    enum = file.enum_type.add(name="EgoFeatureIndex")
    enum.value.add(name="EGO_IS_VALID", number=0)
    enum.value.add(name="EGO_POSITION_X", number=1)
    enum.value.add(name="EGO_POSITION_Y", number=2)
    message = file.message_type.add(name="EgoFeatures")
    if speed is not None:
        enum.value.add(name="EGO_SPEED", number=speed)
        message.field.add(
            name="speed",
            number=speed,
            type=_descriptor_pb2.FieldDescriptorProto.TYPE_INT32,
        )
    return _descriptor_pool.DescriptorPool().AddSerializedFile(file.SerializeToString())


def _make_test_enum(speed=3):
    descriptor = _make_test_descriptors(speed)
    enum_desc = descriptor.enum_types_by_name["EgoFeatureIndex"]
    return enum_type_wrapper.EnumTypeWrapper(enum_desc)


if TEST_PROTOBUF:
    EgoFeatureIndex = _make_test_enum()


@unittest.skipIf(not TEST_PROTOBUF, "Missing protobuf")
@instantiate_parametrized_tests
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

    @parametrize("kind", ("enum", "message"))
    def test_descriptor_mapping_subscription(self, kind):
        def get_descriptor(speed):
            file = _make_test_descriptors(speed)
            if kind == "enum":
                return file.enum_types_by_name["EgoFeatureIndex"]
            return file.message_types_by_name["EgoFeatures"]

        def f(x, descriptor, key):
            try:
                if kind == "enum":
                    return x + descriptor.values_by_name[key].number
                return x + descriptor.fields_by_name[key].number
            except KeyError as exc:
                return x - (1 if exc.args == (key,) else 2)

        compiled = torch.compile(f, backend="eager", fullgraph=True, dynamic=False)
        x = torch.tensor([1, 2])
        key = "EGO_SPEED" if kind == "enum" else "speed"
        original = get_descriptor(3)
        for descriptor, name, offset in (
            (original, key, 3),
            (get_descriptor(7), key, 7),
            (get_descriptor(None), key, -1),
            (original, "missing", -1),
            (original, key, 3),
        ):
            self.assertEqual(compiled(x, descriptor, name), x + offset)
            with torch._dynamo.config.patch(error_on_recompile=True):
                self.assertEqual(compiled(x + 1, descriptor, name), x + 1 + offset)

    @parametrize(
        "key,offset",
        (("EGO_SPEED", 3), ("missing", -1), (b"EGO_SPEED", 3), (b"missing", -1)),
    )
    def test_native_map_replaced_subscription(self, key, offset):
        if api_implementation.Type() != "upb":
            self.skipTest("Requires the UPB descriptor map type")

        mapping = _make_test_enum().DESCRIPTOR.values_by_name
        replacement = mapping["EGO_POSITION_Y"]

        @torch.compile(backend="eager", fullgraph=True)
        def f(x, mapping, key):
            try:
                return x + mapping[key].number
            except KeyError:
                return x - 1

        x = torch.tensor([1, 2])
        self.assertEqual(f(x, mapping, key), x + offset)
        with mock.patch.object(
            type(mapping), "__getitem__", lambda mapping, key: replacement
        ):
            self.assertEqual(f(x, mapping, key), x + 2)
        self.assertEqual(f(x, mapping, key), x + offset)

    def test_native_map_key_with_python_hash(self):
        if api_implementation.Type() != "upb":
            self.skipTest("Requires the UPB descriptor map type")

        calls = []

        class Key:
            def __hash__(self):
                calls.append("hash")
                return 7

        @torch.compile(backend="eager")
        def f(x, mapping, key):
            try:
                return x + mapping[key].number
            except KeyError:
                return x - 1

        mapping = _make_test_enum().DESCRIPTOR.values_by_name
        x = torch.tensor([1, 2])
        self.assertEqual(f(x, mapping, Key()), x - 1)
        self.assertEqual(calls, ["hash"])

    def test_native_map_subscription_changed_in_frame(self):
        if api_implementation.Type() != "upb":
            self.skipTest("Requires the UPB descriptor map type")

        mapping = _make_test_enum().DESCRIPTOR.values_by_name
        replacement = mapping["EGO_POSITION_Y"]
        mapping_type = type(mapping)

        def replacement_getitem(mapping, key):
            return replacement

        def f(x, mapping, mapping_type):
            before = x + mapping["EGO_SPEED"].number
            mapping_type.__getitem__ = replacement_getitem
            return before, x + mapping["EGO_SPEED"].number

        x = torch.tensor([1, 2])
        with mock.patch.object(mapping_type, "__getitem__", mapping_type.__getitem__):
            compiled = torch.compile(f, backend="eager")
            self.assertEqual(compiled(x, mapping, mapping_type), (x + 3, x + 2))


if __name__ == "__main__":
    run_tests()
