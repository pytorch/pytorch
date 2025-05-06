# Owner(s): ["module: inductor"]

import dataclasses

from torch.testing._internal.common_utils import TestCase
from torch.utils._appending_byte_serializer import (
    AppendingByteSerializer,
    BytesReader,
    BytesWriter,
)


class TestAppendingByteSerializer(TestCase):
    def test_write_and_read_int(self) -> None:
        def int_serializer(writer: BytesWriter, i: int) -> None:
            writer.write_uint64(i)

        def int_deserializer(reader: BytesReader) -> int:
            return reader.read_uint64()

        s = AppendingByteSerializer(serialize_fn=int_serializer)

        data = [1, 2, 3, 4]
        s.extend(data)
        self.assertListEqual(
            data,
            AppendingByteSerializer.to_list(
                s.to_bytes(), deserialize_fn=int_deserializer
            ),
        )

        data2 = [8, 9, 10, 11]
        s.extend(data2)
        self.assertListEqual(
            data + data2,
            AppendingByteSerializer.to_list(
                s.to_bytes(), deserialize_fn=int_deserializer
            ),
        )

    def test_write_and_read_class(self) -> None:
        @dataclasses.dataclass(frozen=True, eq=True)
        class Foo:
            x: int
            y: str
            z: bytes

            @staticmethod
            def serialize(writer: BytesWriter, cls: "Foo") -> None:
                writer.write_uint64(cls.x)
                writer.write_str(cls.y)
                writer.write_bytes(cls.z)

            @staticmethod
            def deserialize(reader: BytesReader) -> "Foo":
                x = reader.read_uint64()
                y = reader.read_str()
                z = reader.read_bytes()
                return Foo(x, y, z)

        a = Foo(5, "ok", bytes([15]))
        b = Foo(10, "lol", bytes([25]))

        s = AppendingByteSerializer(serialize_fn=Foo.serialize)
        s.append(a)
        self.assertListEqual(
            [a],
            AppendingByteSerializer.to_list(
                s.to_bytes(), deserialize_fn=Foo.deserialize
            ),
        )

        s.append(b)
        self.assertListEqual(
            [a, b],
            AppendingByteSerializer.to_list(
                s.to_bytes(), deserialize_fn=Foo.deserialize
            ),
        )

    def test_checksum(self) -> None:
        writer = BytesWriter()
        writer.write_str("test")
        b = writer.to_bytes()
        b = bytearray(b)
        b[0:1] = b"\x00"
        b = bytes(b)

        with self.assertRaisesRegex(
            RuntimeError, r"Bytes object is corrupted, checksum does not match.*"
        ):
            BytesReader(b)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
