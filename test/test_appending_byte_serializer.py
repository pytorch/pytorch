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
        def int_serializer(i: int) -> bytes:
            writer = BytesWriter()
            writer.write_int(i)
            return writer.get()

        def int_deserializer(data: bytes) -> int:
            reader = BytesReader(data)
            return reader.read_int()

        s = AppendingByteSerializer(serialize_fn=int_serializer)

        data = [1, 2, 3, 4]
        out = s.appendListAndGet(data)
        self.assertListEqual(
            data, AppendingByteSerializer.to_list(out, deserialize_fn=int_deserializer)
        )

        data2 = [8, 9, 10, 11]
        out = s.appendListAndGet(data2)
        self.assertListEqual(
            data + data2,
            AppendingByteSerializer.to_list(out, deserialize_fn=int_deserializer),
        )

    def test_write_and_read_class(self) -> None:
        @dataclasses.dataclass(frozen=True, eq=True)
        class Foo:
            x: int
            y: str
            z: bytes

            @staticmethod
            def serialize(cls: "Foo") -> bytes:
                writer = BytesWriter()
                writer.write_int(cls.x)
                writer.write_str(cls.y)
                writer.write_int(len(cls.z))
                writer.write_bytes(cls.z)
                return writer.get()

            @staticmethod
            def deserialize(data: bytes) -> "Foo":
                reader = BytesReader(data)
                x = reader.read_int()
                y = reader.read_str()
                bytes_len = reader.read_int()
                z = reader.read_bytes(bytes_len)

                return Foo(x, y, z)

        a = Foo(5, "ok", bytes([15]))
        b = Foo(10, "lol", bytes([25]))

        s = AppendingByteSerializer(serialize_fn=Foo.serialize)
        s.append(a)
        self.assertListEqual(
            [a],
            AppendingByteSerializer.to_list(s.get(), deserialize_fn=Foo.deserialize),
        )

        s.append(b)
        self.assertListEqual(
            [a, b],
            AppendingByteSerializer.to_list(s.get(), deserialize_fn=Foo.deserialize),
        )


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
