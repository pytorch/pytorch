# Owner(s): ["module: meta tensors"]

from torch.testing._internal.common_utils import TestCase, run_tests
import torch
import itertools
from torch.testing._internal.jit_utils import RUN_CUDA
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.utils._python_dispatch import enable_torch_dispatch_mode
import unittest


class FakeTensorTest(TestCase):
    def test_basic(self):
        x = FakeTensor.from_tensor(torch.empty(2, 2, device="cpu"))
        y = x = FakeTensor.from_tensor(torch.empty(4, 2, 2, device="cpu"))
        y = x + x
        self.assertEqual(y.shape, (4, 2, 2))
        self.assertEqual(y.device, torch.device("cpu"))

    @unittest.skipIf(not RUN_CUDA, "requires cuda")
    def test_shape_take_not_device(self):
        x = FakeTensor.from_tensor(torch.empty(1, device="cpu"))
        y = FakeTensor.from_tensor(torch.empty(8, 8, device="cuda"))
        out = x.resize_as_(y)
        self.assertEqual(out.shape, (8, 8))
        self.assertEqual(out.device.type, "cpu")

    @unittest.skipIf(not RUN_CUDA, "requires cuda")
    def test_zero_dim(self):
        x = FakeTensor.from_tensor(torch.tensor(0.0))
        y = FakeTensor.from_tensor(torch.rand([4, 4], device="cuda"))
        out = x + y
        self.assertEqual(out.shape, (4, 4))
        self.assertEqual(out.device, y.device)

    @unittest.skipIf(not RUN_CUDA, "requires cuda")
    def test_throw(self):
        x = FakeTensor.from_tensor(torch.tensor(0.0))
        y = FakeTensor.from_tensor(torch.rand([4, 4], device="cuda"))
        z = FakeTensor.from_tensor(torch.rand([4, 4], device="cpu"))
        self.assertRaises(Exception, lambda: torch.lerp(x, y, z))

    def test_dispatch_device(self):
        x = FakeTensor.from_tensor(torch.rand([4, 4]))
        self.assertEqual(x.device.type, "cpu")

    @unittest.skipIf(not RUN_CUDA, "requires cuda")
    def test_type_as(self):
        x = FakeTensor.from_tensor(torch.rand([16, 1], device="cpu"))
        y = FakeTensor.from_tensor(torch.rand([4, 4], device="cuda"))
        out = x.type_as(y)
        self.assertEqual(out.device.type, "cuda")

    def test_constructor(self):
        with enable_torch_dispatch_mode(FakeTensorMode(inner=None)):
            x = torch.rand([4, 4], device="cpu")

        self.assertTrue(isinstance(x, FakeTensor))
        self.assertTrue(x.device.type == "cpu")

    def test_mode(self):
        x = FakeTensor.from_tensor(torch.rand([1]))
        with enable_torch_dispatch_mode(FakeTensorMode(inner=None)):
            y = torch.rand([4], device="cpu")
            out = x + y

        self.assertTrue(isinstance(y, FakeTensor))

    def test_fake_mode_error(self):
        x = torch.rand([4, 4])

        with self.assertRaisesRegex(Exception, "non-Fake Tensor inputs"):
            with enable_torch_dispatch_mode(FakeTensorMode(inner=None)):
                y = x[0]


def contains_type(type: torch._C.Type, maybe_contained_type: torch._C.Type):
    return maybe_contained_type.isSubtypeOf(type) or any(
        contains_type(e, maybe_contained_type) for e in type.containedTypes()
    )


class FakeTensorOperatorInvariants(TestCase):
    @staticmethod
    def get_aten_op(schema):
        namespace, name = schema.name.split("::")
        overload = schema.overload_name if schema.overload_name else "default"
        assert namespace == "aten"
        return getattr(getattr(torch.ops.aten, name), overload)

    def test_non_kwarg_only_device(self):

        for schema in torch._C._jit_get_all_schemas():
            namespace = schema.name.split("::")[0]
            if namespace != "aten":
                continue

            ten_type = torch._C.TensorType.get()
            if not any(
                contains_type(arg.type, ten_type)
                for arg in itertools.chain(schema.arguments, schema.returns)
            ):
                continue

            opt_device = torch._C.OptionalType(torch._C.DeviceObjType.get())
            has_non_kwarg_device = any(
                not arg.kwarg_only and arg.type.isSubtypeOf(opt_device)
                for arg in schema.arguments
            )
            if has_non_kwarg_device:
                self.assertTrue(
                    self.get_aten_op(schema) in torch._subclasses.fake_tensor._device_not_kwarg_ops
                )

    def test_tensor_constructors_all_have_kwarg_device(self):
        for schema in torch._C._jit_get_all_schemas():
            namespace = schema.name.split("::")[0]
            if namespace != "aten":
                continue

            op = self.get_aten_op(schema)
            if not torch._subclasses.fake_tensor._is_tensor_constructor(op):
                continue

            opt_device = torch._C.OptionalType(torch._C.DeviceObjType.get())
            has_kwarg_device = any(
                arg.kwarg_only and arg.type.isSubtypeOf(opt_device)
                for arg in schema.arguments
            )

            self.assertTrue(
                has_kwarg_device or op == torch.ops.aten._list_to_tensor.default
            )


if __name__ == "__main__":
    run_tests()
