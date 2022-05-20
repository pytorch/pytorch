# Owner(s): ["module: unknown"]

from torch.testing._internal.common_utils import TestCase, run_tests
import torch
import itertools
from torch.testing._internal.jit_utils RUN_CUDA
from torch._subclasses import FakeTensor

from torch._subclasses.fake_tensor import FakeTensor

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


def contains_type(type: torch._C.Type, maybe_contained_type: torch._C.Type):
    return maybe_contained_type.isSubtypeOf(type) or any(
        contains_type(e, maybe_contained_type) for e in type.containedTypes()
    )


class FakeTensorOperatorInvariants(TestCase):
    def test_non_kwarg_only_device(self):
        def get_op(schema):
            namespace, name = schema.name.split("::")
            overload = schema.overload_name if schema.overload_name else "default"
            assert namespace == "aten"
            return getattr(getattr(torch.ops.aten, name), overload)

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
                    get_op(schema) in torch._subclasses.fake_tensor._device_not_kwarg_ops
                )


if __name__ == "__main__":
    run_tests()
