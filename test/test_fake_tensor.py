# Owner(s): ["module: meta tensors"]

from torch.testing._internal.common_utils import TestCase, run_tests, skipIfCrossRef, skipIfRocm
import torch
import itertools
import numpy as np
from torch.testing._internal.jit_utils import RUN_CUDA
from torch._subclasses.fake_tensor import (
    FakeTensor,
    FakeTensorMode,
    FakeTensorConverter,
    DynamicOutputShapeException,
)
from torch.testing import FileCheck
from torch import nn
import unittest
import torch._prims as prims
import contextlib
import weakref
import copy

class FakeTensorTest(TestCase):
    def checkType(self, t, device_str, size):
        self.assertTrue(isinstance(t, FakeTensor))
        self.assertEqual(t.device.type, device_str)
        self.assertEqual(list(t.size()), size)

    def test_basic(self):
        x = torch.empty(2, 2, device="cpu")
        y = torch.empty(4, 2, 2, device="cpu")
        with FakeTensorMode() as mode:
            x = mode.from_tensor(x)
            y = mode.from_tensor(y)
            z = x + y
            self.assertEqual(z.shape, (4, 2, 2))
            self.assertEqual(z.device, torch.device("cpu"))
            self.assertTrue(isinstance(z, FakeTensor))

    def test_parameter_instantiation(self):
        with FakeTensorMode():
            x = torch.rand([4])
            y = torch.nn.parameter.Parameter(x)
            self.assertTrue(isinstance(y, torch.nn.Parameter))

    def test_non_parameter_grad(self):
        mode = FakeTensorMode()
        t = torch.rand([4], requires_grad=True)
        fake_t = mode.from_tensor(t)
        self.assertEqual(fake_t.requires_grad, t.requires_grad)

    @unittest.skipIf(not RUN_CUDA, "requires cuda")
    def test_index_cuda_with_cpu(self):
        with FakeTensorMode():
            x = torch.rand([2048], device='cuda')
            out = x[torch.zeros([36], dtype=torch.int64)]
            self.checkType(out, "cuda", [36])

    @unittest.skipIf(not RUN_CUDA, "requires cuda")
    def test_shape_take_not_device(self):
        with FakeTensorMode():
            x = torch.empty(1, device="cpu")
            y = torch.empty(8, 8, device="cuda")
            out = x.resize_as_(y)
            self.assertEqual(out.shape, (8, 8))
            self.assertEqual(out.device.type, "cpu")
            self.assertTrue(isinstance(out, FakeTensor))

    @unittest.skipIf(not RUN_CUDA, "requires cuda")
    def test_zero_dim(self):
        with FakeTensorMode() as mode:
            x = torch.tensor(0.)
            y = torch.rand([4, 4], device="cuda")
            out = x + y
            self.assertEqual(out.shape, (4, 4))
            self.assertEqual(out.device, y.device)
            self.assertTrue(isinstance(out, FakeTensor))

    def test_nan_to_num(self):
        with FakeTensorMode():
            for dtype in [torch.float16, torch.float32]:
                x = torch.rand([4], dtype=dtype)
                y = torch.nan_to_num(x, nan=None)
                z = torch.nan_to_num(x, 0.0)
                self.assertEqual(dtype, y.dtype)
                self.assertEqual(dtype, z.dtype)

    @unittest.skipIf(not RUN_CUDA, "requires cuda")
    def test_throw(self):
        x = torch.tensor(0.)  # TODO: tensor() errors
        with FakeTensorMode() as mode:
            x_conv = mode.from_tensor(x)
            y = torch.rand([4, 4], device="cuda")
            z = torch.rand([4, 4], device="cpu")
            self.assertRaises(Exception, lambda: torch.lerp(x_conv, y, z))

    @unittest.skipIf(not RUN_CUDA, "requires cuda")
    def test_type_as(self):
        with FakeTensorMode():
            x = torch.rand([16, 1], device="cpu")
            y = torch.rand([4, 4], device="cuda")
            out = x.type_as(y)
            self.assertEqual(out.device.type, "cuda")
            self.assertTrue(isinstance(out, FakeTensor))

    @unittest.skipIf(not RUN_CUDA, "requires cuda")
    def test_setitem(self):
        for device in ["cpu", "cuda"]:
            with FakeTensorMode():
                x = torch.rand([16, 1], device=device)
                x[..., 0] = 0

    def test_fake_dispatch_keys(self):
        with FakeTensorMode():
            x = torch.rand([4])
            f = FileCheck().check("CPU").check("ADInplaceOrView").check("AutogradCPU").check("AutocastCPU")
            f.run(torch._C._dispatch_key_set(x))

            with torch.inference_mode():
                x = torch.rand([4])
                y = x + x
                FileCheck().check("CPU").check("AutocastCPU").run(torch._C._dispatch_key_set(y))
                FileCheck().check_not("ADInplaceOrView").check_not("Autograd").run(torch._C._dispatch_key_set(y))

    def test_constructor(self):
        with FakeTensorMode():
            x = torch.rand([4, 4], device="cpu")

        self.assertTrue(isinstance(x, FakeTensor))
        self.assertTrue(x.device.type == "cpu")

    def test_mode(self):
        with FakeTensorMode():
            y = torch.rand([4], device="cpu")
            out = y + y

        self.assertTrue(isinstance(out, FakeTensor))

    @unittest.skipIf(not RUN_CUDA, "requires cuda")
    def test_non_kwarg_device(self):
        with FakeTensorMode():
            x = torch.rand([16, 1], device="cpu")
            y = x.to(torch.device("cpu"))
            self.assertIs(x, y)
            z = x.to(torch.device("cuda"))
            self.assertEqual(z.device.type, "cuda")

    def test_fake_mode_error(self):
        x = torch.rand([4, 4])

        with self.assertRaisesRegex(Exception, "non-Fake Tensor inputs"):
            with FakeTensorMode():
                y = x[0]

    def test_fake_grad_copy(self):
        x = torch.rand([4, 4], requires_grad=True)
        x.grad = torch.rand([4, 4])
        mode = FakeTensorMode()
        fake_x = mode.from_tensor(x)
        prims.utils.compare_tensor_meta(fake_x, x)
        prims.utils.compare_tensor_meta(fake_x.grad, x.grad)

        self.assertTrue(isinstance(fake_x.grad, FakeTensor))

    @unittest.skipIf(not RUN_CUDA, "requires cuda")
    def test_like_constructor(self):
        with FakeTensorMode():
            x = torch.rand([4, 4])
            y = torch.ones_like(x)
            self.assertTrue(isinstance(y, FakeTensor))
            self.assertEqual(y.device.type, "cpu")
            z = torch.ones_like(x, device="cuda")
            self.assertTrue(isinstance(z, FakeTensor))
            self.assertEqual(z.device.type, "cuda")

    def test_binary_op_type_promotion(self):
        with FakeTensorMode():
            x = torch.empty([2, 2], dtype=torch.float)
            y = torch.empty([2, 2], dtype=torch.int64)
            out = x / y
            self.assertEqual(out.dtype, torch.float)
            self.assertEqual(out.device.type, "cpu")

    def test_from_numpy(self):
        with FakeTensorMode():
            x = torch.tensor(np.zeros([4, 4]))
            self.checkType(x, "cpu", [4, 4])

    def test_randperm(self):
        x = torch.randperm(10)
        y = torch.randperm(5, device="cpu")
        with FakeTensorMode():
            x1 = torch.randperm(10)
            prims.utils.compare_tensor_meta(x, x1)
            y1 = torch.randperm(5, device="cpu")
            prims.utils.compare_tensor_meta(y, y1)

    def test_print_in_fake_mode(self):
        x = torch.zeros(2)
        # does not fail
        with FakeTensorMode():
            out = str(x)
        assert "FakeTensor" not in out

    @unittest.skipIf(not RUN_CUDA, "requires cuda")
    def test_cpu_fallback(self):
        with FakeTensorMode(allow_fallback_kernels=False):
            filters = torch.randn(8, 4, 3, 3).cuda()
            inputs = torch.randn(1, 4, 5, 5).cuda()
            out = torch.nn.functional.conv2d(inputs, filters, padding=1)
            self.assertEqual(out.device.type, "cuda")
            self.assertEqual(list(out.size()), [1, 8, 5, 5])

        with FakeTensorMode(allow_fallback_kernels=True):
            # intentionally bad inputs
            filters = torch.randn(8, 20, 3, 3).cuda()
            inputs = torch.randn(1, 7, 10, 5).cuda()
            with self.assertRaises(RuntimeError):
                torch.nn.functional.conv2d(inputs, filters, padding=1)

        with FakeTensorMode(allow_fallback_kernels=True):
            filters = torch.randn(8, 4, 3, 3).cuda()
            inputs = torch.randn(1, 4, 5, 5).cuda()

            out = torch.nn.functional.conv2d(inputs, filters, padding=1)
            self.assertEqual(out.device.type, "cuda")
            self.assertEqual(list(out.size()), [1, 8, 5, 5])

    @unittest.skipIf(not RUN_CUDA, "requires cuda")
    def test_normalize_device(self):
        with FakeTensorMode():
            x = torch.empty(1, device="cuda")
            y = torch.empty(1, device=f"cuda:{torch.cuda.current_device()}")
            out = x + y
        self.checkType(out, "cuda", [1])

    def test_recursive_invocation(self):
        mode = FakeTensorMode()
        with mode:
            x = torch.tensor(2)
            mode.in_kernel_invocation = True
            y = x + x
            self.assertTrue(mode.in_kernel_invocation)

    @skipIfRocm
    @unittest.skipIf(not RUN_CUDA, "requires cuda")
    def test_cudnn_rnn(self):
        def fn(
            a0,
            b0,
            b1,
            b2,
            b3,
            b4,
            b5,
            b6,
            b7,
            b8,
            b9,
            b10,
            b11,
            b12,
            b13,
            b14,
            b15,
            a3,
            a4,
            a5,
        ):
            a1 = [
                b0,
                b1,
                b2,
                b3,
                b4,
                b5,
                b6,
                b7,
                b8,
                b9,
                b10,
                b11,
                b12,
                b13,
                b14,
                b15,
            ]
            return torch.ops.aten._cudnn_rnn(
                a0,
                a1,
                4,
                a3,
                a4,
                a5,
                2,
                2048,
                0,
                2,
                False,
                0.0,
                False,
                True,
                [],
                None,
            )

        mode = FakeTensorMode()
        for i, context in enumerate([contextlib.nullcontext, lambda: mode]):
            with context():
                inps = (
                    torch.randn([92, 8, 2048]).cuda(),
                    torch.randn([8192, 2048]).cuda(),
                    torch.randn([8192, 2048]).cuda(),
                    torch.randn([8192]).cuda(),
                    torch.randn([8192]).cuda(),
                    torch.randn([8192, 2048]).cuda(),
                    torch.randn([8192, 2048]).cuda(),
                    torch.randn([8192]).cuda(),
                    torch.randn([8192]).cuda(),
                    torch.randn([8192, 4096]).cuda(),
                    torch.randn([8192, 2048]).cuda(),
                    torch.randn([8192]).cuda(),
                    torch.randn([8192]).cuda(),
                    torch.randn([8192, 4096]).cuda(),
                    torch.randn([8192, 2048]).cuda(),
                    torch.randn([8192]).cuda(),
                    torch.randn([8192]).cuda(),
                    torch.randn([167837696]).cuda(),
                    torch.randn([4, 8, 2048]).cuda(),
                    torch.randn([4, 8, 2048]).cuda(),
                )
                out = fn(*inps)
                self.assertIs(out[4], inps[-3])
                for ten in out:
                    if i == 1:
                        self.assertTrue(isinstance(ten, FakeTensor))
                    self.assertEqual(ten.device.type, 'cuda')

    @skipIfRocm
    @unittest.skipIf(not RUN_CUDA, "requires cuda")
    def test_fallback_memory_prop(self):
        m = nn.Conv2d(16, 33, 3, stride=2, device="cuda", dtype=torch.half)
        m = m.to(memory_format=torch.channels_last)
        mode = FakeTensorMode()
        # TODO: module.to() doesn't work because it assigns .data, which is ignored
        with torch._subclasses.fake_tensor.FakeCopyMode(mode):
            mod_copied = copy.deepcopy(m)

        with mode:
            input = torch.rand(20, 16, 50, 100, dtype=torch.half, device="cuda").to(memory_format=torch.channels_last)
            out = mod_copied(input)
            self.assertTrue(out.is_contiguous(memory_format=torch.channels_last))
            self.checkType(out, "cuda", [20, 33, 24, 49])

    def test_data_dependent_operator(self):
        with FakeTensorMode(allow_fallback_kernels=False):
            x = torch.rand([10, 10])

            self.assertRaises(DynamicOutputShapeException, lambda: torch.nonzero(x))

    def checkMetaProps(self, t1, t2):
        prims.utils.compare_tensor_meta(t1, t2)

    @skipIfCrossRef
    def test_deepcopy(self):
        with FakeTensorMode() as mode:
            pass
        mod = torch.nn.BatchNorm2d(10)
        with torch._subclasses.fake_tensor.FakeCopyMode(mode):
            mod_copied = copy.deepcopy(mod)

        def check_copy(mod, mod_copied):
            for name, param in itertools.chain(mod.named_parameters(), mod.named_buffers()):
                param_copied = getattr(mod_copied, name)
                self.checkMetaProps(param, param_copied)
                self.assertTrue(isinstance(param_copied, FakeTensor))
                self.assertEqual(isinstance(param, torch.nn.Parameter), isinstance(param_copied, torch.nn.Parameter))
                self.assertEqual(param.requires_grad, param_copied.requires_grad)

        check_copy(mod, mod_copied)

        class ModuleNew(torch.nn.Module):
            def __init__(self):
                super(ModuleNew, self).__init__()
                self.a = torch.rand([10, 2])
                self.b = self.a
                self.c = self.a[0]

        mod = ModuleNew()
        with torch._subclasses.fake_tensor.FakeCopyMode(mode):
            mod_copied = copy.deepcopy(mod)

        self.assertIs(mod_copied.a, mod_copied.b)
        self.assertEqual(mod_copied.b.storage()._cdata, mod_copied.a.storage()._cdata)

    @unittest.skipIf(not RUN_CUDA, "requires cuda")
    def test_new(self):
        with FakeTensorMode():
            a = torch.rand([16, 1])
            self.checkType(a.new(10, 10), "cpu", [10, 10])
            self.checkType(a.new([1, 2, 3, 4]), "cpu", [4])
            b = torch.rand([4, 4], device='cuda')
            self.checkType(b.new(device='cuda'), "cuda", [0])
            self.checkType(a.new(torch.rand([1])), "cpu", [1])

    def test_scalar_inputs(self):
        with FakeTensorMode():
            self.checkType(torch.div(3, 2), "cpu", [])
            ten = torch.zeros(2, dtype=torch.int32) * 2.0
            self.assertEqual(ten.dtype, torch.float)
            self.checkType(ten, "cpu", [2])


class FakeTensorConstHandling(TestCase):
    def assertConst(self, *args):
        for arg in args:
            self.assertTrue(arg.constant is not None)

    def assertNotConst(self, *args):
        for arg in args:
            self.assertTrue(arg.constant is None)

    def test_simple(self):
        with FakeTensorMode():
            x = torch.tensor(4.)
            self.assertEqual(x.item(), 4.)

    def test_inplace_add(self):
        with FakeTensorMode():
            x = torch.tensor(4.)
            y = x.add_(1)
            self.assertEqual(x.item(), 5.)
            self.assertEqual(y.item(), 5.)
            self.assertConst(x, y)

    def test_shared_storages(self):
        with FakeTensorMode():
            x = torch.tensor([4.])
            y = x[:]

            self.assertEqual(x.storage()._cdata, y.storage()._cdata)
            self.assertEqual(x.constant.storage()._cdata, y.constant.storage()._cdata)

    def test_constant_invalidation(self):
        with FakeTensorMode():
            x = torch.tensor([1.])
            self.assertConst(x)
            y = torch.rand([1])
            x.add_(y)
            self.assertNotConst(x)

    def test_inplace_view_invalidation(self):
        with FakeTensorMode():
            x = torch.tensor([1])
            self.assertConst(x)
            x.resize_([2])
            self.assertEqual(x.size(0), 2)
            self.assertNotConst(x)

    def test_fake_tensor_in_intlist_repro(self):

        def fn(tensors):
            max_size = torch.tensor([800, 1216], dtype=torch.int64)
            batch_shape = [len(tensors)] + list(tensors[0].shape[:-2]) + list(max_size)
            return tensors[0].new_full(batch_shape, 0.0)

        with self.assertRaises(torch._subclasses.fake_tensor.DataDependentOutputException):
            with torch._subclasses.fake_tensor.FakeTensorMode(throw_on_data_dependent_ops=True):
                a = torch.randn(3, 800, 1199)
                b = torch.randn(3, 800, 800)
                inputs = [a, b]
                ref = fn(inputs)

    def test_fake_tensor_batch_norm_cpu(self):
        with torch._subclasses.CrossRefFakeMode():
            m = torch.nn.Sequential(
                torch.nn.BatchNorm2d(10),
                torch.nn.ReLU(),
            )
            m.eval()
            out = m(torch.randn([2, 10, 8, 8]))

    def test_shared_storage_invalidation(self):
        with FakeTensorMode():
            x = torch.tensor([1.])
            y = x[:]
            self.assertConst(x, y)
            y.add_(torch.rand([1]))
            self.assertNotConst(x, y)

    def test_aliased_const_write(self):
        with FakeTensorMode():
            x = torch.tensor([1])
            y = x.expand([4])
            self.assertNotConst(y)
            y[0] = 1
            self.assertNotConst(x)

def contains_type(type: torch._C.Type, maybe_contained_type: torch._C.Type):
    return maybe_contained_type.isSubtypeOf(type) or any(
        contains_type(e, maybe_contained_type) for e in type.containedTypes()
    )


class FakeTensorConverterTest(TestCase):
    def test_memoized_conversion_to_meta(self):
        x = torch.rand(2, 2, 2)
        mode = FakeTensorMode()
        self.assertTrue(mode.from_tensor(x) is mode.from_tensor(x))

    def test_memoized_conversion_from_meta(self):
        x = torch.rand(2, 2).to(device="meta")
        mode = FakeTensorMode()
        converter = mode.fake_tensor_converter
        self.assertTrue(converter(mode, x, "cpu") is converter(mode, x, "cpu"))

    def test_separate_tensor_storages_view(self):
        x = torch.rand(2, 2, 2)
        y = x[0]
        mode = FakeTensorMode()
        converter = mode.fake_tensor_converter
        x_conv = converter(mode, x)
        y_conv = converter(mode, y)
        self.assertEqual(torch._C._storage_id(x_conv), torch._C._storage_id(y_conv))

    def test_separate_tensor_storages_non_view(self):
        x = torch.rand(2, 2, 2)
        y = torch.rand(4, 2)
        y.set_(x.storage())
        mode = FakeTensorMode()
        converter = mode.fake_tensor_converter
        x_conv = converter(mode, x)
        y_conv = converter(mode, y)
        stor_id = torch._C._storage_id(x_conv)
        self.assertEqual(stor_id, torch._C._storage_id(y_conv))
        del x
        self.assertEqual(len(converter.tensor_memo), 1)
        converter.meta_converter.check_for_expired_weak_storages()
        self.assertEqual(len(converter.meta_converter.storage_memo), 1)
        del y
        self.assertEqual(len(converter.tensor_memo), 0)
        converter.meta_converter.check_for_expired_weak_storages()
        self.assertEqual(len(converter.meta_converter.storage_memo), 0)


    def test_dead_weak_ref(self):
        x = torch.rand(2, 2, 2)
        y = x[0]
        mode = FakeTensorMode()
        converter = FakeTensorConverter()
        x_conv = converter(mode, x)
        x_conv_storage = torch._C._storage_id(x_conv)
        del x_conv
        self.assertFalse(x in converter.tensor_memo)
        y_conv = converter(mode, y)
        self.assertEqual(x_conv_storage, torch._C._storage_id(y_conv))

    def test_dead_key(self):
        x = torch.rand(2, 2, 2)
        mode = FakeTensorMode()
        converter = FakeTensorConverter()
        x_conv = converter(mode, x)
        self.assertEqual(len(converter.tensor_memo), 1)
        self.assertEqual(len(converter.meta_converter.tensor_memo), 1)
        del x
        self.assertEqual(len(converter.tensor_memo), 0)
        self.assertEqual(len(converter.meta_converter.tensor_memo), 0)

    def test_no_active_mode(self):
        with FakeTensorMode() as mode:
            x = torch.empty(2, 2, device="cpu")
            y = torch.empty(2, 2, device="cpu")

        out = x + y
        self.assertEqual(mode, out.fake_mode)
        self.assertTrue(isinstance(out, FakeTensor))
        self.assertEqual(out.device.type, "cpu")

    def test_separate_mode_error(self):
        with FakeTensorMode():
            x = torch.empty(2, 2, device="cpu")
        with FakeTensorMode():
            y = torch.empty(2, 2, device="cpu")
        self.assertRaises(Exception, lambda: x, y)

    def test_no_ref_cycle(self):
        x = torch.rand([4])
        mode = FakeTensorMode()
        y = mode.from_tensor(x)
        self.assertEqual(len(mode.fake_tensor_converter.tensor_memo), 1)
        mode_weak = weakref.ref(mode)
        y_weak = weakref.ref(mode)
        del mode
        del y
        assert mode_weak() is None
        assert y_weak() is None


class FakeTensorOperatorInvariants(TestCase):
    @staticmethod
    def get_aten_op(schema):
        namespace, name = schema.name.split("::")
        overload = schema.overload_name if schema.overload_name else "default"
        assert namespace == "aten"
        return getattr(getattr(torch.ops.aten, name), overload)

    @staticmethod
    def get_all_aten_schemas():
        for schema in torch._C._jit_get_all_schemas():
            namespace = schema.name.split("::")[0]
            if namespace != "aten":
                continue
            yield schema

    def test_non_kwarg_only_device(self):
        for schema in self.get_all_aten_schemas():
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
        for schema in self.get_all_aten_schemas():
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

    @unittest.expectedFailure
    def test_sparse_new(self):
        with FakeTensorMode():
            indices = torch.randn(1, 1, dtype=torch.int64)
            values = torch.randn(1)
            extra = (2,)
            sparse = torch.randn(1).to_sparse()
            # This used to segfault, now it does not, but it still raises an
            # error
            sparse2 = sparse.new(indices, values, extra)

    def test_like_ops(self):
        for schema in self.get_all_aten_schemas():
            if "_like" == schema.name[-5:]:
                op = self.get_aten_op(schema)
                self.assertIn(op, torch._subclasses.fake_tensor._like_tensor_constructors)

if __name__ == "__main__":
    run_tests()
