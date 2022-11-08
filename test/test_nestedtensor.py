# Owner(s): ["module: nestedtensor"]

import unittest

import numpy as np
import torch
import torch.nn
from torch.testing._internal.common_device_type import (
    dtypes,
    dtypesIfCUDA,
    instantiate_device_type_tests,
    onlyCPU,
    onlyCUDA,
    skipMeta,
)
from torch.testing._internal.common_dtype import floating_types_and_half
from torch.testing._internal.common_utils import (
    freeze_rng_state,
    gradcheck,
    instantiate_parametrized_tests,
    IS_FBCODE,
    parametrize,
    run_tests,
    subtest,
    TestCase,
)

# Tests are ported from pytorch/nestedtensor.
# This makes porting as_nested_tensor easier in the future.


def _iter_constructors():
    # yield as_nested_tensor
    yield torch.nested.nested_tensor

# Helper function to generate a pair of random nested tensors
# one is contiguous, the other is not, but they appear to have same entries
# an output nested tensor consists of
# * `len(ragged_sizes)` matrices
# * matrices[i].shape == (20, ragged_sizes[i])


def random_nt_noncontiguous_pair(ragged_sizes, device="cpu", dtype=torch.float16):
    xs = []
    for size in ragged_sizes:
        xs.append(torch.randn((size, 20), device=device, dtype=dtype))
    # contiguous nested tensor
    ys = []
    for x in xs:
        ys.append(x.transpose(-1, -2))
    nt_contiguous = torch.nested.nested_tensor(ys)
    # noncontiguous nested tensor
    n = len(ragged_sizes)
    nt_noncontiguous = torch.nested.nested_tensor(xs).transpose(-1, -2)
    return nt_contiguous, nt_noncontiguous

# Helper functions to pad a noncontiguous nested tensor
# can be replaced once to_padded_tensor supports noncontiguous memory


def noncontiguous_to_padded_tensor(input, shape=None):
    tensors = input.unbind()
    ntensors = len(tensors)
    assert ntensors > 0
    if shape is None:
        shape = []
        for size in tensors[0].shape:
            shape.append(size)
        for i in range(1, ntensors):
            new_shape = tensors[i].shape
            for j in range(len(shape)):
                shape[j] = max(shape[j], new_shape[j])
        shape = [ntensors] + shape
    result = tensors[0].new_zeros(shape)
    for itensor in range(ntensors):
        tensor = tensors[itensor]
        view = result[itensor]
        for idim in range(tensor.dim()):
            view = view.narrow(idim, 0, tensor.size(idim))
        view.copy_(tensor)
    return result

# Helper function to generate a random nested tensor


def random_nt(device, dtype, num_tensors, max_dims, min_dims=None):
    if min_dims is None:
        min_dims = tuple([0] * len(max_dims))
    ts1 = []
    for _ in range(num_tensors):
        tensor_dims = tuple([torch.randint(low=min_dim, high=max_dim, size=(1,)).item()
                            for (min_dim, max_dim) in zip(min_dims, max_dims)])
        t1 = torch.randn(tensor_dims, device=device, dtype=dtype)
        ts1.append(t1)
    return torch.nested.nested_tensor(ts1, device=device, dtype=dtype)


class TestNestedTensor(TestCase):
    @parametrize("batch_size", [2, 4])
    @parametrize("max_seq_len", [3, 5])
    @parametrize("vocab_size", [10, 20])
    def test_2d_nested_tensor(self, batch_size, max_seq_len, vocab_size):
        data = []
        nested_tensor_ref_list = []
        for _ in range(batch_size):
            if max_seq_len == 0:
                length = 0
            else:
                length = np.random.randint(low=1, high=max_seq_len)
            row = list(np.random.randint(low=0, high=vocab_size, size=(length,)))
            data.append(row)
            nested_tensor_ref_list.append(torch.tensor(row))
        nested_tensor = torch.nested.nested_tensor(data, dtype=torch.int64)
        nested_tensor_list = nested_tensor.unbind()
        for id in range(batch_size):
            self.assertEqual(
                nested_tensor_list[id],
                nested_tensor_ref_list[id].type(torch.int64)
            )

    @parametrize("batch_size", [2, 4])
    @parametrize("max_seq_len", [3, 5])
    @parametrize("vocab_size", [10, 20])
    def test_3d_nested_tensor(self, batch_size, max_seq_len, vocab_size):
        data = []
        nested_tensor_ref_list = []
        for _ in range(batch_size):
            if max_seq_len == 0:
                length = 0
            else:
                length = np.random.randint(low=1, high=max_seq_len)
            row = list(np.random.randint(low=0, high=vocab_size, size=(length,)))
            row = [list(item * np.arange(max_seq_len)) for item in row]
            data.append(row)
            nested_tensor_ref_list.append(torch.Tensor(row))
        nested_tensor = torch.nested.nested_tensor(data, dtype=torch.int64)
        nested_tensor_list = nested_tensor.unbind()
        for id in range(batch_size):
            self.assertEqual(
                nested_tensor_list[id],
                nested_tensor_ref_list[id].type(torch.int64)
            )

    @parametrize("batch_size", [2, 4])
    @parametrize("max_seq_len", [3, 5])
    @parametrize("vocab_size", [10, 20])
    def test_3d_nested_tensor_float(self, batch_size, max_seq_len, vocab_size):
        data = []
        nested_tensor_ref_list = []
        for _ in range(batch_size):
            if max_seq_len == 0:
                length = 0
            else:
                length = np.random.randint(low=1, high=max_seq_len)
            row = list(
                np.random.randint(low=0, high=vocab_size, size=(length,)).astype(float)
            )
            row = [list(item * np.arange(max_seq_len)) for item in row]
            data.append(row)
            nested_tensor_ref_list.append(torch.Tensor(row))
        nested_tensor = torch.nested.nested_tensor(data, dtype=torch.float)
        nested_tensor_list = nested_tensor.unbind()
        for id in range(batch_size):
            self.assertEqual(
                nested_tensor_list[id],
                nested_tensor_ref_list[id].type(torch.float)
            )


    @torch.inference_mode()
    def _test_unbind_case(self, a, b):
        nt = torch.nested.nested_tensor([a, b])
        a1, b1 = nt.unbind()
        self.assertTrue(a is not a1)
        self.assertTrue(b is not b1)

        nt = torch.nested.nested_tensor([a, b], dtype=a.dtype)
        a1, b1 = nt.unbind(0)
        self.assertEqual(a, a1)
        self.assertEqual(b, b1)

        a = torch.randn((2, 3)).add_(1)
        nt = torch.nested.nested_tensor([a])
        self.assertEqual(a, nt.unbind(0)[0])

    @torch.inference_mode()
    def test_unbind_0(self):
        self._test_unbind_case(
            torch.tensor([1, 2]), torch.tensor([7, 8]),
        )

    @torch.inference_mode()
    def test_unbind_1(self):
        self._test_unbind_case(
            torch.tensor([1]), torch.tensor([7]),
        )

    @torch.inference_mode()
    def test_unbind_3(self):
        self._test_unbind_case(
            torch.tensor([1.0]), torch.tensor([]),
        )

    @torch.inference_mode()
    def test_unbind_4(self):
        self._test_unbind_case(
            torch.tensor([]), torch.tensor([]),
        )

    @torch.inference_mode()
    def test_unbind_dim(self):
        def _test_fn(unbind_fn):
            a = torch.rand(3, 2)
            b = torch.rand(2, 3)
            nt = torch.nested.nested_tensor([a, b])
            self.assertRaises(RuntimeError, lambda: unbind_fn(nt, 1))

        # Both of these tests are necessary, because we're using
        # torch_function.
        _test_fn(lambda x, dim: x.unbind(dim))
        # TODO: Re-enable this once using torch_dispatch
        # _test_fn(lambda x, dim: torch.unbind(x, dim))

    @torch.inference_mode()
    def test_nested_tensor(self):
        self.assertRaises(TypeError, lambda: torch.nested.nested_tensor(torch.tensor([3.0])))
        self.assertRaises(TypeError, lambda: torch.nested.nested_tensor(4.0))

    @torch.inference_mode()
    def test_nested_tensor_matching_dim(self):
        self.assertRaisesRegex(
            RuntimeError,
            "Found dimension 1 for Tensor at index 1 and dimension 0 for Tensor at index 0.",
            lambda: torch.nested.nested_tensor([torch.tensor(1.0), torch.tensor([])]),
        )
        self.assertRaisesRegex(
            RuntimeError,
            "Found dimension 1 for Tensor at index 2 and dimension 0 for Tensor at index 1.",
            lambda: torch.nested.nested_tensor(
                [torch.tensor(1.0), torch.tensor(2.0), torch.tensor([])]
            ),
        )

    @torch.inference_mode()
    def test_default_nested_tensor(self):
        self.assertRaises(TypeError, lambda: torch.nested.nested_tensor())
        default_nested_tensor = torch.nested.nested_tensor([])
        default_tensor = torch.tensor([])
        # self.assertEqual(default_nested_tensor.nested_dim(), 1)
        # self.assertEqual(default_nested_tensor.nested_size(), ())
        self.assertEqual(default_nested_tensor.dim(), default_tensor.dim())
        self.assertEqual(default_nested_tensor.layout, default_tensor.layout)
        self.assertEqual(default_nested_tensor.device, default_tensor.device)
        self.assertEqual(default_nested_tensor.dtype, default_tensor.dtype)
        self.assertEqual(
            default_nested_tensor.requires_grad, default_tensor.requires_grad
        )
        self.assertIsNone(default_tensor.grad)
        # TODO: Re-enable once we have a performance driven
        # use case and implementation.
        # self.assertEqual(default_nested_tensor.is_pinned(),
        #                  default_tensor.is_pinned())

    @torch.inference_mode()
    def test_dim(self):
        for constructor in _iter_constructors():
            a1 = constructor([])
            self.assertEqual(a1.dim(), 1)
            a1 = constructor([torch.tensor(3.0)])
            self.assertEqual(a1.dim(), 1)
            a1 = constructor([torch.tensor([1, 2, 3, 4])])
            self.assertEqual(a1.dim(), 2)

    @unittest.skipIf(IS_FBCODE, "numel is not virtual in fbcode.")
    @torch.inference_mode()
    def test_numel(self):
        for constructor in _iter_constructors():
            a1 = constructor([])
            self.assertEqual(a1.numel(), 0)
            a1 = constructor([torch.tensor(3.0), torch.tensor(4.0)])
            self.assertEqual(a1.numel(), 2)
            a1 = constructor([torch.randn(2, 2, 2)])
            self.assertEqual(a1.numel(), 8)
            a1 = constructor([torch.randn([1, 2, 3]), torch.randn(3, 2, 1)])
            self.assertEqual(a1.numel(), 12)
            a1 = constructor([torch.randn([1, 1, 3]), torch.randn(3, 2, 4)])
            self.assertEqual(a1.numel(), 27)
            a1 = constructor([torch.randn([5, 5, 5]), torch.randn(6, 6, 6)])
            self.assertEqual(a1.numel(), 341)

            # Interesting edge case
            a1 = constructor([torch.randn([1, 2, 3]), torch.randn(1, 2, 0)])
            self.assertEqual(a1.numel(), 6)

    @torch.inference_mode()
    def test_size(self):
        for constructor in _iter_constructors():
            a1 = constructor([])
            self.assertRaisesRegex(
                RuntimeError,
                "NestedTensorImpl doesn't support sizes",
                lambda: a1.size(),
            )

    def test_size_dim(self):
        a = torch.nested.nested_tensor([])
        self.assertEqual(a.size(0), 0)

        a = torch.nested.nested_tensor([torch.tensor(1)])
        self.assertEqual(a.size(0), 1)

        a = torch.nested.nested_tensor([torch.tensor(1), torch.tensor(2)])
        self.assertEqual(a.size(0), 2)

        a = torch.nested.nested_tensor([torch.rand(1, 2),
                                        torch.rand(1, 8)])
        self.assertEqual(a.size(0), 2)
        self.assertEqual(a.size(1), 1)
        self.assertRaisesRegex(
            RuntimeError, "Given dimension 2 is irregular and does not have a size", lambda: a.size(2))

        a = torch.nested.nested_tensor([torch.rand(3, 4),
                                        torch.rand(5, 4)])
        self.assertEqual(a.size(0), 2)
        self.assertRaisesRegex(
            RuntimeError, "Given dimension 1 is irregular and does not have a size", lambda: a.size(1))
        self.assertEqual(a.size(2), 4)

    @unittest.skipIf(IS_FBCODE, "stride is not virtual in fbcode.")
    @torch.inference_mode()
    def test_stride(self):
        for constructor in _iter_constructors():
            a1 = constructor([])
            self.assertRaisesRegex(
                RuntimeError,
                "NestedTensorImpl doesn't support strides",
                lambda: a1.stride(),
            )

    @unittest.skipIf(IS_FBCODE, "is_contiguous is not virtual in fbcode.")
    @torch.inference_mode()
    def test_is_contiguous(self):
        # Test empty case
        nt_empty = torch.nested.nested_tensor([])
        assert nt_empty.is_contiguous()
        self.assertEqual(nt_empty, nt_empty.contiguous())

        nt_contiguous, nt_noncontiguous = random_nt_noncontiguous_pair((2, 3, 6, 7))

        # Test contiguous case
        assert nt_contiguous.is_contiguous()
        self.assertEqual(nt_contiguous, nt_contiguous.contiguous())

        # Test non_contiguous case
        assert not nt_noncontiguous.is_contiguous()
        self.assertEqual(nt_contiguous, nt_noncontiguous.contiguous())

    @torch.inference_mode()
    def test_repr_string(self):
        a = torch.nested.nested_tensor([])
        expected = "nested_tensor([" "\n\n])"
        self.assertEqual(str(a), expected)
        self.assertEqual(repr(a), expected)

        a = torch.nested.nested_tensor([torch.tensor(1.0)])
        expected = "nested_tensor([" "\n  tensor(1.)" "\n])"
        self.assertEqual(str(a), expected)
        self.assertEqual(repr(a), expected)

        a = torch.nested.nested_tensor([torch.tensor([[1, 2]]), torch.tensor([[4, 5]])])
        expected = (
            "nested_tensor([" "\n  tensor([[1, 2]])" "," "\n  tensor([[4, 5]])" "\n])"
        )
        self.assertEqual(str(a), expected)
        self.assertEqual(repr(a), expected)

    def test_to_padded_tensor_on_empty_tensor(self):

        nt = torch.nested.nested_tensor([])
        empty = torch.nested.to_padded_tensor(nt, 4)
        self.assertEqual(empty, torch.tensor([]))

    def test_nested_namespace(self):
        nt = torch.nested.nested_tensor([torch.randn(2, 3), torch.randn(4, 5)])
        result = nt.to_padded_tensor(4)
        nested_namespace_result = torch.nested.to_padded_tensor(nt, 4)
        self.assertEqual(result, nested_namespace_result)

    def test_to(self):
        ntensors = 4
        nt = random_nt(torch.device('cpu'), torch.float32, ntensors, (4, 4))

        def test_copy_behavior(t, non_blocking=False):
            self.assertIs(t, t.to(t, non_blocking=non_blocking))
            self.assertIs(t, t.to(t.dtype, non_blocking=non_blocking))
            self.assertIs(t, t.to(torch.empty_like(t), non_blocking=non_blocking))
            self.assertIsNot(t, t.to(t, non_blocking=non_blocking, copy=True))
            self.assertIsNot(t, t.to(t.dtype, non_blocking=non_blocking, copy=True))
            self.assertIsNot(t, t.to(torch.empty_like(t), non_blocking=non_blocking, copy=True))

            devices = [t.device]
            if t.device.type == 'cuda':
                if t.device.index == -1:
                    devices.append('cuda:{}'.format(torch.cuda.current_device()))
                elif t.device.index == torch.cuda.current_device():
                    devices.append('cuda')
            for device in devices:
                self.assertIs(t, t.to(device, non_blocking=non_blocking))
                self.assertIs(t, t.to(device, t.dtype, non_blocking=non_blocking))
                self.assertIsNot(t, t.to(device, non_blocking=non_blocking, copy=True))
                self.assertIsNot(t, t.to(device, t.dtype, non_blocking=non_blocking, copy=True))

        test_copy_behavior(nt)
        self.assertEqual(nt.device, nt.to('cpu').device)
        self.assertEqual(nt.device, nt.to('cpu', dtype=torch.float32).device)
        self.assertIs(torch.float32, nt.to('cpu', dtype=torch.float32).dtype)
        self.assertEqual(nt.device, nt.to(torch.float32).device)
        self.assertIs(torch.float32, nt.to(dtype=torch.float32).dtype)

        def test_data_ptr(getter):
            self.assertEqual(getter(nt), getter(nt.to('cpu')))
            self.assertEqual(getter(nt), getter(nt.to(dtype=nt.dtype, device=nt.device, copy=False)))
            self.assertEqual(getter(nt), getter(nt.to('cpu', copy=False)))
            self.assertNotEqual(getter(nt), getter(nt.to('cpu', copy=True)))

        test_data_ptr(lambda nt: nt.data_ptr())

        if torch.cuda.is_available():
            for non_blocking in [True, False]:
                for cuda in ['cuda', 'cuda:0' if torch.cuda.device_count() == 1 else 'cuda:1']:
                    nt2 = random_nt(cuda, torch.float32, ntensors, (4, 4))
                    test_copy_behavior(nt2, non_blocking)
                    self.assertEqual(nt2.device, nt2.to(cuda, non_blocking=non_blocking).device)
                    self.assertEqual(nt.device, nt2.to('cpu', non_blocking=non_blocking).device)
                    self.assertEqual(nt2.device, nt.to(cuda, non_blocking=non_blocking).device)
                    self.assertIs(torch.int32, nt2.to('cpu', dtype=torch.int32, non_blocking=non_blocking).dtype)
                    self.assertEqual(nt.device, nt2.to('cpu', dtype=torch.int32, non_blocking=non_blocking).device)
                    self.assertIs(torch.int32, nt2.to(dtype=torch.int32).dtype)
                    self.assertEqual(nt2.device, nt2.to(dtype=torch.int32).device)

    def test_copy_(self):
        ntensors = 4
        nt = random_nt(torch.device('cpu'), torch.float32, ntensors, (4, 4))
        nt_copy = torch.empty_like(nt)
        nt_copy.copy_(nt)

        for (nt_ub, nt_copy_ub) in zip(nt.unbind(), nt_copy):
            self.assertEqual(nt_ub, nt_copy_ub)

        nt_error = torch.nested.nested_tensor([torch.tensor([0, 0])])
        self.assertRaisesRegex(
            RuntimeError,
            "copy_ only supports tensors that are the same size for Nested implementations",
            lambda: nt_error.copy_(nt)
        )

        if torch.cuda.is_available():
            nt = random_nt(torch.device('cuda'), torch.float32, ntensors, (4, 4))
            nt_copy = torch.empty_like(nt, device=torch.device('cpu'))
            nt_copy.copy_(nt, non_blocking=True)
            torch.cuda.current_stream(torch.cuda.current_device()).synchronize()
            for (nt_ub, nt_copy_ub) in zip(nt.unbind(), nt_copy):
                self.assertEqual(nt_ub, nt_copy_ub)

            nt_copy = torch.empty_like(nt, device=torch.device('cpu'))
            nt_copy.copy_(nt, non_blocking=False)
            for (nt_ub, nt_copy_ub) in zip(nt.unbind(), nt_copy):
                self.assertEqual(nt_ub, nt_copy_ub)

    def test_fill_(self):
        ntensors = 4
        nt = random_nt(torch.device('cpu'), torch.float32, ntensors, (4, 4))
        nt.fill_(10.)
        for nt_ub in nt.unbind():
            t = torch.empty_like(nt_ub)
            t.fill_(10.)
            self.assertEqual(nt_ub, t)

        fill_tensor = torch.tensor([11.])
        self.assertRaisesRegex(
            RuntimeError,
            "fill_ only supports 0-dimension value tensor",
            lambda: nt.fill_(fill_tensor)
        )

        nt.fill_(fill_tensor[0])
        for nt_ub in nt.unbind():
            t = torch.empty_like(nt_ub)
            t.fill_(11.)
            self.assertEqual(nt_ub, t)

    def test_ones_like(self):
        ntensors = 4
        nt = random_nt(torch.device('cpu'), torch.float32, ntensors, (4, 4))
        ones_nt = torch.ones_like(nt)

        for nt_ub in ones_nt.unbind():
            t = torch.ones_like(nt_ub)
            self.assertEqual(nt_ub, t)


class TestNestedTensorDeviceType(TestCase):

    # Helper function to generate a pair of random nested tensors
    # the 2 nested tensors have same shapes
    def random_nt_pair(self, device, dtype, num_tensors, max_dims):
        ts1 = []
        ts2 = []
        for _ in range(num_tensors):
            tensor_dims = tuple([torch.randint(low=0, high=max_dim, size=(1,)).item() for max_dim in max_dims])
            t1 = torch.randn(tensor_dims, device=device, dtype=dtype)
            t2 = torch.randn(tensor_dims, device=device, dtype=dtype)
            ts1.append(t1)
            ts2.append(t2)
        return (torch.nested.nested_tensor(ts1, device=device, dtype=dtype),
                torch.nested.nested_tensor(ts2, device=device, dtype=dtype))

    @dtypes(*floating_types_and_half())
    def test_detach(self, device, dtype):
        a = torch.randn(2, 4, device=device, dtype=dtype, requires_grad=False)
        b = torch.randn(5, 4, device=device, dtype=dtype, requires_grad=False)
        x = torch.nested.nested_tensor([a, b], requires_grad=True)

        x_detach = x.detach()

        z = x_detach * 4
        self.assertFalse(x_detach.requires_grad)
        self.assertFalse(z.requires_grad)

        a = torch.randn(2, 4, device=device, dtype=dtype, requires_grad=True)
        b = torch.randn(5, 4, device=device, dtype=dtype, requires_grad=True)
        x = torch.nested.as_nested_tensor([a, b])

        y = x * 2
        y = y.detach()
        self.assertFalse(y.requires_grad)
        self.assertIsNone(y.grad_fn)

        z = x + y
        torch.nested.to_padded_tensor(z, 0).sum().backward()
        # This is an incorrect gradient, but we assume that's what the user
        # wanted. detach() is an advanced option.
        self.assertEqual(a.grad, torch.ones(2, 4, device=device, dtype=dtype))
        self.assertEqual(b.grad, torch.ones(5, 4, device=device, dtype=dtype))

    @dtypes(torch.float, torch.float16, torch.double)
    def test_unbind_noncontiguous(self, device, dtype):
        nt_contiguous, nt_noncontiguous = random_nt_noncontiguous_pair((2, 3, 6, 7), device, dtype)
        ub_contiguous = nt_contiguous.unbind()
        ub_noncontiguous = nt_noncontiguous.unbind()
        self.assertEqual(len(ub_contiguous), len(ub_noncontiguous))
        n = len(ub_contiguous)
        for i in range(n):
            self.assertEqual(ub_contiguous[i], ub_noncontiguous[i])

    @dtypes(torch.float)
    @skipMeta
    def test_to_then_from_padded_tensor_no_transform0213(self, device, dtype):
        t = torch.randn(4, 4, 4, device=device, dtype=dtype)
        ts = list(torch.unbind(t))
        ts[0] = ts[0][:-1]
        nt = torch.nested.nested_tensor(ts, device=device, dtype=dtype)
        padded = torch.nested.to_padded_tensor(nt, 0)

        nt_to = torch._nested_from_padded_and_nested_example(padded, nt)

        for (t1, t2) in zip(nt.unbind(), nt_to.unbind()):
            self.assertEqual(t1, t2)
        self.assertEqual(nt.device, nt_to.device)

    @dtypes(torch.float)
    @dtypesIfCUDA(torch.float, torch.half)
    @skipMeta
    @torch.inference_mode()
    def test_layer_norm(self, device, dtype):
        def _test(size):
            # Simple shapes test
            t0 = torch.randn(2, size, device=device, dtype=dtype, requires_grad=False)
            t1 = torch.randn(2, size, device=device, dtype=dtype, requires_grad=False)
            ts = [t0, t1, t0, t1]
            nt = torch.nested.nested_tensor(ts, device=device, dtype=dtype)
            layer_norm = torch.nn.LayerNorm(size, device=device, dtype=dtype)
            nt_result = layer_norm(nt)
            for (nt_subresult, t) in zip(nt_result.unbind(), ts):
                t_result = layer_norm(t.reshape(1, -1, size).squeeze(0))
                self.assertEqual(nt_subresult, t_result)

            # More complex nt test with different lengths for each tensor
            t0 = torch.randn(4, size, device=device, dtype=dtype, requires_grad=False)
            t1 = torch.randn(10, size, device=device, dtype=dtype, requires_grad=False)
            t2 = torch.randn(7, size, device=device, dtype=dtype, requires_grad=False)
            ts = [t0, t1, t2, t0, t2]
            nt = torch.nested.nested_tensor(ts, device=device, dtype=dtype)
            layer_norm = torch.nn.LayerNorm(size, device=device, dtype=dtype)
            nt_result = layer_norm(nt)
            for (nt_subresult, t) in zip(nt_result.unbind(), ts):
                t_result = layer_norm(t.reshape(1, -1, size).squeeze(0))
                self.assertEqual(nt_subresult, t_result)

            if size <= 128:
                # Test with multidimensional tensors after irregular dim
                # (run only with smaller dimensions to ensure fast execution)
                t0 = torch.randn(4, size, size, 4, device=device, dtype=dtype, requires_grad=False)
                t1 = torch.randn(10, size, size, 4, device=device, dtype=dtype, requires_grad=False)
                t2 = torch.randn(7, size, size, 4, device=device, dtype=dtype, requires_grad=False)
                ts = [t0, t1, t2, t0, t2]
                nt = torch.nested.nested_tensor(ts, device=device, dtype=dtype)
                layer_norm = torch.nn.LayerNorm((size, size, 4), device=device, dtype=dtype)
                nt_result = layer_norm(nt)
                for (nt_subresult, t) in zip(nt_result.unbind(), ts):
                    t_result = layer_norm(t.reshape(1, -1, size, size, 4).squeeze(0))
                    self.assertEqual(nt_subresult, t_result)

                # Test where the normalizing dimensions are not all
                layer_norm = torch.nn.LayerNorm((size, 4), device=device, dtype=dtype)
                nt_result = layer_norm(nt)
                for (nt_subresult, t) in zip(nt_result.unbind(), ts):
                    t_result = layer_norm(t.reshape(1, -1, size, size, 4).squeeze(0))
                    self.assertEqual(nt_subresult, t_result)

        for size in (1024, 1023, 513, 512, 256, 128, 2, 4, 32):
            _test(size)

    @dtypes(torch.float)
    @dtypesIfCUDA(torch.float, torch.half)
    @skipMeta
    @torch.inference_mode()
    def test_layer_norm_breaking(self, device, dtype):
        size = 128
        t0 = torch.randn(4, size, size, 4, device=device, dtype=dtype, requires_grad=False)
        t1 = torch.randn(10, size, size, 4, device=device, dtype=dtype, requires_grad=False)
        t2 = torch.randn(7, size, size, 4, device=device, dtype=dtype, requires_grad=False)
        ts = [t0, t1, t2, t0, t2]
        nt = torch.nested.nested_tensor(ts, device=device, dtype=dtype)
        layer_norm = torch.nn.LayerNorm((4, size, size, 4), device=device, dtype=dtype)
        self.assertRaisesRegex(
            RuntimeError,
            "normalized_shape extends into irregular dimensions for the nested tensor",
            lambda: layer_norm(nt),
        )
        layer_norm = torch.nn.LayerNorm((size + 1, size, 4), device=device, dtype=dtype)
        self.assertRaisesRegex(
            RuntimeError,
            "The shape at dimension 0",
            lambda: layer_norm(nt),
        )

    @skipMeta
    @torch.inference_mode()
    def test_embedding(self, device):
        inputs = [
            torch.randint(100, (L,), device=device, dtype=torch.int64)
            for L in torch.randint(5, 50, (8,))
        ]
        x = torch.nested.nested_tensor(inputs, device=device, dtype=torch.int64)
        emb = torch.nn.Embedding(100, 8, device=device)
        y = emb(x)
        ys = y.unbind()
        for i, inp in enumerate(inputs):
            self.assertEqual(emb(inp), ys[i])

    @dtypes(torch.float, torch.float16)
    def test_to_padded_tensor_simple(self, device, dtype):
        t = torch.randn(4, 4, 4, device=device, dtype=dtype)
        ts = list(torch.unbind(t))
        ts[0] = ts[0][:-1]
        nt = torch.nested.nested_tensor(ts, device=device, dtype=dtype)
        for padding_value in (0, 1):
            padded = torch.nested.to_padded_tensor(nt, padding_value)

            correct_output = t.clone()
            if padding_value == 0:
                correct_output[0][-1] = torch.zeros_like(correct_output[0][-1])
            else:
                correct_output[0][-1] = torch.ones_like(correct_output[0][-1])

            self.assertEqual(padded, correct_output)
            self.assertEqual(padded.device, torch.device(device))
            self.assertEqual(padded.dtype, dtype)

    @dtypes(torch.float, torch.float16)
    def test_to_padded_tensor_output_size(self, device, dtype):
        t = torch.randn(4, 4, 4, device=device, dtype=dtype)
        output_size = (4, 6, 5)
        ts = list(torch.unbind(t))
        ts[0] = ts[0][:-1]
        nt = torch.nested.nested_tensor(ts, device=device, dtype=dtype)
        for padding_value in (0, 1):
            padded = torch.nested.to_padded_tensor(nt, padding_value, output_size=output_size)
            correct_output = torch.ones(output_size, device=device, dtype=dtype) * padding_value
            correct_output[:4:, :4, :4] = t.clone()
            if padding_value == 0:
                correct_output[0][3] = torch.zeros_like(correct_output[0][3])
            else:
                correct_output[0][3] = torch.ones_like(correct_output[0][3])

            self.assertEqual(padded, correct_output)
            self.assertEqual(padded.device, torch.device(device))
            self.assertEqual(padded.dtype, dtype)

    @dtypes(torch.float, torch.float16, torch.double)
    def test_to_padded_tensor_dim2(self, device, dtype):
        ts = [
            torch.randn(160, device=device, dtype=dtype),
            torch.randn(1240, device=device, dtype=dtype),
            torch.randn(2400, device=device, dtype=dtype),
        ]
        nt = torch.nested.nested_tensor(ts, device=device, dtype=dtype)
        pad = 42
        correct_output = []
        for t in ts:
            next_output = torch.ones_like(ts[2]) * pad
            correct_output.append(next_output)
            next_output[:t.size(0)].copy_(t)
        correct_output = torch.stack(correct_output)
        padded = torch.nested.to_padded_tensor(nt, pad)
        self.assertEqual(padded, correct_output)

    @dtypes(torch.float, torch.float16, torch.double)
    def test_to_padded_tensor_dim3(self, device, dtype):
        ts = [
            torch.randn(16, 21, device=device, dtype=dtype),
            torch.randn(24, 32, device=device, dtype=dtype),
            torch.randn(40, 53, device=device, dtype=dtype),
        ]
        nt = torch.nested.nested_tensor(ts, device=device, dtype=dtype)
        pad = 42
        correct_output = []
        for t in ts:
            next_output = torch.ones_like(ts[2]) * pad
            correct_output.append(next_output)
            next_output[:t.size(0), :t.size(1)].copy_(t)
        correct_output = torch.stack(correct_output)
        padded = torch.nested.to_padded_tensor(nt, pad)
        self.assertEqual(padded, correct_output)

    @dtypes(torch.float, torch.float16, torch.double)
    def test_to_padded_tensor_dim4(self, device, dtype):
        ts = [
            torch.randn(16, 21, 13, device=device, dtype=dtype),
            torch.randn(24, 32, 14, device=device, dtype=dtype),
            torch.randn(40, 53, 16, device=device, dtype=dtype),
        ]
        nt = torch.nested.nested_tensor(ts, device=device, dtype=dtype)
        pad = 42
        correct_output = []
        for t in ts:
            next_output = torch.ones_like(ts[2]) * pad
            correct_output.append(next_output)
            next_output[:t.size(0), :t.size(1), :t.size(2)].copy_(t)
        correct_output = torch.stack(correct_output)
        padded = torch.nested.to_padded_tensor(nt, pad)
        self.assertEqual(padded, correct_output)

    # TODO: test noncontiguous to_padded_tensor
    # For now this tests the functionality of noncontiguous_to_padded_tensor
    # and the error message of to_padded_tensor
    # since to_padded_tensor does not support noncontiguous buffer yet
    @dtypes(torch.float, torch.float16, torch.double)
    @torch.inference_mode()
    def test_to_padded_tensor_noncontiguous(self, device, dtype):
        nt_contiguous, nt_noncontiguous = random_nt_noncontiguous_pair((2, 3, 6, 7), device, dtype)
        # test noncontiguous_to_padded_tensor functionality
        self.assertEqual(
            torch.nested.to_padded_tensor(nt_contiguous, 0.0),
            noncontiguous_to_padded_tensor(nt_noncontiguous))
        # test to_padded_tensor error message
        self.assertRaisesRegex(
            RuntimeError,
            r"for now to_padded_tensor only supports contiguous nested tensor",
            lambda: torch.nested.to_padded_tensor(nt_noncontiguous, 0.0)
        )

    @skipMeta
    def test_device_checks(self, device):
        nt = torch.nested.nested_tensor([], device=device)
        is_cuda = 'cuda' in str(device)
        self.assertEqual(nt.is_cuda, is_cuda)

    @dtypes(torch.float, torch.float16, torch.double)
    def test_nested_tensor_indexing(self, device, dtype):
        # edge case: empty nested tensor
        nt0 = torch.nested.nested_tensor([])
        self.assertRaises(IndexError, lambda: nt0[0])
        # normal case
        x0 = torch.randn((2, 5), device=device, dtype=dtype)
        x1 = torch.randn((3, 4), device=device, dtype=dtype)
        nt = torch.nested.nested_tensor([x0, x1])
        # single index: only support integer in the batch dimension
        self.assertEqual(nt[0], x0)
        self.assertEqual(nt[-1], x1)
        self.assertRaises(IndexError, lambda: nt[2])
        self.assertRaises(IndexError, lambda: nt[-3])
        self.assertRaises(NotImplementedError, lambda: nt[:])
        self.assertRaises(NotImplementedError, lambda: nt[...])
        # tuple of indices: only support integer in the batch dimension
        #                 + all possible indexing in the original tensor dimensions
        self.assertEqual(nt[0, 0, 0], x0[0, 0])
        self.assertEqual(nt[0, 1, :], x0[1, :])
        self.assertEqual(nt[1, ...], x1)
        self.assertRaises(IndexError, lambda: nt[1, 4, 2])
        self.assertRaises(NotImplementedError, lambda: nt[:, 1, 1])
        # make sure indexing returns a view
        nt[0].fill_(100.0)
        answer = torch.tensor(100.0, device=device, dtype=dtype).expand((2, 5))
        self.assertEqual(nt[0], answer)
        nt[1, 1, :].fill_(200.0)
        answer = torch.tensor(200.0, device=device, dtype=dtype).expand(4)
        self.assertEqual(nt[1, 1, :], answer)

        # Test that indexing works when requires_grad_(True)
        # previously this was failing because the backward kernel for select.int uses .sizes()
        nt = torch.nested.nested_tensor([x0, x1]).requires_grad_(True)
        self.assertEqual(nt[0], x0)
        self.assertEqual(nt[-1], x1)
        grad_x0 = torch.randn((2, 5), device=device, dtype=dtype)
        nt[0].backward(grad_x0)
        expected_grad = torch.nested.nested_tensor([grad_x0, torch.zeros((3, 4), device=device, dtype=dtype)])
        self.assertEqual(nt.grad, expected_grad)

    @parametrize("func", [subtest(torch.nn.functional.relu, name='relu'),
                          subtest(torch.nn.functional.relu_, name='relu_'),
                          subtest(torch.nn.functional.gelu, name='gelu'),
                          subtest(torch._C._nn.gelu_, name='gelu_'),
                          subtest(torch.tanh, name='tanh'),
                          subtest(torch.tanh_, name='tanh_'),
                          subtest(torch.neg, name='neg')])
    def test_activations(self, device, func):
        nt, nt_noncontiguous = random_nt_noncontiguous_pair((2, 3, 6, 7), device=device, dtype=torch.float32)
        nested_result = func(nt)
        self.assertTrue(nested_result.is_nested)
        for t, t_res in zip(nt.unbind(), nested_result.unbind()):
            self.assertEqual(func(t), t_res)
        self.assertRaisesRegex(
            RuntimeError,
            "NestedTensor must be contiguous to get buffer.",
            lambda: func(nt_noncontiguous))

    @dtypes(*floating_types_and_half())
    def test_nested_tensor_chunk(self, device, dtype):
        # Transformer use case
        a = torch.randn(3, 3 * 4, device=device, dtype=dtype)
        b = torch.randn(2, 3 * 4, device=device, dtype=dtype)
        c = torch.randn(1, 3 * 4, device=device, dtype=dtype)
        a_chunks = a.chunk(3, dim=-1)
        b_chunks = b.chunk(3, dim=-1)
        c_chunks = c.chunk(3, dim=-1)

        a_nt = [a_chunks[0], b_chunks[0], c_chunks[0]]
        b_nt = [a_chunks[1], b_chunks[1], c_chunks[1]]
        c_nt = [a_chunks[2], b_chunks[2], c_chunks[2]]

        nt = torch.nested.nested_tensor([a, b, c])
        chunked = nt.chunk(3, dim=-1)

        self.assertEqual(chunked[0], torch.nested.nested_tensor(a_nt))
        self.assertEqual(chunked[1], torch.nested.nested_tensor(b_nt))
        self.assertEqual(chunked[2], torch.nested.nested_tensor(c_nt))

        for chunk in chunked:
            self.assertFalse(chunk.is_contiguous())

        # Failure chunking on ragged dimensions
        self.assertRaisesRegex(
            RuntimeError, "Chunk for nested tensors is currently only supported for the last dimension.",
            lambda: torch.chunk(nt, 5, dim=1))
        self.assertRaisesRegex(
            RuntimeError, "Chunk for nested tensors is currently only supported for the last dimension.",
            lambda: torch.chunk(nt, 5, dim=0))

        # Failure on non-contiguous nt
        _, nt_noncontiguous = random_nt_noncontiguous_pair((2, 3), device, dtype)
        self.assertRaisesRegex(
            RuntimeError, "chunk expects `self` to be contiguous.", lambda: torch.chunk(nt_noncontiguous, 5, dim=-1))

        # Failure when calling non divisible n_chunks
        self.assertRaisesRegex(
            RuntimeError, "Chunk for nested tensors is only supported for "
            "nested tensors with trailing dimension divisible by chunks.",
            lambda: torch.chunk(nt, 5, dim=-1))

        # Failure when calling backward on a chunk
        a = torch.randn(3, 3 * 4, device=device, dtype=dtype, requires_grad=True)
        b = torch.randn(2, 3 * 4, device=device, dtype=dtype, requires_grad=True)
        nt_grad = torch.nested.as_nested_tensor([a, b])
        chunked = torch.chunk(nt_grad, 2, dim=-1)
        self.assertRaisesRegex(RuntimeError, "derivative for aten::chunk is not implemented",
                               lambda: chunked[0].backward(chunked[0].clone()))

    @dtypes(torch.float, torch.float16, torch.double)
    @torch.inference_mode()
    def test_nested_tensor_indexing_noncontiguous(self, device, dtype):
        nt_contiguous, nt_noncontiguous = random_nt_noncontiguous_pair((2, 3, 6, 7), device, dtype)
        self.assertEqual(nt_contiguous.size(0), nt_noncontiguous.size(0))
        n = nt_contiguous.size(0)
        for i in range(n):
            self.assertEqual(nt_contiguous[i], nt_noncontiguous[i])

    @dtypes(torch.float, torch.float16)
    @skipMeta
    @torch.inference_mode()
    def test_nested_tensor_add(self, device, dtype):
        (nt1, nt2) = self.random_nt_pair(device, dtype, 4, (4, 4))
        ref = torch.nested.nested_tensor([t1 + t2 for (t1, t2) in zip(nt1.unbind(), nt2.unbind())])
        out = nt1 + nt2
        self.assertEqual(ref, out)

    @onlyCUDA
    @dtypes(torch.float, torch.float16)
    @torch.inference_mode()
    @parametrize("embedding_dim", [8, 128, 256, 384])
    def test_nested_tensor_dense_elementwise(self, device, dtype, embedding_dim):
        batch_size = 32
        seq_lens = torch.randint(low=0, high=10, size=(batch_size,))
        ts = [torch.randn((seq_len, embedding_dim)) for seq_len in seq_lens]
        nt = torch.nested.nested_tensor(ts, device=device, dtype=dtype)
        t = torch.randn((batch_size, 1, embedding_dim), device=device, dtype=dtype)
        ref_add = torch.nested.nested_tensor([t1 + t2 for (t1, t2) in zip(nt.unbind(), t.unbind())])
        ref_mul = torch.nested.nested_tensor([t1 * t2 for (t1, t2) in zip(nt.unbind(), t.unbind())])
        self.assertEqual(nt.add(t), ref_add)
        self.assertEqual(nt.mul(t), ref_mul)

    @dtypes(torch.float, torch.float16)
    @skipMeta
    @torch.inference_mode()
    def test_nested_tensor_mul(self, device, dtype):
        # nested tensor * nested tensor
        (nt1, nt2) = self.random_nt_pair(device, dtype, 4, (4, 4))
        ref = torch.nested.nested_tensor([t1 * t2 for (t1, t2) in zip(nt1.unbind(), nt2.unbind())])
        out = nt1 * nt2
        self.assertEqual(ref, out)
        # nested tensor * scalar
        number = 10.0
        scalar = torch.tensor(number).to(dtype).to(device)
        ref = torch.nested.nested_tensor([t * number for t in nt1.unbind()])
        out_number0 = nt1 * number
        out_number1 = number * nt1
        out_scalar0 = nt1 * scalar
        out_scalar1 = scalar * nt1
        self.assertEqual(out_number0, ref)
        self.assertEqual(out_number1, ref)
        self.assertEqual(out_scalar0, ref)
        self.assertEqual(out_scalar1, ref)
        # error case: numel == 1 but dim > 0
        vector = torch.tensor([number]).to(dtype).to(device)
        self.assertRaisesRegex(
            RuntimeError,
            "Expected both self and other to be nested, but got a nested self and non-nested other",
            lambda: nt1.mul(vector)
        )
        self.assertRaisesRegex(
            RuntimeError,
            "Expected both self and other to be nested, but got a non-nested self and nested other",
            lambda: vector.mul(nt1)
        )

    @dtypes(torch.float, torch.float16)
    @skipMeta
    @torch.inference_mode()
    def test_nested_tensor_div(self, device, dtype):
        nt, nt2 = self.random_nt_pair(device, dtype, 4, (4, 4))
        scale = 4.0
        ref = torch.nested.nested_tensor([t / scale for t in nt.unbind()])
        out = nt / 4.0
        self.assertEqual(ref, out)
        ref_transposed = ref.transpose(1, 2)
        out = nt.transpose(1, 2) / 4.0
        self.assertEqual(ref_transposed, out)

        ref = torch.nested.nested_tensor([t / t2 for (t, t2) in zip(nt.unbind(), nt2.unbind())])
        out = nt / nt2
        self.assertEqual(ref, out)

        out = nt.transpose(1, 2) / nt2.transpose(1, 2)
        self.assertEqual(ref.transpose(1, 2), out)

        nt_transpose_copy = torch.nested.nested_tensor([t.transpose(0, 1) for t in nt.unbind()])

        self.assertRaisesRegex(
            RuntimeError, "div requires strides to match when given NestedTensors",
            lambda: nt_transpose_copy.transpose(1, 2) / nt2)

        nt = torch.nested.nested_tensor([torch.randn(i, 4) for i in [3, 4, 5]], device=device, dtype=dtype)
        nt_chunks = nt.chunk(2, -1)
        self.assertRaisesRegex(
            RuntimeError, "div requires offsets to match when given NestedTensors",
            lambda: nt_chunks[0] / nt_chunks[1])

    @dtypes(torch.float, torch.float16)
    @skipMeta
    @torch.inference_mode()
    def test_nested_tensor_add_in_place(self, device, dtype):
        (nt1, nt2) = self.random_nt_pair(device, dtype, 4, (4, 4))
        ref = torch.nested.nested_tensor([t1 + t2 for (t1, t2) in zip(nt1.unbind(), nt2.unbind())])
        nt1 += nt2
        self.assertEqual(ref, nt1)

    @dtypes(torch.float, torch.float16)
    @skipMeta
    @torch.inference_mode()
    def test_nested_tensor_mul_in_place(self, device, dtype):
        # nested tensor * nested tensor
        (nt1, nt2) = self.random_nt_pair(device, dtype, 4, (4, 4))
        ref = torch.nested.nested_tensor([t1 * t2 for (t1, t2) in zip(nt1.unbind(), nt2.unbind())])
        nt1 *= nt2
        self.assertEqual(ref, nt1)
        # nested tensor * scalar
        number = 10.0
        scalar = torch.tensor(number).to(dtype).to(device)
        ref = torch.nested.nested_tensor([t * number for t in nt1.unbind()])
        out_number = nt1.clone()
        out_number *= number
        out_scalar = nt1.clone()
        out_scalar *= scalar
        self.assertEqual(out_number, ref)
        self.assertEqual(out_scalar, ref)
        self.assertRaisesRegex(
            RuntimeError,
            r"output with shape \[.*\] doesn't match the broadcast shape \[.*\]",
            lambda: scalar.mul_(nt1)
        )
        # error case: numel == 1 but dim > 0
        vector = torch.tensor([number]).to(dtype).to(device)
        self.assertRaisesRegex(
            RuntimeError,
            "Expected both self and other to be nested, but got a nested self and non-nested other",
            lambda: nt1.mul_(vector)
        )
        self.assertRaisesRegex(
            RuntimeError,
            "Expected both self and other to be nested, but got a non-nested self and nested other",
            lambda: vector.mul_(nt1)
        )

    @onlyCPU
    @skipMeta
    @dtypes(torch.float)
    def test_nested_tensor_sum_dim(self, device, dtype):
        params = ((2, (1, 1)), ((4), (4, 4)), (10, (3, 5, 7)))

        def test_sum(device, dtype, ntensors, max_sizes, dim, keepdim=True):
            nt = random_nt(device, dtype, ntensors, max_sizes)
            nt2 = nt.clone()
            ub2 = nt2.unbind()
            nt.requires_grad_(True)
            [t.requires_grad_(True) for t in ub2]
            nt_sum = nt.sum(dim=dim, keepdim=keepdim)
            ub2_sum = [t.sum(-1, keepdim=keepdim) for t in ub2]
            self.assertEqual(nt_sum, torch.nested.nested_tensor(ub2_sum))

            # test backward
            # generate gradient tensor that has the same size as the output
            size = nt_sum._nested_tensor_size()
            gt2 = []
            for i in range(ntensors):
                gt2.append(torch.randn(size[i].tolist(), device=device, dtype=dtype))
            gt = torch.nested.nested_tensor(gt2).clone()
            nt_sum.backward(gt)
            for t2, g2 in zip(ub2_sum, gt2):
                t2.backward(g2)
            self.assertEqual(nt.grad, torch.nested.nested_tensor([t.grad for t in ub2]))
            return

        for ntensors, max_sizes in params:
            test_sum(device, dtype, ntensors, max_sizes, len(max_sizes))

        # Test error inputs
        with self.assertRaisesRegex(RuntimeError, "NestedTensor can only be reduced across the last"):
            torch.nested.nested_tensor([torch.tensor([3, 4, 5]), torch.tensor([1, 2])]).sum(0, keepdim=True)

        with self.assertRaisesRegex(RuntimeError, "NestedTensor only allows reduction of a single"):
            torch.nested.nested_tensor([torch.tensor([[3, 4, 5]]), torch.tensor([[1, 2]])]).sum([0, 1], keepdim=True)

        with self.assertRaisesRegex(RuntimeError, "NestedTensor always requires keepdim=True for now."):
            torch.nested.nested_tensor([torch.tensor([3, 4, 5]), torch.tensor([1, 2])]).sum(-1)

    @dtypes(torch.float, torch.float16)
    def test_contiguous(self, device, dtype):
        # Since we don't have access to the buffer in python this is harder to show what
        # we are testing for. When we call chunk on a consistent dim of a NT
        # for chunk_size > 1 the resulting tensors are views of the original NT
        # whose numels is now less than the size of the buffer. Clone was
        # previously creating a new NT with a buffer that was the same size as the
        # original.
        nt_contiguous = torch.nested.nested_tensor([torch.randn(2, 20, device=device, dtype=dtype),
                                                    torch.randn(4, 20, device=device, dtype=dtype)])
        # Split up the last dimension which has a consistent size of 20 into 5 chunks
        chunks = nt_contiguous.chunk(5, dim=-1)

        # # Check chunks are contiguous after calling contiguous
        for chunk in chunks:
            self.assertFalse(chunk.is_contiguous())
            self.assertTrue(chunk.contiguous().is_contiguous())

    @dtypes(torch.float, torch.float16)
    @skipMeta
    def test_clone(self, device, dtype):
        nt1 = random_nt(device, dtype, 4, (4, 4), (1, 1))
        nt2 = nt1.clone()
        # Verify the values match
        self.assertEqual(nt1, nt2)
        # Verify modifying nt2 doesn't affect nt1
        nt2.mul_(nt1)
        ub1 = nt1.unbind()
        ub2 = nt2.unbind()
        for i in range(len(ub1)):
            self.assertNotEqual(ub1[i], ub2[i])

        nt1.clone(memory_format=torch.preserve_format)
        msg = "Nested tensor clone supports Preserve and Contiguous memory formats, called clone with memory format: ChannelsLast"
        with self.assertRaisesRegex(RuntimeError, msg):
            nt1.clone(memory_format=torch.channels_last)

    # cannot test torch.float16 because: RuntimeError: "bernoulli_scalar_cpu_" not implemented for 'Half'
    @dtypes(torch.float, torch.double)
    def test_dropout(self, device, dtype):
        # edge case: empty nested tensor
        nt0 = torch.nested.nested_tensor([])
        y = torch.nn.functional.dropout(nt0, 0.5)
        self.assertEqual(nt0, y)
        # normal nested tensor
        ntensors = 4
        nt = random_nt(device, dtype, ntensors, (4, 4))
        # edge case: invalid dropout
        self.assertRaises(ValueError, lambda: torch.nn.Dropout(-0.1))
        self.assertRaises(ValueError, lambda: torch.nn.Dropout(1.1))
        self.assertRaises(ValueError, lambda: torch.nn.functional.dropout(nt, -0.1))
        self.assertRaises(ValueError, lambda: torch.nn.functional.dropout(nt, 1.1))
        # edge case: no dropout
        dropouter = torch.nn.Dropout(0.0)
        y0 = dropouter(nt)
        y1 = torch.nn.functional.dropout(nt, 0.0)
        self.assertEqual(nt, y0)
        self.assertEqual(nt, y1)
        # edge case: all dropout
        dropouter = torch.nn.Dropout(1.0)
        y0 = dropouter(nt)
        y1 = torch.nn.functional.dropout(nt, 1.0)
        nt0 = nt.clone()
        for i in range(ntensors):
            nt0[i].fill_(0.0)
        self.assertEqual(nt0, y0)
        self.assertEqual(nt0, y1)
        # normal case: normal dropout
        p = 0.2
        y = torch.nn.functional.dropout(nt, p)
        expect = nt.clone()
        for i in range(ntensors):
            actual_tensor = y[i].view(-1)
            expect_tensor = expect[i].view(-1)
            for j in range(actual_tensor.shape[0]):
                if actual_tensor[j].item() == 0.0:
                    expect_tensor[j] = 0.0
                else:
                    expect_tensor[j] /= 1.0 - p
        self.assertEqual(y, expect)
        with freeze_rng_state():
            dropouter = torch.nn.Dropout(p)
            y0 = dropouter(nt)
        with freeze_rng_state():
            y1 = torch.nn.functional.dropout(nt, p)
        self.assertEqual(y0, y1)

    @dtypes(torch.float, torch.double)
    def test_dropout_noncontiguous(self, device, dtype):
        ntensors = 4
        nt0 = random_nt(device, dtype, ntensors, (4, 4))
        nt1 = nt0.transpose(-1, -2)
        p = 0.3
        with freeze_rng_state():
            dropouter = torch.nn.Dropout(p)
            y0 = dropouter(nt0)
        with freeze_rng_state():
            y1 = torch.nn.functional.dropout(nt1, p).transpose(-1, -2)
        self.assertEqual(y0, y1)

    # cannot test torch.float16 because: RuntimeError: "softmax_kernel_impl" not implemented for 'Half'
    @dtypes(torch.float, torch.double)
    def test_softmax(self, device, dtype):
        # normal nested tensor
        ntensors = 4
        nt = random_nt(device, dtype, ntensors, (4, 4))
        # error case: softmax across nested dimension
        self.assertRaisesRegex(
            RuntimeError,
            "Cannot apply softmax across nested dimension 0",
            lambda: torch.nn.functional.softmax(nt, 0)
        )
        self.assertRaisesRegex(
            RuntimeError,
            "Cannot apply softmax across nested dimension 0",
            lambda: torch.nn.functional.softmax(nt, -3)
        )
        # error case: dimension out of range
        self.assertRaises(IndexError, lambda: torch.nn.functional.softmax(nt, 3))
        self.assertRaises(IndexError, lambda: torch.nn.functional.softmax(nt, -4))
        # normal case: should equal to padding -inf
        softmaxer = torch.nn.Softmax(1)
        y0 = softmaxer(nt)
        y1 = torch.nn.functional.softmax(nt, 1)
        self.assertEqual(y0, y1)
        pt = torch.nested.to_padded_tensor(nt, float("-inf"))
        # if an entire slice is padded, then softmax will return 0.0 / 0.0 = nan
        # however, physically speaking that should be 0.0
        expect = torch.nn.functional.softmax(pt, 1).nan_to_num_(0.0)
        self.assertEqual(torch.nested.to_padded_tensor(y0, 0.0), expect)
        # edge case: empty nested tensor
        nt0 = torch.nested.nested_tensor([])
        y = torch.nn.functional.softmax(nt0, 1)
        self.assertEqual(nt0, y)
        # edge case: nesting scalars
        nt1 = torch.nested.nested_tensor([torch.tensor(0.0), torch.tensor(1.0)])
        self.assertRaises(RuntimeError, lambda: torch.nn.functional.softmax(nt1, 0))
        self.assertRaises(IndexError, lambda: torch.nn.functional.softmax(nt1, 1))

    @dtypes(torch.float, torch.double)
    @torch.inference_mode()
    def test_softmax_noncontiguous(self, device, dtype):
        nt_contiguous, nt_noncontiguous = random_nt_noncontiguous_pair((2, 3, 6, 7), device, dtype)
        self.assertEqual(
            torch.nn.functional.softmax(nt_contiguous, -1),
            torch.nn.functional.softmax(nt_noncontiguous, -1))

    def _test_bmm(self, device, dtype):
        # error case: one is nested but the other is not
        nt = torch.nested.nested_tensor([torch.randn(2), torch.randn(3)], device=device, dtype=dtype)
        t = torch.randn(4, device=device, dtype=dtype)
        self.assertRaisesRegex(
            RuntimeError,
            "Expected both to be nested, but got a nested self and non-nested other",
            lambda: nt.bmm(t)
        )
        self.assertRaisesRegex(
            RuntimeError,
            "Expected both to be nested, but got a non-nested self and nested other",
            lambda: t.bmm(nt)
        )
        # error case: not 3D tensors
        nt0 = torch.nested.nested_tensor([], device=device, dtype=dtype)
        nt1 = torch.nested.nested_tensor([torch.randn(2), torch.randn(3)], device=device, dtype=dtype)
        nt2 = torch.nested.nested_tensor([torch.randn((2, 4)), torch.randn((3, 4))], device=device, dtype=dtype)
        self.assertRaisesRegex(
            RuntimeError,
            "batch1 must be a 3D tensor",
            lambda: nt0.bmm(nt0)
        )
        self.assertRaisesRegex(
            RuntimeError,
            "batch1 must be a 3D tensor",
            lambda: nt0.bmm(nt1)
        )
        self.assertRaisesRegex(
            RuntimeError,
            "batch1 must be a 3D tensor",
            lambda: nt0.bmm(nt2)
        )
        self.assertRaisesRegex(
            RuntimeError,
            "batch1 must be a 3D tensor",
            lambda: nt1.bmm(nt0)
        )
        self.assertRaisesRegex(
            RuntimeError,
            "batch1 must be a 3D tensor",
            lambda: nt1.bmm(nt1)
        )
        self.assertRaisesRegex(
            RuntimeError,
            "batch1 must be a 3D tensor",
            lambda: nt1.bmm(nt2)
        )
        self.assertRaisesRegex(
            RuntimeError,
            "batch2 must be a 3D tensor",
            lambda: nt2.bmm(nt0)
        )
        self.assertRaisesRegex(
            RuntimeError,
            "batch2 must be a 3D tensor",
            lambda: nt2.bmm(nt1)
        )
        # error case: incompatible batch size
        nt0 = torch.nested.nested_tensor([torch.randn((2, 4)), torch.randn((3, 4))], device=device, dtype=dtype)
        nt1 = torch.nested.nested_tensor([torch.randn((4, 6)),
                                          torch.randn((4, 5)),
                                          torch.randn((4, 7))],
                                         device=device, dtype=dtype)
        self.assertRaisesRegex(
            RuntimeError,
            "Expected size for the 1st dimension of batch2 tensor to be: 2 but got: 3.",
            lambda: nt0.bmm(nt1)
        )
        self.assertRaisesRegex(
            RuntimeError,
            "Expected size for the 1st dimension of batch2 tensor to be: 3 but got: 2.",
            lambda: nt1.bmm(nt0)
        )
        # error case: underlying matrices cannot be multiplied
        nt0 = torch.nested.nested_tensor([torch.randn((2, 4)), torch.randn((3, 4))], device=device, dtype=dtype)
        self.assertRaisesRegex(
            RuntimeError,
            r"0-th nested matrices in batch cannot be multiplied \(2x4 and 2x4\)",
            lambda: nt0.bmm(nt0)
        )
        # normal nested tensor
        nt0 = torch.nested.nested_tensor([torch.randn((2, 4)), torch.randn((3, 7))], device=device, dtype=dtype)
        nt1 = torch.nested.nested_tensor([torch.randn((4, 6)), torch.randn((7, 5))], device=device, dtype=dtype)
        actual = torch.nested.to_padded_tensor(nt0.bmm(nt1), 0.0)
        expect = torch.nested.to_padded_tensor(nt0, 0.0).bmm(torch.nested.to_padded_tensor(nt1, 0.0))
        if dtype == torch.float16:
            self.assertEqual(actual, expect, rtol=1e-3, atol=1e-3)
        else:
            self.assertEqual(actual, expect)

        # test tensorcore path
        nt0 = torch.nested.nested_tensor([torch.randn((2, 8)), torch.randn((3, 16))], device=device, dtype=dtype)
        nt1 = torch.nested.nested_tensor([torch.randn((8, 8)), torch.randn((16, 8))], device=device, dtype=dtype)
        actual = torch.nested.to_padded_tensor(nt0.bmm(nt1), 0.0)
        expect = torch.nested.to_padded_tensor(nt0, 0.0).bmm(torch.nested.to_padded_tensor(nt1, 0.0))
        if dtype == torch.float16:
            self.assertEqual(actual, expect, rtol=1e-3, atol=1e-3)
        else:
            self.assertEqual(actual, expect)

    @onlyCUDA
    @dtypes(torch.float, torch.double, torch.float16)
    def test_bmm_cuda(self, device, dtype):
        self._test_bmm(device, dtype)

    @onlyCPU
    # cannot test torch.float16 because: RuntimeError: "addmm_impl_cpu_" not implemented for 'Half'
    @dtypes(torch.float, torch.double)
    def test_bmm_cpu(self, device, dtype):
        self._test_bmm(device, dtype)

    # cannot test torch.float16 because: RuntimeError: "addmm_impl_cpu_" not implemented for 'Half'
    @dtypes(torch.float, torch.double)
    def test_bmm_noncontiguous(self, device, dtype):
        nt0_contiguous, nt0_noncontiguous = random_nt_noncontiguous_pair((2, 3), device, dtype)
        nt1_contiguous, nt1_noncontiguous = random_nt_noncontiguous_pair((6, 7), device, dtype)
        self.assertEqual(
            nt0_contiguous.transpose(-1, -2).bmm(nt1_contiguous),
            nt0_noncontiguous.transpose(-1, -2).bmm(nt1_noncontiguous))

    @dtypes(torch.float, torch.double)
    def test_matmul_with_bmm_path(self, device, dtype):
        def unbind_rebind_matmul(nt1, nt2):
            t1s = nt1.unbind()
            t2s = nt2.unbind()
            out_ts = [t1.matmul(t2) for t1, t2 in zip(t1s, t2s)]
            return torch.nested.nested_tensor(out_ts)

        # [N, n_head, *, head_dim], [N, n_head, head_dim, *]
        N = np.random.randint(2, 5)
        n_heads = np.random.randint(2, 5)
        head_dim = 3
        t1s = []
        t2s = []
        for _ in range(N):
            seq_len1 = np.random.randint(2, 5)
            seq_len2 = np.random.randint(2, 5)
            t1s.append(torch.randn(n_heads, seq_len1, head_dim))
            t2s.append(torch.randn(n_heads, head_dim, seq_len2))
        nt1 = torch.nested.nested_tensor(t1s, device=device, dtype=dtype)
        nt2 = torch.nested.nested_tensor(t2s, device=device, dtype=dtype)
        self.assertEqual(torch.matmul(nt1, nt2), unbind_rebind_matmul(nt1, nt2))

        # test with noncontiguous
        t3s = []
        t4s = []
        for _ in range(N):
            seq_len = np.random.randint(2, 5)
            t3s.append(torch.randn(seq_len, n_heads, head_dim))
            t4s.append(torch.randn(seq_len, n_heads, head_dim))
        nt3 = torch.nested.nested_tensor(t3s, device=device, dtype=dtype).transpose(1, 2)
        nt4 = torch.nested.nested_tensor(t4s, device=device, dtype=dtype).transpose(1, 2).transpose(2, 3)
        self.assertEqual(torch.matmul(nt3, nt4), unbind_rebind_matmul(nt3, nt4))

    # cannot test torch.float16 because: RuntimeError: "bmm" not implemented for 'Half'
    @dtypes(torch.float, torch.double)
    def test_matmul(self, device, dtype):
        # error case: one is nested but the other is not
        nt = torch.nested.nested_tensor([torch.randn(2), torch.randn(3)], device=device, dtype=dtype)
        t = torch.randn(4, device=device, dtype=dtype)
        self.assertRaisesRegex(
            RuntimeError,
            "Expected both to be nested, but got a nested self and non-nested other",
            lambda: torch.matmul(nt, t)
        )
        self.assertRaisesRegex(
            RuntimeError,
            "Expected both to be nested, but got a non-nested self and nested other",
            lambda: torch.matmul(t, nt)
        )
        # error case: not 3+D tensors
        nt0 = torch.nested.nested_tensor([], device=device, dtype=dtype)
        nt1 = torch.nested.nested_tensor([torch.randn(2), torch.randn(3)], device=device, dtype=dtype)
        nt2 = torch.nested.nested_tensor([torch.randn((2, 4)), torch.randn((3, 4))], device=device, dtype=dtype)
        self.assertRaisesRegex(
            RuntimeError,
            r"matmul: For nested tensors, only inputs with >= 3 dims are currently supported. 1st input has rank: [0-9]+",
            lambda: torch.matmul(nt0, nt0)
        )
        self.assertRaisesRegex(
            RuntimeError,
            r"matmul: For nested tensors, only inputs with >= 3 dims are currently supported. 1st input has rank: [0-9]+",
            lambda: torch.matmul(nt0, nt1)
        )
        self.assertRaisesRegex(
            RuntimeError,
            r"matmul: For nested tensors, only inputs with >= 3 dims are currently supported. 1st input has rank: [0-9]+",
            lambda: torch.matmul(nt0, nt2)
        )
        self.assertRaisesRegex(
            RuntimeError,
            r"matmul: For nested tensors, only inputs with >= 3 dims are currently supported. 1st input has rank: [0-9]+",
            lambda: torch.matmul(nt1, nt0)
        )
        self.assertRaisesRegex(
            RuntimeError,
            r"matmul: For nested tensors, only inputs with >= 3 dims are currently supported. 1st input has rank: [0-9]+",
            lambda: torch.matmul(nt1, nt1)
        )
        self.assertRaisesRegex(
            RuntimeError,
            r"matmul: For nested tensors, only inputs with >= 3 dims are currently supported. 1st input has rank: [0-9]+",
            lambda: torch.matmul(nt1, nt2)
        )
        self.assertRaisesRegex(
            RuntimeError,
            r"matmul: For nested tensors, only inputs with >= 3 dims are currently supported. 2nd input has rank: [0-9]+",
            lambda: torch.matmul(nt2, nt0)
        )
        self.assertRaisesRegex(
            RuntimeError,
            r"matmul: For nested tensors, only inputs with >= 3 dims are currently supported. 2nd input has rank: [0-9]+",
            lambda: torch.matmul(nt2, nt1)
        )
        # error case: incompatible batch size
        nt0 = torch.nested.nested_tensor([torch.randn((2, 4)), torch.randn((3, 4))], device=device, dtype=dtype)
        nt1 = torch.nested.nested_tensor([torch.randn((4, 6)),
                                          torch.randn((4, 5)),
                                          torch.randn((4, 7))],
                                         device=device, dtype=dtype)
        self.assertRaisesRegex(
            RuntimeError,
            r"matmul: Expected size for the 1st dimension of 2nd input tensor to be: [0-9]+ but got: [0-9]+.",
            lambda: torch.matmul(nt0, nt1)
        )
        self.assertRaisesRegex(
            RuntimeError,
            r"matmul: Expected size for the 1st dimension of 2nd input tensor to be: [0-9]+ but got: [0-9]+.",
            lambda: torch.matmul(nt1, nt0)
        )
        # error case: incompatible (wrong) batch sizes that shouldn't even broadcast?
        nt0 = torch.nested.nested_tensor([torch.randn((2, 2, 4)),
                                          torch.randn((2, 3, 4))],
                                         device=device, dtype=dtype)
        nt1 = torch.nested.nested_tensor([torch.randn((3, 4, 6)),
                                          torch.randn((3, 4, 5))],
                                         device=device, dtype=dtype)
        self.assertRaisesRegex(
            RuntimeError,
            "matmul(): For nested tensors, batch dimensions must have the same sizes,",
            lambda: torch.matmul(nt0, nt1)
        )
        # error case: incompatible batch sizes that should technically broadcast
        nt0 = torch.nested.nested_tensor([torch.randn((2, 2, 4)),
                                          torch.randn((1, 3, 4))],
                                         device=device, dtype=dtype)
        nt1 = torch.nested.nested_tensor([torch.randn((1, 4, 6)),
                                          torch.randn((3, 4, 5))],
                                         device=device, dtype=dtype)
        self.assertRaisesRegex(
            RuntimeError,
            "matmul(): For nested tensors, batch dimensions must have the same sizes,",
            lambda: torch.matmul(nt0, nt1)
        )
        # error case: underlying matrices cannot be multiplied
        nt0 = torch.nested.nested_tensor([torch.randn((2, 4)), torch.randn((3, 4))], device=device, dtype=dtype)
        self.assertRaisesRegex(
            RuntimeError,
            "matmul(): Nested tensors cannot be matrix multiplied",
            lambda: torch.matmul(nt0, nt0)
        )
        # normal nested tensor: 3D
        nt0 = torch.nested.nested_tensor([torch.randn((2, 4)), torch.randn((3, 7))], device=device, dtype=dtype)
        nt1 = torch.nested.nested_tensor([torch.randn((4, 6)), torch.randn((7, 5))], device=device, dtype=dtype)
        actual = torch.nested.to_padded_tensor(torch.matmul(nt0, nt1), 0.0)
        expect = torch.matmul(torch.nested.to_padded_tensor(nt0, 0.0), torch.nested.to_padded_tensor(nt1, 0.0))
        self.assertEqual(actual, expect)
        # normal nested tensor: 4D (with testing for batch_size=1)
        nt0 = torch.nested.nested_tensor([torch.randn((1, 2, 4)),
                                          torch.randn((8, 3, 7))],
                                         device=device, dtype=dtype)
        nt1 = torch.nested.nested_tensor([torch.randn((1, 4, 6)),
                                          torch.randn((8, 7, 5))],
                                         device=device, dtype=dtype)
        actual = torch.nested.to_padded_tensor(torch.matmul(nt0, nt1), 0.0)
        expect = torch.matmul(torch.nested.to_padded_tensor(nt0, 0.0), torch.nested.to_padded_tensor(nt1, 0.0))
        self.assertEqual(actual, expect)
        # normal nested tensor: 5D
        nt0 = torch.nested.nested_tensor([torch.randn((8, 9, 2, 4)),
                                          torch.randn((8, 9, 3, 7))],
                                         device=device, dtype=dtype)
        nt1 = torch.nested.nested_tensor([torch.randn((8, 9, 4, 6)),
                                          torch.randn((8, 9, 7, 5))],
                                         device=device, dtype=dtype)
        actual = torch.nested.to_padded_tensor(torch.matmul(nt0, nt1), 0.0)
        expect = torch.matmul(torch.nested.to_padded_tensor(nt0, 0.0), torch.nested.to_padded_tensor(nt1, 0.0))
        self.assertEqual(actual, expect)

    # cannot test torch.float16 because: RuntimeError: "bmm" not implemented for 'Half'
    @dtypes(torch.float, torch.double)
    def test_matmul_noncontiguous(self, device, dtype):
        nt0_contiguous, nt0_noncontiguous = random_nt_noncontiguous_pair((2, 3), device, dtype)
        nt1_contiguous, nt1_noncontiguous = random_nt_noncontiguous_pair((6, 7), device, dtype)
        self.assertEqual(
            torch.matmul(nt0_contiguous.transpose(-1, -2), nt1_contiguous),
            torch.matmul(nt0_noncontiguous.transpose(-1, -2), nt1_noncontiguous))

    @dtypes(torch.float, torch.double)
    def test_linear(self, device, dtype):
        a = torch.randn(1, 2, device=device, dtype=dtype)
        b = torch.randn(2, 2, device=device, dtype=dtype)
        c = torch.randn(3, 2, device=device, dtype=dtype)
        nt = torch.nested.nested_tensor([a, b, c])

        weight = torch.randn(2, 2, device=device, dtype=dtype)
        bias = torch.randn(2, device=device, dtype=dtype)
        # success case
        torch.functional.F.linear(nt, weight, bias)

        # invalid nested tensor dimension
        msg = r'Linear requires nested_tensor.dim == 3 and dense_matrix.dim == 2. Nested tensor dim: 2. Dense tensor dim: 2'
        nt1 = torch.nested.nested_tensor([torch.randn(1, device=device, dtype=dtype),
                                          torch.randn(2, device=device, dtype=dtype)])
        with self.assertRaisesRegex(RuntimeError, msg):
            torch.functional.F.linear(nt1, weight, bias)

        # invalid weight shape
        msg = r'Linear requires nested_tensor.dim == 3 and dense_matrix.dim == 2. Nested tensor dim: 3. Dense tensor dim: 3'
        weight1 = torch.randn(2, 2, 3, device=device, dtype=dtype)
        with self.assertRaisesRegex(RuntimeError, msg):
            torch.functional.F.linear(nt, weight1, bias)

        # inconsistent last dim of nested tensor
        msg = r"Expected all tensors in nested tensor to have the same trailing dimension, instead last dimension equals:"
        nt2 = torch.nested.nested_tensor([torch.randn(1, 2, device=device, dtype=dtype),
                                          torch.randn(2, 3, device=device, dtype=dtype)])
        with self.assertRaisesRegex(RuntimeError, msg):
            torch.functional.F.linear(nt2, weight, bias)

        # Mismatch of nested tensor last dim and weight dimension
        weight2 = torch.randn(2, 4, device=device, dtype=dtype)
        msg = r"Shape mismatch for NestedTensor Linear: Expected input's \(a nested tensor\) 'last_dim'" \
            r" to equal 'weight.size\(1\), but got: last_dim = 2, and weight.size\(1\) = 4"
        with self.assertRaisesRegex(RuntimeError, msg):
            torch.functional.F.linear(nt, weight2, bias)

        # Nested tensor input and nested weight
        nt_weight = nt.clone()
        msg = r"Linear does not support nested weight when input is a nested tensor."
        with self.assertRaisesRegex(RuntimeError, msg):
            torch.functional.F.linear(nt, nt_weight, bias)

    # TODO: test noncontiguous linear
    # For now this tests the error message of linear
    # since linear does not support noncontiguous buffer yet
    @dtypes(torch.float, torch.double)
    def test_linear_noncontiguous(self, device, dtype):
        nt_contiguous, nt_noncontiguous = random_nt_noncontiguous_pair((2, 3, 6, 7), device, dtype)
        weight = torch.randn((8, 5), device=device, dtype=dtype)
        self.assertRaisesRegex(
            RuntimeError,
            r"for now linear only supports contiguous nested tensor",
            lambda: torch.nn.functional.linear(nt_noncontiguous, weight)
        )

    @dtypes(torch.float, torch.float16, torch.double)
    def test_transpose(self, device, dtype):
        nt = random_nt(device, dtype, 4, (4, 4))
        # error case: transpose nested dimension
        self.assertRaisesRegex(
            RuntimeError,
            "Nested tensor dimension 0 cannot be transposed",
            lambda: nt.transpose(0, 1)
        )
        self.assertRaisesRegex(
            RuntimeError,
            "Nested tensor dimension 0 cannot be transposed",
            lambda: nt.transpose(1, -3)
        )
        # error case: dimension out of range
        self.assertRaises(IndexError, lambda: nt.transpose(1, 3))
        self.assertRaises(IndexError, lambda: nt.transpose(-4, -1))
        # normal case
        ntT = nt.transpose(-1, -2)
        ptT_from_ntT = noncontiguous_to_padded_tensor(ntT)
        pt = torch.nested.to_padded_tensor(nt, 0.0)
        ptT = pt.transpose(-1, -2)
        self.assertEqual(ptT, ptT_from_ntT)

    @dtypes(torch.float, torch.float16, torch.double)
    def test_squeeze_unsqueeze(self, device, dtype):
        a = torch.arange(6).reshape(2, 3)
        b = torch.arange(15).reshape(5, 3)
        nt = torch.nested.nested_tensor([a, b], device=device, dtype=dtype)
        # error case: squeeze no dimension
        self.assertRaisesRegex(
            RuntimeError,
            "For nested tensors, squeeze without the dim argument",
            lambda: nt.squeeze()
        )
        # error case: squeeze nested dimension
        self.assertRaisesRegex(
            RuntimeError,
            "For nested tensors, squeezing dimension 0",
            lambda: nt.squeeze(0)
        )
        # error case: dimension out of range
        self.assertRaises(IndexError, lambda: nt.squeeze(3))
        # error case: squeeze nested tensor of singleton tensors
        c = torch.ones(1)
        nt_singleton = torch.nested.nested_tensor([c, c], device=device, dtype=dtype)
        self.assertRaisesRegex(
            RuntimeError,
            "For nested tensors, squeezing a nested tensor of singleton",
            lambda: nt_singleton.squeeze(1)
        )

        # squeezing a dim which does not have size 1 should be a no-op
        nt2 = nt.squeeze(-1)
        self.assertEqual(nt, nt2)

        # test cases that should work
        for i in range(-2, 3):
            if (i == 0):
                continue
            nt_unsqueezed = nt.unsqueeze(i)
            self.assertTrue(nt_unsqueezed.is_contiguous())
            size_idx = i if i < 0 else i - 1
            self.assertEqual(nt_unsqueezed._nested_tensor_size()[:, size_idx], torch.ones(2, dtype=torch.long))
            nt_squeezed = nt_unsqueezed.squeeze(i)
            self.assertTrue(nt_squeezed.is_contiguous())
            self.assertEqual(nt_squeezed, nt)

    @dtypes(torch.float, torch.float16, torch.double)
    def test_transpose_inference_mode_interaction(self, device, dtype):
        nt = random_nt(device, dtype, 4, (4, 4))
        # Construct in default mode and transpose while in inference mode
        with torch.inference_mode():
            ntT = nt.transpose(-1, -2)
            ptT_from_ntT = noncontiguous_to_padded_tensor(ntT)
            pt = torch.nested.to_padded_tensor(nt, 0.0)
            ptT = pt.transpose(-1, -2)
            self.assertEqual(ptT, ptT_from_ntT)

        # Construct and transpose while in inference mode
        with torch.inference_mode():
            nt = random_nt(device, dtype, 4, (4, 4))
            ntT = nt.transpose(-1, -2)
            ptT_from_ntT = noncontiguous_to_padded_tensor(ntT)
            pt = torch.nested.to_padded_tensor(nt, 0.0)
            ptT = pt.transpose(-1, -2)
            self.assertEqual(ptT, ptT_from_ntT)

    @dtypes(torch.float, torch.float16, torch.double)
    def test_view(self, device, dtype):
        nt = random_nt(device, dtype, 4, (4, 4))
        # error case: empty shape
        self.assertRaisesRegex(
            RuntimeError,
            r"shape '\[\]' is invalid for a nested tensor",
            lambda: nt.view(())
        )
        # error case: empty nested tensor
        nt_empty = torch.nested.nested_tensor([])
        self.assertRaisesRegex(
            RuntimeError,
            "empty nested tensor cannot be reshaped",
            lambda: nt_empty.view(-1)
        )
        # error case: -1 for batch size
        self.assertRaisesRegex(
            RuntimeError,
            r"view: For now nested view cannot change or infer the implicit batch dimension",
            lambda: nt.view(-1, 2, 3)
        )
        self.assertRaisesRegex(
            RuntimeError,
            r"shape '\[.*\]' is invalid for input of size [0-9]+",
            lambda: nt.view(4, 2, 3)
        )
        # normal case
        x0 = torch.randn((2, 20), device=device, dtype=dtype)
        x1 = torch.randn((3, 20), device=device, dtype=dtype)
        nt = torch.nested.nested_tensor([x0, x1])
        pt = torch.nested.to_padded_tensor(nt, 0.0)
        # error case, trying to reshape batch dim to a legit shape
        self.assertRaisesRegex(
            RuntimeError,
            r"For now nested view cannot change or infer the implicit batch dimension",
            lambda: nt.transpose(-1, -2).view(40, -1)
        )
        # inherit only the ragged dimension
        # (2, 20) -> (2, 5, 4)
        # (3, 20) -> (3, 5, 4)
        nt1 = nt.view(2, -1, 5, 4)
        # (2, 3, 20) -> (2, 3, 5, 4) -> (2, 4, 5, 4)
        pt1 = pt.view(2, -1, 5, 4)
        self.assertEqual(noncontiguous_to_padded_tensor(nt1), pt1)

        # more than one -1 (even for "old" dims), should fail
        # this attempts to do # (2, (2, 3), 5, 4) -> (2, (2, 3), 5, 2, 2)
        # but we ban "inherit old behavior" for >1 dimension
        self.assertRaisesRegex(
            RuntimeError,
            r"only one dimension can be inferred",
            lambda: nt1.view(2, -1, -1, 2, 2)
        )

    @dtypes(torch.float, torch.float16, torch.double)
    def test_view_inference_mode_interaction(self, device, dtype):
        # Construct in default mode and view while in inference mode
        nt = torch.nested.nested_tensor([torch.randn((2, 20)), torch.randn((3, 20))], device=device, dtype=dtype)
        with torch.inference_mode():
            ntT = nt.view(2, -1, 4, 5)
            ptT_from_ntT = noncontiguous_to_padded_tensor(ntT)
            pt = torch.nested.to_padded_tensor(nt, 0.0)
            ptT = pt.view(2, -1, 4, 5)
            self.assertEqual(ptT, ptT_from_ntT)
        # Construct and view while in inference mode
        with torch.inference_mode():
            nt = torch.nested.nested_tensor([torch.randn((2, 20)), torch.randn((3, 20))], device=device, dtype=dtype)
            ntT = nt.view(2, -1, 4, 5)
            ptT_from_ntT = noncontiguous_to_padded_tensor(ntT)
            pt = torch.nested.to_padded_tensor(nt, 0.0)
            ptT = pt.view(2, -1, 4, 5)
            self.assertEqual(ptT, ptT_from_ntT)

    @dtypes(torch.float, torch.float16, torch.double)
    def test_reshape(self, device, dtype):
        nt = random_nt(device, dtype, 4, (4, 4))
        # error case: empty shape
        self.assertRaisesRegex(
            RuntimeError,
            r"shape '\[\]' is invalid for a nested tensor",
            lambda: nt.reshape(())
        )
        # error case: empty nested tensor
        nt_empty = torch.nested.nested_tensor([])
        self.assertRaisesRegex(
            RuntimeError,
            "empty nested tensor cannot be reshaped",
            lambda: nt_empty.reshape(-1)
        )
        # error case: -1 for batch size
        self.assertRaisesRegex(
            RuntimeError,
            r"reshape: For now nested reshape cannot change or infer the implicit batch dimension",
            lambda: nt.reshape(-1, 2, 3)
        )
        self.assertRaisesRegex(
            RuntimeError,
            r"shape '\[.*\]' is invalid for input of size [0-9]+",
            lambda: nt.reshape(4, 2, 3)
        )
        # normal case
        x0 = torch.randn((2, 20), device=device, dtype=dtype)
        x1 = torch.randn((3, 20), device=device, dtype=dtype)
        nt = torch.nested.nested_tensor([x0, x1])  # (2, (2, 3), 20)
        pt = torch.nested.to_padded_tensor(nt, 0.0)
        # error case, trying to reshape batch dim to a legit shape
        self.assertRaisesRegex(
            RuntimeError,
            r"reshape: For now nested reshape cannot change or infer the implicit batch dimension",
            lambda: nt.transpose(-1, -2).reshape(40, -1)
        )
        # inherit only the ragged dimension
        # (2, 20) -> (2, 5, 4)
        # (3, 20) -> (3, 5, 4)
        nt1 = nt.reshape(2, -1, 5, 4)
        # (2, 3, 20) -> (2, 3, 5, 4) -> (2, 4, 5, 4)
        pt1 = pt.reshape(2, -1, 5, 4)
        self.assertEqual(noncontiguous_to_padded_tensor(nt1), pt1)

        # more than one -1 (even for "old" dims), should fail
        # this attempts to do # (2, (2, 3), 5, 4) -> (2, (2, 3), 5, 2, 2)
        # but we ban "inherit old behavior" for >1 dimension
        self.assertRaisesRegex(
            RuntimeError,
            r"only one dimension can be inferred",
            lambda: nt1.reshape(2, -1, -1, 2, 2)
        )

    @parametrize("input_dim", [3, 4])
    def test_scaled_dot_product_attention(self, device, input_dim):

        def rand_tensor(*shape):
            return torch.randn(shape, device=device)

        E = 10
        if input_dim == 3:
            # Shape: (N, L, E); ragged L
            query = torch.nested.nested_tensor([rand_tensor(2, E), rand_tensor(3, E), rand_tensor(4, E)])

            # Shape: (N, S, E); ragged S
            key = torch.nested.nested_tensor([rand_tensor(3, E), rand_tensor(4, E), rand_tensor(5, E)])
            value = torch.nested.nested_tensor([rand_tensor(3, E), rand_tensor(4, E), rand_tensor(5, E)])
        elif input_dim == 4:
            # Shape: (N, N', L, E); ragged N' and L
            query = torch.nested.nested_tensor([rand_tensor(2, 2, E), rand_tensor(3, 3, E), rand_tensor(4, 4, E)])
            # Shape: (N, N', S, E); ragged N' and S
            key = torch.nested.nested_tensor([rand_tensor(2, 3, E), rand_tensor(3, 4, E), rand_tensor(4, 5, E)])
            value = torch.nested.nested_tensor([rand_tensor(2, 3, E), rand_tensor(3, 4, E), rand_tensor(4, 5, E)])
        else:
            self.fail(f"Invalid input_dim {input_dim} encountered in SDP test")

        def rand_mask(size):
            return torch.randint(0, 2, size=size, dtype=torch.bool, device=device)

        # Shape: (N, L, S); ragged L and S matching above
        attn_mask = torch.nested.nested_tensor([rand_mask((2, 3)), rand_mask((3, 4)), rand_mask((4, 5))])

        dropout_p = 0.0  # no dropout for reproducibility
        need_attn_weights: bool = True

        # Success case: no attn_mask set and is_causal=False.
        actual = torch.ops.aten._scaled_dot_product_attention(
            query, key, value, attn_mask=None, dropout_p=dropout_p, need_attn_weights=need_attn_weights)

        expected_outputs = []
        expected_attn_weights = []
        for q, k, v in zip(query.unbind(), key.unbind(), value.unbind()):
            (output, attn_weights) = torch.ops.aten._scaled_dot_product_attention(
                q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), attn_mask=None, dropout_p=dropout_p,
                need_attn_weights=need_attn_weights)
            expected_outputs.append(output.squeeze(0))
            expected_attn_weights.append(attn_weights.squeeze(0))
        expected_output_nested = torch.nested.nested_tensor(expected_outputs)
        expected_attn_weight_nested = torch.nested.nested_tensor(expected_attn_weights)
        self.assertEqual(actual[0], expected_output_nested)
        self.assertEqual(actual[1], expected_attn_weight_nested)

        # Error case: explicit attn_mask set.
        with self.assertRaisesRegex(RuntimeError, "not supported when an explicit attn_mask is set"):
            torch.ops.aten._scaled_dot_product_attention(
                query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, need_attn_weights=need_attn_weights)

        # Error case: is_causal=True.
        with self.assertRaisesRegex(RuntimeError, "not supported when is_causal=True"):
            torch.ops.aten._scaled_dot_product_attention(
                query, key, value, dropout_p=dropout_p, need_attn_weights=need_attn_weights, is_causal=True)

    @dtypes(torch.float, torch.float16, torch.double)
    def test_empty_like(self, device, dtype):
        ntensors = 4
        nt = random_nt(device, dtype, ntensors, (4, 4))

        # Create empty on same device as original nested tensor
        nt_empty = torch.empty_like(nt)
        assert nt.is_same_size(nt_empty)
        self.assertEqual(nt.dtype, nt_empty.dtype)
        self.assertEqual(nt.device, nt_empty.device)
        self.assertEqual(nt.layout, nt_empty.layout)

        if torch.cuda.is_available():
            if device == "cpu":
                nt_cuda = torch.empty_like(nt, device='cuda')
                self.assertEqual(torch.device("cuda").type, nt_cuda.device.type)
            else:
                nt_cpu = torch.empty_like(nt, device='cpu')
                self.assertEqual(torch.device("cpu").type, nt_cpu.device.type)

        # Check changing dtype of empty_like nested tensor output
        dtype_set = {torch.float, torch.float16, torch.double}
        for other_dtype in dtype_set - {dtype}:
            nt_empty_other_dtype = torch.empty_like(nt, dtype=other_dtype)
            self.assertEqual(nt.dtype, dtype)
            self.assertEqual(nt_empty_other_dtype.dtype, other_dtype)
            self.assertEqual(nt.device, nt_empty.device)
            self.assertEqual(nt.layout, nt_empty.layout)

        # Create tensor for autograd
        nt_empty_req_grad = torch.empty_like(nt, requires_grad=True)
        self.assertEqual(nt_empty_req_grad.requires_grad, True)

        # Test noncontiguous tensor fails to copy
        nt_cont, nt_noncont = random_nt_noncontiguous_pair((2, 3, 6, 7))
        nt_empty = torch.empty_like(nt_cont)
        assert nt_cont.is_same_size(nt_empty)
        with self.assertRaisesRegex(RuntimeError, "empty_like only supports contiguous memory format for Nested Tensors"):
            nt_empty = torch.empty_like(nt_noncont)


class TestNestedTensorAutograd(TestCase):
    # Note [Gradcheck args check_batched_grad=False] the common_utils testing version of gradcheck
    # includes the default parameters used for testing ops with gradcheck. However nested tensor
    # does not support the stack op therefore we turn it off for these tests
    def _create_leaf_nested_tensor_from_list(self, tensor_device, requires_grad=False):
        return torch.nested.nested_tensor([torch.randn(1, 2,),
                                           torch.randn(7, 8)], requires_grad=requires_grad, device=tensor_device)

    def _create_nested_tensor_from_list(self, tensor_device, requires_grad=False):
        return torch.nested.as_nested_tensor([torch.randn(1, 2, requires_grad=requires_grad),
                                              torch.randn(7, 8, requires_grad=requires_grad)], device=tensor_device)

    def _create_nested_tensor_from_mask(self, tensor_device, requires_grad=False):
        data = torch.randn(2, 3, 4, requires_grad=requires_grad, device=tensor_device)
        mask = torch.ones_like(data[:, :, 0]).bool()
        return torch._nested_tensor_from_mask(data, mask)

    def test_as_nested_tensor_propagates_gradients(self, device):
        a = torch.arange(3, dtype=torch.float, device=device)
        b = torch.arange(5, dtype=torch.float, device=device)
        nt = torch.nested.as_nested_tensor([a, b])
        # tensors with requires_grad=False are leaves
        self.assertTrue(nt.is_leaf)
        self.assertTrue(not nt.requires_grad)

        a = torch.arange(3, dtype=torch.float, requires_grad=True, device=device)
        b = torch.arange(5, dtype=torch.float, requires_grad=True, device=device)
        nt2 = torch.nested.as_nested_tensor([a, b])
        fake_grad = torch.nested.nested_tensor([torch.ones_like(a), torch.zeros_like(b)], device=device)
        nt2.backward(fake_grad)
        self.assertEqual(a.grad, fake_grad[0])
        self.assertEqual(b.grad, fake_grad[1])

    def test_nested_tensor_generates_leaf(self, device):
        a = torch.arange(3, dtype=torch.float, requires_grad=True, device=device)
        b = torch.arange(5, dtype=torch.float, requires_grad=True, device=device)

        nt = torch.nested.nested_tensor([a, b], requires_grad=False)
        self.assertTrue(nt.is_leaf)
        self.assertTrue(not nt.requires_grad)

        nt2 = torch.nested.nested_tensor([a, b], requires_grad=True)
        self.assertTrue(nt2.is_leaf)
        self.assertTrue(nt2.requires_grad)

        fake_grad = torch.nested.nested_tensor([torch.ones_like(a), torch.zeros_like(b)], device=device)
        nt2.backward(fake_grad)
        self.assertEqual(nt2.grad, fake_grad)
        self.assertEqual(a.grad, None)
        self.assertEqual(b.grad, None)

    def test_set_requires_grad_from_list(self, device):
        nt = self._create_nested_tensor_from_list(device)
        nt.requires_grad_()
        assert nt.requires_grad

    def test_set_requires_grad_from_mask(self, device):
        nt = self._create_nested_tensor_from_mask(device)
        nt.requires_grad_()
        assert nt.requires_grad

    def test_backward_for_add_op(self, device):
        nt_1 = self._create_nested_tensor_from_mask(device)
        nt_2 = self._create_nested_tensor_from_mask(device)

        nt_1.requires_grad_()
        c = nt_1 + nt_2

        assert nt_1.requires_grad
        assert c.requires_grad
        grad_output = self._create_nested_tensor_from_mask(device)
        c.backward(grad_output)

        #  Grad check doesn't work with nested yet.
        # d/dnt_1 (nt + nt_1) = 1*grad_output
        self.assertEqual(nt_1.grad, grad_output)

    # Test Factory Functions
    def test_nested_tensor_to_padded_tensor(self, device):
        for padding_val in [0, 1]:
            nt = self._create_leaf_nested_tensor_from_list(tensor_device=device, requires_grad=True)

            out = torch.nested.to_padded_tensor(nt, padding_val)
            grad_output = torch.ones(out.shape, device=device)
            out.backward(grad_output)

            self.assertEqual(nt.grad, torch.nested.nested_tensor([torch.ones(1, 2), torch.ones(7, 8)], device=device))

    def test_nested_tensor_from_mask_and_to_padded(self, device):
        N, L, D = 2, 4, 4
        mask = torch.ones(N, L, device=device)
        for i in range(1, N):
            end = torch.randint(1, L - 1, (1,), device=device)
            mask[i, end:] = 0

        mask[0, :] = 1
        mask = mask.bool()

        data = torch.randn(N, L, D, requires_grad=True, dtype=torch.float64, device=device)

        def grad_test_func(inpt):
            nt = torch._nested_tensor_from_mask(inpt, mask)
            # This implicitly tests to_padded_tensor grads
            return torch.nested.to_padded_tensor(nt, 0)
        assert gradcheck(grad_test_func, inputs=data, check_batched_grad=False)

    def test_nested_tensor_from_padded(self, device):
        nested_size = torch.tensor([[1, 2], [2, 2]])
        padded_tensor = torch.randn(2, 2, 2, dtype=torch.float64, device=device)
        padded_tensor[0, 1, :] = 0
        padded_tensor.requires_grad_()

        def grad_test_func(tensor, nested_size):
            nt = torch._nested_from_padded(tensor, nested_size, fuse_transform_0213=False)
            # This implicitly tests to_padded_tensor grads
            return torch.nested.to_padded_tensor(nt, 0)

        data = (padded_tensor, nested_size)
        assert gradcheck(grad_test_func, inputs=data, check_batched_grad=False)

    def test_nested_tensor_from_padded_fused(self, device):
        nested_size = torch.tensor([[1, 8], [2, 8]])
        padded_tensor = torch.randn(2, 2, 2, 4, dtype=torch.float64, device=device)
        padded_tensor[0, 1, :] = 0
        padded_tensor.requires_grad_()

        def grad_test_func(tensor, nested_size):
            nt = torch._nested_from_padded(tensor, nested_size, fuse_transform_0213=True)
            # This implicitly tests to_padded_tensor grads
            return torch.nested.to_padded_tensor(nt, 0)
        data = (padded_tensor, nested_size)
        assert gradcheck(grad_test_func, inputs=data, check_batched_grad=False)

    def test_nested_tensor_from_list(self, device):

        a = torch.randn(1, 2, requires_grad=True, dtype=torch.float64, device=device)
        b = torch.randn(2, 2, requires_grad=True, dtype=torch.float64, device=device)
        c = torch.randn(10, 2, requires_grad=True, dtype=torch.float64, device=device)

        def grad_test_func(a, b, c):
            c = torch.nested.as_nested_tensor([a, b, c])
            # This implictily tests to_padded_tensor grads
            return torch.nested.to_padded_tensor(c, 0)
        data = (a, b, c)
        assert gradcheck(grad_test_func, inputs=data, check_batched_grad=False)

    def test_dropout_backward(self):
        nt = torch.nested.nested_tensor([torch.randn((2, 5)), torch.randn((3, 4))], requires_grad=True)
        p = 0.2
        y = torch.nn.functional.dropout(nt, p)
        y.backward(nt.clone().detach())
        self.assertEqual(nt.grad, y)

    def test_nested_tensor_bmm_gradcheck(self, device):
        a = torch.randn(2, 6, requires_grad=True, dtype=torch.float64, device=device)
        b = torch.randn(3, 6, requires_grad=True, dtype=torch.float64, device=device)
        c = torch.randn(6, 4, requires_grad=True, dtype=torch.float64, device=device)
        d = torch.randn(6, 5, requires_grad=True, dtype=torch.float64, device=device)

        def grad_test_func(a, b, c, d):
            nt0 = torch.nested.as_nested_tensor([a, b])
            nt1 = torch.nested.as_nested_tensor([c, d])
            result = nt0.bmm(nt1)
            return torch.nested.to_padded_tensor(result, 0.0)

        data = (a, b, c, d)
        assert torch.autograd.gradcheck(grad_test_func, inputs=data)

    def test_nested_tensor_bmm_backward(self, device):
        nt0 = torch.nested.nested_tensor([torch.randn((2, 6)), torch.randn((3, 6))], requires_grad=True, device=device)
        nt1 = torch.nested.nested_tensor([torch.randn((6, 4)), torch.randn((6, 5))], requires_grad=True, device=device)
        with torch.no_grad():
            pt0 = torch.nested.to_padded_tensor(nt0, 0.0).requires_grad_(True)
            pt1 = torch.nested.to_padded_tensor(nt1, 0.0).requires_grad_(True)

        ynt = nt0.bmm(nt1)
        ypt = pt0.bmm(pt1)
        ynt.backward(ynt.clone())
        ypt.backward(ypt.clone())

        self.assertEqual(torch.nested.to_padded_tensor(nt0.grad, 0.0), pt0.grad)
        self.assertEqual(torch.nested.to_padded_tensor(nt1.grad, 0.0), pt1.grad)

    def test_nested_tensor_matmul_gradcheck(self, device):
        a = torch.randn(2, 6, requires_grad=True, dtype=torch.float64, device=device)
        b = torch.randn(3, 6, requires_grad=True, dtype=torch.float64, device=device)
        c = torch.randn(6, 4, requires_grad=True, dtype=torch.float64, device=device)
        d = torch.randn(6, 5, requires_grad=True, dtype=torch.float64, device=device)

        def grad_test_func(a, b, c, d):
            nt0 = torch.nested.as_nested_tensor([a, b])
            nt1 = torch.nested.as_nested_tensor([c, d])
            result = torch.matmul(nt0, nt1)
            return torch.nested.to_padded_tensor(result, 0.0)

        data = (a, b, c, d)
        assert torch.autograd.gradcheck(grad_test_func, inputs=data)

    def test_nested_tensor_matmul_backward(self, device):
        nt0 = torch.nested.nested_tensor([torch.randn((7, 2, 6)), torch.randn((7, 3, 6))], requires_grad=True, device=device)
        nt1 = torch.nested.nested_tensor([torch.randn((7, 6, 4)), torch.randn((7, 6, 5))], requires_grad=True, device=device)
        with torch.no_grad():
            pt0 = torch.nested.to_padded_tensor(nt0, 0.0).requires_grad_(True)
            pt1 = torch.nested.to_padded_tensor(nt1, 0.0).requires_grad_(True)

        ynt = torch.matmul(nt0, nt1)
        ypt = torch.matmul(pt0, pt1)
        ynt.backward(ynt.clone())
        ypt.backward(ypt.clone())

        self.assertEqual(torch.nested.to_padded_tensor(nt0.grad, 0.0), pt0.grad)
        self.assertEqual(torch.nested.to_padded_tensor(nt1.grad, 0.0), pt1.grad)

    def test_nested_tensor_transpose_gradcheck(self, device):
        a = torch.randn(2, 5, requires_grad=True, device=device)
        b = torch.randn(3, 4, requires_grad=True, device=device)

        def grad_test_func(a, b):
            nt = torch.nested.as_nested_tensor([a, b])
            result = nt.transpose(-2, -1).transpose(-2, -1)
            return torch.nested.to_padded_tensor(result, 0.0)

        data = (a, b)
        assert torch.autograd.gradcheck(grad_test_func, inputs=data, eps=1e-3)

    def test_nested_tensor_transpose_backward(self, device):
        nt = torch.nested.nested_tensor([torch.randn((2, 5)), torch.randn((3, 4))], requires_grad=True, device=device)
        with torch.no_grad():
            pt = torch.nested.to_padded_tensor(nt, 0.0).requires_grad_(True)

        ynt = nt.transpose(-2, -1)
        ypt = pt.transpose(-2, -1)
        ynt.backward(ynt.clone())
        ypt.backward(ypt.clone())

        self.assertEqual(torch.nested.to_padded_tensor(nt.grad, 0.0), pt.grad)

    def test_nested_tensor_reshape_gradcheck(self, device):
        a = torch.randn(2, 6, requires_grad=True, device=device)
        b = torch.randn(3, 6, requires_grad=True, device=device)

        def grad_test_func(a, b):
            nt = torch.nested.as_nested_tensor([a, b])
            result = nt.reshape(2, -1, 2, 3)
            return torch.nested.to_padded_tensor(result, 0.0)

        data = (a, b)
        assert torch.autograd.gradcheck(grad_test_func, inputs=data, eps=1e-3)

    def test_nested_tensor_reshape_backward(self):
        nt = torch.nested.nested_tensor([torch.randn((2, 6)), torch.randn((3, 6))], requires_grad=True)
        with torch.no_grad():
            pt = torch.nested.to_padded_tensor(nt, 0.0).requires_grad_(True)

        ynt = nt.reshape(2, -1, 2, 3)
        ypt = pt.reshape(2, -1, 2, 3)
        ynt.backward(ynt.clone())
        ypt.backward(ypt.clone())

        self.assertEqual(torch.nested.to_padded_tensor(nt.grad, 0.0), pt.grad)

    def test_nested_tensor_squeeze_backward(self, device):
        nt = torch.nested.nested_tensor([torch.randn((2, 6, 1)), torch.randn((3, 6, 1))], requires_grad=True, device=device)
        with torch.no_grad():
            pt = torch.nested.to_padded_tensor(nt, 0.0).requires_grad_(True)

        ynt = nt.squeeze(-1)
        ypt = pt.squeeze(-1)
        ynt.backward(ynt.clone())
        ypt.backward(ypt.clone())

        self.assertEqual(torch.nested.to_padded_tensor(nt.grad, 0.0), pt.grad)

    def test_nested_tensor_squeeze_gradcheck(self, device):
        a = torch.randn((2, 6, 1), dtype=torch.float64, requires_grad=True, device=device)
        b = torch.randn((3, 6, 1), dtype=torch.float64, requires_grad=True, device=device)

        def grad_test_func(a, b):
            nt = torch.nested.as_nested_tensor([a, b])
            result = nt.squeeze(-1)
            return torch.nested.to_padded_tensor(result, 0.0)

        assert torch.autograd.gradcheck(grad_test_func, inputs=(a, b), eps=1e-3)

    def test_nested_tensor_unsqueeze_backward(self, device):
        nt = torch.nested.nested_tensor([torch.randn((2, 6)), torch.randn((3, 6))], requires_grad=True, device=device)
        with torch.no_grad():
            pt = torch.nested.to_padded_tensor(nt, 0.0).requires_grad_(True)

        ynt = nt.unsqueeze(2)
        ypt = pt.unsqueeze(2)
        ynt.backward(ynt.clone())
        ypt.backward(ypt.clone())

        self.assertEqual(torch.nested.to_padded_tensor(nt.grad, 0.0), pt.grad)

    def test_nested_tensor_unsqueeze_gradcheck(self, device):
        a = torch.randn((2, 6), dtype=torch.float64, requires_grad=True, device=device)
        b = torch.randn((3, 6), dtype=torch.float64, requires_grad=True, device=device)

        def grad_test_func(a, b):
            nt = torch.nested.as_nested_tensor([a, b])
            result = nt.unsqueeze(-1)
            return torch.nested.to_padded_tensor(result, 0.0)

        assert torch.autograd.gradcheck(grad_test_func, inputs=(a, b), eps=1e-3)

    def test_nested_tensor_linear(self, device):

        a = torch.randn(1, 2, requires_grad=True, dtype=torch.float64, device=device)
        b = torch.randn(2, 2, requires_grad=True, dtype=torch.float64, device=device)
        c = torch.randn(3, 2, requires_grad=True, dtype=torch.float64, device=device)

        weight = torch.randn(2, 2, requires_grad=True, dtype=torch.float64, device=device)
        bias = torch.randn(2, requires_grad=True, dtype=torch.float64, device=device)

        def grad_test_func(a, b, c, weight, bias=None):
            nt = torch.nested.as_nested_tensor([a, b, c])
            # This implicitly tests to_padded_tensor grads
            d = torch.functional.F.linear(nt, weight, bias)
            return torch.nested.to_padded_tensor(d, 0)
        data = (a, b, c, weight, bias)
        assert gradcheck(grad_test_func, inputs=data, check_batched_grad=False)

        # Test linear with no bias added
        data = (a, b, c, weight)
        assert gradcheck(grad_test_func, inputs=data, check_batched_grad=False)

    def test_nested_tensor_softmax(self, device):
        a = torch.randn(1, 2, requires_grad=True, dtype=torch.float64, device=device)
        b = torch.randn(2, 2, requires_grad=True, dtype=torch.float64, device=device)
        c = torch.randn(3, 2, requires_grad=True, dtype=torch.float64, device=device)

        def grad_test_func(a, b, c, dim):
            nt = torch.nested.as_nested_tensor([a, b, c])
            # This implicitly tests to_padded_tensor grads
            d = torch.functional.F.softmax(nt, dim=dim)
            return torch.nested.to_padded_tensor(d, 0)

        # softmax over last dim
        data = (a, b, c, -1)
        assert gradcheck(grad_test_func, inputs=data, check_batched_grad=False)

    def test_nested_tensor_linear_backward(self, device):
        a = torch.randn(1, 2, requires_grad=False, device=device)
        b = torch.randn(2, 2, requires_grad=False, device=device)
        c = torch.randn(3, 2, requires_grad=False, device=device)

        weight = torch.randn(2, 2, requires_grad=True, device=device)
        bias = torch.randn(2, requires_grad=True, device=device)
        nt = torch.nested.as_nested_tensor([a, b, c], device=device)

        out = torch.functional.F.linear(nt, weight, bias)

        out.backward(out.clone())

        assert weight.grad is not None
        assert bias.grad is not None

        assert a.grad is None
        assert b.grad is None
        assert c.grad is None

    def test_values_grad_with_broadcast(self, device):
        a = torch.randn(1, 2, 4, requires_grad=True, dtype=torch.float64, device=device)
        b = torch.randn(2, 2, 4, requires_grad=True, dtype=torch.float64, device=device)
        c = torch.randn(3, 2, 4, requires_grad=True, dtype=torch.float64, device=device)

        def grad_test_func(a, b, c):
            nt = torch.nested.as_nested_tensor([a, b, c])
            buffer = nt.values()
            return buffer.sum()

        data = (a, b, c)
        assert gradcheck(grad_test_func, inputs=data, check_batched_grad=False)

    def test_to_buffer_series_ops_grad_with_broadcast(self, device):
        a = torch.randn(1, 1, 2, requires_grad=True, dtype=torch.float64, device=device)
        b = torch.randn(1, 1, 2, requires_grad=True, dtype=torch.float64, device=device)
        c = torch.randn(1, 1, 2, requires_grad=True, dtype=torch.float64, device=device)

        def grad_test_func(a, b, c):
            nt = torch.nested.as_nested_tensor([a, b, c])
            buffer = nt.values()
            buffer = buffer * 2
            return buffer.exp()

        data = (a, b, c)
        assert gradcheck(grad_test_func, inputs=data, check_batched_grad=False)

    def test_unbind_flow_through(self, device):
        a = torch.randn(1, 2, 4, requires_grad=True, dtype=torch.float64, device=device)
        b = torch.randn(2, 2, 4, requires_grad=True, dtype=torch.float64, device=device)
        c = torch.randn(3, 2, 4, requires_grad=True, dtype=torch.float64, device=device)

        def grad_test_func(a, b, c):
            nt = torch.nested.as_nested_tensor([a, b, c])
            ntT = nt.transpose(-1, -2)
            unbound = ntT.unbind()
            d = unbound[0]
            d = torch.pow(d, 2)
            return d

        data = (a, b, c)
        assert gradcheck(grad_test_func, inputs=data, check_batched_grad=False)

    def test_indexing_backward(self, device):
        x0 = torch.randn((2, 5))
        x1 = torch.randn((3, 4))
        nt = torch.nested.nested_tensor([x0, x1], device=device, requires_grad=True)
        self.assertEqual(nt[0], x0)
        self.assertEqual(nt[-1], x1)
        grad_x0 = torch.randn((2, 5), device=device)
        nt[0].backward(grad_x0)
        expected_grad = torch.nested.nested_tensor([grad_x0, torch.zeros((3, 4), device=device)])
        self.assertEqual(nt.grad, expected_grad)


instantiate_parametrized_tests(TestNestedTensor)
instantiate_device_type_tests(TestNestedTensorDeviceType, globals())
instantiate_device_type_tests(TestNestedTensorAutograd, globals())

if __name__ == '__main__':
    run_tests()
