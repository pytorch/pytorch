# Owner(s): ["module: nestedtensor"]

import torch
import torch.nn
import unittest
from torch.testing._internal.common_device_type import (
    ops,
    dtypes,
    dtypesIfCUDA,
    instantiate_device_type_tests,
    skipMeta,
)
from torch.testing._internal.common_methods_invocations import nested_unary_ufuncs
from torch.testing._internal.common_utils import TestCase, IS_FBCODE, run_tests
from torch import nested_tensor

# Tests are ported from pytorch/nestedtensor.
# This makes porting as_nested_tensor easier in the future.
def _iter_constructors():
    # yield as_nested_tensor
    yield nested_tensor


class TestNestedTensor(TestCase):
    @torch.inference_mode()
    def _test_unbind_case(self, a, b):
        nt = nested_tensor([a, b])
        a1, b1 = nt.unbind()
        self.assertTrue(a is not a1)
        self.assertTrue(b is not b1)

        nt = nested_tensor([a, b], dtype=a.dtype)
        a1, b1 = nt.unbind(0)
        self.assertEqual(a, a1)
        self.assertEqual(b, b1)

        a = torch.randn((2, 3)).add_(1)
        nt = nested_tensor([a])
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

    # @torch.inference_mode()
    # def test_unbind_2(self):
    #     self._test_unbind_case(
    #         torch.tensor(1), torch.tensor(7),
    #     )

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
            nt = nested_tensor([a, b])
            self.assertRaises(RuntimeError, lambda: unbind_fn(nt, 1))

        # Both of these tests are necessary, because we're using
        # torch_function.
        _test_fn(lambda x, dim: x.unbind(dim))
        # TODO: Re-enable this once using torch_dispatch
        # _test_fn(lambda x, dim: torch.unbind(x, dim))

    @torch.inference_mode()
    def test_nested_tensor(self):
        self.assertRaises(TypeError, lambda: nested_tensor([3.0]))
        self.assertRaises(TypeError, lambda: nested_tensor(torch.tensor([3.0])))
        self.assertRaises(TypeError, lambda: nested_tensor(4.0))

    @torch.inference_mode()
    def test_nested_tensor_matching_dim(self):
        self.assertRaisesRegex(
            RuntimeError,
            "Found dimension 1 for Tensor at index 1 and dimension 0 for Tensor at index 0.",
            lambda: nested_tensor([torch.tensor(1.0), torch.tensor([])]),
        )
        self.assertRaisesRegex(
            RuntimeError,
            "Found dimension 1 for Tensor at index 2 and dimension 0 for Tensor at index 1.",
            lambda: nested_tensor(
                [torch.tensor(1.0), torch.tensor(2.0), torch.tensor([])]
            ),
        )

    @torch.inference_mode()
    def test_default_nested_tensor(self):
        self.assertRaises(TypeError, lambda: nested_tensor())
        default_nested_tensor = nested_tensor([])
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
            self.assertRaisesRegex(
                RuntimeError, "numel is disabled", lambda: a1.numel(),
            )

    @torch.inference_mode()
    def test_size(self):
        for constructor in _iter_constructors():
            a1 = constructor([])
            self.assertRaisesRegex(
                RuntimeError,
                "Tensors of type NestedTensorImpl do not have sym sizes"
                if IS_FBCODE
                else "NestedTensorImpl doesn't support sizes",
                lambda: a1.size(),
            )

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
        for constructor in _iter_constructors():
            a1 = constructor([])
            self.assertRaisesRegex(
                RuntimeError, "is_contiguous is disabled", lambda: a1.is_contiguous()
            )

    @torch.inference_mode()
    def test_repr_string(self):
        a = nested_tensor([])
        expected = "nested_tensor([" "\n\n])"
        self.assertEqual(str(a), expected)
        self.assertEqual(repr(a), expected)

        a = nested_tensor([torch.tensor(1.0)])
        expected = "nested_tensor([" "\n  tensor(1.)" "\n])"
        self.assertEqual(str(a), expected)
        self.assertEqual(repr(a), expected)

        a = nested_tensor([torch.tensor([[1, 2]]), torch.tensor([[4, 5]])])
        expected = (
            "nested_tensor([" "\n  tensor([[1, 2]])" "," "\n  tensor([[4, 5]])" "\n])"
        )
        self.assertEqual(str(a), expected)
        self.assertEqual(repr(a), expected)

    @torch.inference_mode()
    def test_activations(self):
        for func in (torch.nn.functional.relu, torch.nn.functional.relu_, torch.nn.functional.gelu, torch._C._nn.gelu_):
            t = torch.tensor([-1, 0, 1], dtype=torch.float)
            nt = nested_tensor([t])
            nested_result = func(nt)
            self.assertTrue(nested_result.is_nested)
            self.assertEqual(func(t), nested_result.unbind()[0])

    def test_to_padded_tensor_on_empty_tensor(self):
        nt = torch.nested_tensor([])
        empty = nt.to_padded_tensor(4)
        self.assertEqual(empty, torch.tensor([]))

class TestNestedTensorDeviceType(TestCase):
    @dtypes(torch.float)
    @skipMeta
    def test_to_then_from_padded_tensor_no_transform0213(self, device, dtype):
        t = torch.randn(4, 4, 4, device=device, dtype=dtype)
        ts = list(torch.unbind(t))
        ts[0] = ts[0][:-1]
        nt = torch.nested_tensor(ts, device=device, dtype=dtype)
        padded = nt.to_padded_tensor(0)

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
            t0 = torch.randn(2, size, device=device, dtype=dtype, requires_grad=False)
            t1 = torch.randn(2, size, device=device, dtype=dtype, requires_grad=False)
            ts = [t0, t1, t0, t1]
            nt = torch.nested_tensor(ts, device=device, dtype=dtype)
            layer_norm = torch.nn.LayerNorm(size, device=device, dtype=dtype)
            nt_result = nt._nested_tensor_layer_norm(
                layer_norm.weight, layer_norm.bias, 1e-5
            )
            for (nt_subresult, t) in zip(nt_result.unbind(), ts):
                t_result = layer_norm(t.reshape(1, -1, size).squeeze(0))
                self.assertEqual(nt_subresult, t_result)

        for size in (1024, 1023, 513, 512, 256, 128, 2, 4, 32):
            _test(size)

    @skipMeta
    @torch.inference_mode()
    def test_embedding(self, device):
        inputs = [
            torch.randint(100, (L,), device=device, dtype=torch.int64)
            for L in torch.randint(5, 50, (8,))
        ]
        x = torch.nested_tensor(inputs, device=device, dtype=torch.int64)
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
        nt = torch.nested_tensor(ts, device=device, dtype=dtype)
        for padding_value in (0, 1):
            padded = nt.to_padded_tensor(padding_value)

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
        nt = torch.nested_tensor(ts, device=device, dtype=dtype)
        for padding_value in (0, 1):
            padded = nt.to_padded_tensor(padding_value, output_size=output_size)
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
        nt = torch.nested_tensor(ts, device=device, dtype=dtype)
        pad = 42
        correct_output = []
        for t in ts:
            next_output = torch.ones_like(ts[2]) * pad
            correct_output.append(next_output)
            next_output[:t.size(0)].copy_(t)
        correct_output = torch.stack(correct_output)
        padded = nt.to_padded_tensor(pad)
        self.assertEqual(padded, correct_output)

    @dtypes(torch.float, torch.float16, torch.double)
    def test_to_padded_tensor_dim3(self, device, dtype):
        ts = [
            torch.randn(16, 21, device=device, dtype=dtype),
            torch.randn(24, 32, device=device, dtype=dtype),
            torch.randn(40, 53, device=device, dtype=dtype),
        ]
        nt = torch.nested_tensor(ts, device=device, dtype=dtype)
        pad = 42
        correct_output = []
        for t in ts:
            next_output = torch.ones_like(ts[2]) * pad
            correct_output.append(next_output)
            next_output[:t.size(0), :t.size(1)].copy_(t)
        correct_output = torch.stack(correct_output)
        padded = nt.to_padded_tensor(pad)
        self.assertEqual(padded, correct_output)

    @dtypes(torch.float, torch.float16, torch.double)
    def test_to_padded_tensor_dim4(self, device, dtype):
        ts = [
            torch.randn(16, 21, 13, device=device, dtype=dtype),
            torch.randn(24, 32, 14, device=device, dtype=dtype),
            torch.randn(40, 53, 16, device=device, dtype=dtype),
        ]
        nt = torch.nested_tensor(ts, device=device, dtype=dtype)
        pad = 42
        correct_output = []
        for t in ts:
            next_output = torch.ones_like(ts[2]) * pad
            correct_output.append(next_output)
            next_output[:t.size(0), :t.size(1), :t.size(2)].copy_(t)
        correct_output = torch.stack(correct_output)
        padded = nt.to_padded_tensor(pad)
        self.assertEqual(padded, correct_output)

    @skipMeta
    def test_device_checks(self, device):
        nt = torch.nested_tensor([], device=device)
        is_cuda = 'cuda' in str(device)
        self.assertEqual(nt.is_cuda, is_cuda)

    @dtypes(torch.float, torch.float16, torch.double)
    def test_nested_tensor_indexing(self, device, dtype):
        # edge case: empty nested tensor
        nt0 = torch.nested_tensor([])
        self.assertRaisesRegex(
            RuntimeError,
            "cannot index an empty nested tensor",
            lambda: nt0[0]
        )
        # normal case
        x0 = torch.randn((2, 5), device=device, dtype=dtype)
        x1 = torch.randn((3, 4), device=device, dtype=dtype)
        nt = torch.nested_tensor([x0, x1])
        # single index: only support integer in the batch dimension
        self.assertEqual(nt[0], x0)
        self.assertEqual(nt[-1], x1)
        self.assertRaises(IndexError, lambda: nt[2])
        self.assertRaises(IndexError, lambda: nt[-3])
        self.assertRaises(NotImplementedError, lambda: nt[:])
        self.assertRaises(NotImplementedError, lambda: nt[None])
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

    # Helper functions for testing elementwise ops
    def random_nt(self, device, dtype, num_tensors, max_dims, min_dims=None):
        if min_dims is None:
            min_dims = tuple([0] * len(max_dims))
        ts1 = []
        for _ in range(num_tensors):
            tensor_dims = tuple([torch.randint(low=min_dim, high=max_dim, size=(1,)).item()
                                for (min_dim, max_dim) in zip(min_dims, max_dims)])
            t1 = torch.randn(tensor_dims, device=device, dtype=dtype)
            ts1.append(t1)
        return torch.nested_tensor(ts1, device=device, dtype=dtype)

    # Helper functions for testing elementwise ops
    def random_nt_pair(self, device, dtype, num_tensors, max_dims):
        ts1 = []
        ts2 = []
        for _ in range(num_tensors):
            tensor_dims = tuple([torch.randint(low=0, high=max_dim, size=(1,)).item() for max_dim in max_dims])
            t1 = torch.randn(tensor_dims, device=device, dtype=dtype)
            t2 = torch.randn(tensor_dims, device=device, dtype=dtype)
            ts1.append(t1)
            ts2.append(t2)
        return (torch.nested_tensor(ts1, device=device, dtype=dtype),
                torch.nested_tensor(ts2, device=device, dtype=dtype))

    def nt_equal(self, nt1, nt2):
        self.assertEqual(nt1.dtype, nt2.dtype)
        self.assertEqual(nt1.device, nt2.device)
        ub1 = nt1.unbind()
        ub2 = nt2.unbind()
        self.assertEqual(len(ub1), len(ub2))
        n = len(ub1)
        for i in range(n):
            self.assertEqual(ub1[i], ub2[i])

    @dtypes(torch.float, torch.float16)
    @skipMeta
    @torch.inference_mode()
    def test_nested_tensor_add(self, device, dtype):
        (nt1, nt2) = self.random_nt_pair(device, dtype, 4, (4, 4))
        ref = torch.nested_tensor([t1 + t2 for (t1, t2) in zip(nt1.unbind(), nt2.unbind())])
        out = nt1 + nt2
        self.nt_equal(ref, out)

    @dtypes(torch.float, torch.float16)
    @skipMeta
    @torch.inference_mode()
    def test_nested_tensor_mul(self, device, dtype):
        (nt1, nt2) = self.random_nt_pair(device, dtype, 4, (4, 4))
        ref = torch.nested_tensor([t1 * t2 for (t1, t2) in zip(nt1.unbind(), nt2.unbind())])
        out = nt1 * nt2
        self.nt_equal(ref, out)

    @dtypes(torch.float, torch.float16)
    @skipMeta
    @torch.inference_mode()
    def test_nested_tensor_add_in_place(self, device, dtype):
        (nt1, nt2) = self.random_nt_pair(device, dtype, 4, (4, 4))
        ref = torch.nested_tensor([t1 + t2 for (t1, t2) in zip(nt1.unbind(), nt2.unbind())])
        nt1 += nt2
        self.nt_equal(ref, nt1)

    @dtypes(torch.float, torch.float16)
    @skipMeta
    @torch.inference_mode()
    def test_nested_tensor_mul_in_place(self, device, dtype):
        (nt1, nt2) = self.random_nt_pair(device, dtype, 4, (4, 4))
        ref = torch.nested_tensor([t1 * t2 for (t1, t2) in zip(nt1.unbind(), nt2.unbind())])
        nt1 *= nt2
        self.nt_equal(ref, nt1)

    @dtypes(torch.float, torch.float16)
    @skipMeta
    @torch.inference_mode()
    def test_clone(self, device, dtype):
        nt1 = self.random_nt(device, dtype, 4, (4, 4), (1, 1))
        nt2 = nt1.clone()
        # Verify the values match
        self.nt_equal(nt1, nt2)
        # Verify modifying nt2 doesn't affect nt1
        nt2.mul_(nt1)
        ub1 = nt1.unbind()
        ub2 = nt2.unbind()
        for i in range(len(ub1)):
            self.assertNotEqual(ub1[i], ub2[i])

        nt1.clone(memory_format=torch.preserve_format)
        msg = "clone_nested only supports memory format Preserve, but got ChannelsLast instead."
        with self.assertRaisesRegex(RuntimeError, msg):
            nt1.clone(memory_format=torch.channels_last)

    @ops(nested_unary_ufuncs)
    def test_nested_unary(self, device, dtype, op):
        samples = op.sample_inputs(device, dtype)
        tensors = []
        expects = []
        for sample in samples:
            # every underlying tensor must have a same dimension; arbitrarily choose as 2
            if sample.input.ndim == 2:
                tensors.append(sample.input)
                expects.append(op(sample.input))
        nt = torch.nested_tensor(tensors)
        expect = torch.nested_tensor(expects)
        print(expect)
        print(op(nt))
        self.nt_equal(expect, op(nt))

    @ops(nested_unary_ufuncs)
    def test_nested_unary_out(self, device, dtype, op):
        if not op.supports_out:
            self.skipTest("Skipped! Out not supported")
        samples = op.sample_inputs(device, dtype)
        tensors = []
        expects = []
        for sample in samples:
            # every underlying tensor must have a same dimension; arbitrarily choose as 2
            if sample.input.ndim == 2:
                tensors.append(sample.input)
                expects.append(op(sample.input))
        nt = torch.nested_tensor(tensors)
        expect = torch.nested_tensor(expects)
        actual = expect.clone()
        op(nt, out=actual)
        self.nt_equal(expect, actual)

    @ops(nested_unary_ufuncs)
    def test_nested_unary_inplace(self, device, dtype, op):
        if op.inplace_variant is None:
            self.skipTest("Skipped! Inplace variant not supported!")
        samples = op.sample_inputs(device, dtype)
        skip = False
        for sample in samples:
            if sample.input.is_complex() and op.name == "abs":
                skip = True
            if skip:
                break
        if skip:
            self.skipTest("Skipped! Data type & inplace variant combination not supported!")
        tensors = []
        expects = []
        for sample in samples:
            # every underlying tensor must have a same dimension; arbitrarily choose as 2
            if sample.input.ndim == 2:
                tensors.append(sample.input)
                expects.append(op(sample.input))
        nt = torch.nested_tensor(tensors)
        expect = torch.nested_tensor(expects)
        op.inplace_variant(nt)
        self.nt_equal(expect, nt)

instantiate_device_type_tests(TestNestedTensorDeviceType, globals())

if __name__ == '__main__':
    run_tests()
