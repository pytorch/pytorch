# Owner(s): ["module: nestedtensor"]

import torch
import unittest
from torch.testing._internal.common_utils import TestCase, IS_FBCODE
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
                "Tensors of type NestedTensorImpl do not have sizes"
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
