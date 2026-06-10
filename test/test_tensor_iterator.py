# Owner(s): ["module: tests"]

import gc
import unittest

import torch
from torch._tensor_iterator import (
    binary_float_op,
    binary_op,
    comparison_op,
    nullary_op,
    reduce_op,
    TensorIterator,
    unary_float_op,
    unary_op,
)
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


class TestTensorIteratorBuild(TestCase):
    """Build-pipeline tests that don't depend on a particular device type."""

    def test_lifetime_owns_operands(self):
        # Core claim: the iterator wraps operands with the *owning*
        # MaybeOwned<TensorBase> path, so the underlying storage outlives
        # the Python tensor handles even if they're dropped between
        # build() and inspection. We verify via storage data_ptr staying
        # valid -- not via weakref, because PyTorch's TensorImpl pyobj_slot
        # resurrects the Python wrapper as long as the C++ side holds the
        # TensorImpl alive.
        a = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        b = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        a_ptr = a.data_ptr()

        it = TensorIterator(
            outputs=[None],
            const_inputs=[a, b],
            promote_inputs_to_common_dtype=True,
        )

        del a, b
        gc.collect()

        # If the iterator had borrowed (not owned) the operands, the
        # storage would be freed here and the assertion below would
        # either crash or read garbage.
        self.assertEqual(it.input(0).data_ptr(), a_ptr)
        self.assertEqual(it.input(0).sum().item(), float(sum(range(12))))

    def test_is_reduction_requires_defined_output(self):
        # at::TensorIterator::reduce_op gates is_reduction with
        # TORCH_INTERNAL_ASSERT(out.defined()). The named-constructor
        # gate isn't reachable when a Python user composes the flag
        # manually; without our build-time check the iterator silently
        # produces a non-reducing iterator with a wrong-shape output.
        a = torch.zeros(3, 4)
        with self.assertRaisesRegex(
            RuntimeError, "is_reduction.*requires every output"
        ):
            TensorIterator(
                outputs=[None],
                const_inputs=[a],
                is_reduction=True,
            )

    def test_is_reduction_with_defined_output_ok(self):
        # Same flag combination, defined output of reduced shape: builds.
        a = torch.zeros(3, 4)
        out = torch.empty(3, 1)
        it = TensorIterator(
            outputs=[out],
            const_inputs=[a],
            resize_outputs=False,
            is_reduction=True,
        )
        self.assertEqual(it.noutputs, 1)

    def test_enforce_safe_casting_to_output_raises(self):
        # float -> integer is not a safe cast; the build must reject when
        # safe-casting-to-output is on.
        a = torch.zeros(3, dtype=torch.float32)
        b = torch.zeros(3, dtype=torch.float32)
        out = torch.empty(3, dtype=torch.int32)
        with self.assertRaises(RuntimeError):
            TensorIterator(
                outputs=[out],
                const_inputs=[a, b],
                promote_inputs_to_common_dtype=True,
                enforce_safe_casting_to_output=True,
                resize_outputs=False,
            )

    def test_enforce_linear_iteration_preserves_c_order(self):
        # Probe with a non-contiguous input: a transposed (3, 4) tensor.
        # The default reorder permutes dims into memory order so the iter
        # collapses to one contiguous dim of length 12. With
        # enforce_linear_iteration the perm is fixed at [n-1, ..., 0], so
        # the iter cannot coalesce -- ndim stays at 2.
        a = torch.zeros(4, 3).t()
        no_flag = unary_op(None, a)
        self.assertEqual(no_flag.ndim, 1)
        self.assertEqual(no_flag.numel, 12)
        with_flag = TensorIterator(
            outputs=[None],
            const_inputs=[a],
            enforce_linear_iteration=True,
        )
        self.assertEqual(with_flag.ndim, 2)
        self.assertEqual(with_flag.numel, 12)

    def test_declare_static_dtype_and_device(self):
        a = torch.zeros(3, dtype=torch.float32)
        out = torch.empty(3, dtype=torch.float64)
        it = TensorIterator(
            outputs=[out],
            const_inputs=[a],
            check_all_same_dtype=False,
            static_dtype=torch.float64,
            static_device=torch.device("cpu"),
        )
        self.assertEqual(it.dtype(0), torch.float64)
        self.assertEqual(it.device(0).type, "cpu")

    def test_declare_static_device(self):
        a = torch.zeros(3)
        out = torch.empty(3)
        it = TensorIterator(
            outputs=[out],
            const_inputs=[a],
            static_device=torch.device("cpu"),
        )
        self.assertEqual(it.device(0).type, "cpu")

    def test_mixed_dtype_rejected_without_promotion(self):
        # check_all_same_dtype is on by default. Mixed dtypes should fail.
        a = torch.zeros(3, dtype=torch.float32)
        b = torch.zeros(3, dtype=torch.float64)
        with self.assertRaises(RuntimeError):
            TensorIterator(outputs=[None], const_inputs=[a, b])

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_cross_device_check_raises(self):
        # Default check_all_same_device=True rejects mixed CPU+CUDA.
        a = torch.zeros(3, device="cpu")
        b = torch.zeros(3, device="cuda")
        with self.assertRaises(RuntimeError):
            binary_op(None, a, b)

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_allow_cpu_scalars(self):
        # A 0-dim CPU tensor is a "CPU scalar". With allow_cpu_scalars=True
        # the iterator accepts it alongside a CUDA tensor without violating
        # the same-device check (the same-device check itself allows CPU
        # scalars when this flag is on).
        a = torch.zeros(3, device="cuda")
        b = torch.tensor(2.0, device="cpu")  # 0-dim CPU scalar
        it = TensorIterator(
            outputs=[None],
            const_inputs=[a, b],
            allow_cpu_scalars=True,
            promote_inputs_to_common_dtype=True,
        )
        self.assertEqual(it.numel, 3)


class TestTensorIterator(TestCase):
    @parametrize(
        "shape_a,shape_b,expected",
        [
            ((3, 1), (1, 4), (3, 4)),
            ((1,), (5,), (5,)),
            ((2, 3, 4), (3, 4), (2, 3, 4)),
            ((1, 1, 5), (4, 1, 5), (4, 1, 5)),
        ],
    )
    def test_broadcast_shape(self, device, shape_a, shape_b, expected):
        a = torch.zeros(shape_a, device=device)
        b = torch.zeros(shape_b, device=device)
        it = binary_op(None, a, b)
        # The post-coalesce shape may collapse contiguous dims, but its numel
        # has to equal the broadcast numel and the output tensor must end up at
        # the broadcast shape after build.
        expected_numel = 1
        for s in expected:
            expected_numel *= s
        self.assertEqual(it.numel, expected_numel)
        self.assertEqual(tuple(it.output(0).shape), expected)

    def test_promote_inputs_to_common_dtype(self, device):
        a = torch.zeros(3, dtype=torch.int32, device=device)
        b = torch.zeros(3, dtype=torch.float32, device=device)
        it = binary_op(None, a, b)
        self.assertEqual(it.common_dtype, torch.float32)

    def test_promote_integer_inputs_to_float(self, device):
        a = torch.zeros(3, dtype=torch.int64, device=device)
        b = torch.zeros(3, dtype=torch.int32, device=device)
        it = TensorIterator(
            outputs=[None],
            const_inputs=[a, b],
            promote_inputs_to_common_dtype=True,
            promote_integer_inputs_to_float=True,
        )
        # Integer common dtype gets bumped to default float scalar type.
        self.assertEqual(it.common_dtype, torch.get_default_dtype())

    def test_common_dtype_when_all_match(self, device):
        # check_all_same_dtype still produces a common dtype: it's just the
        # single dtype shared by all operands.
        a = torch.zeros(3, dtype=torch.float64, device=device)
        b = torch.zeros(3, dtype=torch.float64, device=device)
        it = TensorIterator(outputs=[None], const_inputs=[a, b])
        self.assertEqual(it.common_dtype, torch.float64)

    def test_resize_outputs(self, device):
        a = torch.zeros(3, 4, device=device)
        b = torch.zeros(3, 4, device=device)
        # Pre-allocated zero-element output is reshaped to the broadcast
        # shape (the docs-recommended pattern for resizable outputs).
        out = torch.empty(0, device=device)
        binary_op(out, a, b)
        self.assertEqual(tuple(out.shape), (3, 4))

    def test_no_resize_outputs(self, device):
        a = torch.zeros(3, 4, device=device)
        b = torch.zeros(3, 4, device=device)
        out = torch.zeros(3, 4, device=device)
        it = TensorIterator(
            outputs=[out],
            const_inputs=[a, b],
            resize_outputs=False,
        )
        self.assertEqual(it.numel, 12)
        self.assertEqual(tuple(out.shape), (3, 4))

    def test_declare_static_shape(self, device):
        a = torch.zeros(2, 6, device=device)
        out = torch.empty(2, 6, device=device)
        it = TensorIterator(
            outputs=[out],
            const_inputs=[a],
            resize_outputs=False,
            static_shape=(2, 6),
        )
        # Shape is reported post-coalesce: a 2x6 contiguous tensor collapses.
        self.assertEqual(it.numel, 12)

    def test_declare_static_dtype(self, device):
        a = torch.zeros(3, dtype=torch.float64, device=device)
        out = torch.empty(3, dtype=torch.float64, device=device)
        it = TensorIterator(
            outputs=[out],
            const_inputs=[a],
            check_all_same_dtype=False,
            static_dtype=torch.float64,
        )
        self.assertEqual(it.dtype(0), torch.float64)

    def test_mem_overlap_detected(self, device):
        # Broadcast view -> internal overlap on output.
        out = torch.empty(2, 3, device=device).expand(4, 2, 3)
        a = torch.zeros(4, 2, 3, device=device)
        b = torch.zeros(4, 2, 3, device=device)
        with self.assertRaises(RuntimeError):
            TensorIterator(
                outputs=[out],
                const_inputs=[a, b],
                resize_outputs=False,
            )

    def test_mem_overlap_disabled(self, device):
        # Same setup as above but with the overlap check turned off.
        out = torch.empty(2, 3, device=device).expand(4, 2, 3)
        a = torch.zeros(4, 2, 3, device=device)
        b = torch.zeros(4, 2, 3, device=device)
        it = TensorIterator(
            outputs=[out],
            const_inputs=[a, b],
            resize_outputs=False,
            check_mem_overlap=False,
        )
        self.assertEqual(it.numel, 24)

    def test_comparison_op_bool_output(self, device):
        a = torch.zeros(3, dtype=torch.float32, device=device)
        b = torch.zeros(3, dtype=torch.float32, device=device)
        # Undefined output -> static dtype is bool.
        it = comparison_op(None, a, b)
        self.assertEqual(it.dtype(0), torch.bool)

    def test_unary_op_dtype_passthrough(self, device):
        a = torch.zeros(3, 4, dtype=torch.float64, device=device)
        # unary_op uses check_all_same_dtype, so the output (when
        # auto-allocated) is allocated in the input's dtype.
        it = unary_op(None, a)
        self.assertEqual(it.dtype(0), torch.float64)
        self.assertEqual(it.numel, 12)

    def test_nullary_op(self, device):
        out = torch.empty(3, 4, device=device)
        it = nullary_op(out)
        self.assertEqual(it.ntensors, 1)
        self.assertEqual(it.noutputs, 1)
        self.assertEqual(it.ninputs, 0)

    def test_reduce_op(self, device):
        a = torch.zeros(3, 4, device=device)
        # reduce_op requires the output to broadcast with the input. To reduce
        # along the trailing dim we pre-allocate a (3, 1) output.
        out = torch.empty(3, 1, device=device)
        it = reduce_op(out, a)
        self.assertEqual(it.ntensors, 2)
        self.assertEqual(it.noutputs, 1)

    def test_reduce_op_two_outputs(self, device):
        # The (out, out2) overload covers things like at::min returning
        # (values, indices). We don't need integer indices to exercise the
        # binding; we just need two outputs that share the reduced shape.
        a = torch.zeros(3, 4, device=device)
        out = torch.empty(3, 1, device=device)
        out2 = torch.empty(3, 1, device=device)
        it = reduce_op(out, a, out2=out2)
        self.assertEqual(it.ntensors, 3)
        self.assertEqual(it.noutputs, 2)

    def test_reduce_op_two_outputs_shape_mismatch_raises(self, device):
        a = torch.zeros(3, 4, device=device)
        out = torch.empty(3, 1, device=device)
        out2 = torch.empty(1, 3, device=device)
        with self.assertRaisesRegex(RuntimeError, "identical sizes and strides"):
            reduce_op(out, a, out2=out2)

    def test_binary_float_op_promotes_integers(self, device):
        # binary_float_op promotes integer inputs to the default float dtype
        # so the kernel can dispatch a single float arithmetic path.
        a = torch.zeros(3, 4, dtype=torch.int32, device=device)
        b = torch.zeros(3, 4, dtype=torch.int64, device=device)
        it = binary_float_op(None, a, b)
        self.assertTrue(it.common_dtype.is_floating_point)
        # The auto-allocated output picks up the promoted dtype.
        self.assertTrue(it.dtype(0).is_floating_point)

    def test_unary_float_op_promotes_integers(self, device):
        a = torch.zeros(3, 4, dtype=torch.int32, device=device)
        it = unary_float_op(None, a)
        self.assertTrue(it.common_dtype.is_floating_point)
        self.assertTrue(it.dtype(0).is_floating_point)

    def test_strides_are_bytes(self, device):
        a = torch.zeros(3, 4, dtype=torch.float32, device=device)
        b = torch.zeros(3, 4, dtype=torch.float32, device=device)
        it = binary_op(None, a, b)
        # Element size is 4 bytes; element_strides should equal byte strides / 4.
        byte_strides = it.strides(0)
        elt_strides = it.element_strides(0)
        self.assertEqual(len(byte_strides), len(elt_strides))
        for bs, es in zip(byte_strides, elt_strides):
            self.assertEqual(bs, es * 4)

    def test_repr(self, device):
        a = torch.zeros(3, 4, device=device)
        b = torch.zeros(3, 4, device=device)
        it = binary_op(None, a, b)
        r = repr(it)
        self.assertIn("ndim=", r)
        self.assertIn("ntensors=3", r)


instantiate_device_type_tests(TestTensorIterator, globals())


if __name__ == "__main__":
    run_tests()
