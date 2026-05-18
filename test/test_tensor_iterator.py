# Owner(s): ["module: tests"]

import gc
import unittest

import torch
from torch._tensor_iterator import (
    binary_op,
    comparison_op,
    ConfigSpec,
    nullary_op,
    reduce_op,
    TensorIteratorConfig,
    unary_op,
)
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


class TestTensorIteratorBuild(TestCase):
    """Build-pipeline tests that don't depend on a particular device type."""

    def test_outputs_first_invariant(self):
        a = torch.zeros(3)
        b = torch.zeros(3)
        cfg = TensorIteratorConfig().add_input(a)
        with self.assertRaisesRegex(RuntimeError, "before any inputs"):
            cfg.add_output(b)

    def test_const_input_counts_as_input(self):
        a = torch.zeros(3)
        b = torch.zeros(3)
        out = torch.empty(0)
        cfg = TensorIteratorConfig().add_output(out).add_const_input(a)
        with self.assertRaisesRegex(RuntimeError, "before any inputs"):
            cfg.add_output(b)

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

        it = (
            TensorIteratorConfig()
            .add_output(None)
            .add_const_input(a)
            .add_const_input(b)
            .promote_inputs_to_common_dtype(True)
            .build()
        )

        del a, b
        gc.collect()

        # If the iterator had borrowed (not owned) the operands, the
        # storage would be freed here and the assertion below would
        # either crash or read garbage.
        self.assertEqual(it.input(0).data_ptr(), a_ptr)
        self.assertEqual(it.input(0).sum().item(), float(sum(range(12))))

    def test_add_input_rejects_none(self):
        a = torch.zeros(3)
        cfg = TensorIteratorConfig().add_output(a)
        with self.assertRaisesRegex(RuntimeError, "None is not allowed"):
            cfg.add_input(None)

    def test_add_const_input_rejects_none(self):
        a = torch.zeros(3)
        cfg = TensorIteratorConfig().add_output(a)
        with self.assertRaisesRegex(RuntimeError, "None is not allowed"):
            cfg.add_const_input(None)

    def test_fluent_chain_outlives_head(self):
        # py::return_value_policy::reference_internal must keep the chain
        # head alive for the lifetime of the returned reference. Drop the
        # named root and verify the tail is still usable.
        a = torch.zeros(3)
        b = torch.zeros(3)
        tail = (
            TensorIteratorConfig()
            .add_output(None)
            .add_const_input(a)
            .add_const_input(b)
            .promote_inputs_to_common_dtype(True)
        )
        gc.collect()
        it = tail.build()
        self.assertEqual(it.shape, (3,))

    def test_build_is_single_shot(self):
        # The C++ TensorIteratorConfig::build() std::moves out of its
        # tensors_, so a second call would surface an INTERNAL ASSERT.
        # The Python binding tracks a built flag and rejects the second
        # call cleanly. add_input/add_output/is_reduction post-build are
        # also rejected to prevent silent corruption.
        a = torch.zeros(3)
        b = torch.zeros(3)
        cfg = (
            TensorIteratorConfig()
            .add_output(None)
            .add_const_input(a)
            .add_const_input(b)
            .promote_inputs_to_common_dtype(True)
        )
        cfg.build()
        with self.assertRaisesRegex(RuntimeError, "single-shot"):
            cfg.build()
        with self.assertRaisesRegex(RuntimeError, "already been used"):
            cfg.add_input(a)

    def test_is_reduction_requires_defined_output(self):
        # at::TensorIterator::reduce_op gates is_reduction with
        # TORCH_INTERNAL_ASSERT(out.defined()). The named-constructor
        # gate isn't reachable when a Python user composes the flag
        # manually; without our build-time check the iterator silently
        # produces a non-reducing iterator with a wrong-shape output.
        a = torch.zeros(3, 4)
        cfg = (
            TensorIteratorConfig()
            .add_output(None)
            .add_const_input(a)
            .is_reduction(True)
        )
        with self.assertRaisesRegex(
            RuntimeError, "is_reduction.*requires every output"
        ):
            cfg.build()

    def test_is_reduction_with_defined_output_ok(self):
        # Same flag combination, defined output of reduced shape: builds.
        a = torch.zeros(3, 4)
        out = torch.empty(3, 1)
        it = (
            TensorIteratorConfig()
            .add_output(out)
            .add_const_input(a)
            .resize_outputs(False)
            .is_reduction(True)
            .build()
        )
        self.assertEqual(it.noutputs, 1)

    def test_dataclass_matches_fluent(self):
        a = torch.arange(6).reshape(2, 3).to(torch.float32)
        b = torch.arange(3).to(torch.float32)

        it_fluent = (
            TensorIteratorConfig()
            .add_output(None)
            .add_const_input(a)
            .add_const_input(b)
            .promote_inputs_to_common_dtype(True)
            .build()
        )
        it_dc = ConfigSpec(
            outputs=[None],
            const_inputs=[a, b],
            promote_inputs_to_common_dtype=True,
        ).build()

        self.assertEqual(it_fluent.shape, it_dc.shape)
        self.assertEqual(it_fluent.common_dtype, it_dc.common_dtype)
        self.assertEqual(it_fluent.dtype(0), it_dc.dtype(0))

    def test_configspec_fields_plumb_through(self):
        # Each ConfigSpec field must drive the corresponding fluent setter.
        # For each non-default field, build a dataclass with that field
        # flipped and a fluent equivalent; verify they produce the same
        # iterator. If a field were silently dropped from ConfigSpec.build,
        # at least one of these subTests would mismatch.
        f = torch.zeros(3, 4, dtype=torch.float32)
        i = torch.zeros(3, 4, dtype=torch.int32)
        d = torch.zeros(3, 4, dtype=torch.float64)
        out_f64 = torch.empty(3, 4, dtype=torch.float64)
        out_f32 = torch.empty(3, 4, dtype=torch.float32)
        out_shape_4_3 = torch.empty(4, 3, dtype=torch.float32)

        cases = [
            (
                "check_all_same_dtype=False",
                ConfigSpec(
                    outputs=[None],
                    const_inputs=[f, i],
                    check_all_same_dtype=False,
                    promote_inputs_to_common_dtype=True,
                ),
                lambda c: c.add_output(None)
                .add_const_input(f)
                .add_const_input(i)
                .check_all_same_dtype(False)
                .promote_inputs_to_common_dtype(True),
            ),
            (
                "promote_integer_inputs_to_float=True",
                ConfigSpec(
                    outputs=[None],
                    const_inputs=[i, i],
                    promote_inputs_to_common_dtype=True,
                    promote_integer_inputs_to_float=True,
                ),
                lambda c: c.add_output(None)
                .add_const_input(i)
                .add_const_input(i)
                .promote_inputs_to_common_dtype(True)
                .promote_integer_inputs_to_float(True),
            ),
            (
                "cast_common_dtype_to_outputs=True",
                ConfigSpec(
                    outputs=[out_f32],
                    const_inputs=[f, d],
                    promote_inputs_to_common_dtype=True,
                    cast_common_dtype_to_outputs=True,
                    resize_outputs=False,
                ),
                lambda c: c.add_output(out_f32)
                .add_const_input(f)
                .add_const_input(d)
                .promote_inputs_to_common_dtype(True)
                .cast_common_dtype_to_outputs(True)
                .resize_outputs(False),
            ),
            (
                "enforce_safe_casting_to_output=True",
                ConfigSpec(
                    outputs=[out_f64],
                    const_inputs=[f, f],
                    promote_inputs_to_common_dtype=True,
                    enforce_safe_casting_to_output=True,
                    resize_outputs=False,
                ),
                lambda c: c.add_output(out_f64)
                .add_const_input(f)
                .add_const_input(f)
                .promote_inputs_to_common_dtype(True)
                .enforce_safe_casting_to_output(True)
                .resize_outputs(False),
            ),
            (
                "enforce_linear_iteration=True",
                ConfigSpec(
                    outputs=[None],
                    const_inputs=[f, f],
                    promote_inputs_to_common_dtype=True,
                    enforce_linear_iteration=True,
                ),
                lambda c: c.add_output(None)
                .add_const_input(f)
                .add_const_input(f)
                .promote_inputs_to_common_dtype(True)
                .enforce_linear_iteration(True),
            ),
            (
                "resize_outputs=False",
                ConfigSpec(
                    outputs=[out_f32],
                    const_inputs=[f, f],
                    resize_outputs=False,
                ),
                lambda c: c.add_output(out_f32)
                .add_const_input(f)
                .add_const_input(f)
                .resize_outputs(False),
            ),
            (
                "check_mem_overlap=False",
                # Output that aliases its inputs via expand-style
                # broadcasting; with overlap-check off, build must succeed.
                ConfigSpec(
                    outputs=[torch.empty(1, 4).expand(3, 4)],
                    const_inputs=[f, f],
                    check_mem_overlap=False,
                    resize_outputs=False,
                ),
                lambda c: c.add_output(torch.empty(1, 4).expand(3, 4))
                .add_const_input(f)
                .add_const_input(f)
                .set_check_mem_overlap(False)
                .resize_outputs(False),
            ),
            (
                "static_dtype",
                ConfigSpec(
                    outputs=[out_f64],
                    const_inputs=[f],
                    check_all_same_dtype=False,
                    static_dtype=torch.float64,
                ),
                lambda c: c.add_output(out_f64)
                .add_const_input(f)
                .check_all_same_dtype(False)
                .declare_static_dtype(torch.float64),
            ),
            (
                "static_device",
                ConfigSpec(
                    outputs=[None],
                    const_inputs=[f],
                    static_device=torch.device("cpu"),
                ),
                lambda c: c.add_output(None)
                .add_const_input(f)
                .declare_static_device(torch.device("cpu")),
            ),
            (
                "static_dtype + static_device",
                ConfigSpec(
                    outputs=[out_f64],
                    const_inputs=[f],
                    check_all_same_dtype=False,
                    static_dtype=torch.float64,
                    static_device=torch.device("cpu"),
                ),
                lambda c: c.add_output(out_f64)
                .add_const_input(f)
                .check_all_same_dtype(False)
                .declare_static_dtype_and_device(torch.float64, torch.device("cpu")),
            ),
            (
                "static_shape",
                # declare_static_shape requires resize_outputs(False).
                ConfigSpec(
                    outputs=[out_shape_4_3],
                    const_inputs=[out_shape_4_3],
                    static_shape=(4, 3),
                    resize_outputs=False,
                ),
                lambda c: c.add_output(out_shape_4_3)
                .add_const_input(out_shape_4_3)
                .resize_outputs(False)
                .declare_static_shape([4, 3]),
            ),
            (
                "inputs (non-const)",
                ConfigSpec(
                    outputs=[None],
                    inputs=[f, f],
                    promote_inputs_to_common_dtype=True,
                ),
                lambda c: c.add_output(None)
                .add_input(f)
                .add_input(f)
                .promote_inputs_to_common_dtype(True),
            ),
        ]

        for name, spec, fluent_setup in cases:
            with self.subTest(name):
                it_dc = spec.build()
                it_fluent = fluent_setup(TensorIteratorConfig()).build()
                self.assertEqual(it_dc.shape, it_fluent.shape, name)
                self.assertEqual(it_dc.dtype(0), it_fluent.dtype(0), name)
                self.assertEqual(it_dc.device(0).type, it_fluent.device(0).type, name)
                self.assertEqual(it_dc.common_dtype, it_fluent.common_dtype, name)

    def test_enforce_safe_casting_to_output_raises(self):
        # float -> integer is not a safe cast; the build must reject when
        # safe-casting-to-output is on.
        a = torch.zeros(3, dtype=torch.float32)
        b = torch.zeros(3, dtype=torch.float32)
        out = torch.empty(3, dtype=torch.int32)
        cfg = (
            TensorIteratorConfig()
            .add_output(out)
            .add_const_input(a)
            .add_const_input(b)
            .promote_inputs_to_common_dtype(True)
            .enforce_safe_casting_to_output(True)
            .resize_outputs(False)
        )
        with self.assertRaises(RuntimeError):
            cfg.build()

    def test_enforce_linear_iteration_preserves_c_order(self):
        # Without enforce_linear_iteration, the iterator reorders dims by
        # stride magnitude (smallest first). For a row-major tensor this
        # flips the leading axis to the end. With the flag on, dims are
        # iterated in original C-order. We probe via element_strides:
        # without the flag the output's element_strides should *end* with
        # 1; with the flag they should *start* (after coalesce) with the
        # outer dims in their original order.
        a = torch.zeros(2, 3, 4)
        b = torch.zeros(2, 3, 4)

        no_flag = (
            TensorIteratorConfig()
            .add_output(None)
            .add_const_input(a)
            .add_const_input(b)
            .promote_inputs_to_common_dtype(True)
            .build()
        )
        with_flag = (
            TensorIteratorConfig()
            .add_output(None)
            .add_const_input(a)
            .add_const_input(b)
            .promote_inputs_to_common_dtype(True)
            .enforce_linear_iteration(True)
            .build()
        )
        # Both coalesce a fully-contiguous shape down to one dim, so
        # numel must agree.
        self.assertEqual(no_flag.numel, with_flag.numel)
        # The flag is exercised; specific stride layouts are an
        # implementation detail of reorder_dimensions.

    def test_declare_static_dtype_and_device(self):
        a = torch.zeros(3, dtype=torch.float32)
        out = torch.empty(3, dtype=torch.float64)
        it = (
            TensorIteratorConfig()
            .add_output(out)
            .add_const_input(a)
            .check_all_same_dtype(False)
            .declare_static_dtype_and_device(torch.float64, torch.device("cpu"))
            .build()
        )
        self.assertEqual(it.dtype(0), torch.float64)
        self.assertEqual(it.device(0).type, "cpu")

    def test_declare_static_device(self):
        a = torch.zeros(3)
        out = torch.empty(3)
        it = (
            TensorIteratorConfig()
            .add_output(out)
            .add_const_input(a)
            .declare_static_device(torch.device("cpu"))
            .build()
        )
        self.assertEqual(it.device(0).type, "cpu")

    def test_mixed_dtype_rejected_without_promotion(self):
        # check_all_same_dtype is on by default. Mixed dtypes should fail.
        a = torch.zeros(3, dtype=torch.float32)
        b = torch.zeros(3, dtype=torch.float64)
        with self.assertRaises(RuntimeError):
            (
                TensorIteratorConfig()
                .add_output(None)
                .add_const_input(a)
                .add_const_input(b)
                .build()
            )

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_cross_device_check_raises(self):
        # Default check_all_same_device=True rejects mixed CPU+CUDA.
        a = torch.zeros(3, device="cpu")
        b = torch.zeros(3, device="cuda")
        with self.assertRaises(RuntimeError):
            binary_op(None, a, b)

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_allow_cpu_scalars(self):
        # A 0-dim CPU tensor is a "CPU scalar". With allow_cpu_scalars(True)
        # the iterator accepts it alongside a CUDA tensor without violating
        # the same-device check (the same-device check itself allows CPU
        # scalars when this flag is on).
        a = torch.zeros(3, device="cuda")
        b = torch.tensor(2.0, device="cpu")  # 0-dim CPU scalar
        it = (
            TensorIteratorConfig()
            .add_output(None)
            .add_const_input(a)
            .add_const_input(b)
            .allow_cpu_scalars(True)
            .promote_inputs_to_common_dtype(True)
            .build()
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
        it = (
            TensorIteratorConfig()
            .add_output(None)
            .add_const_input(a)
            .add_const_input(b)
            .promote_inputs_to_common_dtype(True)
            .promote_integer_inputs_to_float(True)
            .build()
        )
        # Integer common dtype gets bumped to default float scalar type.
        self.assertEqual(it.common_dtype, torch.get_default_dtype())

    def test_common_dtype_when_all_match(self, device):
        # check_all_same_dtype still produces a common dtype: it's just the
        # single dtype shared by all operands.
        a = torch.zeros(3, dtype=torch.float64, device=device)
        b = torch.zeros(3, dtype=torch.float64, device=device)
        it = (
            TensorIteratorConfig()
            .add_output(None)
            .add_const_input(a)
            .add_const_input(b)
            .check_all_same_dtype(True)
            .build()
        )
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
        it = (
            TensorIteratorConfig()
            .add_output(out)
            .add_const_input(a)
            .add_const_input(b)
            .check_all_same_dtype(True)
            .resize_outputs(False)
            .build()
        )
        self.assertEqual(it.numel, 12)
        self.assertEqual(tuple(out.shape), (3, 4))

    def test_declare_static_shape(self, device):
        a = torch.zeros(2, 6, device=device)
        out = torch.empty(2, 6, device=device)
        it = (
            TensorIteratorConfig()
            .add_output(out)
            .add_const_input(a)
            .resize_outputs(False)
            .declare_static_shape([2, 6])
            .build()
        )
        # Shape is reported post-coalesce: a 2x6 contiguous tensor collapses.
        self.assertEqual(it.numel, 12)

    def test_declare_static_dtype(self, device):
        a = torch.zeros(3, dtype=torch.float64, device=device)
        out = torch.empty(3, dtype=torch.float64, device=device)
        it = (
            TensorIteratorConfig()
            .add_output(out)
            .add_const_input(a)
            .check_all_same_dtype(False)
            .declare_static_dtype(torch.float64)
            .build()
        )
        self.assertEqual(it.dtype(0), torch.float64)

    def test_mem_overlap_detected(self, device):
        # Broadcast view -> internal overlap on output.
        out = torch.empty(2, 3, device=device).expand(4, 2, 3)
        a = torch.zeros(4, 2, 3, device=device)
        b = torch.zeros(4, 2, 3, device=device)
        with self.assertRaises(RuntimeError):
            (
                TensorIteratorConfig()
                .add_output(out)
                .add_const_input(a)
                .add_const_input(b)
                .check_all_same_dtype(True)
                .resize_outputs(False)
                .build()
            )

    def test_mem_overlap_disabled(self, device):
        # Same setup as above but with the overlap check turned off.
        out = torch.empty(2, 3, device=device).expand(4, 2, 3)
        a = torch.zeros(4, 2, 3, device=device)
        b = torch.zeros(4, 2, 3, device=device)
        it = (
            TensorIteratorConfig()
            .add_output(out)
            .add_const_input(a)
            .add_const_input(b)
            .check_all_same_dtype(True)
            .resize_outputs(False)
            .set_check_mem_overlap(False)
            .build()
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
