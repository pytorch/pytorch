# Owner(s): ["module: onnx"]
"""Factory and Tensor[] translations built directly with ONNX IR."""

from __future__ import annotations

import operator

import numpy as np

import onnx
import onnx_ir as ir
import onnxruntime as ort
from onnx.reference import ReferenceEvaluator
from test_native_torchlib import no_script

import torch
from torch.onnx._internal.exporter import _ir_builder, _registration
from torch.onnx._internal.exporter._torchlib.ops import factories, linalg, sequences
from torch.testing._internal import common_utils
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import HardwareClassification


def _evaluate(function, tensors, expected, *, dynamic=True, runtime=False):
    graph = ir.Graph([], [], nodes=[], name="factories", opset_imports={"": 18})
    builder = _ir_builder.GraphBuilder(graph)
    values = [
        ir.Value(
            name=f"input_{index}",
            type=ir.TensorType(ir.DataType.from_numpy(tensor.numpy().dtype)),
            shape=ir.Shape([None] * tensor.ndim if dynamic else tensor.shape),
        )
        for index, tensor in enumerate(tensors)
    ]
    graph.inputs.extend(values)
    with no_script(), builder.activate():
        result = function(*values)
    if isinstance(result, (list, tuple)):
        graph.outputs.extend(result)
        expected_values = expected
    else:
        graph.outputs.append(result)
        expected_values = [expected]
    for output, reference in zip(graph.outputs, expected_values):
        if not isinstance(output.type, ir.SequenceType):
            output.shape = ir.Shape([None] * reference.ndim)
    proto = onnx.shape_inference.infer_shapes(
        ir.to_proto(ir.Model(graph, ir_version=10)), strict_mode=True
    )
    onnx.checker.check_model(proto, full_check=True)
    feeds = {value.name: tensor.numpy() for value, tensor in zip(values, tensors)}
    if runtime:
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        outputs = ort.InferenceSession(
            proto.SerializeToString(), options, providers=["CPUExecutionProvider"]
        ).run(None, feeds)
    else:
        outputs = ReferenceEvaluator(proto).run(None, feeds)

    def convert(value):
        if isinstance(value, list):
            return [convert(element) for element in value]
        return torch.from_numpy(np.asarray(value).copy())

    outputs = [convert(value) for value in outputs]
    return outputs if isinstance(result, (list, tuple)) else outputs[0]


class TestNativeFactorySequenceOps(common_utils.TestCase):
    hw_classification = HardwareClassification.CPU

    @common_utils.parametrize(
        "dtype", [torch.float32, torch.float64, torch.int64, torch.int32, torch.int8]
    )
    def test_arange_and_linspace(self, device, dtype):
        onnx_dtype = ir.DataType.from_numpy(torch.empty((), dtype=dtype).numpy().dtype)
        cases = [
            (
                lambda: factories.aten_arange(3.1, dtype=onnx_dtype),
                torch.arange(3.1, dtype=dtype),
            ),
            (
                lambda: factories.aten_arange_start(-2, 4, dtype=onnx_dtype),
                torch.arange(-2, 4, dtype=dtype),
            ),
            (
                lambda: factories.aten_arange_start_step(
                    3.2, -2.1, -1.2, dtype=onnx_dtype
                ),
                torch.arange(3.2, -2.1, -1.2, dtype=dtype),
            ),
        ]
        for steps in (0, 1, 2, 7):
            cases.append(
                (
                    lambda steps=steps: factories.aten_linspace(
                        -2.2, 5.3, steps, dtype=onnx_dtype
                    ),
                    torch.linspace(-2.2, 5.3, steps, dtype=dtype),
                )
            )
        for function, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(_evaluate(function, [], expected), expected)

    def test_dynamic_range_and_linspace(self, device):
        for size in (0, 1, 6):
            x = torch.empty(size, device=device)
            for native, reference in (
                (lambda n: factories.aten_arange(n), torch.arange(size)),
                (
                    lambda n: factories.aten_linspace(2.0, 8.0, n),
                    torch.linspace(2, 8, size),
                ),
            ):

                def function(a):
                    builder = _ir_builder.current_builder()
                    size_value = builder.call("Size", a)[0]
                    return native(size_value)

                self.assertEqual(_evaluate(function, [x], reference), reference)

    @common_utils.parametrize("shape", [(), (0,), (2, 0, 3), (2, 3)])
    @common_utils.parametrize("dtype", [torch.bool, torch.int64, torch.float64])
    def test_factories(self, device, shape, dtype):
        x = torch.ones(shape, device=device, dtype=dtype)
        onnx_dtype = ir.DataType.from_numpy(x.numpy().dtype)
        cases = [
            (
                lambda a, size: factories.aten_empty(size, dtype=onnx_dtype),
                torch.zeros_like(x),
            ),
            (
                lambda a, size: factories.aten_empty_strided(
                    size, [1] * len(shape), dtype=onnx_dtype
                ),
                torch.zeros_like(x),
            ),
            (
                lambda a, size: factories.aten_zeros(size, dtype=onnx_dtype),
                torch.zeros_like(x),
            ),
            (
                lambda a, size: factories.aten_ones(size, dtype=onnx_dtype),
                torch.ones_like(x),
            ),
            (
                lambda a, size: factories.aten_full(size, 3, dtype=onnx_dtype),
                torch.full_like(x, 3),
            ),
            (lambda a, size: factories.aten_new_empty(a, size), torch.zeros_like(x)),
            (
                lambda a, size: factories.aten_new_empty_strided(
                    a, size, [1] * len(shape)
                ),
                torch.zeros_like(x),
            ),
            (lambda a, size: factories.aten_new_zeros(a, size), torch.zeros_like(x)),
            (lambda a, size: factories.aten_new_ones(a, size), torch.ones_like(x)),
            (
                lambda a, size: factories.aten_new_full(a, size, 3),
                torch.full_like(x, 3),
            ),
            (lambda a, size: factories.aten_empty_like(a), torch.zeros_like(x)),
            (lambda a, size: factories.aten_zeros_like(a), torch.zeros_like(x)),
            (lambda a, size: factories.aten_ones_like(a), torch.ones_like(x)),
            (lambda a, size: factories.aten_full_like(a, 3), torch.full_like(x, 3)),
            (lambda a, size: factories.aten_fill(a, 3), torch.full_like(x, 3)),
        ]
        for index, (native, expected) in enumerate(cases):
            with self.subTest(case=index):

                def function(a):
                    return native(a, _ir_builder.current_builder().call("Shape", a)[0])

                self.assertEqual(_evaluate(function, [x], expected), expected)

    @common_utils.parametrize("dim", [0, 1, -1])
    def test_splits_and_sequence_consumers(self, device, dim):
        x = torch.arange(30, device=device).reshape(5, 6)
        expected = list(torch.split(x, 2, dim))
        self.assertEqual(
            _evaluate(lambda a: sequences.aten_split(a, 2, dim), [x], expected),
            expected,
        )
        self.assertEqual(
            _evaluate(lambda a: sequences.aten_unsafe_split(a, 2, dim), [x], expected),
            expected,
        )
        sizes = [1, x.shape[dim] - 1]
        expected = list(torch.split(x, sizes, dim))
        self.assertEqual(
            _evaluate(
                lambda a: sequences.aten_split_with_sizes(a, sizes, dim), [x], expected
            ),
            expected,
        )
        self.assertEqual(
            _evaluate(
                lambda a: sequences.aten_cat(sequences.aten_split(a, 2, dim), dim),
                [x],
                x,
            ),
            x,
        )
        self.assertEqual(
            _evaluate(
                lambda a: sequences.aten_getitem(sequences.aten_split(a, 2, dim), -1),
                [x],
                expected[-1],
            ),
            torch.split(x, 2, dim)[-1],
        )
        expected = list(torch.unbind(x, dim))
        self.assertEqual(
            _evaluate(lambda a: sequences.aten_unbind(a, dim), [x], expected), expected
        )

    @common_utils.parametrize("ndim", [1, 2, 3])
    def test_atleast_runtime_sequence(self, device, ndim):
        native = getattr(sequences, f"aten_atleast_{ndim}d_sequence")
        reference = getattr(torch, f"atleast_{ndim}d")
        for shape in ((), (0,), (3,), (2, 3), (2, 3, 4)):
            x = torch.ones(shape, device=device)
            expected = [reference(x)]

            def function(a):
                builder = _ir_builder.current_builder()
                sequence = builder.call(
                    "SequenceConstruct", a, _output_types=[ir.SequenceType(a.type)]
                )[0]
                return native(sequence)

            self.assertEqual(_evaluate(function, [x], expected), expected)
            self.assertEqual(_evaluate(lambda a: native([a]), [x], expected), expected)

    @common_utils.parametrize("dim", [None, 0, 1, -1])
    def test_repeat_interleave(self, device, dim):
        x = torch.arange(12, device=device).reshape(3, 4)
        for repeats in (0, 1, 3):
            expected = x.repeat_interleave(repeats, dim)
            self.assertEqual(
                _evaluate(
                    lambda a: sequences.aten_repeat_interleave_self_int(
                        a, repeats, dim
                    ),
                    [x],
                    expected,
                ),
                expected,
            )
            count = x.numel() if dim is None else x.shape[dim]
            vector = torch.arange(count, device=device) % 3
            expected = x.repeat_interleave(vector, dim)
            self.assertEqual(
                _evaluate(
                    lambda a, b: sequences.aten_repeat_interleave_Tensor(a, b, dim),
                    [x, vector],
                    expected,
                ),
                expected,
            )
        repeats_value = torch.tensor(2, device=device)
        expected = x.repeat_interleave(2, dim)
        self.assertEqual(
            _evaluate(
                lambda a, b: sequences.aten_repeat_interleave_self_int(a, b, dim),
                [x, repeats_value],
                expected,
            ),
            expected,
        )

    def test_empty_repeat_indices(self, device):
        for values in ([], [0, 0], [1, 0, 3]):
            repeats = torch.tensor(values, dtype=torch.int64, device=device)
            expected = torch.repeat_interleave(repeats)
            self.assertEqual(
                _evaluate(sequences.aten_repeat_interleave_Tensor, [repeats], expected),
                expected,
            )

    def test_dynamic_chunks_and_split_sizes(self, device):
        for size in (0, 1, 5, 6):
            x = torch.arange(size * 3, device=device).reshape(size, 3)
            for chunks in (1, 2, 4, 8):
                expected = list(torch.chunk(x, chunks))
                self.assertEqual(
                    _evaluate(lambda a: sequences.aten_chunk(a, chunks), [x], expected),
                    expected,
                )
            sizes = torch.tensor([0, size], dtype=torch.int64, device=device)
            expected = list(torch.split(x, sizes.tolist()))
            self.assertEqual(
                _evaluate(
                    lambda a, b: sequences.aten_split_with_sizes(a, b),
                    [x, sizes],
                    expected,
                ),
                expected,
            )

    def test_dynamic_repeat_and_tile(self, device):
        x = torch.arange(6, device=device).reshape(2, 3)
        repeat = torch.tensor(2, device=device)
        expected = x.tile(2, 1)
        self.assertEqual(
            _evaluate(
                lambda a, b: sequences.aten_tile(a, [b, 1]), [x, repeat], expected
            ),
            expected,
        )
        for shape in ((0, 3), (2, 0), (0,)):
            empty = torch.empty(shape, device=device)
            expected = empty.repeat_interleave(0, -1)
            self.assertEqual(
                _evaluate(
                    lambda a: sequences.aten_repeat_interleave_self_int(a, 0, -1),
                    [empty],
                    expected,
                ),
                expected,
            )

    def test_stack_helpers_and_empty_cat(self, device):
        for shape in ((), (3,), (2, 3)):
            x = torch.ones(shape, device=device)
            for native, reference in (
                (sequences.aten_hstack, torch.hstack),
                (sequences.aten_vstack, torch.vstack),
            ):
                expected = reference([x, x])
                self.assertEqual(
                    _evaluate(lambda a: native([a, a]), [x], expected), expected
                )

                def function(a):
                    builder = _ir_builder.current_builder()
                    seq = builder.call(
                        "SequenceConstruct",
                        a,
                        a,
                        _output_types=[ir.SequenceType(a.type)],
                    )[0]
                    return native(seq)

                self.assertEqual(_evaluate(function, [x], expected), expected)
        empty = torch.empty(0, device=device)
        self.assertEqual(
            _evaluate(
                lambda a: sequences.aten_cat([a, a]), [empty], empty, dynamic=False
            ),
            empty,
        )

    def test_windows_and_scalar_tensor(self, device):
        for native, reference in (
            (factories.aten_blackman_window, torch.blackman_window),
            (factories.aten_hann_window, torch.hann_window),
            (factories.aten_hamming_window, torch.hamming_window),
        ):
            for size in (0, 1, 7):
                expected = reference(size, device=device)
                output = _evaluate(lambda: native(size), [], expected)
                self.assertEqual(output, expected, rtol=8e-2, atol=6e-3)
                length = torch.tensor(size, device=device)
                output = _evaluate(native, [length], expected)
                self.assertEqual(output, expected, rtol=8e-2, atol=6e-3)
        for native, value in (
            (factories.aten_tensor_bool, True),
            (factories.aten_tensor_int, 12),
            (factories.aten_tensor_float, 1.25),
        ):
            expected = torch.tensor(value, device=device)
            self.assertEqual(
                _evaluate(lambda: native(value, None), [], expected), expected
            )

    def test_matrix_products(self, device):
        for left_shape, right_shape in (
            ((3,), (3,)),
            ((2, 3), (3,)),
            ((3,), (3, 4)),
            ((2, 3), (3, 4)),
            ((2, 2, 3), (2, 3, 4)),
        ):
            a = torch.randn(left_shape, dtype=torch.complex64, device=device)
            b = torch.randn(right_shape, dtype=torch.complex64, device=device)
            expected = torch.view_as_real(a @ b)
            self.assertEqual(
                _evaluate(
                    linalg.aten_matmul_complex,
                    [torch.view_as_real(a), torch.view_as_real(b)],
                    expected,
                ),
                expected,
            )
        a = torch.randn(2, 3, 4, device=device)
        b = torch.randn(2, 4, 5, device=device)
        for native, reference, shape in (
            (linalg.aten_addbmm, torch.addbmm, (3, 5)),
            (linalg.aten_baddbmm, torch.baddbmm, (2, 3, 5)),
        ):
            bias = torch.full(shape, float("nan"), device=device)
            expected = reference(bias, a, b, beta=0)
            self.assertEqual(
                _evaluate(
                    lambda x, y, z: native(x, y, z, beta=0), [bias, a, b], expected
                ),
                expected,
            )

    @common_utils.parametrize("dim", [0, 1, -1])
    def test_complex_cat_stack(self, device, dim):
        x = torch.complex(
            torch.arange(6, device=device).reshape(2, 3).float(),
            torch.ones(2, 3, device=device),
        )
        real = torch.view_as_real(x)
        for native, reference in (
            (sequences.aten_cat_complex, torch.cat),
            (sequences.aten_stack_complex, torch.stack),
        ):
            expected = torch.view_as_real(reference([x, x], dim))
            self.assertEqual(
                _evaluate(lambda a: native([a, a], dim), [real], expected), expected
            )

    @common_utils.parametrize("shape", [(), (0,), (3, 4)])
    def test_random_shapes_ranges(self, device, shape):
        x = torch.ones(shape, device=device, dtype=torch.float64)
        for native in (factories.aten_rand, factories.aten_randn):
            expected = torch.empty(shape, dtype=torch.float64)
            output = _evaluate(
                lambda a: native(
                    _ir_builder.current_builder().call("Shape", a)[0],
                    dtype=ir.DataType.DOUBLE,
                ),
                [x],
                expected,
                runtime=True,
            )
            self.assertEqual(output.shape, expected.shape)
            self.assertEqual(output.dtype, expected.dtype)
            self.assertTrue(torch.isfinite(output).all())
            if native is factories.aten_rand:
                self.assertTrue(((output >= 0) & (output < 1)).all())
        output = _evaluate(
            lambda a: factories.aten_randint_low(
                -3, 5, _ir_builder.current_builder().call("Shape", a)[0]
            ),
            [x],
            torch.empty(shape, dtype=torch.int64),
            runtime=True,
        )
        self.assertTrue(((output >= -3) & (output < 5)).all())
        output = _evaluate(
            lambda a: factories.aten_bernoulli_p(a, 0.5), [x], x, runtime=True
        )
        self.assertTrue(((output == 0) | (output == 1)).all())
        output = _evaluate(lambda a: factories.aten_normal(a), [x], x, runtime=True)
        self.assertEqual(output.shape, x.shape)
        output = _evaluate(
            lambda a: factories.aten_normal_float_tensor(2.0, a), [x], x, runtime=True
        )
        self.assertEqual(output.shape, x.shape)

    @common_utils.parametrize("optimize", [False, True])
    def test_export_sequences_factories(self, device, optimize):
        class Model(torch.nn.Module):
            def forward(self, x):
                pieces = torch.split(x, [1, 2], dim=1)
                return torch.cat(pieces, dim=1) + torch.ones_like(x), torch.stack(
                    torch.unbind(x, dim=1), dim=-1
                )

        x = torch.arange(12, device=device).reshape(4, 3).float()
        with no_script():
            program = torch.onnx.export(
                Model(), (x,), optimize=optimize, opset_version=18
            )
        onnx.checker.check_model(program.model_proto, full_check=True)
        expected = Model()(x)
        actual = ReferenceEvaluator(program.model_proto).run(
            None, {program.model.graph.inputs[0].name: x.numpy()}
        )
        self.assertEqual(tuple(torch.from_numpy(a) for a in actual), expected)

    @common_utils.parametrize("optimize", [False, True])
    def test_export_dynamic_factories(self, device, optimize):
        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.full((x.shape[0], 3), 2.0) + torch.arange(
                    x.shape[0], dtype=x.dtype
                ).unsqueeze(-1)

        x = torch.ones(4, 3, device=device)
        with no_script():
            program = torch.onnx.export(
                Model(),
                (x,),
                optimize=optimize,
                opset_version=18,
                dynamic_shapes={"x": {0: torch.export.Dim("batch", min=1)}},
            )
        onnx.checker.check_model(program.model_proto, full_check=True)
        for size in (2, 5):
            value = torch.ones(size, 3, device=device)
            arrays = ReferenceEvaluator(program.model_proto).run(
                None, {program.model.graph.inputs[0].name: value.numpy()}
            )
            self.assertEqual(torch.from_numpy(arrays[0]), Model()(value))

    @common_utils.parametrize("opset", [18, 20, 23])
    def test_aliases_are_native(self, device, opset):
        registry = _registration.ONNXRegistry.from_torchlib(opset_version=opset)
        seen = set()
        for function, complex_input, targets in _SOURCE_TARGETS:
            for target in targets:
                self.assertNotIn((target, complex_input), seen)
                seen.add((target, complex_input))
                entries = [
                    entry
                    for entry in registry.get_decomps(target)
                    if entry.is_complex == complex_input
                ]
                self.assertEqual(len(entries), 1)
                self.assertIs(entries[0].onnx_function.function, function)
                self.assertEqual(entries[0].opset_introduced, 18)


instantiate_device_type_tests(TestNativeFactorySequenceOps, globals(), only_for="cpu")


aten = torch.ops.aten
_SOURCE_TARGETS = (
    (factories.aten_arange, False, (aten.arange.default,)),
    (factories.aten_arange_start, False, (aten.arange.start,)),
    (factories.aten_arange_start_step, False, (aten.arange.start_step,)),
    (factories.aten_linspace, False, (aten.linspace.default,)),
    (factories.aten_empty, False, (aten.empty.memory_format,)),
    (factories.aten_empty_like, False, (aten.empty_like.default,)),
    (factories.aten_empty_strided, False, (aten.empty_strided.default,)),
    (
        factories.aten_fill,
        False,
        (
            aten.fill.Tensor,
            aten.fill.Scalar,
        ),
    ),
    (factories.aten_full, False, (aten.full.default,)),
    (factories.aten_full_like, False, (aten.full_like.default,)),
    (factories.aten_new_empty, False, (aten.new_empty.default,)),
    (factories.aten_new_empty_strided, False, (aten.new_empty_strided.default,)),
    (factories.aten_new_full, False, (aten.new_full.default,)),
    (factories.aten_new_ones, False, (aten.new_ones.default,)),
    (factories.aten_new_zeros, False, (aten.new_zeros.default,)),
    (factories.aten_ones, False, (aten.ones.default,)),
    (factories.aten_ones_like, False, (aten.ones_like.default,)),
    (factories.aten_zeros, False, (aten.zeros.default,)),
    (factories.aten_zeros_like, False, (aten.zeros_like.default,)),
    (factories.aten_tensor_bool, False, (aten.tensor.bool,)),
    (factories.aten_tensor_float, False, (aten.tensor.float,)),
    (factories.aten_tensor_int, False, (aten.tensor.int,)),
    (factories.aten_blackman_window, False, (aten.blackman_window.default,)),
    (factories.aten_hamming_window, False, (aten.hamming_window.default,)),
    (factories.aten_hann_window, False, (aten.hann_window.default,)),
    (factories.aten_bernoulli, False, (aten.bernoulli.default,)),
    (factories.aten_bernoulli_p, False, (aten.bernoulli.p,)),
    (factories.aten_normal, False, (aten.normal_functional.default,)),
    (factories.aten_normal_float_float, False, (aten.normal.float_float,)),
    (factories.aten_normal_float_tensor, False, (aten.normal.float_Tensor,)),
    (factories.aten_normal_tensor_float, False, (aten.normal.Tensor_float,)),
    (factories.aten_normal_tensor_tensor, False, (aten.normal.Tensor_Tensor,)),
    (factories.aten_rand, False, (aten.rand.default,)),
    (factories.aten_rand_like, False, (aten.rand_like.default,)),
    (factories.aten_randint, False, (aten.randint.default,)),
    (factories.aten_randint_low, False, (aten.randint.low,)),
    (factories.aten_randint_like, False, (aten.randint_like.default,)),
    (factories.aten_randint_like_low_dtype, False, (aten.randint_like.low_dtype,)),
    (factories.aten_randn, False, (aten.randn.default,)),
    (factories.aten_randn_like, False, (aten.randn_like.default,)),
    (factories.aten_multinomial, False, (aten.multinomial.default,)),
    (sequences.aten_atleast_1d_sequence, False, (aten.atleast_1d.Sequence,)),
    (sequences.aten_atleast_2d_sequence, False, (aten.atleast_2d.Sequence,)),
    (sequences.aten_atleast_3d_sequence, False, (aten.atleast_3d.Sequence,)),
    (
        sequences.aten_cat_complex,
        True,
        (
            aten.cat.default,
            aten.concat.default,
            aten.concatenate.default,
        ),
    ),
    (
        sequences.aten_cat,
        False,
        (
            aten.cat.default,
            aten.concat.default,
            aten.concatenate.default,
        ),
    ),
    (sequences.aten_getitem, False, (operator.getitem,)),
    (
        sequences.aten_repeat_interleave_self_int,
        False,
        (aten.repeat_interleave.self_int,),
    ),
    (sequences.aten_repeat_interleave_Tensor, False, (aten.repeat_interleave.Tensor,)),
    (
        sequences.aten_split,
        False,
        (
            aten.split.default,
            aten.split.Tensor,
        ),
    ),
    (sequences.aten_split_with_sizes, False, (aten.split_with_sizes.default,)),
    (sequences.aten_stack_complex, True, (aten.stack.default,)),
    (sequences.aten_stack, False, (aten.stack.default,)),
    (sequences.aten_tile, False, (aten.tile.default,)),
    (sequences.aten_unbind, False, (aten.unbind.int,)),
    (linalg.aten_mm, False, (aten.mm.default,)),
    (linalg.aten_bmm, False, (aten.bmm.default,)),
    (linalg.aten_mv, False, (aten.mv.default,)),
    (linalg.aten_dot, False, (aten.dot.default,)),
    (linalg.aten_matmul, False, (aten.matmul.default,)),
    (linalg.aten_matmul_complex, True, (aten.matmul.default,)),
    (linalg.aten_addmm, False, (aten.addmm.default,)),
    (linalg.aten_addbmm, False, (aten.addbmm.default,)),
    (linalg.aten_baddbmm, False, (aten.baddbmm.default,)),
    (linalg.aten_addmv, False, (aten.addmv.default,)),
    (linalg.aten_addr, False, (aten.addr.default,)),
    (linalg.aten_einsum, False, (aten.einsum.default,)),
    (linalg.aten_contiguous, False, (aten.contiguous.default,)),
    (linalg.aten_copy, False, (aten.copy.default,)),
    (linalg.aten_detach, False, (aten.detach.default,)),
    (linalg.aten_lift_fresh_copy, False, (aten.lift_fresh_copy.default,)),
    (linalg.aten_resolve_conj, False, (aten.resolve_conj.default,)),
    (linalg.aten_resolve_neg, False, (aten.resolve_neg.default,)),
    (sequences.aten_chunk, False, (aten.chunk.default,)),
    (sequences.aten_unsafe_split, False, (aten.unsafe_split.Tensor,)),
)


if __name__ == "__main__":
    common_utils.run_tests()
