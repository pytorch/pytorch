# Owner(s): ["module: onnx"]
"""Native shape, indexing, and reduction translations, without ONNX Script tracing."""

from __future__ import annotations

from unittest import mock

import numpy as np

import onnx
import onnx_ir as ir
import onnxruntime as ort
from onnx.reference import ReferenceEvaluator
from test_native_torchlib import no_script

import torch
from torch.onnx._internal.exporter import (
    _core,
    _custom_translation,
    _ir_builder,
    _registration,
)
from torch.onnx._internal.exporter._torchlib import _torchlib_registry
from torch.onnx._internal.exporter._torchlib.ops import core
from torch.testing._internal import common_utils
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import HardwareClassification


aten = torch.ops.aten


def _evaluate(function, tensors, *, dynamic=False, expected=None, runtime=False):
    graph = ir.Graph([], [], nodes=[], name="structural", opset_imports={"": 18})
    builder = _ir_builder.GraphBuilder(graph, opset=18)
    op = _ir_builder.TranslationContext(builder)
    values = [
        ir.Value(
            name=f"input_{i}",
            shape=ir.Shape(
                [f"d{i}_{j}" for j in range(tensor.ndim)] if dynamic else tensor.shape
            ),
            type=ir.TensorType(ir.DataType.from_numpy(tensor.numpy().dtype)),
        )
        for i, tensor in enumerate(tensors)
    ]
    graph.inputs.extend(values)
    with no_script():
        result = function(op, *values)
    graph.outputs.extend(result if isinstance(result, tuple) else (result,))
    if expected is not None:
        expected_outputs = expected if isinstance(expected, tuple) else (expected,)
        for output, tensor in zip(graph.outputs, expected_outputs):
            # The exporter supplies FX output ranks; dynamic dimensions remain unknown.
            output.shape = ir.Shape([None] * tensor.ndim)
    proto = onnx.shape_inference.infer_shapes(
        ir.to_proto(ir.Model(graph, ir_version=10)), strict_mode=True, data_prop=True
    )
    onnx.checker.check_model(proto, full_check=True)
    feeds = {value.name: tensor.numpy() for value, tensor in zip(values, tensors)}
    if runtime:
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        arrays = ort.InferenceSession(
            proto.SerializeToString(), options, providers=["CPUExecutionProvider"]
        ).run(None, feeds)
    else:
        arrays = ReferenceEvaluator(proto).run(None, feeds)
    outputs = tuple(torch.from_numpy(np.asarray(array).copy()) for array in arrays)
    return outputs if isinstance(result, tuple) else outputs[0]


class TestNativeStructuralOps(common_utils.TestCase):
    hw_classification = HardwareClassification.CPU

    @common_utils.parametrize("opset", [18, 20, 23])
    def test_source_aliases(self, device, opset):
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
                self.assertIsInstance(
                    entries[0].onnx_function, _custom_translation.NativeTranslation
                )
                self.assertIs(entries[0].onnx_function.function, function)
                self.assertEqual(entries[0].opset_introduced, 18)
        self.assertEqual(len(seen), 95)

    @common_utils.parametrize("dynamic", [False, True])
    @common_utils.parametrize("keepdim", [False, True])
    def test_multiple_boolean_reduction_axes(self, device, dynamic, keepdim):
        for native, reference in (
            (core.aten_all_dims, torch.all),
            (core.aten_any_dims, torch.any),
        ):
            for shape, dims in (((2, 3, 4), (0, -1)), ((2, 0, 3), (0, 1)), ((), (0,))):
                with self.subTest(op=reference.__name__, shape=shape):
                    x = torch.ones(shape, device=device, dtype=torch.int64)
                    expected = reference(x, dim=dims, keepdim=keepdim)
                    self.assertEqual(
                        _evaluate(
                            lambda op, a: native(op, a, dims, keepdim),
                            [x],
                            dynamic=dynamic,
                            expected=expected,
                        ),
                        expected,
                    )

    @common_utils.parametrize("dynamic", [False, True])
    @common_utils.parametrize("dtype", [torch.bool, torch.int64, torch.float32])
    def test_shapes_and_negative_axes(self, device, dynamic, dtype):
        x = torch.arange(24, device=device).reshape(2, 3, 4).to(dtype)
        functions = (
            (lambda op, a: core.aten_transpose(op, a, -1, -3), x.transpose(-1, -3)),
            (lambda op, a: core.aten_permute(op, a, [-1, 0, 1]), x.permute(-1, 0, 1)),
            (lambda op, a: core.aten_flatten(op, a, -2, -1), x.flatten(-2, -1)),
            (
                lambda op, a: core.aten_unflatten(op, a, -1, [2, 2]),
                x.unflatten(-1, (2, 2)),
            ),
            (lambda op, a: core.aten_view(op, a, [6, 4]), x.view(6, 4)),
            (lambda op, a: core.aten_flip(op, a, [-1, 0]), x.flip(-1, 0)),
            (lambda op, a: core.aten_slice(op, a, -1, -3, None, 2), x[..., -3::2]),
            (lambda op, a: core.aten_sym_size(op, a, -1), torch.tensor(x.shape[-1])),
            (lambda op, a: core.aten_unfold(op, a, -1, 2, 2), x.unfold(-1, 2, 2)),
        )
        for i, (function, expected) in enumerate(functions):
            with self.subTest(case=i):
                self.assertEqual(
                    _evaluate(function, [x], dynamic=dynamic, expected=expected),
                    expected,
                )

    @common_utils.parametrize("dynamic", [False, True])
    @common_utils.parametrize("dtype", [torch.bool, torch.int64, torch.float32])
    def test_gather_and_index(self, device, dynamic, dtype):
        x = torch.arange(24, device=device).reshape(2, 3, 4).to(dtype)
        first = torch.tensor([1, 0], device=device)
        last = torch.tensor([-1, 1], device=device)
        self.assertEqual(
            _evaluate(
                lambda op, a, i, j: core.aten_index(op, a, [i, None, j]),
                [x, first, last],
                dynamic=dynamic,
            ),
            x[first, :, last],
        )
        mask = torch.tensor([True, False], device=device)
        self.assertEqual(
            _evaluate(
                lambda op, a, m: core.aten_index(op, a, [m]), [x, mask], dynamic=dynamic
            ),
            x[mask],
        )
        indices = torch.tensor([2, 0, 2], device=device)
        self.assertEqual(
            _evaluate(
                lambda op, a, i: core.aten_index_select(op, a, -2, i),
                [x, indices],
                dynamic=dynamic,
            ),
            x.index_select(-2, indices),
        )
        gather_index = torch.tensor([[[3, 1], [1, 1], [0, 2]]], device=device).expand(
            2, -1, -1
        )
        if dtype != torch.bool:
            self.assertEqual(
                _evaluate(
                    lambda op, a, i: core.aten_gather(op, a, -1, i),
                    [x, gather_index],
                    dynamic=dynamic,
                ),
                x.gather(-1, gather_index),
            )

    @common_utils.parametrize("dynamic", [False, True])
    @common_utils.parametrize("accumulate", [False, True])
    def test_index_put(self, device, dynamic, accumulate):
        x = torch.arange(24, device=device, dtype=torch.float32).reshape(2, 3, 4)
        indices = torch.tensor([1, 1, 0] if accumulate else [1, 0], device=device)
        values = torch.ones((len(indices), 4), device=device)
        self.assertEqual(
            _evaluate(
                lambda op, a, i, v: core.aten_index_put(
                    op, a, [None, i], v, accumulate
                ),
                [x, indices, values],
                dynamic=dynamic,
            ),
            aten.index_put.default(x, [None, indices], values, accumulate),
        )
        mask = torch.tensor([True, False], device=device)
        values = torch.ones((3, 4), device=device)
        self.assertEqual(
            _evaluate(
                lambda op, a, m, v: core.aten_index_put(op, a, [m], v, accumulate),
                [x, mask, values],
                dynamic=dynamic,
            ),
            x.index_put((mask,), values, accumulate=accumulate),
        )

    @common_utils.parametrize("dynamic", [False, True])
    @common_utils.parametrize("dtype", [torch.bool, torch.int64, torch.float32])
    def test_masking_and_slice_scatter(self, device, dynamic, dtype):
        x = torch.arange(12, device=device).reshape(3, 4).to(dtype)
        mask = torch.tensor([[True], [False], [True]], device=device)
        source = torch.arange(10, device=device).to(dtype)
        self.assertEqual(
            _evaluate(core.aten_masked_scatter, [x, mask, source], dynamic=dynamic),
            x.masked_scatter(mask, source),
        )
        self.assertEqual(
            _evaluate(
                lambda op, a, s: core.aten_slice_scatter(op, a, s, -1, None, None, 2),
                [x, x[:, ::2]],
                dynamic=dynamic,
            ),
            torch.slice_scatter(x, x[:, ::2], -1, None, None, 2),
        )

    @common_utils.parametrize("reduce", ["sum", "prod", "amin", "amax", "mean"])
    @common_utils.parametrize("include_self", [False, True])
    @common_utils.parametrize("dtype", [torch.float32, torch.int64])
    def test_duplicate_scatter_reductions(self, device, reduce, include_self, dtype):
        x = torch.tensor([3, 6, 9], device=device, dtype=dtype)
        indices = torch.tensor([0, 0, 2], device=device)
        updates = torch.tensor([2, -4, -8], device=device, dtype=dtype)
        self.assertEqual(
            _evaluate(
                lambda op, a, i, s: core.aten_scatter_reduce(
                    op, a, -1, i, s, reduce, include_self
                ),
                [x, indices, updates],
            ),
            x.scatter_reduce(-1, indices, updates, reduce, include_self=include_self),
        )

    def test_scatter_mean_int64_boundary(self, device):
        x = torch.ones(3, dtype=torch.int64, device=device)
        index = torch.tensor([0, 0, 0], device=device)
        src = torch.tensor([torch.iinfo(torch.int64).min, 0, 0], device=device)
        self.assertEqual(
            _evaluate(
                lambda op, a, i, s: core.aten_scatter_reduce(
                    op, a, 0, i, s, "mean", False
                ),
                [x, index, src],
                runtime=True,
            ),
            x.scatter_reduce(0, index, src, "mean", include_self=False),
        )

    @common_utils.parametrize("reduce", ["amin", "amax"])
    def test_scatter_infinite_identity(self, device, reduce):
        x = torch.tensor([float("nan"), 1.0], device=device)
        index = torch.tensor([0, 1], device=device)
        src = torch.tensor([float("-inf"), float("inf")], device=device)
        self.assertEqual(
            _evaluate(
                lambda op, a, i, s: core.aten_scatter_reduce(
                    op, a, 0, i, s, reduce, False
                ),
                [x, index, src],
            ),
            x.scatter_reduce(0, index, src, reduce, include_self=False),
        )

    @common_utils.parametrize("shape", [(), (2, 3), (0, 3)])
    @common_utils.parametrize("dtype", [torch.bool, torch.int32, torch.float32])
    def test_reductions(self, device, shape, dtype):
        x = torch.ones(shape, dtype=dtype, device=device)
        for i, (function, expected) in enumerate(
            (
                (core.aten_all, torch.all(x)),
                (core.aten_any, torch.any(x)),
                (core.aten_sum, torch.sum(x)),
                (core.aten_prod, torch.prod(x)),
            )
        ):
            with self.subTest(case=i):
                self.assertEqual(_evaluate(function, [x], expected=expected), expected)

    @common_utils.parametrize("keepdim", [False, True])
    def test_multiple_outputs(self, device, keepdim):
        x = torch.tensor([[2.0, 3.0, 3.0], [1.0, -2.0, 4.0]], device=device)
        for i, (function, expected) in enumerate(
            (
                (
                    lambda op, a: core.aten_max_dim(op, a, -1, keepdim),
                    tuple(x.max(-1, keepdim=keepdim)),
                ),
                (
                    lambda op, a: core.aten_min_dim(op, a, -1, keepdim),
                    tuple(x.min(-1, keepdim=keepdim)),
                ),
                (lambda op, a: core.aten_topk(op, a, 2, -1), tuple(x.topk(2, -1))),
                (lambda op, a: core.aten_sort(op, a, -1), tuple(x.sort(-1))),
            )
        ):
            with self.subTest(case=i):
                self.assertEqual(_evaluate(function, [x], expected=expected), expected)

    @common_utils.parametrize("shape", [(), (2, 3), (0, 3)])
    def test_complex_shapes(self, device, shape):
        x = torch.ones(shape, dtype=torch.complex64, device=device) * (2 + 3j)
        real = torch.view_as_real(x)
        for function, expected in (
            (lambda op, a: core.aten_sum_complex(op, a), x.sum()),
            (lambda op, a: core.aten_sum_dim_IntList_complex(op, a, []), x.sum(dim=[])),
            (lambda op, a: core.aten_unsqueeze_complex(op, a, -1), x.unsqueeze(-1)),
            (lambda op, a: core.aten_view_complex(op, a, [-1]), x.view(-1)),
        ):
            self.assertEqual(_evaluate(function, [real]), torch.view_as_real(expected))

    @common_utils.parametrize("dynamic", [False, True])
    def test_strided_indices_and_empty_views(self, device, dynamic):
        x = torch.arange(24, device=device, dtype=torch.float32)
        sizes = torch.tensor([2, 3], device=device)
        strides = torch.tensor([6, 2], device=device)
        self.assertEqual(
            _evaluate(
                lambda op, a, s, t: core.aten_as_strided(
                    op,
                    a,
                    [core.aten_select(op, s, 0, 0), core.aten_select(op, s, 0, 1)],
                    [core.aten_select(op, t, 0, 0), core.aten_select(op, t, 0, 1)],
                    1,
                ),
                [x, sizes, strides],
                dynamic=dynamic,
                expected=x.as_strided((2, 3), (6, 2), 1),
            ),
            x.as_strided((2, 3), (6, 2), 1),
        )
        empty = x[:0]
        self.assertEqual(
            _evaluate(lambda op, a: core.aten_view(op, a, [0, 3]), [empty]),
            empty.view(0, 3),
        )

    @common_utils.parametrize("keepdim", [False, True])
    def test_reduction_axes_and_accumulation(self, device, keepdim):
        x = torch.tensor([[1e8, 1.0, -1e8]], device=device)
        expected = x.mean(dim=-1, keepdim=keepdim, dtype=torch.float64)
        self.assertEqual(
            _evaluate(
                lambda op, a: core.aten_mean_dim(
                    op, a, [-1], keepdim, ir.DataType.DOUBLE
                ),
                [x],
                expected=expected,
            ),
            expected,
        )
        integers = torch.tensor([2**30, 2**30, -1], dtype=torch.int32, device=device)
        self.assertEqual(
            _evaluate(lambda op, a: core.aten_cumsum(op, a, 0), [integers]),
            integers.cumsum(0),
        )
        self.assertEqual(
            _evaluate(lambda op, a: core.aten_sum_dim_IntList(op, a, [-1]), [integers]),
            integers.sum(-1),
        )
        for function in (core.aten_all_dim, core.aten_any_dim):
            scalar = torch.tensor(True, device=device)
            self.assertEqual(
                _evaluate(lambda op, a: function(op, a, -1, keepdim), [scalar]), scalar
            )

    @common_utils.parametrize("dynamic", [False, True])
    def test_complex_axes(self, device, dynamic):
        x = torch.arange(24, dtype=torch.float32, device=device).reshape(2, 3, 4)
        x = torch.complex(x, -x)
        real = torch.view_as_real(x)
        functions = (
            (
                lambda op, a: core.aten_transpose_complex(op, a, -1, -2),
                x.transpose(-1, -2),
            ),
            (
                lambda op, a: core.aten_permute_complex(op, a, [-1, 0, -2]),
                x.permute(-1, 0, -2),
            ),
            (lambda op, a: core.aten_slice_complex(op, a, -1, 1, 3), x[..., 1:3]),
            (lambda op, a: core.aten_flip_complex(op, a, [-1]), x.flip(-1)),
            (core.aten_mH_complex, x.mH.resolve_conj()),
            (
                lambda op, a: core.aten_sum_dim_IntList_complex(op, a, [-1], True),
                x.sum(-1, keepdim=True),
            ),
            (core.aten_mean_complex, x.mean()),
            (
                lambda op, a: core.aten_mean_dim_complex(op, a, [-1], True),
                x.mean(-1, keepdim=True),
            ),
            (lambda op, a: core.aten_mean_dim_complex(op, a, []), x.mean()),
            (lambda op, a: core.aten_mean_dim_complex(op, a, None), x.mean()),
        )
        for i, (function, expected) in enumerate(functions):
            with self.subTest(case=i):
                self.assertEqual(
                    _evaluate(
                        function,
                        [real],
                        dynamic=dynamic,
                        expected=torch.view_as_real(expected),
                    ),
                    torch.view_as_real(expected),
                )
        scalar = real[0, 0, 0]
        for function in (
            core.aten_mean_complex,
            lambda op, a: core.aten_mean_dim_complex(op, a, []),
        ):
            self.assertEqual(_evaluate(function, [scalar]), scalar)
        with self.assertRaisesRegex(NotImplementedError, "cast the input"):
            _evaluate(
                lambda op, a: core.aten_mean_complex(op, a, ir.DataType.COMPLEX128),
                [real],
            )

    def test_bounds_and_scalar_gather(self, device):
        x = torch.tensor([1.0, 2.0], device=device)
        with self.assertRaises(
            (IndexError, ValueError, onnx.shape_inference.InferenceError)
        ):
            _evaluate(lambda op, a: core.aten_select(op, a, 0, 2), [x])
        scalar = x[0]
        indices = torch.tensor([0], device=device)
        self.assertEqual(
            _evaluate(
                lambda op, a, i: core.aten_gather(op, a, 0, i), [scalar, indices]
            ),
            scalar.gather(0, indices),
        )

    def test_lower_opset_preserves_external_registry(self, device):
        registry = _registration.ONNXRegistry.from_torchlib(opset_version=17)
        functions = {function for function, _, _ in _SOURCE_TARGETS}
        prior = [
            meta
            for meta in _torchlib_registry.get_torchlib_ops()
            if not isinstance(meta.onnx_function, _custom_translation.NativeTranslation)
            or meta.onnx_function.function not in functions
        ]
        with mock.patch.object(
            _torchlib_registry, "get_torchlib_ops", return_value=prior
        ):
            baseline = _registration.ONNXRegistry.from_torchlib(opset_version=17)
        for target in (aten.view.default, aten.sum.dim_IntList, aten.index.Tensor):
            entries = registry.get_decomps(target)
            self.assertEqual(entries, baseline.get_decomps(target))

    @common_utils.parametrize("optimize", [False, True])
    def test_dynamic_export_without_legacy(self, device, optimize):
        class Model(torch.nn.Module):
            def forward(self, x, index):
                y = x.transpose(-1, -2).reshape(x.shape[0], -1)
                return y.index_select(-1, index).sum(-1), y.max(-1)

        x = torch.arange(24, dtype=torch.float32, device=device).reshape(2, 3, 4)
        indices = torch.tensor([0, 3, 3, 5], device=device)
        _registration.ONNXRegistry.from_torchlib()
        with (
            no_script(),
            mock.patch.object(
                _core,
                "_call_legacy_translation",
                side_effect=AssertionError("legacy translation called"),
            ),
        ):
            program = torch.onnx.export(
                Model(),
                (x, indices),
                optimize=optimize,
                opset_version=20,
                verbose=False,
                dynamic_shapes=({0: torch.export.Dim("batch")}, None),
            )
        proto = ir.to_proto(program.model)
        onnx.checker.check_model(proto, full_check=True)
        for batch in (2, 5):
            value = torch.arange(batch * 12, dtype=x.dtype).reshape(batch, 3, 4)
            actual = ReferenceEvaluator(proto).run(
                None, {"x": value.numpy(), "index": indices.numpy()}
            )
            expected_sum, (expected_max, expected_indices) = Model()(value, indices)
            self.assertEqual(
                tuple(torch.from_numpy(a) for a in actual),
                (expected_sum, expected_max, expected_indices),
            )


# Source aliases are listed independently of the runtime registry.
_SOURCE_TARGETS = (
    (core.aten_all, False, (aten.all.default,)),
    (core.aten_all_dim, False, (aten.all.dim,)),
    (core.aten_all_dims, False, (aten.all.dims,)),
    (core.aten_amax, False, (aten.amax.default,)),
    (core.aten_amin, False, (aten.amin.default,)),
    (core.aten_any, False, (aten.any.default,)),
    (core.aten_any_dim, False, (aten.any.dim,)),
    (core.aten_any_dims, False, (aten.any.dims,)),
    (core.aten_argmax, False, (aten.argmax.default,)),
    (core.aten_argmin, False, (aten.argmin.default,)),
    (core.aten_atleast_1d, False, (aten.atleast_1d.default,)),
    (core.aten_atleast_2d, False, (aten.atleast_2d.default,)),
    (core.aten_atleast_3d, False, (aten.atleast_3d.default,)),
    (core.aten_broadcast_to, False, (aten.broadcast_to.default,)),
    (core.aten_cumsum, False, (aten.cumsum.default,)),
    (core.aten_expand, False, (aten.expand.default,)),
    (core.aten_expand_as, False, (aten.expand_as.default,)),
    (core.aten_flatten, False, (aten.flatten.using_ints,)),
    (core.aten_flip, False, (aten.flip.default,)),
    (core.aten_flip_complex, True, (aten.flip.default,)),
    (core.aten_gather, False, (aten.gather.default,)),
    (core.aten_index, False, (aten.index.Tensor, aten._unsafe_index.Tensor)),
    (core.aten_index_select, False, (aten.index_select.default,)),
    (core.aten_logsumexp, False, (aten.logsumexp.default,)),
    (core.aten_mH, False, (aten.mH.default,)),
    (core.aten_mH_complex, True, (aten.mH.default,)),
    (core.aten_mT, False, (aten.mT.default,)),
    (core.aten_mT_complex, True, (aten.mT.default,)),
    (core.aten_masked_fill, False, (aten.masked_fill.Scalar, aten.masked_fill.Tensor)),
    (core.aten_masked_scatter, False, (aten.masked_scatter.default,)),
    (core.aten_max, False, (aten.max.default,)),
    (core.aten_max_dim, False, (aten.max.dim,)),
    (core.aten_mean, False, (aten.mean.default,)),
    (core.aten_mean_complex, True, (aten.mean.default,)),
    (core.aten_mean_dim, False, (aten.mean.dim,)),
    (core.aten_mean_dim_complex, True, (aten.mean.dim,)),
    (core.aten_min, False, (aten.min.default,)),
    (core.aten_min_dim, False, (aten.min.dim,)),
    (core.aten_narrow, False, (aten.narrow.default,)),
    (core.aten_nonzero, False, (aten.nonzero.default,)),
    (core.aten_permute, False, (aten.permute.default,)),
    (core.aten_permute_complex, True, (aten.permute.default,)),
    (core.aten_prod, False, (aten.prod.default,)),
    (core.aten_prod_dim_int, False, (aten.prod.dim_int,)),
    (core.aten_repeat, False, (aten.repeat.default,)),
    (core.aten_reshape, False, (aten.reshape.default,)),
    (core.aten_scatter_src, False, (aten.scatter.src,)),
    (core.aten_scatter_value, False, (aten.scatter.value,)),
    (core.aten_scatter_add, False, (aten.scatter_add.default,)),
    (core.aten_scatter_reduce, False, (aten.scatter_reduce.two,)),
    (core.aten_select, False, (aten.select.int,)),
    (core.aten_select_scatter, False, (aten.select_scatter.default,)),
    (core.aten_slice_complex, True, (aten.slice.Tensor,)),
    (core.aten_slice, False, (aten.slice.Tensor,)),
    (core.aten_slice_scatter, False, (aten.slice_scatter.default,)),
    (core.aten_sort, False, (aten.sort.default,)),
    (core.aten_squeeze, False, (aten.squeeze.default,)),
    (core.aten_squeeze_dim, False, (aten.squeeze.dim,)),
    (core.aten_squeeze_dim_complex, True, (aten.squeeze.dim,)),
    (core.aten_sum, False, (aten.sum.default,)),
    (core.aten_sum_complex, True, (aten.sum.default,)),
    (core.aten_sum_dim_IntList, False, (aten.sum.dim_IntList,)),
    (core.aten_sum_dim_IntList_complex, True, (aten.sum.dim_IntList,)),
    (core.aten_sym_size, False, (aten.sym_size.int,)),
    (core.aten_sym_storage_offset, False, (aten.sym_storage_offset.default,)),
    (core.aten_t, False, (aten.t.default,)),
    (core.aten_t_complex, True, (aten.t.default,)),
    (core.aten_topk, False, (aten.topk.default,)),
    (core.aten_transpose, False, (aten.transpose.int,)),
    (core.aten_transpose_complex, True, (aten.transpose.int,)),
    (core.aten_unflatten, False, (aten.unflatten.int,)),
    (core.aten_unfold, False, (aten.unfold.default,)),
    (core.aten_unsqueeze, False, (aten.unsqueeze.default,)),
    (core.aten_unsqueeze_complex, True, (aten.unsqueeze.default,)),
    (core.aten_view, False, (aten.view.default, aten._unsafe_view.default)),
    (core.aten_view_complex, True, (aten.view.default, aten._unsafe_view.default)),
    (core.aten_view_as, False, (aten.view_as.default,)),
    (core.aten_view_as_complex, False, (aten.view_as_complex.default,)),
    (core.aten_view_as_complex_copy, False, (aten.view_as_complex_copy.default,)),
    (core.aten_view_as_real, True, (aten.view_as_real.default,)),
    (core.aten_view_as_real_copy, True, (aten.view_as_real_copy.default,)),
    (core.aten_view_copy, False, (aten.view_copy.default,)),
    (
        core.aten_where,
        False,
        (
            aten.where.Scalar,
            aten.where.ScalarSelf,
            aten.where.ScalarOther,
            aten.where.self,
        ),
    ),
    (core.aten_as_strided, False, (aten.as_strided.default,)),
    (core.aten_diagonal, False, (aten.diagonal.default, aten.diagonal_copy.default)),
    (
        core.aten_index_put,
        False,
        (aten.index_put.default, aten._unsafe_index_put.default),
    ),
)


instantiate_device_type_tests(TestNativeStructuralOps, globals(), only_for=["cpu"])

if __name__ == "__main__":
    common_utils.run_tests()
