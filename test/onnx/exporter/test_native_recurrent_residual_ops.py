# Owner(s): ["module: onnx"]
"""Recurrent, embedding, and residual core operators built with native ONNX IR."""

from __future__ import annotations

from unittest import mock

import numpy as np

import onnx
import onnx_ir as ir
import onnxruntime as ort
from test_native_factory_sequence_ops import _evaluate
from test_native_torchlib import no_script

import torch
from torch.onnx._internal._lazy_import import onnxscript_apis
from torch.onnx._internal.exporter import _registration
from torch.onnx._internal.exporter._torchlib.ops import embedding, recurrent, residual
from torch.testing._internal import common_utils
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import HardwareClassification


class TestNativeRecurrentResidualOps(common_utils.TestCase):
    hw_classification = HardwareClassification.CPU

    @common_utils.parametrize("kind", ["GRU", "LSTM"])
    @common_utils.parametrize("bias", [False, True])
    @common_utils.parametrize("layers", [1, 2])
    @common_utils.parametrize("bidirectional", [False, True])
    @common_utils.parametrize("batch_first", [False, True])
    @common_utils.parametrize("dropout", [0.0, 1.0])
    def test_recurrent(
        self, device, kind, bias, layers, bidirectional, batch_first, dropout
    ):
        torch.manual_seed(0)
        module = (
            getattr(torch.nn, kind)(
                3,
                4,
                num_layers=layers,
                bias=bias,
                bidirectional=bidirectional,
                batch_first=batch_first,
                dropout=dropout if layers > 1 else 0.0,
            )
            .to(device)
            .eval()
        )
        module.train(dropout > 0)
        states = [
            torch.randn(layers * (2 if bidirectional else 1), 2, 4, device=device)
        ]
        if kind == "LSTM":
            states.append(torch.randn_like(states[0]))
        params = [parameter.detach() for parameter in module.parameters()]
        for sequence in (1, 5):
            x = torch.randn(
                (2, sequence, 3) if batch_first else (sequence, 2, 3), device=device
            )
            expected = module(x, states[0] if kind == "GRU" else tuple(states))
            if kind == "LSTM":
                expected = (expected[0], *expected[1])

            def function(op, a, *values):
                a.shape = ir.Shape([2, None, 3] if batch_first else [None, 2, 3])
                state = values[0] if kind == "GRU" else values[:2]
                impl = recurrent.aten_gru if kind == "GRU" else recurrent.aten_lstm
                return impl(
                    op,
                    a,
                    state,
                    values[len(states) :],
                    bias,
                    layers,
                    dropout,
                    dropout > 0,
                    bidirectional,
                    batch_first,
                )

            actual = _evaluate(
                function, [x, *states, *params], expected, dynamic=False, runtime=True
            )
            self.assertEqual(actual, list(expected), atol=2e-6, rtol=2e-5)

    @common_utils.parametrize("mode", [0, 1, 2])
    @common_utils.parametrize("padding_idx", [None, 0, -1])
    @common_utils.parametrize("include_last_offset", [False, True])
    def test_embedding_bag(self, device, mode, padding_idx, include_last_offset):
        weight = torch.randn(5, 3, device=device)
        indices = torch.tensor([1, 0, 2, 4, 0], device=device)
        offsets = torch.tensor(
            [0, 2, 2, 5] if include_last_offset else [0, 2, 2], device=device
        )
        expected = torch.ops.aten.embedding_bag.padding_idx(
            weight,
            indices,
            offsets,
            False,
            mode,
            False,
            None,
            include_last_offset,
            padding_idx,
        )
        actual = _evaluate(
            lambda op, w, i, o: embedding.aten_embedding_bag_padding_idx(
                op,
                w,
                i,
                o,
                mode=mode,
                include_last_offset=include_last_offset,
                padding_idx=padding_idx,
            ),
            [weight, indices, offsets],
            expected,
            runtime=True,
        )
        self.assertEqual(actual, list(expected), atol=2e-6, rtol=2e-5)

    @common_utils.parametrize("include_last_offset", [False, True])
    def test_embedding_bag_weights_and_no_bags(self, device, include_last_offset):
        weight = torch.randn(5, 3, device=device)
        indices = torch.tensor([1, 2, 4], device=device)
        offsets = torch.tensor(
            [0, 1, 3] if include_last_offset else [0, 1], device=device
        )
        weights = torch.randn(3, device=device)
        expected = torch.ops.aten.embedding_bag.default(
            weight, indices, offsets, False, 0, False, weights, include_last_offset
        )
        actual = _evaluate(
            lambda op, w, i, o, p: embedding.aten_embedding_bag(
                op,
                w,
                i,
                o,
                per_sample_weights=p,
                include_last_offset=include_last_offset,
            ),
            [weight, indices, offsets, weights],
            expected,
            runtime=True,
        )
        self.assertEqual(actual, list(expected))
        indices = indices[:0]
        offsets = torch.tensor(
            [0] if include_last_offset else [], dtype=torch.int64, device=device
        )
        expected = torch.ops.aten.embedding_bag.default(
            weight, indices, offsets, False, 0, False, None, include_last_offset
        )
        actual = _evaluate(
            lambda op, w, i, o: embedding.aten_embedding_bag(
                op, w, i, o, include_last_offset=include_last_offset
            ),
            [weight, indices, offsets],
            expected,
            runtime=True,
        )
        self.assertEqual(actual, list(expected))

    @common_utils.parametrize("norm", [1.0, 2.0, 3.0])
    def test_embedding(self, device, norm):
        weight = torch.randn(5, 3, device=device)
        indices = torch.tensor([1, 1, 4], device=device)
        self.assertEqual(
            _evaluate(
                embedding.aten_embedding,
                [weight, indices],
                torch.embedding(weight, indices),
            ),
            torch.embedding(weight, indices),
        )
        expected = torch.embedding_renorm_(weight.clone(), indices, 1.0, norm)
        self.assertEqual(
            _evaluate(
                lambda op, w, i: embedding.aten_embedding_renorm(op, w, i, 1.0, norm),
                [weight, indices],
                expected,
            ),
            expected,
        )

    @common_utils.parametrize("shape", [(), (2, 3)])
    def test_activations_softmax_and_triangles(self, device, shape):
        x = torch.randn(shape, device=device)
        for native, reference in (
            (residual.aten_selu, torch.selu),
            (residual.aten_sigmoid, torch.sigmoid),
            (
                lambda op, a: residual.aten__softmax(op, a, -1, False),
                lambda a: torch.softmax(a, -1),
            ),
            (
                lambda op, a: residual.aten__log_softmax(op, a, -1, False),
                lambda a: torch.log_softmax(a, -1),
            ),
            (
                lambda op, a: residual.aten_softmax(op, a, -1),
                lambda a: torch.softmax(a, -1),
            ),
            (
                lambda op, a: residual.aten_logcumsumexp(op, a, -1),
                lambda a: torch.logcumsumexp(a, -1),
            ),
            (lambda op, a: residual.aten_dropout(op, a, 0.0, False), lambda a: a),
        ):
            expected = reference(x)
            self.assertEqual(_evaluate(native, [x], expected), expected)
        if x.ndim == 2:
            for native, reference in (
                (residual.aten_tril, torch.tril),
                (residual.aten_triu, torch.triu),
            ):
                self.assertEqual(
                    _evaluate(lambda op, a: native(op, a, 1), [x], reference(x, 1)),
                    reference(x, 1),
                )

    @common_utils.parametrize("dtype", [torch.int32, torch.int64])
    @common_utils.parametrize("empty", [False, True])
    def test_bincount_and_unique(self, device, dtype, empty):
        x = torch.tensor(
            [] if empty else [2147483647, 2147483647, 2, 2, 3, 2],
            dtype=dtype,
            device=device,
        )
        for inverse in (False, True):
            for counts in (False, True):
                expected = torch.unique_consecutive(
                    x, return_inverse=True, return_counts=True
                )
                expected = (
                    expected[0],
                    expected[1] if inverse else torch.empty(0, dtype=torch.int64),
                    expected[2] if counts else torch.empty(0, dtype=torch.int64),
                )
                actual = _evaluate(
                    lambda op, a: residual.aten_unique_consecutive(
                        op, a, inverse, counts
                    ),
                    [x],
                    expected,
                )
                self.assertEqual(actual, list(expected))
                for native in (residual.aten__unique2,):
                    expected = torch.ops.aten._unique2(x, True, inverse, counts)
                    actual = _evaluate(
                        lambda op, a: native(op, a, True, inverse, counts),
                        [x],
                        expected,
                    )
                    self.assertEqual(actual, list(expected))
        x = torch.tensor([] if empty else [0, 1, 1, 3, 0], dtype=dtype, device=device)
        for length in (0, 6):
            expected = torch.bincount(x, minlength=length)
            self.assertEqual(
                _evaluate(
                    lambda op, a: residual.aten_bincount(op, a, minlength=length),
                    [x],
                    expected,
                ),
                expected,
            )

    def test_unique_dim(self, device):
        x = torch.tensor([[1, 2], [1, 2], [3, 4]], device=device)
        expected = torch.ops.aten.unique_dim(x, 0, True, True, True)
        self.assertEqual(
            _evaluate(
                lambda op, a: residual.aten_unique_dim(op, a, 0, True, True, True),
                [x],
                expected,
            ),
            list(expected),
        )

    def test_bilinear_and_grouped_mm(self, device):
        a = torch.randn(2, 3, device=device)
        b = torch.randn(2, 4, device=device)
        w = torch.randn(5, 3, 4, device=device)
        bias = torch.randn(5, device=device)
        expected = torch.nn.functional.bilinear(a, b, w, bias)
        self.assertEqual(
            _evaluate(residual.aten_bilinear, [a, b, w, bias], expected), expected
        )
        a = torch.randn(2, 3, 4, device=device)
        b = torch.randn(4, 5, device=device)
        offsets = torch.tensor([1, 4], dtype=torch.int32, device=device)
        expected = torch.cat((a[0, :, :1] @ b[:1], a[1, :, 1:] @ b[1:]), dim=0).reshape(
            2, 3, 5
        )
        a2 = torch.cat((a[0, :, :1], a[1, :, 1:]), dim=1)
        self.assertEqual(
            _evaluate(
                lambda op, x, y, o: residual.aten_grouped_mm(op, x, y, o),
                [a2, b, offsets],
                expected,
                dynamic=False,
            ),
            expected,
        )

    @common_utils.parametrize("complex", [False, True])
    @common_utils.parametrize("shape", [(2, 3), (0, 3), (2, 0)])
    def test_roll(self, device, complex, shape):
        x = torch.randn(shape, device=device)
        if complex:
            x = torch.complex(x, torch.randn_like(x))
        for dims, shifts in (((), (2,)), ((0, -1), (1, -2))):
            expected = torch.roll(x, shifts, dims)
            tensors = [torch.view_as_real(x)] if complex else [x]
            expected = torch.view_as_real(expected) if complex else expected
            impl = residual.aten_roll_complex if complex else residual.aten_roll
            self.assertEqual(
                _evaluate(lambda op, a: impl(op, a, shifts, dims), tensors, expected),
                expected,
            )

    @common_utils.parametrize("normalized", [False, True])
    @common_utils.parametrize("window_size", [None, 4, 8])
    def test_stft(self, device, normalized, window_size):
        x = torch.randn(24, device=device)
        window = torch.hann_window(window_size, device=device) if window_size else None
        expected = torch.view_as_real(
            torch.stft(
                x,
                8,
                2,
                win_length=window_size,
                window=window,
                center=False,
                normalized=normalized,
                return_complex=True,
            )
        )

        def function(op, a, *w):
            return residual.aten_stft(
                op, a, 8, 2, window_size, w[0] if w else None, normalized
            )

        self.assertEqual(
            _evaluate(
                function,
                [x] + ([window] if window is not None else []),
                expected,
                runtime=True,
            ),
            expected,
            atol=2e-5,
            rtol=2e-5,
        )

    def test_pixel_padding_and_complex(self, device):
        x = torch.randn(2, 8, 3, 4, device=device)
        expected = torch.pixel_shuffle(x, 2)
        self.assertEqual(
            _evaluate(
                lambda op, a: residual.aten_pixel_shuffle(op, a, 2), [x], expected
            ),
            expected,
        )
        self.assertEqual(
            _evaluate(
                lambda op, a: residual.aten_pixel_unshuffle(op, a, 2), [expected], x
            ),
            x,
        )
        expected = torch.nn.functional.pad(x, (1, -1, 2, 0), value=3.0)
        self.assertEqual(
            _evaluate(
                lambda op, a: residual.aten_constant_pad_nd(op, a, [1, -1, 2, 0], 3.0),
                [x],
                expected,
            ),
            expected,
        )
        real = torch.randn(2, 1, device=device)
        imag = torch.randn(3, device=device)
        expected = torch.view_as_real(torch.complex(real, imag))
        self.assertEqual(
            _evaluate(residual.aten_complex, [real, imag], expected), expected
        )
        expected = torch.view_as_real(torch.polar(real.abs(), imag))
        self.assertEqual(
            _evaluate(residual.aten_polar, [real.abs(), imag], expected), expected
        )

    @common_utils.parametrize("quant_min,quant_max", [(0, 255), (0, 127), (-128, 127)])
    def test_fake_quantization(self, device, quant_min, quant_max):
        x = torch.randn(2, 3, device=device)
        z = 2 if quant_min == 0 else -2
        expected = torch.fake_quantize_per_tensor_affine(
            x, 0.1, z, quant_min, quant_max
        )
        self.assertEqual(
            _evaluate(
                lambda op, a: residual.aten_fake_quantize_per_tensor_affine(
                    op, a, 0.1, z, quant_min, quant_max
                ),
                [x],
                expected,
                runtime=True,
            ),
            expected,
        )
        scales = torch.full((3,), 0.1, device=device)
        scale = torch.tensor(0.1, device=device)
        zero = torch.tensor(z, device=device)
        expected = torch.fake_quantize_per_tensor_affine(
            x, scale, zero, quant_min, quant_max
        )
        self.assertEqual(
            _evaluate(
                lambda op,
                a,
                s,
                z: residual.aten_fake_quantize_per_tensor_affine_tensor_qparams(
                    op, a, s, z, quant_min, quant_max
                ),
                [x, scale, zero],
                expected,
                runtime=True,
            ),
            expected,
        )
        zeros = torch.full((3,), z, dtype=torch.int32, device=device)
        expected = torch.fake_quantize_per_channel_affine(
            x, scales, zeros, 1, quant_min, quant_max
        )
        self.assertEqual(
            _evaluate(
                lambda op, a, s, z: residual.aten_fake_quantize_per_channel_affine(
                    op, a, s, z, 1, quant_min, quant_max
                ),
                [x, scales, zeros],
                expected,
                runtime=True,
            ),
            expected,
        )

    def test_miscellaneous(self, device):
        a = torch.tensor([-1.0, 0.0, 1.0], device=device)
        b = torch.tensor([2.0, 3.0, 4.0], device=device)
        for native, reference in (
            (residual.aten_heaviside, torch.heaviside),
            (residual.aten_allclose, torch.allclose),
            (residual.aten_equal, torch.equal),
            (residual.aten_cross, torch.linalg.cross),
        ):
            expected = torch.as_tensor(reference(a, b))
            self.assertEqual(_evaluate(native, [a, b], expected), expected)
        for native, expected in (
            (residual.aten_signbit, torch.signbit(a)),
            (
                lambda op, x: residual.aten_histc(op, x, 4, -2, 2),
                torch.histc(a, 4, -2, 2),
            ),
        ):
            self.assertEqual(_evaluate(native, [a], expected), expected)
        expected = torch.nn.functional.prelu(a, torch.tensor([0.2]))
        self.assertEqual(
            _evaluate(residual.aten_prelu, [a, torch.tensor([0.2])], expected), expected
        )
        scalar = torch.tensor(2.0, device=device)
        self.assertEqual(
            _evaluate(residual.aten_is_nonzero, [scalar], torch.tensor(True)),
            torch.tensor(True),
        )
        matrix = torch.eye(3, device=device) * 2
        self.assertEqual(
            _evaluate(residual.aten_logdet, [matrix], torch.logdet(matrix)),
            torch.logdet(matrix),
        )
        for p, train in ((0.0, False), (1.0, True)):
            expected = torch.ops.aten.native_dropout(a, p, train)
            self.assertEqual(
                _evaluate(
                    lambda op, x: residual.aten_native_dropout(op, x, p, train),
                    [a],
                    expected,
                ),
                list(expected),
            )

    @common_utils.parametrize("optimize", [False, True])
    def test_export(self, device, optimize):
        class Model(torch.nn.Module):
            def forward(self, weight, indices, offsets):
                embedded = torch.ops.aten._embedding_bag(
                    weight, indices, offsets, False, 1, False, None, False, -1
                )[0]
                return torch.sigmoid(torch.roll(embedded, 1, 0))

        model = Model().eval()
        args = (
            torch.randn(5, 3, device=device),
            torch.tensor([1, 2, 4], device=device),
            torch.tensor([0, 1], device=device),
        )
        with (
            mock.patch.object(
                onnxscript_apis,
                "get_torchlib_ops",
                side_effect=AssertionError("legacy fallback"),
            ),
            no_script(),
        ):
            program = torch.onnx.export(
                model,
                args,
                dynamo=True,
                optimize=optimize,
                verbose=False,
                opset_version=18,
            )
        onnx.checker.check_model(program.model_proto, full_check=True)
        session = ort.InferenceSession(
            program.model_proto.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        feeds = {
            value.name: tensor.numpy()
            for value, tensor in zip(session.get_inputs(), args)
        }
        actual = session.run(None, feeds)[0]
        self.assertEqual(torch.from_numpy(np.asarray(actual)), model(*args))

    @common_utils.parametrize("opset", [18, 20, 23])
    def test_native_registration(self, device, opset):
        registry = _registration.ONNXRegistry.from_torchlib(opset_version=opset)
        seen = set()
        for implementation, complex_input, targets in _SOURCE_TARGETS:
            for target in targets:
                self.assertNotIn((target, complex_input), seen)
                seen.add((target, complex_input))
                entries = [
                    entry
                    for entry in registry.get_decomps(target)
                    if entry.is_complex == complex_input
                ]
                self.assertIs(entries[0].onnx_function.function, implementation)


_SOURCE_TARGETS = (
    (recurrent.aten_gru, False, (torch.ops.aten.gru.input,)),
    (recurrent.aten_lstm, False, (torch.ops.aten.lstm.input,)),
    (embedding.aten_embedding, False, (torch.ops.aten.embedding.default,)),
    (
        embedding.aten_embedding_renorm,
        False,
        (torch.ops.aten.embedding_renorm.default,),
    ),
    (residual.aten__log_softmax, False, (torch.ops.aten._log_softmax.default,)),
    (residual.aten__softmax, False, (torch.ops.aten._softmax.default,)),
    (residual.aten_allclose, False, (torch.ops.aten.allclose.default,)),
    (residual.aten_bilinear, False, (torch.ops.aten.bilinear.default,)),
    (residual.aten_bincount, False, (torch.ops.aten.bincount.default,)),
    (residual.aten_complex, False, (torch.ops.aten.complex.default,)),
    (residual.aten_constant_pad_nd, False, (torch.ops.aten.constant_pad_nd.default,)),
    (
        residual.aten_cross,
        False,
        (
            torch.ops.aten.cross.default,
            torch.ops.aten.linalg_cross.default,
        ),
    ),
    (residual.aten_dropout, False, (torch.ops.aten.dropout.default,)),
    (residual.aten_equal, False, (torch.ops.aten.equal.default,)),
    (
        residual.aten_fake_quantize_per_channel_affine,
        False,
        (torch.ops.aten.fake_quantize_per_channel_affine.default,),
    ),
    (
        residual.aten_fake_quantize_per_tensor_affine,
        False,
        (torch.ops.aten.fake_quantize_per_tensor_affine.default,),
    ),
    (
        residual.aten_fake_quantize_per_tensor_affine_tensor_qparams,
        False,
        (torch.ops.aten.fake_quantize_per_tensor_affine.tensor_qparams,),
    ),
    (residual.aten_grid_sampler, False, (torch.ops.aten.grid_sampler.default,)),
    (residual.aten_grid_sampler_2d, False, (torch.ops.aten.grid_sampler_2d.default,)),
    (residual.aten_grouped_mm, False, (torch.ops.aten._grouped_mm.default,)),
    (residual.aten_heaviside, False, (torch.ops.aten.heaviside.default,)),
    (residual.aten_histc, False, (torch.ops.aten.histc.default,)),
    (residual.aten_is_nonzero, False, (torch.ops.aten.is_nonzero.default,)),
    (residual.aten_logcumsumexp, False, (torch.ops.aten.logcumsumexp.default,)),
    (residual.aten_logdet, False, (torch.ops.aten.logdet.default,)),
    (residual.aten_native_dropout, False, (torch.ops.aten.native_dropout.default,)),
    (residual.aten_pixel_shuffle, False, (torch.ops.aten.pixel_shuffle.default,)),
    (residual.aten_pixel_unshuffle, False, (torch.ops.aten.pixel_unshuffle.default,)),
    (residual.aten_polar, False, (torch.ops.aten.polar.default,)),
    (
        residual.aten_prelu,
        False,
        (
            torch.ops.aten.prelu.default,
            torch.ops.aten._prelu_kernel.default,
        ),
    ),
    (residual.aten_roll, False, (torch.ops.aten.roll.default,)),
    (residual.aten_roll_complex, True, (torch.ops.aten.roll.default,)),
    (residual.aten_selu, False, (torch.ops.aten.selu.default,)),
    (residual.aten_sigmoid, False, (torch.ops.aten.sigmoid.default,)),
    (residual.aten_signbit, False, (torch.ops.aten.signbit.default,)),
    (
        residual.aten_softmax,
        False,
        (
            torch.ops.aten.softmax.int,
            torch.ops.aten.special_softmax.default,
        ),
    ),
    (residual.aten_stft, False, (torch.ops.aten.stft.default,)),
    (residual.aten_tril, False, (torch.ops.aten.tril.default,)),
    (residual.aten_triu, False, (torch.ops.aten.triu.default,)),
    (
        residual.aten_unique_consecutive,
        False,
        (torch.ops.aten.unique_consecutive.default,),
    ),
    (residual.aten__unique, False, (torch.ops.aten._unique.default,)),
    (residual.aten__unique2, False, (torch.ops.aten._unique2.default,)),
    (residual.aten_unique_dim, False, (torch.ops.aten.unique_dim.default,)),
    (embedding.aten_embedding_bag, False, (torch.ops.aten.embedding_bag.default,)),
    (
        embedding.aten_embedding_bag_padding_idx,
        False,
        (
            torch.ops.aten.embedding_bag.padding_idx,
            torch.ops.aten._embedding_bag.default,
            torch.ops.aten._embedding_bag_forward_only.default,
        ),
    ),
)


instantiate_device_type_tests(TestNativeRecurrentResidualOps, globals(), only_for="cpu")


if __name__ == "__main__":
    common_utils.run_tests()
