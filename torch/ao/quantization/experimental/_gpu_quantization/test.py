# mypy: ignore-errors
import copy
import unittest

import torch
import torch.nn as nn
from torch._inductor.utils import run_and_get_code

from torch.ao.quantization import MinMaxObserver, QConfigMapping

from torch.ao.quantization.experimental._gpu_quantization.dynamic_quant import (
    DynamicallyPerAxisQuantizedLinear,
)
from torch.ao.quantization.experimental._gpu_quantization.quant_api import (
    apply_dynamic_quant,
    apply_weight_only_int8_quant,
    change_linear_weights_to_dqtensors,
)
from torch.ao.quantization.experimental._gpu_quantization.quant_primitives import (
    dequantize_per_channel,
    dequantize_per_tensor,
    dynamically_quantize_per_channel,
    dynamically_quantize_per_tensor,
    quant_int8_dynamic_linear,
    quant_int8_dynamic_per_token_linear,
    quantize_activation_per_token_absmax,
    safe_int_mm,
)

from torch.ao.quantization.experimental._gpu_quantization.smoothquant import (
    get_scale,
    replace_with_custom_fn_if_matches_filter,
    smooth_fq_linear_to_inference,
    SmoothFakeDynamicallyQuantizedLinear,
    swap_linear_with_smooth_fq_linear,
)
from torch.ao.quantization.experimental._gpu_quantization.subclass import (
    DynamicallyQuantizedLinearWeight,
)
from torch.ao.quantization.experimental._gpu_quantization.utils import (
    apply_logging_hook,
    compute_error,
    compute_error as SQNR,
    fqn_to_op_to_shape_to_count,
    LoggingTensorMode,
)
from torch.ao.quantization.quantize_fx import convert_to_reference_fx, prepare_fx

torch.manual_seed(0)


class SmoothquantUnitTest(unittest.TestCase):
    # first, let's reproduce the graphic from the paper, Figure 4, to ensure
    # we are calculating the scales correctly
    def test_figure_4(self):
        X = torch.FloatTensor([1, -16, 2, 6, -2, 8, -1, -9]).reshape(1, 2, 4)
        W = torch.FloatTensor([2, 1, -2, 1, -1, -1, 2, -1, -2, -1, -1, 1]).reshape(4, 3)
        X_mul_W = torch.matmul(X, W)

        smoothquant_scale = get_scale(
            torch.amax(torch.abs(X), dim=(0, 1)),
            torch.amax(torch.abs(W), dim=1),
            alpha=0.5,
        )

        # reproduce scaled calculation
        X_scaled = X / smoothquant_scale.reshape(1, 1, -1)
        W_scaled = torch.matmul(torch.diag(smoothquant_scale), W)
        X_scaled_mul_scaled_W = torch.matmul(X_scaled, W_scaled)
        assert torch.allclose(X_mul_W, X_scaled_mul_scaled_W), "not close!"
        assert X_mul_W.shape == X_scaled_mul_scaled_W.shape

    # next, run the above test on a sample of representative inputs
    def test_tensors(self):
        x_shape = (1, 5, 7)
        w_shape = (7, 9)
        for i in range(3):
            X = torch.randn(x_shape) * 10
            W = torch.randn(w_shape)
            s = get_scale(
                torch.amax(torch.abs(X), dim=(0, 1)),
                torch.amax(torch.abs(W), dim=1),
                alpha=0.5,
            )

            Y = torch.matmul(X, W)
            Y_ref = torch.matmul(
                X / s.reshape(1, 1, -1),
                torch.matmul(torch.diag(s), W),
            )
            assert torch.allclose(Y, Y_ref, atol=1e-3, rtol=1e-3), "not close!"

    def _test_smooth_linear_impl(self, x_shape, lin_shape, device):
        # so we can use the full range
        torch.backends.quantized.engine = "qnnpack"

        x = torch.randn(*x_shape, device=device) * 9 + 10

        lin_fp32 = nn.Linear(*lin_shape, device=device)  # misc: ignore
        lin_smooth = SmoothFakeDynamicallyQuantizedLinear.from_float(
            copy.deepcopy(lin_fp32), alpha=0.25
        )
        lin_smooth_skip_scaling = SmoothFakeDynamicallyQuantizedLinear.from_float(
            copy.deepcopy(lin_fp32), alpha=0.25
        )

        lin_fp32_copy = copy.deepcopy(lin_fp32)  # assignment: ignore
        lin_fp32_copy.qconfig = torch.ao.quantization.QConfig(  # assignment: ignore
            activation=None,
            weight=torch.ao.quantization.default_per_channel_weight_observer,
        )
        lin_dynamic_q = torch.ao.nn.quantized.dynamic.Linear.from_float(
            lin_fp32_copy.cpu()
        )

        y_ref = lin_fp32(x)

        # calibrate the smoothquant versions
        y_smooth_nocalib = lin_smooth(x)
        _ = lin_smooth_skip_scaling(x)
        lin_smooth.to_inference()
        lin_smooth_skip_scaling.debug_skip_scaling = True
        lin_smooth_skip_scaling.to_inference()

        # verify that with scaling turned off, numerics match quantized version
        y_smooth_fq_only = lin_smooth_skip_scaling(x)
        y_smooth_fq = lin_smooth(x)
        y_dynamic_q = lin_dynamic_q(x.cpu()).to(device)

        # print('y_ref', y_ref)
        # print('y_smooth_nocalib', y_smooth_nocalib)
        # print('y_smooth_fq', y_smooth_fq)
        # print('y_smooth_fq_only', y_smooth_fq_only)
        # print('y_dynamic_q', y_dynamic_q)

        sqnr_smooth_fq = compute_error(y_ref, y_smooth_fq)
        sqnr_dynamic_q = compute_error(y_ref, y_dynamic_q)
        sqnr_fq = compute_error(y_smooth_fq_only, y_dynamic_q)
        # print('sqnr_smooth', sqnr_smooth_fq, 'sqnr_dynamic', sqnr_dynamic_q, 'sqnr_fq', sqnr_fq)

        assert torch.allclose(
            y_ref, y_smooth_nocalib
        ), "y_ref not close to y_smooth_nocalib"
        # after https://github.com/pytorch-labs/ao_benchmarks/pull/32,
        # numerics do not match exactly between production c++ code
        # and this Python code
        # assert torch.allclose(
        #     y_smooth_fq_only, y_dynamic_q,
        #     atol=torch.max(y_smooth_fq_only).item()*0.01,
        #     rtol=0.00001), \
        #     'y_smooth_fq_only not close to y_dynamic_q'

        self.assertTrue(sqnr_smooth_fq.item() >= 40.0)
        self.assertTrue(sqnr_dynamic_q.item() >= 40.0)
        self.assertTrue(sqnr_fq.item() >= 40.0)

    def test_smooth_linear_cpu(self):
        self._test_smooth_linear_impl((1, 5, 3), (3, 4), "cpu")

    def test_smooth_linear_cuda(self):
        if not torch.cuda.is_available():
            print("no cuda, skip")
            return
        self._test_smooth_linear_impl((1, 32, 32), (32, 16), "cuda")

    def test_smooth_linear_edge_cases(self):
        # so we can use the full range
        torch.backends.quantized.engine = "qnnpack"
        lin_fp32 = nn.Linear(3, 4)
        lin_smooth = SmoothFakeDynamicallyQuantizedLinear.from_float(
            lin_fp32, alpha=0.25
        )

        # test different ranks
        x0 = torch.randn(4, 5, 3)
        x1 = torch.randn(1, 8, 5, 3)
        x2 = torch.randn(2, 3, 7, 5, 3)

        # calibrate
        _ = lin_smooth(x0)
        _ = lin_smooth(x1)
        _ = lin_smooth(x2)

        # inference
        lin_smooth.to_inference()
        _ = lin_smooth(x0)
        _ = lin_smooth(x1)
        _ = lin_smooth(x2)

    def test_swap(self):
        m = nn.Sequential(
            nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4)),
            nn.Linear(4, 4),
        )
        m_copy = copy.deepcopy(m)
        swap_linear_with_smooth_fq_linear(m_copy, skip_fqn_list=["0.2"])

        # verify all linears are swapped
        assert isinstance(m_copy[0][0], SmoothFakeDynamicallyQuantizedLinear)
        assert isinstance(m_copy[0][1], nn.ReLU)
        # this one was skipped
        assert isinstance(m_copy[0][2], nn.Linear)
        assert isinstance(m_copy[1], SmoothFakeDynamicallyQuantizedLinear)

        # verify results do not change without smoothing
        x = torch.randn(4, 4)
        y_ref = m(x)
        y = m_copy(x)
        assert torch.allclose(y_ref, y)

    def test_weight_t_and_non_t_numerics_match(self):
        # verify that numerics match whether weight is stored
        # in transposed format (for cuBLAS) vs non-transposed format
        # (for torch.compile)
        if not torch.cuda.is_available():
            print("no cuda, skip")
            return
        dtype = torch.half
        device = "cuda"
        lin_ref = nn.Linear(32, 16, dtype=dtype, device=device)
        lin_eager_t = copy.deepcopy(lin_ref)
        lin_opt_t = copy.deepcopy(lin_eager_t)
        lin_opt = copy.deepcopy(lin_eager_t)
        lin_eager_t = SmoothFakeDynamicallyQuantizedLinear.from_float(lin_eager_t)
        lin_opt_t = SmoothFakeDynamicallyQuantizedLinear.from_float(lin_opt_t)
        lin_opt = SmoothFakeDynamicallyQuantizedLinear.from_float(lin_opt)
        lin_opt.store_w_int_repr_t = False

        x = torch.randn(32, 32, dtype=dtype, device=device)

        y_calib_eager_t = lin_eager_t(x)
        y_calib_opt_t = lin_opt_t(x)
        y_calib_opt = lin_opt(x)
        torch.testing.assert_close(y_calib_eager_t, y_calib_opt_t)
        torch.testing.assert_close(y_calib_eager_t, y_calib_opt)

        lin_eager_t.to_inference()
        lin_opt_t.to_inference()
        lin_opt.to_inference()

        torch.testing.assert_close(lin_eager_t.W_int_repr, lin_opt_t.W_int_repr)
        torch.testing.assert_close(lin_eager_t.W_int_repr, lin_opt.W_int_repr)

        lin_opt_t = torch.compile(lin_opt_t, mode="max-autotune")
        lin_opt = torch.compile(lin_opt, mode="max-autotune")

        y_ref = lin_ref(x)
        y_eager = lin_eager_t(x)
        y_opt_t = lin_opt_t(x)
        y_opt = lin_opt(x)

        if not torch.any(torch.isinf(y_ref)) and torch.any(torch.isinf(y_eager)):
            # eager mode torch._int_mm is sometimes buggy, when this happens
            # we can't really compare the compiled version against it properly
            print("eager mode torch._int_mm known bad, test is inconclusive")
            return

        sqnr_ref_eager = compute_error(y_ref, y_eager)
        sqnr_eager_opt_t = compute_error(y_eager, y_opt_t)
        sqnr_eager_opt = compute_error(y_eager, y_opt)
        # since torch.compile for a torch.half model can
        # change numerics significantly, we can only test for a high SQNR here
        # and not for closeness
        self.assertTrue(sqnr_eager_opt_t >= 45.0)
        self.assertTrue(sqnr_eager_opt >= 45.0)
        # y_opt_t and y_opt should be equivalent
        torch.testing.assert_close(y_opt_t, y_opt)

    def test_selective_torch_compile(self):
        m = nn.Sequential(
            nn.Linear(4, 4),
            nn.Sequential(
                nn.Linear(4, 4),
                nn.Linear(4, 4),
            ),
            nn.Linear(4, 4),
        )
        x = torch.randn(4, 4)
        y_ref = m(x)

        replace_with_custom_fn_if_matches_filter(
            m,
            lambda mod: torch.compile(mod),
            lambda mod, fqn: isinstance(mod, nn.Linear) and fqn != "1.0",
        )

        self.assertTrue(isinstance(m[0], torch._dynamo.eval_frame.OptimizedModule))
        self.assertTrue(isinstance(m[1][0], nn.Linear))
        self.assertTrue(isinstance(m[1][1], torch._dynamo.eval_frame.OptimizedModule))
        self.assertTrue(isinstance(m[2], torch._dynamo.eval_frame.OptimizedModule))

        y = m(x)
        torch.testing.assert_close(y, y_ref)

    def test_debug_x_absmax(self):
        m = nn.Sequential(nn.Linear(3, 4))
        x0 = torch.randn(4, 5, 3)
        y0 = m(x0)
        swap_linear_with_smooth_fq_linear(m)
        # no calibration, straight to inference, should not crash
        smooth_fq_linear_to_inference(m, debug_skip_calibration=True)
        y1 = m(x0)


class PythonQuantPrimitivesUnitTest(unittest.TestCase):
    def _test_dynamic_quant_per_tensor_numerics_impl(
        self, qmin, qmax, int_dtype, qint_dtype, float_dtype, device, qscheme
    ):
        x = torch.randn(256, dtype=float_dtype, device=device)
        y_vals, y_scale, y_zero_point = dynamically_quantize_per_tensor(
            x, qmin, qmax, int_dtype, qscheme
        )

        # reference
        # quantize_per_tensor_dynamic doesn't work for half, so we cast there and back
        x_for_ref = x.half().float() if float_dtype == torch.float16 else x

        # quantize_per_tensor_dynamic doesn't support qscheme, so we just do dynamic
        # quant manually with observers + static quant
        obs = MinMaxObserver(
            dtype=qint_dtype, qscheme=qscheme, quant_min=qmin, quant_max=qmax
        ).to(device)
        obs(x_for_ref)
        ref_scale, ref_zero_point = obs.calculate_qparams()
        y_ref = torch.quantize_per_tensor(
            x_for_ref, ref_scale, ref_zero_point, qint_dtype
        )

        # y_ref = torch.quantize_per_tensor_dynamic(x_for_ref, qint_dtype, False)
        # print(y_ref)
        if float_dtype == torch.float:
            assert torch.equal(y_vals, y_ref.int_repr())
        else:
            # numerics are not exactly aligned yet, off-by-one probably due
            # to rounding
            assert torch.max(torch.abs(y_vals - y_ref.int_repr())).item() <= 1
        torch.testing.assert_close(
            y_scale, torch.tensor([y_ref.q_scale()], device=device, dtype=float_dtype)
        )
        if y_zero_point is not None:
            assert torch.equal(
                y_zero_point, torch.tensor([y_ref.q_zero_point()], device=device)
            )
        else:
            self.assertTrue(y_ref.q_zero_point() == 0)

        # dequantize and check again
        x_dq = dequantize_per_tensor(y_vals, y_scale, y_zero_point, float_dtype)
        y_ref_dq = y_ref.dequantize().to(float_dtype)
        if float_dtype == torch.float:
            torch.testing.assert_close(x_dq, y_ref_dq)
        else:
            sqnr = compute_error(x_dq, y_ref_dq)
            self.assertTrue(sqnr.item() > 45.0)

    def test_dynamic_quant_per_tensor_numerics_cpu(self):
        # verifies that dynamic quant per tensor in plain pytorch matches
        # numerics of production AO code
        # TODO(future): test this on cpu-half, need to first make
        # torch.aminmax support half on cpu
        test_cases = (
            (
                0,
                255,
                torch.uint8,
                torch.quint8,
                torch.float32,
                "cpu",
                torch.per_tensor_affine,
            ),
            (
                -128,
                127,
                torch.int8,
                torch.qint8,
                torch.float32,
                "cpu",
                torch.per_tensor_affine,
            ),
            (
                -128,
                127,
                torch.int8,
                torch.qint8,
                torch.float32,
                "cpu",
                torch.per_tensor_symmetric,
            ),
            (
                -127,
                127,
                torch.int8,
                torch.qint8,
                torch.float32,
                "cpu",
                torch.per_tensor_symmetric,
            ),
        )
        for row in test_cases:
            self._test_dynamic_quant_per_tensor_numerics_impl(*row)

    def test_dynamic_quant_per_tensor_numerics_cuda(self):
        # verifies that dynamic quant per tensor in plain pytorch matches
        # numerics of production AO code
        if not torch.cuda.is_available():
            print("no cuda, skip")
            return
        test_cases = (
            (
                -128,
                127,
                torch.int8,
                torch.qint8,
                torch.float32,
                "cuda",
                torch.per_tensor_affine,
            ),
            (
                -128,
                127,
                torch.int8,
                torch.qint8,
                torch.float16,
                "cuda",
                torch.per_tensor_affine,
            ),
            (
                -128,
                127,
                torch.int8,
                torch.qint8,
                torch.float32,
                "cuda",
                torch.per_tensor_symmetric,
            ),
            (
                -128,
                127,
                torch.int8,
                torch.qint8,
                torch.float16,
                "cuda",
                torch.per_tensor_symmetric,
            ),
            (
                -127,
                127,
                torch.int8,
                torch.qint8,
                torch.float32,
                "cuda",
                torch.per_tensor_symmetric,
            ),
            (
                -127,
                127,
                torch.int8,
                torch.qint8,
                torch.float16,
                "cuda",
                torch.per_tensor_symmetric,
            ),
        )
        for row in test_cases:
            self._test_dynamic_quant_per_tensor_numerics_impl(*row)

    def _test_dynamic_quant_per_channel_numerics_impl(
        self, qmin, qmax, int_dtype, qint_dtype, float_dtype, device
    ):
        # verifies that dynamic quant per channel in plain pytorch matches
        # numerics of production AO code
        # TODO(future): test this on cpu-half, need to first make
        # torch.aminmax support half on cpu

        x = torch.randn(16, 32, device=device, dtype=float_dtype)
        y_vals, y_scale, y_zero_point = dynamically_quantize_per_channel(
            x, qmin, qmax, int_dtype
        )

        min_val, max_val = torch.aminmax(x, dim=1)

        # reference
        weight_obs = torch.ao.quantization.MovingAveragePerChannelMinMaxObserver(
            dtype=qint_dtype,
            quant_min=qmin,
            quant_max=qmax,
            qscheme=torch.per_channel_symmetric,
            averaging_constant=1.0,  # make it ignore previous iterations
        )
        weight_obs(x)
        y_ref_scale, y_ref_zp = weight_obs.calculate_qparams()
        y_ref_scale = y_ref_scale.to(device)
        y_ref_zp = y_ref_zp.to(device)
        # quantize_per_channel doesn't work for half, so we cast there and back
        x_for_ref = x.half().float() if float_dtype == torch.float16 else x
        y_ref = torch.quantize_per_channel(
            x_for_ref, y_ref_scale, y_ref_zp, 0, qint_dtype
        )

        torch.testing.assert_close(
            y_scale, y_ref.q_per_channel_scales().to(float_dtype)
        )
        assert torch.equal(y_zero_point, y_ref.q_per_channel_zero_points())
        # this test case has one element where the rounding is off by one
        # from Python-only code vs the c++ code, it's easy to repro with
        # various shapes.
        # Discussion here is relevant: https://github.com/pytorch/pytorch/issues/16498
        # TODO(future): figure out what to do about this
        # assert torch.equal(int_vals, q_reference.int_repr())
        assert torch.max(torch.abs(y_vals - y_ref.int_repr())) <= 1

        # dequantize
        x_dq = dequantize_per_channel(y_vals, y_scale, y_zero_point)
        x_ref_dq = y_ref.dequantize()
        # off-by-one for scale is okay
        torch.testing.assert_close(
            x_dq, x_ref_dq, atol=torch.max(y_scale).item() * 1.01, rtol=0.0001
        )

    def test_dynamic_quant_per_channel_numerics_cpu(self):
        test_cases = ((-128, 127, torch.int8, torch.qint8, torch.float32, "cpu"),)
        for row in test_cases:
            self._test_dynamic_quant_per_channel_numerics_impl(*row)

    def test_dynamic_quant_per_channel_numerics_cuda(self):
        if not torch.cuda.is_available():
            print("no cuda, skip")
            return
        test_cases = (
            (-128, 127, torch.int8, torch.qint8, torch.float32, "cuda"),
            (-128, 127, torch.int8, torch.qint8, torch.float16, "cuda"),
        )
        for row in test_cases:
            self._test_dynamic_quant_per_channel_numerics_impl(*row)

    def _test_quantize_per_token_impl(self, device, dtype):
        x = torch.randn(3, 3, 3, device=device, dtype=dtype)
        xq, scales = quantize_activation_per_token_absmax(x)
        x_dq = dequantize_per_tensor(xq, scales, None).to(x.dtype)
        sqnr = compute_error(x, x_dq)
        self.assertTrue(sqnr >= 45.0)

    def test_quantize_per_token_cpu(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            self._test_quantize_per_token_impl("cpu", dtype)

    def test_quantize_per_token_cuda(self):
        if not torch.cuda.is_available():
            print("no cuda, skip")
            return
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            self._test_quantize_per_token_impl("cuda", dtype)

    def _test_per_token_linear_impl(self, device, dtype):
        x = torch.randn(2, 16, 8, device=device, dtype=dtype)
        w = torch.randn(16, 8, device=device, dtype=dtype)
        wq, w_scales, _w_zp = dynamically_quantize_per_channel(w, -127, 127, torch.int8)
        # Note: need to make the weight contiguous because we are
        # testing in eager mode and cuBlas will not give correct results
        # for a transposed weight
        y = quant_int8_dynamic_per_token_linear(
            x, wq.t().contiguous(), w_scales, None, dtype
        )
        y_ref = torch.matmul(x, w.t())
        sqnr = compute_error(y_ref, y)
        self.assertTrue(sqnr >= 42.0)

    def test_per_token_linear_cpu(self):
        for dtype in (torch.float32,):
            self._test_per_token_linear_impl("cpu", dtype)

    def test_per_token_linear_cuda(self):
        if not torch.cuda.is_available():
            print("no cuda, skip")
            return
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            self._test_per_token_linear_impl("cuda", dtype)

    def test__int_mm(self):
        # TODO(future): figure out what here needs to move to PT core,
        # if it's not already tested there
        if not torch.cuda.is_available():
            print("no cuda, skip")
            return

        m, k, n = 32, 32, 16
        x = torch.randint(-128, 127, (m, k), dtype=torch.int8, device="cuda")
        w = torch.randint(-128, 127, (k, n), dtype=torch.int8, device="cuda")

        y_ref = torch.matmul(x.float(), w.float()).to(torch.int32)
        y_raw = safe_int_mm(x, w)

        wrap_in_mm_opt = torch.compile(safe_int_mm, mode="max-autotune")
        # note: triton chokes on the line below on k == 8 and n == 8 with
        # https://www.internalfb.com/phabricator/paste/view/P683467944
        # TODO(future): file an issue
        y_opt = wrap_in_mm_opt(x, w)

        torch.testing.assert_close(y_ref, y_raw, atol=0, rtol=0)
        torch.testing.assert_close(y_ref, y_opt, atol=0, rtol=0)

    def test__int_mm_eager_and_torch_compile_numerics(self):
        if not torch.cuda.is_available():
            print("no cuda, skip")
            return

        def __int_mm_ref(x, w):
            x = x.cpu().to(torch.int32)
            w = w.cpu().to(torch.int32)
            y = torch.matmul(x, w)
            return y.cuda()

        shapes = (
            # minimal test shape
            ((1, 32, 32), (32, 16)),
            # paste of real linear shapes from LLaMa 1.5b
            ((17, 1, 1536), (1536, 1536)),
            ((17, 8, 4096), (4096, 1536)),
            ((17, 1, 1536), (1536, 4096)),
            ((17, 8, 1536), (1536, 1536)),
            ((17, 1, 4096), (4096, 1536)),
            ((17, 8, 1536), (1536, 4096)),
        )

        for x_shape, w_shape in shapes:

            def wrap_torch_int_mm(x, w):
                b, n, k = x.shape
                k, m = w.shape
                x = x.reshape(b * n, k)
                res = safe_int_mm(x, w)
                res = res.reshape(b, n, m)
                return res

            wrap_torch_int_mm_opt = torch.compile(
                wrap_torch_int_mm, mode="max-autotune"
            )

            x = torch.randint(-128, 127, x_shape, dtype=torch.int8, device="cuda")
            w = torch.randint(-128, 127, w_shape, dtype=torch.int8, device="cuda")

            z_ref = __int_mm_ref(x, w)
            z_eager = wrap_torch_int_mm(x, w)
            z_torch_compile = wrap_torch_int_mm_opt(x, w)
            # print(z_ref)
            # print(z_eager)
            # print(z_torch_compile)

            torch.testing.assert_close(z_ref, z_eager, atol=0, rtol=0)
            torch.testing.assert_close(z_ref, z_torch_compile, atol=0, rtol=0)

    def _test_qlinear_per_channel_numerics(
        self, x_shape, lin_shape, qmin, qmax, int_dtype, qint_dtype, float_dtype, device
    ):
        qconfig = torch.ao.quantization.per_channel_dynamic_qconfig

        x = torch.randn(*x_shape, device=device, dtype=float_dtype)

        # TODO: test bias true and false
        # Note: reference path only works on float because lack of aten quant primitives
        # support of half, so we cast back and forth to emulate
        lin_ref = (
            nn.Sequential(nn.Linear(*lin_shape))
            .eval()
            .to(float_dtype)
            .float()
            .to(device)
        )
        y_ref = lin_ref(x.float())
        weight = lin_ref[0].weight
        bias = lin_ref[0].bias

        qconfig_mapping = QConfigMapping().set_global(qconfig)
        lin_ref_p = prepare_fx(lin_ref, qconfig_mapping, (torch.randn(1, 1),))
        lin_ref_q = convert_to_reference_fx(lin_ref_p)
        y_q_ref = lin_ref_q(x.float())

        # scale, zp of weight (get from reference model)
        w_obs = qconfig.weight()
        w_obs(weight)
        lin_ref_w_scale, lin_ref_w_zp = w_obs.calculate_qparams()
        lin_ref_w_scale = lin_ref_w_scale.to(device).to(float_dtype)
        # print('lin_ref_w', 'scale', lin_ref_w_scale, 'zp', lin_ref_w_zp)

        w_vals, _s, _z = dynamically_quantize_per_channel(
            getattr(lin_ref_q, "0").weight.to(float_dtype), -128, 127, torch.int8
        )
        w_vals = w_vals.t().contiguous()
        w_vals_sums = w_vals.sum(dim=0)

        # do our version of the quantized linear operator
        y = quant_int8_dynamic_linear(
            x,
            qmin,
            qmax,
            int_dtype,
            w_vals,
            lin_ref_w_scale,
            w_vals_sums,
            bias,
            float_dtype,
        )

        # print('y', y)
        # print('y_q_ref', y_q_ref)
        # print('y_ref', y_ref)

        sqnr_ref = compute_error(y_ref, y_q_ref)
        sqnr_our = compute_error(y_ref, y)
        # print('sqnr_ref', sqnr_ref, 'sqnr_our', sqnr_our)
        # for large shapes, sqnr can be in the high 30s for float32 and float16
        self.assertTrue(sqnr_our.item() >= 37.5)

    def test_qlinear_per_channel_numerics_cpu(self):
        # Note: the AO codebase doesn't easily support qint8 activations,
        # so the test cases below are for the quant primitives defined in
        # this file only. The AO reference is using quint8 here.
        test_cases = (
            ((2, 3), (3, 4), 0, 255, torch.uint8, torch.quint8, torch.float32, "cpu"),
            ((2, 3), (3, 4), -128, 127, torch.int8, torch.qint8, torch.float32, "cpu"),
        )
        for test_case in test_cases:
            self._test_qlinear_per_channel_numerics(*test_case)

    def test_qlinear_per_channel_numerics_cuda(self):
        if not torch.cuda.is_available():
            print("no cuda, skip")
            return
        test_cases = (
            # Note:  torch._int_mm needs int8 activations, so we don't test uint8
            # activations on CUDA at all
            (
                (32, 32),
                (32, 16),
                -128,
                127,
                torch.int8,
                torch.qint8,
                torch.float32,
                "cuda",
            ),
            (
                (32, 32),
                (32, 16),
                -128,
                127,
                torch.int8,
                torch.qint8,
                torch.float16,
                "cuda",
            ),
            # a large shape from LLaMa 1.5B - currently fails for float16
            (
                (17, 4096),
                (4096, 1536),
                -128,
                127,
                torch.int8,
                torch.qint8,
                torch.float32,
                "cuda",
            ),
            (
                (17, 4096),
                (4096, 1536),
                -128,
                127,
                torch.int8,
                torch.qint8,
                torch.float16,
                "cuda",
            ),
        )
        for test_case in test_cases:
            self._test_qlinear_per_channel_numerics(*test_case)


class TestSubclass(unittest.TestCase):
    def test_dq_lin_weight_subclass_aot(self):
        m, k, n = 32, 64, 32
        x = torch.randn(m, k, device="cuda", dtype=torch.float32)
        lin = torch.nn.Linear(k, n, device="cuda")

        import copy

        linq = DynamicallyPerAxisQuantizedLinear.from_float(copy.deepcopy(lin))

        ref_f = lin(x)
        ref_q = linq(x)

        print(SQNR(ref_f, ref_q), "float to dq")

        lin.weight = torch.nn.Parameter(
            DynamicallyQuantizedLinearWeight.from_float(lin.weight), requires_grad=False
        )
        test = lin(x)
        print(SQNR(ref_f, test), "float to dq class")
        print(SQNR(ref_q, test), "dq to dq class")
        assert SQNR(ref_f, test) > 35
        assert SQNR(ref_q, test) > 35

        lin_comp = torch.compile(lin, backend="aot_eager")
        linq_comp = torch.compile(linq, backend="aot_eager")
        test_comp = lin_comp(x)
        ref_q_comp = linq_comp(x)
        print(SQNR(ref_f, test_comp), "float to dq class compiled")
        print(SQNR(ref_q_comp, test_comp), "dq compiled to dq class compiled")
        assert SQNR(ref_f, test_comp) > 35
        assert SQNR(ref_q_comp, test_comp) > 35

    def test_dq_lin_weight_subclass_max_autotune(self):
        m, k, n = 32, 64, 32
        x = torch.randn(m, k, device="cuda", dtype=torch.float32)
        lin = torch.nn.Linear(k, n, device="cuda")

        import copy

        linq = DynamicallyPerAxisQuantizedLinear.from_float(copy.deepcopy(lin))

        ref_f = lin(x)
        ref_q = linq(x)

        print(SQNR(ref_f, ref_q), "float to dq")

        lin.weight = torch.nn.Parameter(
            DynamicallyQuantizedLinearWeight.from_float(lin.weight), requires_grad=False
        )
        test = lin(x)
        print(SQNR(ref_f, test), "float to dq class")
        print(SQNR(ref_q, test), "dq to dq class")
        assert SQNR(ref_f, test) > 35
        assert SQNR(ref_q, test) > 35

        lin_comp = torch.compile(lin, mode="max-autotune")
        linq_comp = torch.compile(linq, mode="max-autotune")

        test_comp = lin_comp(x)
        ref_q_comp = linq_comp(x)
        print(SQNR(ref_f, test_comp), "float to dq class compiled")
        print(SQNR(ref_q_comp, test_comp), "dq compiled to dq class compiled")
        assert SQNR(ref_f, test_comp) > 35
        assert SQNR(ref_q_comp, test_comp) > 35

    @torch.no_grad()
    def test_dq_lin_weight_subclass_max_autotune_api(self):
        m, k, n = 32, 64, 32
        x = torch.randn(m, k, device="cuda", dtype=torch.float32)

        mod = nn.Sequential(
            nn.Linear(k, n, device="cuda"), nn.ReLU(), nn.Linear(n, n, device="cuda")
        )
        change_linear_weights_to_dqtensors(mod)
        mod_qc = torch.compile(mod, mode="max-autotune")
        mod_qc(x)
        mod_qc(x)


class TestDynamicQuant(unittest.TestCase):
    def test_dynamic_quant(self):
        M, K, N = 8, 16, 8
        x = torch.randn(M, K)
        m = nn.Sequential(nn.Linear(K, N))

        y_ref = m(x)
        apply_dynamic_quant(m)
        y_test = m(x)

        sqnr = compute_error(y_ref, y_test)
        self.assertGreater(sqnr, 40.0)
        self.assertTrue(isinstance(m[0], DynamicallyPerAxisQuantizedLinear))


class TestWeightOnlyInt8Quant(unittest.TestCase):
    def test_weight_only_quant(self):
        for x_shape in [[2, 4], [5, 5, 5, 4], [1, 4, 4]]:
            x = torch.randn(*x_shape)
            m = nn.Sequential(nn.Linear(4, 5))
            y_ref = m(x)
            apply_weight_only_int8_quant(m)
            y_wo = m(x)
            sqnr = compute_error(y_ref, y_wo)
            self.assertGreater(sqnr, 44.0)

    @torch.no_grad()
    def test_weight_only_quant_force_mixed_mm(self):
        torch._inductor.config.epilogue_fusion = True
        torch._inductor.config.force_mixed_mm = True
        for x_dtype in [torch.float16, torch.bfloat16, torch.float32]:
            for x_shape in [[2, 4], [5, 5, 5, 4], [1, 4, 4]]:
                torch._dynamo.reset()
                x = torch.randn(*x_shape).to("cuda").to(x_dtype)
                m = nn.Sequential(nn.Linear(4, 5)).to("cuda").to(x_dtype)
                y_ref = m(x)
                apply_weight_only_int8_quant(m)
                m(x)
                m_c = torch.compile(m, mode="max-autotune")
                y_wo, (code,) = run_and_get_code(m_c, x)
                sqnr = compute_error(y_ref, y_wo)
                self.assertGreater(sqnr, 43.0)
                self.assertTrue("mixed_mm" in code)

    def test_weight_only_quant_use_mixed_mm(self):
        torch._inductor.config.epilogue_fusion = False
        torch._inductor.config.use_mixed_mm = True
        for x_dtype in [torch.float32, torch.float16, torch.bfloat16]:
            for x_shape in [[2, 4], [5, 5, 5, 4], [1, 4, 4]]:
                torch._dynamo.reset()
                x = torch.randn(*x_shape).to("cuda").to(x_dtype)
                m = nn.Sequential(nn.Linear(4, 5)).to("cuda").to(x_dtype)
                y_ref = m(x)
                apply_weight_only_int8_quant(m)
                m_c = torch.compile(m, mode="max-autotune")
                y_wo, (code,) = run_and_get_code(m_c, x)
                sqnr = compute_error(y_ref, y_wo)
                self.assertGreater(sqnr, 43.0)


class TorchCompileUnitTest(unittest.TestCase):
    def test_fullgraph(self):
        if not torch.cuda.is_available():
            print("no cuda, skip")
            return
        lin_fp16 = nn.Linear(32, 16, device="cuda", dtype=torch.float16)
        lin_smooth = SmoothFakeDynamicallyQuantizedLinear.from_float(
            lin_fp16, alpha=0.25
        )

        x0 = torch.randn(17, 1, 32, device="cuda", dtype=torch.float16)

        # calibrate
        _ = lin_smooth(x0)

        # inference
        lin_smooth.to_inference()

        # torch.compile
        lin_smooth_opt = torch.compile(lin_smooth, fullgraph=True)
        # print(lin_smooth_opt)

        y = lin_smooth_opt(x0)
        # print(y)


class UtilsUnitTest(unittest.TestCase):
    def test_shape_logger(self):
        x = torch.randn(4, 4)

        m = nn.Sequential(
            nn.Linear(4, 4),
            nn.Sequential(
                nn.Linear(4, 4),
            ),
        )

        apply_logging_hook(m)
        with LoggingTensorMode():
            m(x)
            m(x)

        for fqn, d1 in fqn_to_op_to_shape_to_count.items():  # noqa: PERF102
            for op, d2 in d1.items():  # noqa: PERF102
                for shape, count in d2.items():  # noqa: PERF102
                    # print(fqn, op, shape, count)
                    pass


class SmoothquantIntegrationTest(unittest.TestCase):
    @torch.inference_mode()
    def test_on_dummy_distilbert(self):
        # https://huggingface.co/distilbert-base-uncased#how-to-use
        from transformers import (  # type: ignore[import-untyped]
            DistilBertModel,
            DistilBertTokenizer,
        )

        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        # print(model)
        text = "Replace me by any text you'd like."
        encoded_input = tokenizer(text, return_tensors="pt")
        output_ref = model(**encoded_input)
        # print(output_ref)

        #
        # smooth_quant
        #
        model_copy = copy.deepcopy(model)
        swap_linear_with_smooth_fq_linear(model_copy, alpha=0.75)
        # calibrate
        output_1_1 = model_copy(**encoded_input)
        # inference
        smooth_fq_linear_to_inference(model_copy)
        output_1_2 = model_copy(**encoded_input)
        # print(output_1_1)
        # print(output_1_2)
        sqnr_sq = compute_error(
            output_ref.last_hidden_state, output_1_2.last_hidden_state
        )
        print("sqnr_sq", sqnr_sq)
        self.assertTrue(sqnr_sq >= 20.0)

        #
        # reference - dynamic linear quant
        #
        model_copy2 = copy.deepcopy(model)
        qconfig = torch.ao.quantization.QConfig(
            activation=None,
            weight=torch.ao.quantization.default_per_channel_weight_observer,
        )
        model_copy2 = torch.ao.quantization.quantize_dynamic(
            model_copy2,
            {torch.nn.Linear: qconfig},
        )
        output_2_2 = model_copy2(**encoded_input)
        # print(output_2_2)
        sqnr_pt_quant = compute_error(
            output_ref.last_hidden_state, output_2_2.last_hidden_state
        )
        print("sqnr_pt_quant", sqnr_pt_quant)
        self.assertTrue(sqnr_sq >= 8.0)


if __name__ == "__main__":
    unittest.main()
