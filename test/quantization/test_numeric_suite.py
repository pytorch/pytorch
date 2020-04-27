import torch
import torch.nn as nn
import torch.nn.quantized as nnq
from torch.quantization import (
    DeQuantStub,
    QuantStub,
    default_eval_fn,
    default_qconfig,
    quantize,
    prepare,
    convert,
)
from torch.quantization._numeric_suite import (
    RecordingLogger,
    Shadow,
    compare_model_stub,
    compare_weights,
)
from torch.testing._internal.common_quantization import (
    AnnotatedConvBnReLUModel,
    AnnotatedConvModel,
    QuantizationTestCase,
)
from torch.testing._internal.common_quantized import override_quantized_engine
import unittest
from hypothesis import given
from hypothesis import strategies as st

class SubModule(torch.nn.Module):
    def __init__(self):
        super(SubModule, self).__init__()
        self.mod1 = nn.Identity()
        self.mod2 = nn.ReLU()

    def forward(self, x):
        x = self.mod1(x)
        x = self.mod2(x)
        return x


class ModelWithSubModules(torch.nn.Module):
    def __init__(self):
        super(ModelWithSubModules, self).__init__()
        self.qconfig = default_qconfig
        self.mod1 = SubModule()
        self.conv = torch.nn.Conv2d(3, 5, 3, bias=False).to(dtype=torch.float)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.mod1(x)
        x = self.conv(x)
        x = self.dequant(x)
        return x


class ModelWithFunctionals(torch.nn.Module):
    def __init__(self):
        super(ModelWithFunctionals, self).__init__()
        self.mycat = nnq.FloatFunctional()
        self.myadd = nnq.FloatFunctional()
        self.mymul = nnq.FloatFunctional()
        self.myadd_relu = nnq.FloatFunctional()
        self.my_scalar_add = nnq.FloatFunctional()
        self.my_scalar_mul = nnq.FloatFunctional()
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.mycat.cat([x, x, x])
        x = self.myadd.add(x, x)
        x = self.mymul.mul(x, x)
        x = self.myadd_relu.add_relu(x, x)
        w = self.my_scalar_add.add_scalar(x, -0.5)
        w = self.my_scalar_mul.mul_scalar(w, 0.5)

        w = self.dequant(w)
        return w


class TestEagerModeNumericSuite(QuantizationTestCase):
    @unittest.skipUnless(
        'fbgemm' in torch.backends.quantized.supported_engines or
        'qnnpack' in torch.backends.quantized.supported_engines,
        " Quantized operations require FBGEMM."
    )
    @given(qengine=st.sampled_from(("qnnpack", "fbgemm")))
    def test_compare_weights(self, qengine):
        r"""Compare the weights of float and quantized conv layer
        """

        def compare_and_validate_results(float_model, q_model):
            weight_dict = compare_weights(
                float_model.state_dict(), q_model.state_dict()
            )
            self.assertEqual(len(weight_dict), 1)
            for k, v in weight_dict.items():
                self.assertTrue(v["float"].shape == v["quantized"].shape)

        if qengine in torch.backends.quantized.supported_engines:
            with override_quantized_engine(qengine):
                model_list = [AnnotatedConvModel(qengine), AnnotatedConvBnReLUModel(qengine)]
                for model in model_list:
                    model.eval()
                    if hasattr(model, "fuse_model"):
                        model.fuse_model()
                    q_model = quantize(model, default_eval_fn, self.img_data)
                    compare_and_validate_results(model, q_model)

    @unittest.skipUnless(
        'fbgemm' in torch.backends.quantized.supported_engines or
        'qnnpack' in torch.backends.quantized.supported_engines,
        " Quantized operations require FBGEMM."
    )
    @given(qengine=st.sampled_from(("qnnpack", "fbgemm")))
    def test_compare_model_stub(self, qengine):
        r"""Compare the output of quantized conv layer and its float shadow module
        """

        def compare_and_validate_results(float_model, q_model, module_swap_list, data):
            ob_dict = compare_model_stub(
                float_model, q_model, module_swap_list, data, RecordingLogger
            )
            self.assertEqual(len(ob_dict), 1)
            for k, v in ob_dict.items():
                self.assertTrue(v["float"].shape == v["quantized"].shape)

        if qengine in torch.backends.quantized.supported_engines:
            with override_quantized_engine(qengine):
                model_list = [AnnotatedConvModel(qengine), AnnotatedConvBnReLUModel(qengine)]
                data = self.img_data[0][0]
                module_swap_list = [nn.Conv2d, nn.intrinsic.modules.fused.ConvReLU2d]
                for model in model_list:
                    model.eval()
                    if hasattr(model, "fuse_model"):
                        model.fuse_model()
                    q_model = quantize(model, default_eval_fn, self.img_data)
                    compare_and_validate_results(model, q_model, module_swap_list, data)

                # Test adding stub to sub module
                model = ModelWithSubModules().eval()
                q_model = quantize(model, default_eval_fn, self.img_data)
                module_swap_list = [SubModule]
                ob_dict = compare_model_stub(
                    model, q_model, module_swap_list, data, RecordingLogger
                )
                self.assertTrue(isinstance(q_model.mod1, Shadow))
                self.assertFalse(isinstance(q_model.conv, Shadow))
                for k, v in ob_dict.items():
                    torch.testing.assert_allclose(v["float"], v["quantized"])

                # Test adding stub to functionals
                model = ModelWithFunctionals().eval()
                model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
                q_model = prepare(model, inplace=False)
                q_model(data)
                q_model = convert(q_model)
                module_swap_list = [nnq.FloatFunctional]
                ob_dict = compare_model_stub(
                    model, q_model, module_swap_list, data, RecordingLogger
                )
                self.assertEqual(len(ob_dict), 6)
                self.assertTrue(isinstance(q_model.mycat, Shadow))
                self.assertTrue(isinstance(q_model.myadd, Shadow))
                self.assertTrue(isinstance(q_model.mymul, Shadow))
                self.assertTrue(isinstance(q_model.myadd_relu, Shadow))
                self.assertTrue(isinstance(q_model.my_scalar_add, Shadow))
                self.assertTrue(isinstance(q_model.my_scalar_mul, Shadow))
                for k, v in ob_dict.items():
                    self.assertTrue(v["float"].shape == v["quantized"].shape)
