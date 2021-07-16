import torch
import torch.nn as nn
import torch.nn.quantized as nnq
from torch.quantization import (
    DeQuantStub,
    QuantStub,
    convert,
    default_qconfig,
    prepare,
    quantize,
    quantize_dynamic,
)
from torch.quantization._numeric_suite import (
    OutputLogger,
    Shadow,
    ShadowLogger,
    compare_model_outputs,
    compare_model_stub,
    compare_weights,
)
from torch.testing._internal.common_quantization import (
    AnnotatedConvBnReLUModel,
    AnnotatedConvModel,
    AnnotatedSingleLayerLinearModel,
    LSTMwithHiddenDynamicModel,
    AnnotatedTwoLayerLinearModel,
    QuantizationTestCase,
    SingleLayerLinearDynamicModel,
    test_only_eval_fn,
)
from torch.testing._internal.common_quantized import override_qengines


class SubModule(torch.nn.Module):
    def __init__(self):
        super(SubModule, self).__init__()
        self.qconfig = default_qconfig
        self.mod1 = torch.nn.Conv2d(3, 3, 3, bias=False).to(dtype=torch.float)
        self.mod2 = nn.ReLU()
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.mod1(x)
        x = self.mod2(x)
        x = self.dequant(x)
        return x


class ModelWithSubModules(torch.nn.Module):
    def __init__(self):
        super(ModelWithSubModules, self).__init__()
        self.mod1 = SubModule()
        self.conv = torch.nn.Conv2d(3, 5, 3, bias=False).to(dtype=torch.float)

    def forward(self, x):
        x = self.mod1(x)
        x = self.conv(x)
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
    @override_qengines
    def test_compare_weights_conv_static(self):
        r"""Compare the weights of float and static quantized conv layer"""

        qengine = torch.backends.quantized.engine

        def compare_and_validate_results(float_model, q_model):
            weight_dict = compare_weights(
                float_model.state_dict(), q_model.state_dict()
            )
            self.assertEqual(len(weight_dict), 1)
            for k, v in weight_dict.items():
                self.assertTrue(v["float"].shape == v["quantized"].shape)

        model_list = [AnnotatedConvModel(qengine), AnnotatedConvBnReLUModel(qengine)]
        for model in model_list:
            model.eval()
            if hasattr(model, "fuse_model"):
                model.fuse_model()
            q_model = quantize(model, test_only_eval_fn, [self.img_data_2d])
            compare_and_validate_results(model, q_model)

    @override_qengines
    def test_compare_weights_linear_static(self):
        r"""Compare the weights of float and static quantized linear layer"""

        qengine = torch.backends.quantized.engine

        def compare_and_validate_results(float_model, q_model):
            weight_dict = compare_weights(
                float_model.state_dict(), q_model.state_dict()
            )
            self.assertEqual(len(weight_dict), 1)
            for k, v in weight_dict.items():
                self.assertTrue(v["float"].shape == v["quantized"].shape)

        model_list = [AnnotatedSingleLayerLinearModel(qengine)]
        for model in model_list:
            model.eval()
            if hasattr(model, "fuse_model"):
                model.fuse_model()
            q_model = quantize(model, test_only_eval_fn, [self.calib_data])
            compare_and_validate_results(model, q_model)

    @override_qengines
    def test_compare_weights_linear_dynamic(self):
        r"""Compare the weights of float and dynamic quantized linear layer"""

        qengine = torch.backends.quantized.engine

        def compare_and_validate_results(float_model, q_model):
            weight_dict = compare_weights(
                float_model.state_dict(), q_model.state_dict()
            )
            self.assertEqual(len(weight_dict), 1)
            for k, v in weight_dict.items():
                self.assertTrue(len(v["float"]) == len(v["quantized"]))
                for i, val in enumerate(v["quantized"]):
                    self.assertTrue(v["float"][i].shape == v["quantized"][i].shape)

        model_list = [SingleLayerLinearDynamicModel(qengine)]
        for model in model_list:
            model.eval()
            if hasattr(model, "fuse_model"):
                model.fuse_model()
            q_model = quantize_dynamic(model)
            compare_and_validate_results(model, q_model)

    @override_qengines
    def test_compare_weights_lstm_dynamic(self):
        r"""Compare the weights of float and dynamic quantized LSTM layer"""

        qengine = torch.backends.quantized.engine

        def compare_and_validate_results(float_model, q_model):
            weight_dict = compare_weights(
                float_model.state_dict(), q_model.state_dict()
            )
            self.assertEqual(len(weight_dict), 1)
            for k, v in weight_dict.items():
                self.assertTrue(len(v["float"]) == len(v["quantized"]))
                for i, val in enumerate(v["quantized"]):
                    self.assertTrue(v["float"][i].shape == v["quantized"][i].shape)

        model_list = [LSTMwithHiddenDynamicModel(qengine)]
        for model in model_list:
            model.eval()
            if hasattr(model, "fuse_model"):
                model.fuse_model()
            q_model = quantize_dynamic(model)
            compare_and_validate_results(model, q_model)

    @override_qengines
    def test_compare_model_stub_conv_static(self):
        r"""Compare the output of static quantized conv layer and its float shadow module"""

        qengine = torch.backends.quantized.engine

        def compare_and_validate_results(float_model, q_model, module_swap_list, data):
            ob_dict = compare_model_stub(float_model, q_model, module_swap_list, data)
            self.assertEqual(len(ob_dict), 1)
            for k, v in ob_dict.items():
                self.assertTrue(len(v["float"]) == len(v["quantized"]))
                for i, val in enumerate(v["quantized"]):
                    self.assertTrue(v["float"][i].shape == v["quantized"][i].shape)

        model_list = [AnnotatedConvModel(qengine), AnnotatedConvBnReLUModel(qengine)]
        module_swap_list = [nn.Conv2d, nn.intrinsic.modules.fused.ConvReLU2d]
        for model in model_list:
            model.eval()
            if hasattr(model, "fuse_model"):
                model.fuse_model()
            q_model = quantize(model, test_only_eval_fn, [self.img_data_2d])
            compare_and_validate_results(
                model, q_model, module_swap_list, self.img_data_2d[0][0]
            )

    @override_qengines
    def test_compare_model_stub_linear_static(self):
        r"""Compare the output of static quantized linear layer and its float shadow module"""

        qengine = torch.backends.quantized.engine

        def compare_and_validate_results(float_model, q_model, module_swap_list, data):
            ob_dict = compare_model_stub(float_model, q_model, module_swap_list, data)
            self.assertEqual(len(ob_dict), 1)
            for k, v in ob_dict.items():
                self.assertTrue(len(v["float"]) == len(v["quantized"]))
                for i, val in enumerate(v["quantized"]):
                    self.assertTrue(v["float"][i].shape == v["quantized"][i].shape)

        linear_data = self.calib_data[0][0]
        module_swap_list = [nn.Linear]
        model_list = [AnnotatedSingleLayerLinearModel(qengine)]
        for model in model_list:
            model.eval()
            if hasattr(model, "fuse_model"):
                model.fuse_model()
            q_model = quantize(model, test_only_eval_fn, [self.calib_data])
            compare_and_validate_results(model, q_model, module_swap_list, linear_data)

    @override_qengines
    def test_compare_model_stub_partial(self):
        r"""Compare the output of static quantized linear layer and its float shadow module"""

        qengine = torch.backends.quantized.engine
        # TODO: Rebase on top of PR to remove compare and validate results here

        def compare_and_validate_results(float_model, q_model, module_swap_list, data):
            ob_dict = compare_model_stub(float_model, q_model, module_swap_list, data)
            self.assertEqual(len(ob_dict), 1)
            for k, v in ob_dict.items():
                self.assertTrue(len(v["float"]) == len(v["quantized"]))
                for i, val in enumerate(v["quantized"]):
                    self.assertTrue(v["float"][i].shape == v["quantized"][i].shape)

        linear_data = self.calib_data[0][0]
        module_swap_list = [nn.Linear]
        model_list = [AnnotatedTwoLayerLinearModel()]
        for model in model_list:
            model.eval()
            if hasattr(model, "fuse_model"):
                model.fuse_model()
            q_model = quantize(model, test_only_eval_fn, [self.calib_data])
            compare_and_validate_results(model, q_model, module_swap_list, linear_data)

    @override_qengines
    def test_compare_model_stub_submodule_static(self):
        r"""Compare the output of static quantized submodule and its float shadow module"""

        qengine = torch.backends.quantized.engine

        model = ModelWithSubModules().eval()
        q_model = quantize(model, test_only_eval_fn, [self.img_data_2d])
        module_swap_list = [SubModule, nn.Conv2d]
        ob_dict = compare_model_stub(
            model, q_model, module_swap_list, self.img_data_2d[0][0]
        )
        # Since conv is not quantized, we do not insert a shadow module
        # mod1 contains a linear that is quantized, so we insert a shadow module
        self.assertTrue(isinstance(q_model.mod1, Shadow))
        self.assertFalse(isinstance(q_model.conv, Shadow))


    @override_qengines
    def test_compare_model_stub_functional_static(self):
        r"""Compare the output of static quantized functional layer and its float shadow module"""

        qengine = torch.backends.quantized.engine

        model = ModelWithFunctionals().eval()
        model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
        q_model = prepare(model, inplace=False)
        q_model(self.img_data_2d[0][0])
        q_model = convert(q_model)
        module_swap_list = [nnq.FloatFunctional]
        ob_dict = compare_model_stub(
            model, q_model, module_swap_list, self.img_data_2d[0][0]
        )
        self.assertEqual(len(ob_dict), 6)
        self.assertTrue(isinstance(q_model.mycat, Shadow))
        self.assertTrue(isinstance(q_model.myadd, Shadow))
        self.assertTrue(isinstance(q_model.mymul, Shadow))
        self.assertTrue(isinstance(q_model.myadd_relu, Shadow))
        self.assertTrue(isinstance(q_model.my_scalar_add, Shadow))
        self.assertTrue(isinstance(q_model.my_scalar_mul, Shadow))
        for k, v in ob_dict.items():
            self.assertTrue(len(v["float"]) == len(v["quantized"]))
            for i, val in enumerate(v["quantized"]):
                self.assertTrue(v["float"][i].shape == v["quantized"][i].shape)

    @override_qengines
    def test_compare_model_stub_linear_dynamic(self):
        r"""Compare the output of dynamic quantized linear layer and its float shadow module"""

        qengine = torch.backends.quantized.engine

        def compare_and_validate_results(float_model, q_model, module_swap_list, data):
            ob_dict = compare_model_stub(float_model, q_model, module_swap_list, data)
            self.assertEqual(len(ob_dict), 1)
            for k, v in ob_dict.items():
                self.assertTrue(len(v["float"]) == len(v["quantized"]))
                for i, val in enumerate(v["quantized"]):
                    self.assertTrue(v["float"][i].shape == v["quantized"][i].shape)

        linear_data = self.calib_data[0][0]

        model_list = [SingleLayerLinearDynamicModel(qengine)]
        module_swap_list = [nn.Linear, nn.LSTM]
        for model in model_list:
            model.eval()
            if hasattr(model, "fuse_model"):
                model.fuse_model()
            q_model = quantize_dynamic(model)
            compare_and_validate_results(model, q_model, module_swap_list, linear_data)

    @override_qengines
    def test_compare_model_stub_lstm_dynamic(self):
        r"""Compare the output of dynamic quantized LSTM layer and its float shadow module"""

        qengine = torch.backends.quantized.engine

        def compare_and_validate_results(
            float_model, q_model, module_swap_list, input, hidden
        ):
            ob_dict = compare_model_stub(
                float_model, q_model, module_swap_list, input, hidden
            )
            self.assertEqual(len(ob_dict), 1)
            for k, v in ob_dict.items():
                self.assertTrue(len(v["float"]) == len(v["quantized"]))
                for i, val in enumerate(v["quantized"]):
                    self.assertTrue(v["float"][i].shape == v["quantized"][i].shape)

        lstm_input = torch.rand((1, 1, 2))
        lstm_hidden = (torch.rand(1, 1, 2), torch.rand(1, 1, 2))

        model_list = [LSTMwithHiddenDynamicModel(qengine)]
        module_swap_list = [nn.Linear, nn.LSTM]
        for model in model_list:
            model.eval()
            if hasattr(model, "fuse_model"):
                model.fuse_model()
            q_model = quantize_dynamic(model)
            compare_and_validate_results(
                model, q_model, module_swap_list, lstm_input, lstm_hidden
            )

    @override_qengines
    def test_compare_model_outputs_conv_static(self):
        r"""Compare the output of conv layer in stataic quantized model and corresponding
        output of conv layer in float model
        """
        qengine = torch.backends.quantized.engine

        def compare_and_validate_results(float_model, q_model, data):
            act_compare_dict = compare_model_outputs(float_model, q_model, data)
            expected_act_compare_dict_keys = {"conv.stats", "quant.stats"}

            self.assertTrue(act_compare_dict.keys() == expected_act_compare_dict_keys)
            for k, v in act_compare_dict.items():
                self.assertTrue(v["float"][0].shape == v["quantized"][0].shape)

        model_list = [AnnotatedConvModel(qengine), AnnotatedConvBnReLUModel(qengine)]
        for model in model_list:
            model.eval()
            if hasattr(model, "fuse_model"):
                model.fuse_model()
            q_model = quantize(model, test_only_eval_fn, [self.img_data_2d])
            compare_and_validate_results(model, q_model, self.img_data_2d[0][0])

    @override_qengines
    def test_compare_model_outputs_linear_static(self):
        r"""Compare the output of linear layer in static quantized model and corresponding
        output of conv layer in float model
        """
        qengine = torch.backends.quantized.engine

        def compare_and_validate_results(float_model, q_model, data):
            act_compare_dict = compare_model_outputs(float_model, q_model, data)
            expected_act_compare_dict_keys = {"fc1.quant.stats", "fc1.module.stats"}

            self.assertTrue(act_compare_dict.keys() == expected_act_compare_dict_keys)
            for k, v in act_compare_dict.items():
                self.assertTrue(len(v["float"]) == len(v["quantized"]))
                for i, val in enumerate(v["quantized"]):
                    self.assertTrue(v["float"][i].shape == v["quantized"][i].shape)

        linear_data = self.calib_data[0][0]
        model_list = [AnnotatedSingleLayerLinearModel(qengine)]
        for model in model_list:
            model.eval()
            if hasattr(model, "fuse_model"):
                model.fuse_model()
            q_model = quantize(model, test_only_eval_fn, [self.calib_data])
            compare_and_validate_results(model, q_model, linear_data)

    @override_qengines
    def test_compare_model_outputs_functional_static(self):
        r"""Compare the output of functional layer in static quantized model and corresponding
        output of conv layer in float model
        """
        qengine = torch.backends.quantized.engine

        model = ModelWithFunctionals().eval()
        model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
        q_model = prepare(model, inplace=False)
        q_model(self.img_data_2d[0][0])
        q_model = convert(q_model)
        act_compare_dict = compare_model_outputs(model, q_model, self.img_data_2d[0][0])
        self.assertEqual(len(act_compare_dict), 7)
        expected_act_compare_dict_keys = {
            "mycat.stats",
            "myadd.stats",
            "mymul.stats",
            "myadd_relu.stats",
            "my_scalar_add.stats",
            "my_scalar_mul.stats",
            "quant.stats",
        }
        self.assertTrue(act_compare_dict.keys() == expected_act_compare_dict_keys)
        for k, v in act_compare_dict.items():
            self.assertTrue(len(v["float"]) == len(v["quantized"]))
            for i, val in enumerate(v["quantized"]):
                self.assertTrue(v["float"][i].shape == v["quantized"][i].shape)

    @override_qengines
    def test_compare_model_outputs_linear_dynamic(self):
        r"""Compare the output of linear layer in dynamic quantized model and corresponding
        output of conv layer in float model
        """
        qengine = torch.backends.quantized.engine

        def compare_and_validate_results(float_model, q_model, data):
            act_compare_dict = compare_model_outputs(float_model, q_model, data)
            expected_act_compare_dict_keys = {"fc1.stats"}

            self.assertTrue(act_compare_dict.keys() == expected_act_compare_dict_keys)
            for k, v in act_compare_dict.items():
                self.assertTrue(len(v["float"]) == len(v["quantized"]))
                for i, val in enumerate(v["quantized"]):
                    self.assertTrue(v["float"][i].shape == v["quantized"][i].shape)

        linear_data = self.calib_data[0][0]

        model_list = [SingleLayerLinearDynamicModel(qengine)]
        for model in model_list:
            model.eval()
            if hasattr(model, "fuse_model"):
                model.fuse_model()
            q_model = quantize_dynamic(model)
            compare_and_validate_results(model, q_model, linear_data)

    @override_qengines
    def test_compare_model_outputs_lstm_dynamic(self):
        r"""Compare the output of LSTM layer in dynamic quantized model and corresponding
        output of conv layer in float model
        """
        qengine = torch.backends.quantized.engine

        def compare_and_validate_results(float_model, q_model, input, hidden):
            act_compare_dict = compare_model_outputs(
                float_model, q_model, input, hidden
            )
            expected_act_compare_dict_keys = {"lstm.stats"}

            self.assertTrue(act_compare_dict.keys() == expected_act_compare_dict_keys)
            for k, v in act_compare_dict.items():
                self.assertTrue(len(v["float"]) == len(v["quantized"]))
                for i, val in enumerate(v["quantized"]):
                    self.assertTrue(len(v["float"][i]) == len(v["quantized"][i]))
                    if i == 0:
                        self.assertTrue(v["float"][i][0].shape == v["quantized"][i][0].shape)
                    else:
                        self.assertTrue(
                            v["float"][i][0].shape == v["quantized"][i][0].shape
                        )
                        self.assertTrue(
                            v["float"][i][1].shape == v["quantized"][i][1].shape
                        )

        lstm_input = torch.rand((1, 1, 2))
        lstm_hidden = (torch.rand(1, 1, 2), torch.rand(1, 1, 2))

        model_list = [LSTMwithHiddenDynamicModel(qengine)]
        for model in model_list:
            model.eval()
            if hasattr(model, "fuse_model"):
                model.fuse_model()
            q_model = quantize_dynamic(model)
            compare_and_validate_results(model, q_model, lstm_input, lstm_hidden)

    @override_qengines
    def test_output_logger(self):
        r"""Compare output from OutputLogger with the expected results"""
        x = torch.rand(2, 2)
        y = torch.rand(2, 1)

        l = []
        l.append(x)
        l.append(y)

        logger = OutputLogger()
        logger.forward(x)
        logger.forward(y)

        self.assertEqual(l, logger.stats["tensor_val"])

    @override_qengines
    def test_shadow_logger(self):
        r"""Compare output from ShawdowLogger with the expected results"""
        a_float = torch.rand(2, 2)
        a_quantized = torch.rand(2, 2)

        b_float = torch.rand(3, 2, 2)
        b_quantized = torch.rand(3, 2, 2)

        logger = ShadowLogger()
        logger.forward(a_float, a_quantized)
        logger.forward(b_float, b_quantized)

        self.assertEqual(len(logger.stats["float"]), 2)
        self.assertEqual(len(logger.stats["quantized"]), 2)
