import torch.nn as nn
from torch.quantization import default_eval_fn, quantize
from torch.quantization._numeric_suite import (
    compare_model_outputs,
    compare_model_stub,
    compare_weights,
)
from torch.testing._internal.common_quantization import (
    AnnotatedConvBnReLUModel,
    AnnotatedConvModel,
    QuantizationTestCase,
)


class EagerModeNumericSuiteTest(QuantizationTestCase):
    def test_compare_weights(self):
        r"""Compare the weights of float and quantized conv layer
        """

        def compare_and_validate_results(float_model, q_model):
            weight_dict = compare_weights(
                float_model.state_dict(), q_model.state_dict()
            )
            self.assertEqual(len(weight_dict), 1)
            for k, v in weight_dict.items():
                self.assertTrue(v["float"].shape == v["quantized"].shape)

        # eager mode
        annotated_conv_model = AnnotatedConvModel().eval()
        quantized_annotated_conv_model = quantize(
            annotated_conv_model, default_eval_fn, self.img_data
        )
        compare_and_validate_results(
            annotated_conv_model, quantized_annotated_conv_model
        )

        annotated_conv_bn_relu_model = AnnotatedConvBnReLUModel().eval()
        annotated_conv_bn_relu_model.fuse_model()
        quantized_annotated_conv_bn_relu_model = quantize(
            annotated_conv_bn_relu_model, default_eval_fn, self.img_data
        )
        compare_and_validate_results(
            annotated_conv_bn_relu_model, quantized_annotated_conv_bn_relu_model
        )

    def test_compare_model_stub(self):
        r"""Compare the output of quantized conv layer and its float shadow module
        """

        def compare_and_validate_results(float_model, q_model, module_swap_list, data):
            ob_dict = compare_model_stub(float_model, q_model, module_swap_list, data)
            self.assertEqual(len(ob_dict), 1)
            for k, v in ob_dict.items():
                self.assertTrue(v["float"].shape == v["quantized"].shape)

        # eager mode
        annotated_conv_model = AnnotatedConvModel().eval()
        quantized_annotated_conv_model = quantize(
            annotated_conv_model, default_eval_fn, self.img_data
        )
        data = self.img_data[0][0]
        module_swap_list = [nn.Conv2d]
        compare_and_validate_results(
            annotated_conv_model, quantized_annotated_conv_model, module_swap_list, data
        )

        annotated_conv_bn_relu_model = AnnotatedConvBnReLUModel().eval()
        annotated_conv_bn_relu_model.fuse_model()
        quantized_annotated_conv_bn_relu_model = quantize(
            annotated_conv_bn_relu_model, default_eval_fn, self.img_data
        )
        module_swap_list = [nn.intrinsic.modules.fused.ConvReLU2d]
        compare_and_validate_results(
            annotated_conv_bn_relu_model,
            quantized_annotated_conv_bn_relu_model,
            module_swap_list,
            data,
        )

    def test_compare_model_outputs(self):
        r"""Compare the output of conv layer in quantized model and corresponding
        output of conv layer in float model
        """

        def compare_and_validate_results(float_model, q_model, data):
            act_compare_dict = compare_model_outputs(float_model, q_model, data)
            self.assertEqual(len(act_compare_dict), 2)
            for k, v in act_compare_dict.items():
                self.assertTrue(v["float"].shape == v["quantized"].shape)

        # eager mode
        annotated_conv_model = AnnotatedConvModel().eval()
        quantized_annotated_conv_model = quantize(
            annotated_conv_model, default_eval_fn, self.img_data
        )
        data = self.img_data[0][0]
        compare_and_validate_results(
            annotated_conv_model, quantized_annotated_conv_model, data
        )

        annotated_conv_bn_relu_model = AnnotatedConvBnReLUModel().eval()
        annotated_conv_bn_relu_model.fuse_model()
        quantized_annotated_conv_bn_relu_model = quantize(
            annotated_conv_bn_relu_model, default_eval_fn, self.img_data
        )
        compare_and_validate_results(
            annotated_conv_bn_relu_model, quantized_annotated_conv_bn_relu_model, data
        )
