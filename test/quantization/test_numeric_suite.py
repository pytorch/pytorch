# torch
import torch

# torch quantization
from torch.quantization import default_eval_fn, quantize
from torch.quantization._numeric_suite import compare_weights

# Testing Utils
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.common_quantization import (
    AnnotatedConvModel,
    QuantizationTestCase,
)

# Standard library
import unittest

class TestEagerModeNumericSuite(QuantizationTestCase):
    @unittest.skipUnless(
        'fbgemm' in torch.backends.quantized.supported_engines,
        " Quantized operations require FBGEMM."
    )
    def test_compare_weights(self):
        r"""Compare the weights of float and quantized conv layer
        """
        # eager mode
        annotated_conv_model = AnnotatedConvModel().eval()
        quantized_annotated_conv_model = quantize(
            annotated_conv_model, default_eval_fn, self.img_data
        )
        weight_dict = compare_weights(
            annotated_conv_model.state_dict(),
            quantized_annotated_conv_model.state_dict(),
        )
        self.assertEqual(len(weight_dict), 1)
        for k, v in weight_dict.items():
            self.assertTrue(v["float"].shape == v["quantized"].shape)

if __name__ == '__main__':
    run_tests()
