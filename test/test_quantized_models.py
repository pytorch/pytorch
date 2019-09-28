import torch
import torch.jit
import unittest
from common_utils import run_tests
from common_quantization import QuantizationTestCase, ModelMultipleOps

@unittest.skipUnless('fbgemm' in torch.backends.quantized.supported_engines,
                     "Quantized operations require FBGEMM. FBGEMM is only optimized for CPUs"
                     " with instruction set support avx2 or newer.")
class ModelNumerics(QuantizationTestCase):
    def test_float_quant_compare_per_tensor(self):
        torch.manual_seed(42)
        my_model = ModelMultipleOps().to(torch.float32)
        my_model.eval()
        calib_data = torch.rand(1024, 3, 15, 15, dtype=torch.float32)
        eval_data = torch.rand(1, 3, 15, 15, dtype=torch.float32)
        out_ref = my_model(eval_data)
        q_model = torch.quantization.QuantWrapper(my_model)
        q_model.eval()
        q_model.qconfig = torch.quantization.default_qconfig
        torch.quantization.fuse_modules(q_model.module, [['conv1', 'bn1', 'relu1']])
        torch.quantization.prepare(q_model)
        q_model(calib_data)
        torch.quantization.convert(q_model)
        out_q = q_model(eval_data)
        SQNRdB = 20 * torch.log10(torch.norm(out_ref) / torch.norm(out_ref - out_q))
        # Quantized model output should be close to floating point model output numerically
        # Setting target SQNR to be 30 dB so that relative error is 1e-3 below the desired
        # output
        self.assertGreater(SQNRdB, 30, msg='Quantized model numerics diverge from float, expect SQNR > 30 dB')

    def test_float_quant_compare_per_channel(self):
        # Test for per-channel Quant
        torch.manual_seed(67)
        my_model = ModelMultipleOps().to(torch.float32)
        my_model.eval()
        calib_data = torch.rand(2048, 3, 15, 15, dtype=torch.float32)
        eval_data = torch.rand(10, 3, 15, 15, dtype=torch.float32)
        out_ref = my_model(eval_data)
        q_model = torch.quantization.QuantWrapper(my_model)
        q_model.eval()
        q_model.qconfig = torch.quantization.default_per_channel_qconfig
        torch.quantization.fuse_modules(q_model.module, [['conv1', 'bn1', 'relu1']])
        torch.quantization.prepare(q_model)
        q_model(calib_data)
        torch.quantization.convert(q_model)
        out_q = q_model(eval_data)
        SQNRdB = 20 * torch.log10(torch.norm(out_ref) / torch.norm(out_ref - out_q))
        # Quantized model output should be close to floating point model output numerically
        # Setting target SQNR to be 35 dB
        self.assertGreater(SQNRdB, 35, msg='Quantized model numerics diverge from float, expect SQNR > 35 dB')

if __name__ == "__main__":
    run_tests()
