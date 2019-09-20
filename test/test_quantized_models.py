import torch
import torch.jit
from common_utils import run_tests
from common_quantization import QuantizationTestCase, ModelMultipleOps

class ModelNumerics(QuantizationTestCase):
    def test_float_quant_compare_per_tensor(self):
        torch.manual_seed(42)
        myModel = ModelMultipleOps().to(torch.float32)
        myModel.eval()
        calib_data = torch.rand(1024, 3, 15, 15, dtype=torch.float32)
        eval_data = torch.rand(1, 3, 15, 15, dtype=torch.float32)
        out_ref = myModel(eval_data)
        qModel = torch.quantization.QuantWrapper(myModel)
        qModel.eval()
        qModel.qconfig = torch.quantization.default_qconfig
        torch.quantization.fuse_modules(qModel.module, [['conv1', 'bn1', 'relu1']])
        torch.quantization.prepare(qModel)
        qModel(calib_data)
        torch.quantization.convert(qModel)
        out_q = qModel(eval_data)
        SQNRdB = 20 * torch.log10(torch.norm(out_ref) / torch.norm(out_ref - out_q))
        # Quantized model output should be close to floating point model output numerically
        # Setting target SQNR to be 30 dB so that relative error is 1e-3 below the desired
        # output
        self.assertGreater(SQNRdB, 30, msg='Quantized model numerics diverge from float, expect SQNR > 30 dB')

    def test_float_quant_compare_per_channel(self):
        # Test for per-channel Quant
        torch.manual_seed(67)
        myModel = ModelMultipleOps().to(torch.float32)
        myModel.eval()
        calib_data = torch.rand(2048, 3, 15, 15, dtype=torch.float32)
        eval_data = torch.rand(10, 3, 15, 15, dtype=torch.float32)
        out_ref = myModel(eval_data)
        qModel = torch.quantization.QuantWrapper(myModel)
        qModel.eval()
        qModel.qconfig = torch.quantization.default_per_channel_qconfig
        torch.quantization.fuse_modules(qModel.module, [['conv1', 'bn1', 'relu1']])
        torch.quantization.prepare(qModel)
        qModel(calib_data)
        torch.quantization.convert(qModel)
        out_q = qModel(eval_data)
        SQNRdB = 20 * torch.log10(torch.norm(out_ref) / torch.norm(out_ref - out_q))
        # Quantized model output should be close to floating point model output numerically
        # Setting target SQNR to be 35 dB
        self.assertGreater(SQNRdB, 35, msg='Quantized model numerics diverge from float, expect SQNR > 35 dB')

if __name__ == "__main__":
    run_tests()
