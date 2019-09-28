import torch
import torch.jit
from common_utils import run_tests, TEST_WITH_UBSAN, IS_WINDOWS, IS_PPC
from common_quantization import QuantizationTestCase, ModelMultipleOps
from common_quantized import enable_mobile_quantized_engine

def test_float_quant_compare_op(self):
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
    torch.quantization.prepare(qModel, inplace=True)
    qModel(calib_data)
    torch.quantization.convert(qModel, inplace=True)
    out_q = qModel(eval_data)
    SQNRdB = 20 * torch.log10(torch.norm(out_ref) / torch.norm(out_ref - out_q))
    # Quantized model output should be close to floating point model output numerically
    # Setting target SQNR to be 30 dB so that relative error is 1e-3 below the desired
    # output
    self.assertGreater(SQNRdB, 30, msg='Quantized model numerics diverge from float, expect SQNR > 30 dB')

class ModelNumerics(QuantizationTestCase):
    def test_float_quant_compare(self):
        if 'qnnpack' in torch.backends.quantized.supported_engines:
            if not IS_WINDOWS and not IS_PPC and not TEST_WITH_UBSAN:
                with enable_mobile_quantized_engine():
                    test_float_quant_compare_op(self)
        if 'fbgemm' in torch.backends.quantized.supported_engines:
            test_float_quant_compare_op(self)


if __name__ == "__main__":
    run_tests()
