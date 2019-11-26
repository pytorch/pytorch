from __future__ import print_function

import numpy as np
import unittest
import torch.onnx
import io

import onnx
import caffe2.python.onnx.backend as c2

class TestQuantizedOps(unittest.TestCase):
    def generic_test(self, model, sample_inputs, input_names=None):
        pt_inputs = tuple(torch.from_numpy(x) for x in sample_inputs)
        model.qconfig = torch.quantization.default_qconfig
        q_model = torch.quantization.prepare(model, inplace=False)
        q_model = torch.quantization.convert(q_model, inplace=False)

        pytorch_res = q_model(*pt_inputs)
        f = io.BytesIO()
        torch.onnx.export(q_model, pt_inputs, f, input_names=input_names, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
        f.seek(0)
        onnx_model = onnx.load(f)

        caffe_res = c2.run_model(onnx_model, dict(zip(input_names, sample_inputs)))[0]
        np.testing.assert_almost_equal(pytorch_res.numpy(), caffe_res, decimal=3)

    def generic_unary_test(self, op):
        class QModule(torch.nn.Module):
            def __init__(self, op):
                super(QModule, self).__init__()
                self.quant1 = torch.quantization.QuantStub()
                self.op = op
                self.dequant = torch.quantization.DeQuantStub()

            def forward(self, x):
                res = self.op(self.quant1(x))
                return self.dequant(res)

        x = np.random.random((1, 2)).astype("float32")
        self.generic_test(QModule(op), (x,), input_names=["x"])

    def test_quantized_add(self):
        class QAddModule(torch.nn.Module):
            def __init__(self):
                super(QAddModule, self).__init__()
                self.quant1 = torch.quantization.QuantStub()
                self.quant2 = torch.quantization.QuantStub()
                self.dequant = torch.quantization.DeQuantStub()

            def forward(self, x, y):
                res = torch.ops.quantized.add(self.quant1(x), self.quant2(y), 1.0, 0)
                return self.dequant(res)

        x = np.random.random(2).astype("float32")
        y = np.random.random(2).astype("float32")
        self.generic_test(QAddModule(), (x, y), input_names=["x", "y"])

    def test_quantized_relu(self):
        self.generic_unary_test(torch.nn.ReLU())

    def export_to_onnx(self, model, input, input_names):
        outputs = model(input)

        traced = torch.jit.trace(model, input)
        buf = io.BytesIO()
        torch.jit.save(traced, buf)
        buf.seek(0)

        model = torch.jit.load(buf)
        f = io.BytesIO()
        torch.onnx.export(model, input, f, input_names=input_names, example_outputs=outputs, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
        f.seek(0)

        onnx_model = onnx.load(f)
        return onnx_model

    def test_qlinear_model(self):
        class LinearModel(torch.nn.Module):
            def __init__(self):
                super(LinearModel, self).__init__()
                self.qconfig = torch.quantization.default_qconfig
                self.fc1 = torch.quantization.QuantWrapper(torch.nn.Linear(5, 10).to(dtype=torch.float))

            def forward(self, x):
                x = self.fc1(x)
                return x

        torch.backends.quantized.engine = "qnnpack"
        qconfig = torch.quantization.default_qconfig
        model = LinearModel()
        model.qconfig = qconfig
        model = torch.quantization.prepare(model)
        model = torch.quantization.convert(model)

        x_numpy = np.random.rand(1, 2, 5).astype(np.float32)
        x = torch.from_numpy(x_numpy).to(dtype=torch.float)
        outputs = model(x)
        input_names = ["x"]
        onnx_model = self.export_to_onnx(model, x, input_names)

        caffe_res = c2.run_model(onnx_model, dict(zip(input_names, x_numpy)))[0]
        np.testing.assert_almost_equal(np.squeeze(outputs.numpy()), caffe_res, decimal=3)

    def test_qconv_model(self):
        class ConvModel(torch.nn.Module):
            def __init__(self):
                super(ConvModel, self).__init__()
                self.qconfig = torch.quantization.default_qconfig
                self.fc1 = torch.quantization.QuantWrapper(torch.nn.Conv2d(3, 5, 2, bias=True).to(dtype=torch.float))

            def forward(self, x):
                x = self.fc1(x)
                return x
        torch.backends.quantized.engine = "qnnpack"
        qconfig = torch.quantization.default_qconfig
        model = ConvModel()
        model.qconfig = qconfig
        model = torch.quantization.prepare(model)
        model = torch.quantization.convert(model)

        x_numpy = np.random.rand(1, 3, 6, 6).astype(np.float32)
        x = torch.from_numpy(x_numpy).to(dtype=torch.float)
        outputs = model(x)
        input_names = ["x"]
        onnx_model = self.export_to_onnx(model, x, input_names)

        y = np.expand_dims(x_numpy, axis=0)
        caffe_res = c2.run_model(onnx_model, dict(zip(input_names, y)))[0]

        np.testing.assert_almost_equal(outputs, caffe_res, decimal=3)

    def test_upsample(self):
        class QUpsampleModule(torch.nn.Module):
            def __init__(self):
                super(QUpsampleModule, self).__init__()
                self.quant1 = torch.quantization.QuantStub()
                self.dequant = torch.quantization.DeQuantStub()

            def forward(self, x):
                res = torch.nn.quantized.functional.interpolate(self.quant1(x), size=[6, 8], mode='nearest')
                return self.dequant(res)

        x = np.random.rand(1, 2, 3, 4).astype("float32")
        self.generic_test(QUpsampleModule(), (x,), input_names=["x"])
    '''
    def test_quantized_ts(self):
        torch.backends.quantized.engine = "qnnpack"
        module_quant = torch.jit.load("/home/supriyar/pytorch/quantized_ts_V6.pt")

        input_img = torch.from_numpy(np.random.random((1, 3, 48, 64)).astype("float32"))
        input_fp = torch.from_numpy(np.random.random((1, 12, 48, 64)).astype("float32"))
        X = torch.from_numpy(np.random.random((1, 2)).astype("float32"))
        module_quant.eval()
        output = module_quant(input_img, input_fp, X)
        torch.set_default_tensor_type(torch.FloatTensor)
        input_names = ["img", "fp", "ang"]
        f = io.BytesIO()
        torch.onnx.export(module_quant, (input_img, input_fp, X), f, verbose=False, input_names=input_names, example_outputs=output, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK, opset_version=9)
        f.seek(0)
        onnx_model = onnx.load(f)
        sample_inputs = (
            input_img.numpy(),
            input_fp.numpy(),
            X.numpy(),
        )

        caffe_res = c2.run_model(onnx_model, dict(zip(input_names, sample_inputs)))[0]
        np.testing.assert_almost_equal(output[0].numpy(), np.squeeze(caffe_res), decimal=3)

    def test_quantized_fbn(self):
        torch.backends.quantized.engine = "qnnpack"
        module_quant = torch.jit.load("/home/supriyar/pytorch/fbn.pt")

        x_numpy = np.random.random((2, 3, 8, 8)).astype("float32")
        X = torch.from_numpy(x_numpy)
        module_quant.eval()
        output = module_quant(X)
        torch.set_default_tensor_type(torch.FloatTensor)
        input_names = ["x"]
        f = io.BytesIO()
        torch.onnx.export(module_quant, (X), f, verbose=False, example_outputs=output, input_names=input_names, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK, opset_version=9)
        f.seek(0)
        onnx_model = onnx.load(f)
        sample_inputs = (
            x_numpy,
        )
        caffe_res = c2.run_model(onnx_model, dict(zip(input_names, sample_inputs)))[0]
        np.testing.assert_almost_equal(output.numpy(), caffe_res, decimal=3)

    '''

if __name__ == '__main__':
    unittest.main()
