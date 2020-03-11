from __future__ import print_function

import numpy as np
import unittest
import torch.onnx
import torch.nn as nn
import io

import onnx
import caffe2.python.onnx.backend as c2

class TestQuantizedOps(unittest.TestCase):
    def generic_test(self, model, sample_inputs, input_names=None):
        torch.backends.quantized.engine = "qnnpack"
        pt_inputs = tuple(torch.from_numpy(x) for x in sample_inputs)
        model.qconfig = torch.quantization.default_qconfig
        q_model = torch.quantization.prepare(model, inplace=False)
        q_model = torch.quantization.convert(q_model, inplace=False)

        pytorch_res = q_model(*pt_inputs)
        f = io.BytesIO()
        torch.onnx.export(q_model, pt_inputs, f, input_names=input_names,
                          operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
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

    def generic_model_test(self, model, sample_inputs, input_names=None):
        torch.backends.quantized.engine = "qnnpack"
        pt_inputs = tuple(torch.from_numpy(x) for x in sample_inputs)
        model.qconfig = torch.quantization.default_qconfig
        q_model = torch.quantization.prepare(model, inplace=False)
        q_model = torch.quantization.convert(q_model, inplace=False)

        traced_model = torch.jit.trace(q_model, *pt_inputs)
        buf = io.BytesIO()
        torch.jit.save(traced_model, buf)
        buf.seek(0)
        q_model = torch.jit.load(buf)

        q_model.eval()
        output = q_model(*pt_inputs)

        f = io.BytesIO()
        torch.onnx.export(q_model, pt_inputs, f, input_names=input_names, example_outputs=output,
                          operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
        f.seek(0)
        onnx_model = onnx.load(f)
        caffe_res = c2.run_model(onnx_model, dict(zip(input_names, sample_inputs)))[0]
        np.testing.assert_almost_equal(output.detach().numpy(), caffe_res, decimal=3)

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
        torch.onnx.export(model, input, f, input_names=input_names, example_outputs=outputs,
                          operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
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

        # Permute pytorch output to NHWC
        np.testing.assert_almost_equal(outputs.numpy(), caffe_res, decimal=3)

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

    def test_avg_pool2d(self):
        class QAvgPool2dModule(torch.nn.Module):
            def __init__(self):
                super(QAvgPool2dModule, self).__init__()
                self.quant1 = torch.quantization.QuantStub()
                self.dequant = torch.quantization.DeQuantStub()

            def forward(self, x):
                res = torch.nn.functional.avg_pool2d(self.quant1(x), kernel_size=2, stride=1, padding=0)
                return self.dequant(res)

        x = np.random.rand(1, 2, 8, 8).astype("float32")
        self.generic_test(QAvgPool2dModule(), (x,), input_names=["x"])

    def test_reshape(self):
        class QReshapeModule(torch.nn.Module):
            def __init__(self):
                super(QReshapeModule, self).__init__()
                self.quant1 = torch.quantization.QuantStub()
                self.dequant = torch.quantization.DeQuantStub()

            def forward(self, x):
                res = self.quant1(x).reshape((1, 2, 1, 12))
                return self.dequant(res)

        x = np.random.rand(1, 2, 3, 4).astype("float32")
        self.generic_test(QReshapeModule(), (x,), input_names=["x"])

    def test_slice(self):
        class QSliceModule(torch.nn.Module):
            def __init__(self):
                super(QSliceModule, self).__init__()
                self.quant1 = torch.quantization.QuantStub()
                self.dequant = torch.quantization.DeQuantStub()

            def forward(self, x):
                qx = self.quant1(x)
                res = qx[:, 1:2]
                return self.dequant(res)

        x = np.random.rand(1, 2, 3, 4).astype("float32")
        self.generic_test(QSliceModule(), (x,), input_names=["x"])

    def test_cat(self):
        class QConcatModule(torch.nn.Module):
            def __init__(self):
                super(QConcatModule, self).__init__()
                self.quant1 = torch.quantization.QuantStub()
                self.dequant = torch.quantization.DeQuantStub()

            def forward(self, x, y):
                res = torch.ops.quantized.cat([self.quant1(x), self.quant1(y)], dim=1, scale=1.0, zero_point=0)
                return self.dequant(res)

        x = np.random.rand(1, 2, 3, 4).astype("float32")
        y = np.random.rand(1, 4, 3, 4).astype("float32")
        self.generic_test(QConcatModule(), (x, y,), input_names=["x", "y"])

    def test_max_pool2d(self):
        class QMaxPool2dModule(torch.nn.Module):
            def __init__(self):
                super(QMaxPool2dModule, self).__init__()
                self.quant1 = torch.quantization.QuantStub()
                self.dequant = torch.quantization.DeQuantStub()

            def forward(self, x):
                res = torch.nn.functional.max_pool2d(self.quant1(x), kernel_size=2, stride=1, padding=0)
                return self.dequant(res)

        x = np.random.rand(1, 2, 8, 8).astype("float32")
        self.generic_test(QMaxPool2dModule(), (x,), input_names=["x"])

    def test_quantized_sigmoid(self):
        self.generic_unary_test(torch.nn.Sigmoid())

    def test_small_model(self):
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super(SimpleModel, self).__init__()
                self.quant = torch.quantization.QuantStub()
                self.dequant = torch.quantization.DeQuantStub()
                self.conv1 = nn.Conv2d(3, 2, 5, bias=None).to(dtype=torch.float)
                self.act1 = nn.Sigmoid()
                self.conv2 = nn.Conv2d(2, 2, 1, bias=None).to(dtype=torch.float)
                self.fc = nn.Linear(72, 10).to(dtype=torch.float)
                self.fc.qconfig = None

            def forward(self, x):
                x = self.quant(x)
                x = self.conv1(x)
                x = self.act1(x)
                x = self.conv2(x)
                x = self.dequant(x)
                x = x.view(-1, 72).contiguous()
                x = self.fc(x)
                return x

        x = np.random.rand(2, 3, 10, 10).astype("float32")
        self.generic_model_test(SimpleModel(), (x,), input_names=["x"])

if __name__ == '__main__':
    unittest.main()
