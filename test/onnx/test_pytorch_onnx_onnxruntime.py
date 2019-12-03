from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import onnxruntime  # noqa
import torch

import numpy as np
import io
import itertools
import copy

from torch.nn.utils import rnn as rnn_utils
from model_defs.lstm_flattening_result import LstmFlatteningResult
from model_defs.rnn_model_with_packed_sequence import RnnModelWithPackedSequence
from test_pytorch_common import skipIfUnsupportedMinOpsetVersion, skipIfNoLapack
from test_pytorch_common import BATCH_SIZE
from test_pytorch_common import RNN_BATCH_SIZE, RNN_SEQUENCE_LENGTH, RNN_INPUT_SIZE, RNN_HIDDEN_SIZE
import model_defs.word_language_model as word_language_model
import torchvision


def ort_test_with_input(ort_sess, input, output, rtol, atol):
    input, _ = torch.jit._flatten(input)
    output, _ = torch.jit._flatten(output)

    def to_numpy(tensor):
        if tensor.requires_grad:
            return tensor.detach().cpu().numpy()
        else:
            return tensor.cpu().numpy()

    inputs = list(map(to_numpy, input))
    outputs = list(map(to_numpy, output))

    ort_inputs = dict((ort_sess.get_inputs()[i].name, input) for i, input in enumerate(inputs))
    ort_outs = ort_sess.run(None, ort_inputs)

    # compare onnxruntime and PyTorch results
    assert len(outputs) == len(ort_outs), "number of outputs differ"

    # compare onnxruntime and PyTorch results
    [np.testing.assert_allclose(out, ort_out, rtol=rtol, atol=atol) for out, ort_out in zip(outputs, ort_outs)]


def run_model_test(self, model, batch_size=2, state_dict=None,
                   input=None, use_gpu=True, rtol=0.001, atol=1e-7,
                   example_outputs=None, do_constant_folding=True,
                   dynamic_axes=None, test_with_inputs=None,
                   input_names=None, output_names=None,
                   fixed_batch_size=False):
    model.eval()

    if input is None:
        input = torch.randn(batch_size, 3, 224, 224, requires_grad=True)

    with torch.no_grad():
        if isinstance(input, torch.Tensor):
            input = (input,)
        # In-place operators will update input tensor data as well.
        # Thus inputs are replicated before every forward call.
        input_copy = copy.deepcopy(input)
        output = model(*input_copy)
        if isinstance(output, torch.Tensor):
            output = (output,)

        # export the model to ONNX
        f = io.BytesIO()
        input_copy = copy.deepcopy(input)
        torch.onnx._export(model, input_copy, f,
                           opset_version=self.opset_version,
                           example_outputs=output,
                           do_constant_folding=do_constant_folding,
                           keep_initializers_as_inputs=self.keep_initializers_as_inputs,
                           dynamic_axes=dynamic_axes,
                           input_names=input_names, output_names=output_names,
                           fixed_batch_size=fixed_batch_size)

        # compute onnxruntime output prediction
        ort_sess = onnxruntime.InferenceSession(f.getvalue())
        input_copy = copy.deepcopy(input)
        ort_test_with_input(ort_sess, input_copy, output, rtol, atol)

        # if addiional test inputs are provided run the onnx
        # model with these inputs and check the outputs
        if test_with_inputs is not None:
            for test_input in test_with_inputs:
                if isinstance(test_input, torch.Tensor):
                    test_input = (test_input,)
                test_input_copy = copy.deepcopy(test_input)
                output = model(*test_input_copy)
                if isinstance(output, torch.Tensor):
                    output = (output,)
                ort_test_with_input(ort_sess, test_input, output, rtol, atol)


class TestONNXRuntime(unittest.TestCase):
    from torch.onnx.symbolic_helper import _export_onnx_opset_version
    opset_version = _export_onnx_opset_version
    keep_initializers_as_inputs = True  # For IR version 3 type export.

    def setUp(self):
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
        np.random.seed(seed=0)

    def run_test(self, model, input, rtol=1e-3, atol=1e-7, do_constant_folding=True,
                 batch_size=2, use_gpu=True, dynamic_axes=None, test_with_inputs=None,
                 input_names=None, output_names=None, fixed_batch_size=False):
        return run_model_test(self, model, batch_size=batch_size,
                              input=input, use_gpu=use_gpu, rtol=rtol, atol=atol,
                              do_constant_folding=do_constant_folding,
                              dynamic_axes=dynamic_axes, test_with_inputs=test_with_inputs,
                              input_names=input_names, output_names=output_names,
                              fixed_batch_size=fixed_batch_size)

    # Export Torchvision models

    def test_alexnet(self):
        model = torchvision.models.alexnet(pretrained=True)
        x = torch.randn(2, 3, 224, 224, requires_grad=True)
        self.run_test(model, (x,))

    def test_densenets(self):
        model = torchvision.models.densenet121(pretrained=True)
        x = torch.randn(2, 3, 224, 224, requires_grad=True)
        self.run_test(model, (x,), rtol=1e-3, atol=1e-5)

    def test_googlenet(self):
        model = torchvision.models.googlenet(pretrained=True)
        x = torch.randn(2, 3, 224, 224, requires_grad=True)
        self.run_test(model, (x,), rtol=1e-3, atol=1e-5)

    def test_inception(self):
        model = torchvision.models.inception_v3(pretrained=True)
        x = torch.randn(2, 3, 224, 224, requires_grad=True)
        self.run_test(model, (x,), rtol=1e-3, atol=1e-5)

    def test_mnasnet(self):
        model = torchvision.models.mnasnet1_0(pretrained=True)
        x = torch.randn(2, 3, 224, 224, requires_grad=True)
        self.run_test(model, (x,), rtol=1e-3, atol=1e-5)

    def test_mobilenet(self):
        model = torchvision.models.mobilenet_v2(pretrained=True)
        x = torch.randn(2, 3, 224, 224, requires_grad=True)
        self.run_test(model, (x,), rtol=1e-3, atol=1e-5)

    def test_resnet(self):
        model = torchvision.models.resnet50(pretrained=True)
        x = torch.randn(2, 3, 224, 224, requires_grad=True)
        self.run_test(model, (x,))

    def test_shufflenet(self):
        model = torchvision.models.shufflenet_v2_x1_0(pretrained=True)
        x = torch.randn(2, 3, 224, 224, requires_grad=True)
        self.run_test(model, (x,), rtol=1e-3, atol=1e-5)

    def test_squeezenet(self):
        model = torchvision.models.squeezenet1_1(pretrained=True)
        x = torch.randn(2, 3, 224, 224, requires_grad=True)
        self.run_test(model, (x,))

    def test_vgg(self):
        model = torchvision.models.vgg19(pretrained=True)
        x = torch.randn(2, 3, 224, 224, requires_grad=True)
        self.run_test(model, (x,), rtol=1e-3, atol=1e-5)
        model = torchvision.models.vgg19_bn(pretrained=True)
        self.run_test(model, (x,), rtol=1e-3, atol=1e-5)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_fcn(self):
        model = torchvision.models.segmentation.segmentation.fcn_resnet101(pretrained=True)
        x = torch.randn(2, 3, 224, 224, requires_grad=True)
        self.run_test(model, (x,), rtol=1e-3, atol=1e-5)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_deeplab(self):
        model = torchvision.models.segmentation.segmentation.deeplabv3_resnet101(pretrained=True)
        x = torch.randn(2, 3, 224, 224, requires_grad=True)
        self.run_test(model, (x,), rtol=1e-3, atol=1e-5)

    def test_googlenet_quantization(self):
        model = torchvision.models.quantization.googlenet(pretrained=True)
        x = torch.randn(2, 3, 224, 224, requires_grad=True)
        self.run_test(model, (x,), rtol=1e-3, atol=1e-5)

    def test_inception_quantization(self):
        model = torchvision.models.quantization.inception_v3(pretrained=True)
        x = torch.randn(2, 3, 224, 224, requires_grad=True)
        self.run_test(model, (x,), rtol=1e-3, atol=1e-5)

    def test_mobilenet_quantization(self):
        model = torchvision.models.quantization.mobilenet_v2(pretrained=True)
        x = torch.randn(2, 3, 224, 224, requires_grad=True)
        self.run_test(model, (x,), rtol=1e-3, atol=1e-5)

    def test_resnet_quantization(self):
        model = torchvision.models.quantization.resnet50(pretrained=True)
        x = torch.randn(2, 3, 224, 224, requires_grad=True)
        self.run_test(model, (x,))

    def test_shufflenet_quantization(self):
        model = torchvision.models.quantization.shufflenet_v2_x1_0(pretrained=True)
        x = torch.randn(2, 3, 224, 224, requires_grad=True)
        self.run_test(model, (x,), rtol=1e-3, atol=1e-5)

    def test_r3d_18_video(self):
        model = torchvision.models.video.r3d_18(pretrained=True)
        x = torch.randn(1, 3, 4, 112, 112, requires_grad=True)
        self.run_test(model, (x,), rtol=1e-3, atol=1e-5)

    def test_mc3_18_video(self):
        model = torchvision.models.video.mc3_18(pretrained=True)
        x = torch.randn(1, 3, 4, 112, 112, requires_grad=True)
        self.run_test(model, (x,), rtol=1e-3, atol=1e-5)

    def test_r2plus1d_18_video(self):
        model = torchvision.models.video.r2plus1d_18(pretrained=True)
        x = torch.randn(1, 3, 4, 112, 112, requires_grad=True)
        self.run_test(model, (x,), rtol=1e-3, atol=1e-5)

    def run_word_language_model(self, model_name):
        ntokens = 50
        emsize = 5
        nhid = 5
        nlayers = 5
        dropout = 0.2
        tied = False
        batchsize = 5
        model = word_language_model.RNNModel(model_name, ntokens, emsize,
                                             nhid, nlayers, dropout, tied,
                                             batchsize)
        x = torch.arange(0, ntokens).long().view(-1, batchsize)
        # Only support CPU version, since tracer is not working in GPU RNN.
        self.run_test(model, (x, model.hidden))

    def test_word_language_model_RNN_TANH(self):
        self.run_word_language_model("RNN_TANH")

    def test_word_language_model_RNN_RELU(self):
        self.run_word_language_model("RNN_RELU")

    def test_word_language_model_LSTM(self):
        self.run_word_language_model("LSTM")

    def test_word_language_model_GRU(self):
        self.run_word_language_model("GRU")

    def test_index_1d(self):
        self._test_index_generic(lambda input: input[0])

    def test_index_2d_1dimslice(self):
        self._test_index_generic(lambda input: input[0:1, :])

    def test_index_2d_sliceint(self):
        self._test_index_generic(lambda input: input[1, :])

    def test_index_2d_neg_slice(self):
        self._test_index_generic(lambda input: input[0:-1, :])

    def test_dict(self):
        class MyModel(torch.nn.Module):
            def forward(self, x_in):
                x_out = {}
                x_out["test_key_out"] = torch.add(x_in[list(x_in.keys())[0]], list(x_in.keys())[0])
                return x_out

        x = {torch.tensor(1.): torch.randn(1, 2, 3)}
        self.run_test(MyModel(), (x,))

    def test_dict_str(self):
        class MyModel(torch.nn.Module):
            def forward(self, x_in):
                x_out = {}
                x_out["test_key_out"] = torch.add(x_in["test_key_in"], 2.)
                return x_out

        x = {"test_key_in": torch.randn(1, 2, 3)}
        self.run_test(MyModel(), (x,))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_cste_script(self):
        class MyModel(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x):
                return torch.zeros(x.size(0)), torch.ones((x.size(1), x.size(0)), dtype=torch.int64)

        x = torch.randn(3, 4)
        self.run_test(MyModel(), x)

    def test_scalar_tensor(self):
        class test(torch.nn.Module):
            def forward(self, input):
                return torch.scalar_tensor(input.size(0)), \
                    torch.scalar_tensor(input.size(1), dtype=torch.int64)

        x = torch.randn(2, 3, 4)
        y = torch.randn(7, 8, 9)
        model = test()
        self.run_test(model, x, test_with_inputs=[y],
                      input_names=['input_1'],
                      dynamic_axes={'input_1': [0, 1, 2]})

    def test_hardtanh(self):
        model = torch.nn.Hardtanh(-1.5, 2.5)
        x = torch.arange(-5, 5).to(dtype=torch.float32)
        self.run_test(model, x)

    def test_hardtanh_script_with_default_values(self):
        class MyModel(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x):
                return torch.nn.functional.hardtanh(x)

        x = torch.arange(-5, 5).to(dtype=torch.float32)
        self.run_test(MyModel(), x)

    def test_clamp(self):
        class ClampModel(torch.nn.Module):
            def forward(self, x):
                return x.clamp(-0.5, 0.5)

        x = torch.randn(3, 4)
        self.run_test(ClampModel(), x)

        class ClampMinModel(torch.nn.Module):
            def forward(self, x):
                return x.clamp(min=-0.5)

        x = torch.randn(3, 4)
        self.run_test(ClampMinModel(), x)

        class ClampMaxModel(torch.nn.Module):
            def forward(self, x):
                return x.clamp(max=0.5)

        x = torch.randn(3, 4)
        self.run_test(ClampMaxModel(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_clamp_dyn(self):
        class ClampMaxModel(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x):
                return x.clamp(None, x.size(0))

        x = torch.arange(16).view(4, 4).float()
        self.run_test(ClampMaxModel(), x)


        class ClampMinModel(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x):
                return x.clamp(x.size(0), None)

        x = torch.arange(16).view(4, 4).float()
        self.run_test(ClampMinModel(), x)

        class ClampMinMaxModel(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x):
                return x.clamp(x.size(0), x.size(1))

        x = torch.arange(16).view(2, 8).float()
        self.run_test(ClampMinMaxModel(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_full_trace(self):
        class FullModel(torch.nn.Module):
            def forward(self, x):
                return torch.full((3, 4), x, dtype=torch.long)

        x = torch.tensor(12)
        self.run_test(FullModel(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_full_script(self):
        class FullModelScripting(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x):
                return torch.full((3, 4), x, dtype=torch.long)

        x = torch.tensor(12)
        self.run_test(FullModelScripting(), x)

    def test_fuse_addmm(self):
        class AddmmModel(torch.nn.Module):
            def forward(self, x):
                return torch.mm(x, x) + x

        x = torch.ones(3, 3)
        self.run_test(AddmmModel(), x)

    def test_maxpool(self):
        model = torch.nn.MaxPool1d(2, stride=1)
        x = torch.randn(20, 16, 50)
        self.run_test(model, x)

    def test_conv(self):
        class TraceModel(torch.nn.Module):
            def __init__(self):
                super(TraceModel, self).__init__()
                self.conv1 = torch.nn.Conv1d(16, 33, 3, stride=2)
                self.conv2 = torch.nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
                self.conv3 = torch.nn.Conv3d(16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(4, 2, 0))

            def forward(self, input1, input2, input3):
                return self.conv1(input1), self.conv2(input2), self.conv3(input3)

        class ScriptModel(torch.jit.ScriptModule):
            def __init__(self):
                super(ScriptModel, self).__init__()
                self.conv1 = torch.nn.Conv1d(16, 33, 3, stride=2)
                self.conv2 = torch.nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
                self.conv3 = torch.nn.Conv3d(16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(4, 2, 0))

            @torch.jit.script_method
            def forward(self, input1, input2, input3):
                return self.conv1(input1), self.conv2(input2), self.conv3(input3)

        x1 = torch.randn(20, 16, 50)
        x2 = torch.randn(20, 16, 50, 100)
        x3 = torch.randn(20, 16, 10, 50, 100)

        self.run_test(TraceModel(), (x1, x2, x3), atol=10e-5)
        self.run_test(ScriptModel(), (x1, x2, x3), atol=10e-5)

    # TODO: Add ConvTranspose1d and ConvTranspose3d when supported in ORT
    # TODO : Add test with dilation != 1 when ORT fixed
    def test_conv_transpose(self):
        class TraceModel(torch.nn.Module):
            def __init__(self):
                super(TraceModel, self).__init__()
                self.conv2 = torch.nn.ConvTranspose2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(1, 1))

            def forward(self, input2):
                return self.conv2(input2)

        class ScriptModel(torch.jit.ScriptModule):
            def __init__(self):
                super(ScriptModel, self).__init__()
                self.conv2 = torch.nn.ConvTranspose2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(1, 1))

            @torch.jit.script_method
            def forward(self, input2):
                return self.conv2(input2)

        x2 = torch.randn(20, 16, 50, 100)

        self.run_test(TraceModel(), (x2,), atol=10e-5)
        self.run_test(ScriptModel(), (x2,), atol=10e-5)

    def test_squeeze(self):
        class Squeeze(torch.nn.Module):
            def forward(self, x):
                return torch.torch.squeeze(x, dim=-2)

        x = torch.randn(2, 1, 4)
        self.run_test(Squeeze(), x)

    def test_unsqueeze(self):
        class Unsqueeze(torch.nn.Module):
            def forward(self, x):
                return torch.unsqueeze(x, dim=-2)

        x = torch.randn(2, 3, 4)
        self.run_test(Unsqueeze(), x)

    @skipIfUnsupportedMinOpsetVersion(8)
    def test_maxpool_adaptive(self):
        model = torch.nn.AdaptiveMaxPool1d((5), return_indices=False)
        x = torch.randn(20, 16, 50, requires_grad=True)
        self.run_test(model, x)

    def test_maxpool_2d(self):
        model = torch.nn.MaxPool2d(5, padding=(1, 2))
        x = torch.randn(1, 20, 16, 50, requires_grad=True)
        self.run_test(model, x)

    def test_maxpool_1d_ceil(self):
        model = torch.nn.MaxPool1d(3, 2, ceil_mode=True)
        x = torch.randn(20, 16, 50)
        self.run_test(model, x)

    def test_maxpool_2d_ceil(self):
        model = torch.nn.MaxPool2d(3, 2, ceil_mode=True)
        x = torch.randn(20, 16, 50, 32)
        self.run_test(model, x)

    def test_maxpool_3d_ceil(self):
        model = torch.nn.MaxPool3d(3, 2, ceil_mode=True)
        x = torch.randn(20, 16, 50, 44, 31)
        self.run_test(model, x)

    @skipIfUnsupportedMinOpsetVersion(8)
    def test_maxpool_with_indices(self):
        model = torch.nn.MaxPool1d(2, stride=1, return_indices=True)
        x = torch.randn(20, 16, 50)
        self.run_test(model, x)

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_maxpool_dilation(self):
        model = torch.nn.MaxPool1d(2, stride=1, dilation=2)
        x = torch.randn(20, 16, 50)
        self.run_test(model, x)

    def test_avgpool(self):
        model = torch.nn.AvgPool1d(2, stride=1)
        x = torch.randn(20, 16, 50)
        self.run_test(model, x)

    def test_avgpool_1d_ceil(self):
        model = torch.nn.AvgPool1d(3, 2, ceil_mode=True)
        x = torch.randn(1, 1, 7)
        self.run_test(model, x)

    def test_avgpool_2d_ceil(self):
        model = torch.nn.AvgPool2d(3, 2, ceil_mode=True)
        x = torch.randn(20, 16, 50, 32)
        self.run_test(model, x)

    def test_avgpool_3d_ceil(self):
        model = torch.nn.AvgPool3d(3, 2, ceil_mode=True)
        x = torch.randn(20, 16, 50, 44, 31)
        self.run_test(model, x)

    def test_arithmetic(self):
        class ArithmeticModule(torch.nn.Module):
            def forward(self, x):
                x = x + 2
                x = x - 4
                x = x * 6
                x = x / 8
                return x

        x = torch.randn(2, 3, 4)
        self.run_test(ArithmeticModule(), x)

    def test_slice_trace(self):
        class MyModule(torch.nn.Module):
            def forward(self, x):
                return x[0:1]

        x = torch.randn(3)
        self.run_test(MyModule(), x)

    def test_slice_neg(self):
        class NegSlice(torch.nn.Module):
            def forward(self, x):
                return x[-1:]

        x = torch.randn(3, 4, 5)
        self.run_test(NegSlice(), x)

    def test_slice_neg_large(self):
        class NegSlice(torch.nn.Module):
            def forward(self, x):
                return x[:, :, -3:-1, :, -1]

        x = torch.randn(3, 4, 5, 6, 7)
        self.run_test(NegSlice(), x)

    @unittest.skip('https://github.com/pytorch/pytorch/issues/10984')
    def test_slice_neg_large_negone(self):
        class NegSlice(torch.nn.Module):
            def forward(self, x):
                return x[:, :, :, :, -1]

        x = torch.randn(3, 4, 5, 6, 7)
        self.run_test(NegSlice(), x)

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_slice_dynamic(self):
        class DynamicSliceExportMod(torch.nn.Module):
            def forward(self, x):
                results = []
                for i in range(4):
                    results.append(x[:x.size(0) - i, i:x.size(2), i:3])
                return tuple(results)

        x = torch.rand(5, 5, 5)
        y = torch.randn(6, 7, 8)
        self.run_test(DynamicSliceExportMod(), x, test_with_inputs=[y],
                      input_names=['input_1'],
                      output_names=['output_1'],
                      dynamic_axes={'input_1': [0, 1, 2],
                                    'output_1': [0, 1, 2]})

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_slice_dynamic_script(self):
        class DynamicSliceModel(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x):
                return x[1:x.size(0)]

        x = torch.rand(1, 2)
        self.run_test(DynamicSliceModel(), x)

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_slice_dynamic_to_end(self):
        class DynamicSliceExportMod(torch.nn.Module):
            def forward(self, x):
                results = []
                for i in range(4):
                    results.append(x[:, i:, x.size(2) - 5])
                return tuple(results)

        x = torch.rand(5, 5, 5)
        self.run_test(DynamicSliceExportMod(), x,
                      dynamic_axes={'input_1': [0, 1, 2],
                      'output_1': [0, 1, 2]})

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_arange_dynamic(self):
        class ArangeModel(torch.nn.Module):
            def forward(self, input):
                return torch.arange(input.shape[0]), \
                    torch.arange(12), \
                    torch.arange(start=input.shape[0], end=input.shape[0] + 5)

        x = torch.randn(5, 3, 2)
        y = torch.randn(8, 3, 2)
        self.run_test(ArangeModel(), x, test_with_inputs=[y],
                      input_names=['input_1'],
                      output_names=['output_1', 'output_2', 'output_3'],
                      dynamic_axes={'input_1': [0],
                                    'output_1': [0]})

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_arange(self):
        class ArangeModel(torch.nn.Module):
            def forward(self, start, end):
                return torch.arange(start.size(0), end, 1.5, dtype=torch.int64)

        x = torch.randn(2, 3, 4)
        y = torch.tensor(8.5, dtype=torch.float)
        self.run_test(ArangeModel(), (x, y))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_arange_no_type(self):
        class ArangeModel(torch.nn.Module):
            def forward(self, end):
                return torch.arange(end), \
                    torch.arange(0, end)

        x = torch.tensor(6.2, dtype=torch.float)
        self.run_test(ArangeModel(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_size(self):
        class SizeModel(torch.nn.Module):
            def forward(self, input):
                return torch.arange(input.size(0)), torch.arange(input.size(-1))

        x = torch.randn(5, 3, 2)
        self.run_test(SizeModel(), x)

    def _test_index_generic(self, fn):
        class MyModel(torch.nn.Module):
            def __init__(self):
                super(MyModel, self).__init__()

            def forward(self, input):
                return fn(input)

        m1 = torch.randn(3, 4, 5, 6, 7)
        self.run_test(MyModel(), m1)

    def test_tensor_index_advanced_indexing(self):
        self._test_index_generic(
            lambda input: input[:, torch.tensor([[0, 2], [1, 1]]), :, torch.tensor([2, 1]), torch.tensor([0, 3])])
        self._test_index_generic(lambda input: input[..., torch.tensor([2, 1]), torch.tensor([0, 3])])
        self._test_index_generic(lambda input: input[:, torch.tensor([0, 2]), None, 2:4, torch.tensor([[1, 3], [4, 0]])])
        self._test_index_generic(lambda input: input[:, torch.tensor([0, 2]), torch.tensor([1]), 2:4, torch.tensor([[1], [4]])])

    def test_tensor_index_advanced_indexing_consecutive(self):
        self._test_index_generic(lambda input: input[:, torch.tensor([0, 2]), torch.tensor([[1, 3], [4, 0]]), None])

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_flip(self):
        class MyModule(torch.nn.Module):
            def forward(self, x):
                return torch.flip(x, dims=[0])

        x = torch.tensor(np.arange(6.0).reshape(2, 3))
        self.run_test(MyModule(), x)

    def test_random(self):
        class RandN(torch.nn.Module):
            def forward(self, x):
                return torch.mul(x, (torch.rand(2, 3, 4) + x).size(0))

        x = torch.randn(2, 3, 4)
        self.run_test(RandN(), x)

        class Rand(torch.nn.Module):
            def forward(self, x):
                return torch.mul(x, (torch.rand(2, 3, 4) + x).size(0))

        x = torch.randn(2, 3, 4)
        self.run_test(Rand(), x)

    def test_random_like(self):
        class RandNLike(torch.nn.Module):
            def forward(self, x):
                return torch.mul(x, torch.randn_like(x).size(0))

        x = torch.randn(2, 3, 4)
        self.run_test(RandNLike(), x)

        class RandLike(torch.nn.Module):
            def forward(self, x):
                return torch.mul(x, torch.rand_like(x).size(0))

        x = torch.randn(2, 3, 4)
        self.run_test(RandLike(), x)

    def test_random_like_dtype(self):
        class RandNLike(torch.nn.Module):
            def forward(self, x):
                return torch.mul(x.to(torch.double), torch.randn_like(x, dtype=torch.double).size(0))

        x = torch.randn(2, 3, 4)
        self.run_test(RandNLike(), x)

        class RandLike(torch.nn.Module):
            def forward(self, x):
                return torch.mul(x.to(torch.double), torch.rand_like(x, dtype=torch.double).size(0))

        x = torch.randn(2, 3, 4)
        self.run_test(RandLike(), x)

    def _interpolate(self, x, mode, use_size, is_upsample):
        class MyModel(torch.nn.Module):
            def forward(self, x):
                scale = 2.3 if is_upsample else 0.3
                if use_size:
                    size = [int(float(v) * scale) for v in x.size()[2:]]
                    return torch.nn.functional.interpolate(x, mode=mode, size=size)
                return torch.nn.functional.interpolate(x, mode=mode, scale_factor=scale)

        self.run_test(MyModel(), x)

    def _interpolate_script(self, x, mode, use_size, is_upsample):

        class MyModel(torch.jit.ScriptModule):
            __constants__ = ['mode', 'use_size', 'is_upsample', 'size', 'scale', 'size_array', 'scale_array']

            def __init__(self, mode, use_size, is_upsample):
                super(MyModel, self).__init__()
                self.mode = mode
                self.use_size = use_size
                self.is_upsample = is_upsample
                self.scale = 2.0 if self.is_upsample else 0.5
                self.size = 24 if self.is_upsample else 1
                if x.dim() == 3:
                    self.scale_array = [2.]
                    self.size_array = [16]
                elif x.dim() == 4:
                    self.scale_array = [2., 3.]
                    self.size_array = [16, 32]
                else:
                    self.scale_array = [2., 3., 4.]
                    self.size_array = [16, 32, 64]

            @torch.jit.script_method
            def forward(self, x):
                if self.use_size:
                    out = torch.nn.functional.interpolate(x, mode=self.mode, size=self.size)
                    out_array = torch.nn.functional.interpolate(x, mode=self.mode, size=self.size_array)
                    return out, out_array
                out = torch.nn.functional.interpolate(x, mode=self.mode, scale_factor=self.scale)
                out_array = torch.nn.functional.interpolate(x, mode=self.mode, scale_factor=self.scale_array)
                return out, out_array

        model = MyModel(mode, use_size, is_upsample)
        self.run_test(model, x)

    # TODO: Enable bicubic, linear1d and linear3d when implemented in ORT
    def _interpolate_tests(self, is_upsample):
        # - cubic mode is not supported for opsets below 11;
        # - linear mode does not match for opsets below 11;
        # - nearest mode does not match for opsets below 11,
        # for some cases where the nearest pixel's index is
        # not calculated the same way for ONNX and PyTorch
        # (the operation involves a floor in PyTorch vs
        # in round_prefer_floor ONNX). (The below tests
        # do not  show this error for nearest mode for
        # all opsets)
        modes = ["nearest", "linear"]  # TODO : add "bicubic" when enabled in ORT
        if self.opset_version < 11:
            modes = ["nearest"]
        x = [torch.randn(1, 2, 4, requires_grad=True),
             torch.randn(1, 2, 4, 4, requires_grad=True),
             torch.randn(1, 2, 4, 4, 6, requires_grad=True)]

        for mode in modes:
            for xi in x:
                mode_i = mode
                if mode == "bicubic" and xi.dim() != 4:
                    continue
                elif mode == "linear":
                    if xi.dim() == 3:
                        # TODO : enable when linear mode is implemented for 1d inputs in ORT
                        continue
                    elif xi.dim() == 4:
                        mode_i = "bilinear"
                    elif xi.dim() == 5:
                        # TODO : enable when linear mode is implemented for 3d inputs in ORT
                        mode_i = "trilinear"
                        continue
                self._interpolate(xi, mode_i, True, is_upsample)
                # the following cases, require dynamic sizes/scales,
                # which which is not supported for opset_version < 9
                if self.opset_version >= 9:
                    self._interpolate_script(xi, mode_i, True, is_upsample)
                    self._interpolate(xi, mode_i, False, is_upsample)
                    self._interpolate_script(xi, mode_i, False, is_upsample)

    def test_interpolate_upsample(self):
        self._interpolate_tests(True)

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_interpolate_downsample(self):
        self._interpolate_tests(False)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_interpolate_no_shape(self):
        class MyModel(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x, y):
                x = torch.add(x, x)
                out1 = torch.nn.functional.interpolate(x, mode="bilinear", size=(16, 16), align_corners=False)
                out2 = torch.nn.functional.interpolate(x, mode="nearest", size=(int(y.size(0)), int(y.size(1))))
                return out1, out2

        x = torch.randn(1, 2, 4, 4, requires_grad=True)
        y = torch.randn(16, 16, requires_grad=True)
        self.run_test(MyModel(), (x, y))

    def test_groupnorm(self):
        model = torch.nn.GroupNorm(3, 6, 0.002)
        x = torch.randn(4, 6, 180, 180, 180)
        self.run_test(model, x)

        model = torch.nn.GroupNorm(1, 6, 0.002)
        x = torch.randn(4, 6, 180, 180)
        self.run_test(model, x)

        model = torch.nn.GroupNorm(6, 6, 0.002)
        x = torch.randn(4, 6, 180, 180)
        self.run_test(model, x)

    def test_groupnorm_noaffine(self):
        model = torch.nn.GroupNorm(4, 8, 0.002, affine=False)
        x = torch.randn(3, 8, 224, 224)
        self.run_test(model, x)

        model = torch.nn.GroupNorm(1, 6, 0.002, affine=False)
        x = torch.randn(4, 6, 180, 180)
        self.run_test(model, x)

        model = torch.nn.GroupNorm(6, 6, 0.002, affine=False)
        x = torch.randn(4, 6, 180, 180)
        self.run_test(model, x)

    def test_std(self):
        class StandardDeviation(torch.nn.Module):
            def forward(self, input):
                return torch.std(input, unbiased=False)

        x = torch.randn(2, 3, 4)
        model = StandardDeviation()
        self.run_test(model, x)

    def test_std_along_dims(self):
        class StandardDeviation(torch.nn.Module):
            def forward(self, input):
                return torch.std(input, dim=(0, 1), unbiased=False)

        x = torch.randn(2, 3, 4)
        model = StandardDeviation()
        self.run_test(model, x)

    def test_std_keepdim(self):
        class StandardDeviation(torch.nn.Module):
            def forward(self, input):
                return torch.std(input, dim=(0, 1), unbiased=False, keepdim=True)

        x = torch.randn(2, 3, 4)
        model = StandardDeviation()
        self.run_test(model, x)

    def test_bitshift(self):
        class BitshiftModel(torch.nn.Module):
            def forward(self, input, input2):
                return input >> 1, input << 3.1, \
                    input2 >> torch.tensor([1, 2]), input2 << 4.2
        input = torch.arange(24, dtype=torch.float32).reshape(3, 4, 2)
        input2 = torch.arange(24, dtype=torch.int64).reshape(3, 4, 2)
        self.run_test(BitshiftModel(), (input, input2))

    def test_bitshift_other_fp(self):
        class BitshiftModel(torch.nn.Module):
            def forward(self, input):
                return input << 2.4
        input = torch.arange(24, dtype=torch.int64).reshape(3, 4, 2)
        self.run_test(BitshiftModel(), input)

    # uint8 not implemented in ORT for Mul used in
    # exporting bitshift for opset_version < 10
    @skipIfUnsupportedMinOpsetVersion(11)
    def test_bitshift_uint8(self):
        class BitshiftModel(torch.nn.Module):
            def forward(self, input, input2):
                return input >> 1, input << 3., \
                    input2 >> torch.tensor([1, 2], dtype=torch.uint8), input2 << 4.
        input = torch.arange(24, dtype=torch.uint8).reshape(3, 4, 2)
        input2 = torch.arange(24, dtype=torch.uint8).reshape(3, 4, 2)
        self.run_test(BitshiftModel(), (input, input2))

    def test_narrow(self):
        class NarrowModel(torch.nn.Module):
            def forward(self, input):
                return torch.narrow(input, 0, 0, 2)

        x = torch.randn(3, 3, requires_grad=True)
        self.run_test(NarrowModel(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_index_fill(self):
        class IndexFillModel(torch.nn.Module):
            def forward(self, input):
                index = torch.tensor([2, 0])
                return input.index_fill(2, index, -1)

        x = torch.randn(3, 4, 5, requires_grad=True)
        self.run_test(IndexFillModel(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_index_copy(self):
        class IndexCopyModel(torch.nn.Module):
            def forward(self, input):
                index = torch.tensor([2, 0])
                source = torch.ones(3, 2, 5)
                return input.index_copy(1, index, source)

        x = torch.randn(3, 4, 5, requires_grad=True)
        self.run_test(IndexCopyModel(), x)

    # TODO: enable for opset 10 when ONNXRuntime version will be updated

    def test_index_select_constant_scaler_index(self):
        class IndexSelectScalerIndexModel(torch.nn.Module):
            def forward(self, x):
                index = 2
                return torch.index_select(x, 1, torch.tensor(index))
        x = torch.randn(3, 4)
        self.run_test(IndexSelectScalerIndexModel(), x)

    def test_index_select_scaler_index(self):
        class IndexSelectScalerIndexModel(torch.nn.Module):
            def __init__(self, index_base):
                super(IndexSelectScalerIndexModel, self).__init__()
                self.index_base = torch.tensor(index_base)

            def forward(self, x, index_offset):
                index = self.index_base + index_offset
                return torch.index_select(x, 1, index)
        x = torch.randn(3, 4)
        offset = 2
        index_offset = torch.tensor(offset)
        base = 1
        self.run_test(IndexSelectScalerIndexModel(base), (x, index_offset))

    def test_topk(self):
        class MyModule(torch.nn.Module):
            def forward(self, x):
                return torch.topk(x, 3)

        x = torch.arange(1., 6., requires_grad=True)
        self.run_test(MyModule(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_topk_smallest_unsorted(self):
        class MyModule(torch.nn.Module):
            def forward(self, x, k):
                return torch.topk(x, k, largest=False, sorted=False)

        x = torch.arange(1., 6., requires_grad=True)
        k = torch.tensor(3)
        self.run_test(MyModule(), (x, k))

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_topk_script(self):
        class MyModuleDynamic(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x, k):
                return torch.topk(x, k)

        x = torch.arange(1., 6., requires_grad=True)
        k = torch.tensor(3)
        self.run_test(MyModuleDynamic(), [x, k])

    def test_layer_norm(self):
        model = torch.nn.LayerNorm([10, 10])
        x = torch.randn(20, 5, 10, 10)
        self.run_test(model, x)

    def test_batchnorm1d(self):
        x = torch.randn(10, 10)
        model = torch.nn.BatchNorm1d(10, affine=True)
        self.run_test(model, x)

        x = torch.randn(10, 10, 128)
        self.run_test(model, x)

    def test_batchnorm1d_noaffine(self):
        x = torch.randn(10, 10)
        model = torch.nn.BatchNorm1d(10, affine=False)
        self.run_test(model, x)

        x = torch.randn(10, 10, 128)
        self.run_test(model, x)

    def test_batchnorm2d(self):
        x = torch.randn(10, 3, 128, 128)
        model = torch.nn.BatchNorm2d(3, affine=True)
        self.run_test(model, x)

    def test_batchnorm2d_noaffine(self):
        x = torch.randn(10, 3, 128, 128)
        model = torch.nn.BatchNorm2d(3, affine=False)
        self.run_test(model, x)

    def test_batchnorm3d(self):
        x = torch.randn(10, 3, 128, 128, 128)
        model = torch.nn.BatchNorm3d(3, affine=True)
        self.run_test(model, x)

    def test_batchnorm3d_noaffine(self):
        x = torch.randn(10, 3, 128, 128, 128)
        model = torch.nn.BatchNorm3d(3, affine=False)
        self.run_test(model, x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_scatter(self):
        class ScatterModel(torch.nn.Module):
            def forward(self, input, indices, values):
                return input.scatter(1, indices, values)

        input = torch.tensor([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
        indices = torch.tensor([[1, 0], [0, 1], [0, 1]], dtype=torch.int64)
        values = torch.tensor([[1.0, 1.1], [2.0, 2.1], [3.0, 3.1]])
        self.run_test(ScatterModel(), input=(input, indices, values))

        input = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        indices = torch.tensor([[1, 0], [0, 2], [0, 1]], dtype=torch.int64)
        values = torch.tensor([[1.0, 1.1], [2.0, 2.1], [3.0, 3.1]])
        self.run_test(ScatterModel(), (input, indices, values))

        input = torch.zeros(3, 4, 5, 6)
        indices = torch.tensor([[1, 0], [0, 2], [0, 1]], dtype=torch.int64)
        indices = indices.view(3, 2, 1, 1).expand(3, 2, 5, 6)
        values = torch.arange(3 * 2 * 5 * 6, dtype=torch.float32).view(3, 2, 5, 6)
        self.run_test(ScatterModel(), (input, indices, values))

        input = torch.zeros(3, 4, 2)
        indices = torch.tensor([[[1, 0], [0, 2]], [[1, 1], [0, 1]], [[2, 1], [2, 2]]])
        values = torch.arange(3 * 2 * 2, dtype=torch.float32).view(3, 2, 2)
        self.run_test(ScatterModel(), (input, indices, values))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_scatter_add(self):
        class ScatterModel(torch.nn.Module):
            def forward(self, input, indices, values):
                return input.scatter_add(1, indices, values)

        input = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        indices = torch.tensor([[1, 0], [0, 1], [0, 1]], dtype=torch.int64)
        values = torch.tensor([[1.0, 1.1], [2.0, 2.1], [3.0, 3.1]])
        self.run_test(ScatterModel(), input=(input, indices, values))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_gather(self):
        class GatherModel(torch.nn.Module):
            def forward(self, input, indices):
                return input.gather(1, indices)

        input = torch.tensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
        indices = torch.tensor([[1, 0], [0, 1], [0, 1]], dtype=torch.int64)
        self.run_test(GatherModel(), input=(input, indices))

    def test_multinomial(self):
        class Multinomial(torch.nn.Module):
            def forward(self, weight):
                return torch.multinomial(weight, 3, replacement=True)

        class MultinomialNoReplacement(torch.nn.Module):
            def forward(self, weight):
                return torch.multinomial(weight, 1)

        weight = torch.tensor([[0, 10, 0, 0], [0, 0, 100, 0]], dtype=torch.float)
        self.run_test(Multinomial(), (weight,))
        self.run_test(MultinomialNoReplacement(), (weight,))

    def test_reduce_log_sum_exp(self):
        class ReduceLogSumExpModel(torch.nn.Module):
            def forward(self, input):
                a = torch.logsumexp(input, dim=0)
                b = torch.logsumexp(input, dim=(0, 1))
                return a + b

        x = torch.randn(4, 4, requires_grad=True)
        self.run_test(ReduceLogSumExpModel(), x)

    def test_logsoftmax(self):
        for i in range(7)[2:]:
            model = torch.nn.LogSoftmax(dim=i - 1)
            dims = [2] * (i - 2) + [3, 4]
            input = torch.ones(*dims, requires_grad=True)
            self.run_test(model, input)

    def test_logsoftmax_dim(self):
        for i in range(-4, 3):
            model = torch.nn.LogSoftmax(dim=i)
            input = torch.randn(3, 4, 5, 6)
            self.run_test(model, input)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_lstm(self):
        model = torch.nn.LSTM(RNN_INPUT_SIZE, RNN_HIDDEN_SIZE, 1, bidirectional=False)
        input = torch.randn(RNN_SEQUENCE_LENGTH, BATCH_SIZE, RNN_INPUT_SIZE)
        h0 = torch.randn(1, BATCH_SIZE, RNN_HIDDEN_SIZE)
        c0 = torch.randn(1, BATCH_SIZE, RNN_HIDDEN_SIZE)
        self.run_test(model, (input, (h0, c0)))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_lstm_default_init_state(self):
        model = torch.nn.LSTM(RNN_INPUT_SIZE, RNN_HIDDEN_SIZE, 1, bidirectional=False)
        input = torch.randn(RNN_SEQUENCE_LENGTH, BATCH_SIZE, RNN_INPUT_SIZE)
        self.run_test(model, input)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_lstm_fixed_batch_size(self):
        class LSTMModel(torch.nn.Module):
            def __init__(self):
                super(LSTMModel, self).__init__()
                self.lstm = torch.nn.LSTM(RNN_INPUT_SIZE, RNN_HIDDEN_SIZE, 1, bidirectional=False)

            def forward(self, input):
                batch_size = input.size()[1]
                h0_np = np.ones([1, batch_size, RNN_HIDDEN_SIZE]).astype(np.float32)
                c0_np = np.ones([1, batch_size, RNN_HIDDEN_SIZE]).astype(np.float32)
                h0 = torch.from_numpy(h0_np)
                c0 = torch.from_numpy(c0_np)
                return self.lstm(input, (h0, c0))

        input = torch.randn(RNN_SEQUENCE_LENGTH, BATCH_SIZE, RNN_INPUT_SIZE)
        # verify with different input of same batch size
        input2 = torch.randn(RNN_SEQUENCE_LENGTH, BATCH_SIZE, RNN_INPUT_SIZE)
        self.run_test(LSTMModel(), input, fixed_batch_size=True, test_with_inputs=[input2])

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_lstm_post_fix_init_state(self):
        class LSTMModel(torch.nn.Module):
            def __init__(self):
                super(LSTMModel, self).__init__()
                self.lstm = torch.nn.LSTM(RNN_INPUT_SIZE, RNN_HIDDEN_SIZE,
                                          1, bidirectional=False)

            def forward(self, input):
                batch_size = input.size()[1]
                h0_np = np.ones([1, batch_size, RNN_HIDDEN_SIZE]).astype(np.float32)
                c0_np = np.ones([1, batch_size, RNN_HIDDEN_SIZE]).astype(np.float32)
                h0 = torch.from_numpy(h0_np)
                c0 = torch.from_numpy(c0_np)
                return self.lstm(input, (h0, c0))

        model = LSTMModel()
        input = torch.randn(RNN_SEQUENCE_LENGTH, 1, RNN_INPUT_SIZE)
        # verify with different input of different batch size
        input2 = torch.randn(RNN_SEQUENCE_LENGTH, BATCH_SIZE, RNN_INPUT_SIZE)
        self.run_test(model, input, dynamic_axes={'input' : {0 : 'seq', 1 : 'batch'}},
                      test_with_inputs=[input2])

    def test_lstm_constant_folding(self):
        class LstmNet(torch.nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, bidirectional):
                super(LstmNet, self).__init__()
                self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, bidirectional=bidirectional)

            def forward(self, input, initial_state):
                return self.lstm(input, initial_state)

        def get_LstmNet_model_and_inputs(input_size, hidden_size, num_layers, batch_size,
                                         seq_len, bidirectional):
            num_directions = 2 if bidirectional else 1
            model = LstmNet(input_size, hidden_size, num_layers, bidirectional)
            input = torch.randn(seq_len, batch_size, input_size)
            h0 = torch.randn(num_layers * num_directions, batch_size, hidden_size)
            c0 = torch.randn(num_layers * num_directions, batch_size, hidden_size)
            return model, (input, (h0, c0))

        batch_size1 = 3
        model1, input1 = get_LstmNet_model_and_inputs(7, 3, 2, batch_size1, 5, True)
        self.run_test(model1, input1, do_constant_folding=True)

        batch_size2 = 4
        model2, input2 = get_LstmNet_model_and_inputs(5, 4, 3, batch_size2, 7, False)
        self.run_test(model2, input2, do_constant_folding=True)

    def test_gru_constant_folding(self):
        class GruNet(torch.nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, bidirectional):
                super(GruNet, self).__init__()
                self.mygru = torch.nn.GRU(input_size, hidden_size, num_layers, bidirectional=bidirectional)

            def forward(self, input, initial_state):
                out = self.mygru(input, initial_state)
                return out

        def get_GruNet_model_and_inputs(input_size, hidden_size, num_layers, batch_size,
                                        seq_len, bidirectional):
            num_directions = 2 if bidirectional else 1
            model = GruNet(input_size, hidden_size, num_layers, bidirectional)
            input = torch.randn(seq_len, batch_size, input_size)
            h0 = torch.randn(num_layers * num_directions, batch_size, hidden_size)
            return model, (input, h0)

        batch_size1 = 3
        model1, input1 = get_GruNet_model_and_inputs(7, 3, 2, batch_size1, 5, True)
        self.run_test(model1, input1, do_constant_folding=True)

        batch_size2 = 4
        model2, input2 = get_GruNet_model_and_inputs(5, 4, 3, batch_size2, 7, False)
        self.run_test(model2, input2, do_constant_folding=True)

    @skipIfUnsupportedMinOpsetVersion(8)
    def test_max_tensors(self):
        class MaxModel(torch.nn.Module):
            def forward(self, input, other):
                return torch.max(input, other)

        model = MaxModel()
        x = torch.randn(4, 4, requires_grad=True)
        y = torch.randn(4, 1, requires_grad=True)
        self.run_test(model, (x, y))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_arange_end(self):
        class ArangeScript(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, a):
                return torch.arange(a.size(0), dtype=torch.float).view(-1, 1) + a

        x = torch.randn(3, 4, requires_grad=True)
        outputs = ArangeScript()(x)
        self.run_test(ArangeScript(), x)

        class ArangeModel(torch.nn.Module):
            def forward(self, a):
                return torch.arange(a.size(0), dtype=torch.float).view(-1, 1) + a

        self.run_test(ArangeModel(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_arange_end_notype(self):
        class ArangeScript(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, a):
                return torch.arange(a.size(0))

        x = torch.randn(3, 4, requires_grad=True)
        outputs = ArangeScript()(x)
        self.run_test(ArangeScript(), x)

        class ArangeModel(torch.nn.Module):
            def forward(self, a):
                return torch.arange(a.size(0))

        self.run_test(ArangeModel(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_arange_start_end(self):
        class ArangeScript(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, a):
                return torch.arange(2, a.size(0) + 2, dtype=torch.float).view(-1, 1) + a

        x = torch.randn(3, 4, requires_grad=True)
        self.run_test(ArangeScript(), x)

        class ArangeModel(torch.nn.Module):
            def forward(self, a):
                return torch.arange(2, a.size(0) + 2, dtype=torch.float).view(-1, 1) + a

        self.run_test(ArangeModel(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_arange_start_end_notype(self):
        class ArangeScript(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, a):
                return torch.arange(2.7, a.size(0) + 2).view(-1, 1) + a

        x = torch.randn(3, 4, requires_grad=True)
        self.run_test(ArangeScript(), x)

        class ArangeModel(torch.nn.Module):
            def forward(self, a):
                return torch.arange(2.7, a.size(0) + 2).view(-1, 1) + a

        self.run_test(ArangeModel(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_arange_start_end_step(self):
        class ArangeScript(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, a):
                return torch.arange(2, a.size(0) * a.size(1) + 2, a.size(1), dtype=torch.float).view(-1, 1) + a

        x = torch.randn(3, 4, requires_grad=True)
        self.run_test(ArangeScript(), x)

        class ArangeModel(torch.nn.Module):
            def forward(self, a):
                return torch.arange(2, a.size(0) * a.size(1) + 2, a.size(1), dtype=torch.float).view(-1, 1) + a

        self.run_test(ArangeModel(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_arange_start_end_step_notype(self):
        class ArangeScript(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, a):
                return torch.arange(2.7, a.size(0) * a.size(1) + 2, a.size(1)).view(-1, 1) + a

        x = torch.randn(3, 4, requires_grad=True)
        self.run_test(ArangeScript(), x)

        class ArangeModel(torch.nn.Module):
            def forward(self, a):
                return torch.arange(2.7, a.size(0) * a.size(1) + 2, a.size(1)).view(-1, 1) + a

        self.run_test(ArangeModel(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test__dim_arange(self):
        class DimArange(torch.nn.Module):
            def forward(self, input):
                return torch._dim_arange(input, 1)

        x = torch.ones(5, 6)
        self.run_test(DimArange(), x)

    def test_gt(self):
        class GreaterModel(torch.nn.Module):
            def forward(self, input, other):
                return input > other

        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        y = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.run_test(GreaterModel(), (x, y))

        x = torch.randint(10, (3, 4), dtype=torch.int32)
        y = torch.randint(10, (3, 4), dtype=torch.int32)
        self.run_test(GreaterModel(), (x, y))

    def test_gt_scalar(self):
        class GreaterModel(torch.nn.Module):
            def forward(self, input):
                return input > 1

        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.run_test(GreaterModel(), x)

        x = torch.randint(10, (3, 4), dtype=torch.int32)
        self.run_test(GreaterModel(), x)

    def test_lt(self):
        class LessModel(torch.nn.Module):
            def forward(self, input, other):
                return input > other

        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        y = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.run_test(LessModel(), (x, y))

        x = torch.randint(10, (3, 4), dtype=torch.int32)
        y = torch.randint(10, (3, 4), dtype=torch.int32)
        self.run_test(LessModel(), (x, y))

    def test_matmul(self):
        class MatmulModel(torch.nn.Module):
            def forward(self, input, other):
                return torch.matmul(input, other)

        x = torch.randn(3, 4, requires_grad=True)
        y = torch.randn(4, 5, requires_grad=True)
        self.run_test(MatmulModel(), (x, y))

        x = torch.randint(10, (3, 4))
        y = torch.randint(10, (4, 5))
        self.run_test(MatmulModel(), (x, y))

    def test_matmul_batch(self):
        class MatmulModel(torch.nn.Module):
            def forward(self, input, other):
                return torch.matmul(input, other)

        x = torch.randn(2, 3, 4, requires_grad=True)
        y = torch.randn(2, 4, 5, requires_grad=True)
        self.run_test(MatmulModel(), (x, y))

        x = torch.randint(10, (2, 3, 4))
        y = torch.randint(10, (2, 4, 5))
        self.run_test(MatmulModel(), (x, y))

    def test_view(self):
        class ViewModel(torch.nn.Module):
            def forward(self, input):
                return input.view(4, 24)

        x = torch.randint(10, (4, 2, 3, 4), dtype=torch.int32)
        self.run_test(ViewModel(), x)

    def test_weight_norm(self):
        model = torch.nn.utils.weight_norm(torch.nn.Linear(5, 10), dim=1)
        x = torch.randn(3, 4, 5, requires_grad=True)
        self.run_test(model, x)

        model = torch.nn.utils.weight_norm(torch.nn.Conv1d(1, 1, 3))
        x = torch.randn(1, 1, 5, requires_grad=True)
        self.run_test(model, x)

        model = torch.nn.utils.weight_norm(torch.nn.Conv1d(1, 1, 3), dim=-2)
        x = torch.randn(1, 1, 5, requires_grad=True)
        self.run_test(model, x)

    def test_weight_norm_nodim(self):
        model = torch.nn.utils.weight_norm(torch.nn.Linear(5, 10), dim=None)
        x = torch.randn(3, 4, 5, requires_grad=True)
        self.run_test(model, x)


    def test_flatten(self):
        class FlattenModel(torch.nn.Module):
            def forward(self, input):
                return torch.flatten(input)

        x = torch.randint(10, (1, 2, 3, 4))
        self.run_test(FlattenModel(), x)

    def test_flatten2d(self):
        class FlattenModel(torch.nn.Module):
            def forward(self, input):
                return torch.flatten(input, 1)

        x = torch.randint(10, (1, 2, 3, 4))
        self.run_test(FlattenModel(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_getitem(self):
        class GetItemModel(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x, y, z, ind):
                # this will create prim::ListConstruct(x, y, z) + aten::__getitem__
                arr = [x, y, z]
                return arr[ind]

        x = torch.randn(3, 4, 5)
        y = torch.randn(1, 4, 5)
        z = torch.randn(2, 4, 5)
        ind = torch.tensor(1, dtype=torch.long)
        self.run_test(GetItemModel(), (x, y, z, ind))

        ind = torch.tensor(-2, dtype=torch.long)
        self.run_test(GetItemModel(), (x, y, z, ind))

    def test_unbind(self):
        class UnbindModel(torch.nn.Module):
            def forward(self, input):
                return input.unbind()

        x = torch.randn(3, 4, 5)
        self.run_test(UnbindModel(), x)

        class UnbindModel2(torch.nn.Module):
            def forward(self, input):
                _, out, _, _ = input.unbind(1)
                return out

        x = torch.randn(3, 4, 5)
        self.run_test(UnbindModel2(), x)

        class UnbindModel3(torch.nn.Module):
            def forward(self, input):
                _, out, _, _ = input.unbind(-2)
                return out

        x = torch.randn(3, 4, 5)
        self.run_test(UnbindModel3(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_unbind_dynamic(self):
        class UnbindModel(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, input):
                return input.unbind()[1]

        x = torch.randn(3, 4, 5)
        self.run_test(UnbindModel(), x)

        class UnbindModel2(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, input):
                return input.unbind(-1)[1]

        x = torch.randn(3, 4, 5)
        self.run_test(UnbindModel2(), x)

    def test_split(self):
        class SplitModel(torch.nn.Module):
            def forward(self, input):
                return input.split([2, 1, 2])

        x = torch.randn(5, 4, 3)
        self.run_test(SplitModel(), x)

        class SplitModel2(torch.nn.Module):
            def forward(self, input):
                return input.split([2, 1, 1], -2)

        x = torch.randn(5, 4, 3)
        self.run_test(SplitModel2(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_split_dynamic(self):
        class SplitModel(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, input):
                return input.split(2)[1]

        x = torch.randn(5, 4, 3)
        self.run_test(SplitModel(), x)

        class SplitModel2(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, input):
                return input.split(2, -3)[1]

        x = torch.randn(5, 4, 3)
        self.run_test(SplitModel2(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_tensor_factories(self):
        class TensorFactory(torch.nn.Module):
            def forward(self, x):
                return torch.zeros(x.size()) + torch.ones(x.size())

        x = torch.randn(2, 3, 4)
        self.run_test(TensorFactory(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_tensor_factories_script(self):
        class TensorFactory(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x):
                return torch.zeros(x.shape, dtype=torch.float) + torch.ones(x.shape, dtype=torch.float)

        x = torch.randn(2, 3, 4)
        self.run_test(TensorFactory(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_tensor_like_factories_script(self):
        class TensorFactory(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x):
                zeros = torch.zeros_like(x, dtype=torch.float, layout=torch.strided, device=torch.device('cpu'))
                ones = torch.ones_like(x, dtype=torch.float, layout=torch.strided, device=torch.device('cpu'))
                return zeros + ones

        x = torch.randn(2, 3, 4)
        self.run_test(TensorFactory(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_inplace_zero(self):
        class Zero_(torch.nn.Module):
            def forward(self, x):
                return x.zero_(), x

        x = torch.randn(2, 3, 4)
        self.run_test(Zero_(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_inplace_fill(self):
        class Fill_(torch.nn.Module):
            def forward(self, x):
                return x.fill_(3), x

        x = torch.randn(2, 3, 4)
        self.run_test(Fill_(), x)

    def test_inplace_arithmetic(self):
        class Arithmetic(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x, y):
                x.add_(3)
                y.mul_(x)
                return x, y

        x = torch.randn(2, 3, 4)
        y = torch.randn(2, 3, 4)
        self.run_test(Arithmetic(), (x, y))

    def test_sort(self):
        class SortModel(torch.nn.Module):
            def __init__(self, dim):
                super(SortModel, self).__init__()
                self.dim = dim

            def forward(self, x):
                return torch.sort(x, dim=self.dim, descending=True)

        dim = 1
        x = torch.randn(3, 4)
        self.run_test(SortModel(dim), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_sort_ascending(self):
        class SortModel(torch.nn.Module):
            def __init__(self, dim):
                super(SortModel, self).__init__()
                self.dim = dim

            def forward(self, x):
                return torch.sort(x, dim=self.dim, descending=False)

        dim = 1
        x = torch.randn(3, 4)
        self.run_test(SortModel(dim), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_masked_fill(self):
        class MaskedFillModel(torch.nn.Module):
            def forward(self, x):
                mask = torch.tensor([[0, 0, 1], [1, 1, 0]], dtype=torch.uint8)
                return x.masked_fill(mask, 2)

        x = torch.zeros(4, 2, 3, requires_grad=True)
        self.run_test(MaskedFillModel(), x)

        class MaskedFillModel2(torch.nn.Module):
            def forward(self, x):
                return x.masked_fill(x > 3, -1)

        x = torch.arange(16).view(2, 2, 4).to(torch.float32)
        self.run_test(MaskedFillModel2(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_masked_scatter(self):
        class MaskedScatterModel(torch.nn.Module):
            def forward(self, x):
                return torch.masked_scatter(x, x.ge(0.5), torch.ones(100, 100) * 5)

        x = torch.randn(3, 4, 5, requires_grad=True)
        self.run_test(MaskedScatterModel(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_masked_select(self):
        class MaskedSelectModel(torch.nn.Module):
            def forward(self, x):
                return torch.masked_select(x, x.ge(0.5))

        x = torch.randn(3, 4, 5, requires_grad=True)
        self.run_test(MaskedSelectModel(), x)

    @unittest.skip("Enable this once depthToSpace attr 'mode' is supported in ORT")
    @skipIfUnsupportedMinOpsetVersion(9)
    def test_pixel_shuffle(self):
        class PixelShuffle(torch.nn.Module):
            def forward(self, x):
                return torch.pixel_shuffle(x, upscale_factor=2)

        x = torch.randn(2, 16, 4, 3, requires_grad=True)
        self.run_test(PixelShuffle(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_scalar_type(self):
        class ArithmeticModel(torch.nn.Module):
            def forward(self, x):
                return x.size(0) * 2 * x

        x = torch.ones(2, 3, dtype=torch.float32)
        self.run_test(ArithmeticModel(), x)

        class ReciprocalModel(torch.nn.Module):
            def forward(self, x):
                return torch.reciprocal(x)

        x = torch.tensor([2.0, 4.0], dtype=torch.double)
        self.run_test(ReciprocalModel(), x)

        class ComparisonModel(torch.nn.Module):
            def forward(self, x, y):
                return x.ge(0.5) & y.le(2)

        x = torch.ones(2, 3, dtype=torch.int32)
        y = torch.ones(2, 3, dtype=torch.float32)
        self.run_test(ComparisonModel(), (x, y))

        class MatMulModel(torch.nn.Module):
            def forward(self, x):
                return (torch.mm(x, x) + x + torch.mm(x, x) + x)

        x = torch.ones(3, 3)
        self.run_test(MatMulModel(), x)

        class AddMMModel(torch.nn.Module):
            def forward(self, x):
                return torch.mm(x, x) + x

        x = torch.ones(3, 3)
        self.run_test(AddMMModel(), x)

        class FullModel(torch.nn.Module):
            # add is used for exporting full
            def forward(self, x):
                return torch.full((3, 4), x)
        x = torch.tensor(12)
        self.run_test(FullModel(), x)

    def test_frobenius_norm(self):
        class NormModel(torch.nn.Module):
            def forward(self, x):
                return torch.norm(x, p="fro", dim=0, keepdim=False)

        x = torch.randn(4, 2, 3, requires_grad=True)
        self.run_test(NormModel(), x)

    def test_frobenius_norm_keepdim(self):
        class NormModel(torch.nn.Module):
            def forward(self, x):
                return torch.norm(x, p="fro", dim=(0, 1), keepdim=True)

        x = torch.randn(4, 2, 3, requires_grad=True)
        self.run_test(NormModel(), x)

    def test_unfold(self):
        class UnfoldModel(torch.nn.Module):
            def forward(self, x):
                return x.unfold(dimension=2, size=2, step=2)

        x = torch.randn(4, 2, 3, requires_grad=True)
        self.run_test(UnfoldModel(), x)

    def test_remainder(self):
        class RemainderModel(torch.nn.Module):
            def forward(self, input, other):
                return torch.remainder(input, other)

        x = torch.randn(4, 2, 3)
        y = torch.randn(1, 2, 1)
        self.run_test(RemainderModel(), (x, y))

    def test_remainder_scalar(self):
        class RemainderModel(torch.nn.Module):
            def forward(self, input):
                return torch.remainder(input, 2.55)

        x = torch.randint(10, (2, 3))
        self.run_test(RemainderModel(), x)

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_fmod(self):
        class FModModel(torch.nn.Module):
            def forward(self, input, other):
                return torch.fmod(input, other)

        x = torch.randn(4, 2, 3)
        y = torch.randn(1, 2, 1)
        self.run_test(FModModel(), (x, y))

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_fmod_scalar(self):
        class FModModel(torch.nn.Module):
            def forward(self, input):
                return torch.fmod(input, 2.55)

        x = torch.randint(10, (2, 3))
        self.run_test(FModModel(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_gelu(self):
        class GeluModel(torch.nn.Module):
            def forward(self, x):
                return torch.nn.functional.gelu(x)

        x = torch.randn(2, 4, 5, 6, requires_grad=True)
        self.run_test(GeluModel(), x)

    def test_add_inplace(self):
        class InplaceAddModel(torch.nn.Module):
            def forward(self, x):
                x += 12
                return x

        x = torch.randn(4, 2, 3, requires_grad=True)
        self.run_test(InplaceAddModel(), x)

    def test_rsqrt(self):
        class RsqrtModel(torch.nn.Module):
            def forward(self, x):
                return x.rsqrt()

        x = torch.randn(4, 2, 3, requires_grad=True, dtype=torch.float64)
        self.run_test(RsqrtModel(), x)

    def test_rsqrt_zeros(self):
        class RsqrtModel(torch.nn.Module):
            def forward(self, x):
                return x.rsqrt()
        x = torch.zeros(4, 2, 3, requires_grad=True, dtype=torch.float64)
        self.run_test(RsqrtModel(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_unique(self):
        class UniqueModel(torch.nn.Module):
            def forward(self, x):
                return torch.unique(x, sorted=True, return_inverse=False, return_counts=True)

        x = torch.tensor([1, 3, 2, 3], dtype=torch.long)
        self.run_test(UniqueModel(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_unique_along_dim(self):
        class UniqueModel(torch.nn.Module):
            def forward(self, x):
                return torch.unique(x, dim=0, sorted=True, return_inverse=True, return_counts=False)

        x = torch.tensor([1, 3, 2, 3], dtype=torch.long)
        self.run_test(UniqueModel(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_cumsum(self):
        class CumSum(torch.nn.Module):
            def forward(self, input):
                return torch.cumsum(input, dim=0)
        x = torch.randn(2, 3, 4)
        model = CumSum()
        self.run_test(model, x)

    @skipIfUnsupportedMinOpsetVersion(8)
    def test_meshgrid(self):
        class Meshgrid(torch.nn.Module):
            def forward(self, x, y, z):
                return torch.meshgrid(x, y, z)

        x = torch.randn(3, requires_grad=True)
        y = torch.zeros(4, requires_grad=True)
        z = torch.randn(5, requires_grad=True)
        self.run_test(Meshgrid(), (x, y, z))

    @skipIfUnsupportedMinOpsetVersion(8)
    def test_meshgrid_scalar(self):
        class Meshgrid(torch.nn.Module):
            def forward(self, x, y, z):
                return torch.meshgrid(x, y, z)

        x = torch.ones(3, requires_grad=True)
        y = torch.zeros(4, requires_grad=True)
        z = torch.tensor(2.0)
        self.run_test(Meshgrid(), (x, y, z))

    def test_baddbmm(self):
        class MyModule(torch.nn.Module):
            def forward(self, input, batch1, batch2):
                return torch.baddbmm(input, batch1, batch2, alpha=torch.tensor(5), beta=3.5)
        x = torch.randn(10, 3, 5)
        batch1 = torch.randn(10, 3, 4)
        batch2 = torch.randn(10, 4, 5)
        model = MyModule()
        self.run_test(model, (x, batch1, batch2))

    def test_baddbmm_dynamic(self):
        class MyModule(torch.nn.Module):
            def forward(self, input, batch1, batch2, alpha, beta):
                return torch.baddbmm(input, batch1, batch2, alpha=alpha, beta=beta)
        x = torch.randn(10, 3, 5)
        batch1 = torch.randn(10, 3, 4)
        batch2 = torch.randn(10, 4, 5)
        alpha = torch.tensor(5)
        beta = torch.tensor(3.5)
        model = MyModule()
        self.run_test(model, (x, batch1, batch2, alpha, beta))

    def test_log(self):
        class Log(torch.nn.Module):
            def forward(self, input):
                return torch.log(input)
        x = torch.rand(2, 3, 4)
        model = Log()
        self.run_test(model, x)

    def test_log1p(self):
        class Log1p(torch.nn.Module):
            def forward(self, input):
                return torch.log1p(input)
        x = torch.rand(2, 3, 4)
        model = Log1p()
        self.run_test(model, x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_round(self):
        class Round(torch.nn.Module):
            def forward(self, x):
                return torch.round(x)

        x = torch.tensor([0.9920, -1.0362, -1.5000, 3.5000], requires_grad=True)
        self.run_test(Round(), x)

    def test_constant_pad(self):
        model = torch.nn.ConstantPad1d(2, 3.5)
        x = torch.randn(2, 4, 4)
        self.run_test(model, x)

        model = torch.nn.ConstantPad2d((3, 0, 2, 1), 3.5)
        x = torch.randn(2, 2, 4, 4)
        self.run_test(model, x)

    # Dynamic padding is added in opset 11
    @skipIfUnsupportedMinOpsetVersion(11)
    def test_pad_types(self):
        # Test for different pad integer types
        class Pad(torch.nn.Module):
            def forward(self, x, pad):
                return torch.nn.functional.pad(x, pad)

        x = torch.randn(2, 2, 4, 4)
        y = pad = (torch.tensor(2, dtype=torch.int32), torch.tensor(4, dtype=torch.int32))
        self.run_test(Pad(), (x, y))

        y = pad = (torch.tensor(2, dtype=torch.int64), torch.tensor(4, dtype=torch.int64))
        self.run_test(Pad(), (x, y))


    def test_reflection_pad(self):
        model = torch.nn.ReflectionPad1d(2)
        x = torch.randn(2, 4, 4)
        self.run_test(model, x)

        model = torch.nn.ReflectionPad2d((3, 0, 2, 1))
        x = torch.randn(2, 2, 4, 4)
        self.run_test(model, x)

    def test_replication_pad(self):
        model = torch.nn.ReplicationPad1d(2)
        x = torch.randn(2, 4, 4)
        self.run_test(model, x)

        model = torch.nn.ReplicationPad2d((3, 0, 2, 1))
        x = torch.randn(2, 2, 4, 4)
        self.run_test(model, x)

    @skipIfNoLapack
    @skipIfUnsupportedMinOpsetVersion(11)
    def test_det(self):
        class Det(torch.nn.Module):
            def forward(self, x):
                return torch.det(x)

        x = torch.randn(2, 3, 5, 5)
        self.run_test(Det(), x)

    # This test checks output scalar type in the ONNX graph should not be null
    # https://github.com/pytorch/pytorch/issues/28607
    @skipIfUnsupportedMinOpsetVersion(10)
    def test_trace_script(self):
        @torch.jit.script
        def center_slice_helper(input, h_offset):
            return input[:, h_offset:]

        class CenterCrop(torch.nn.Module):
            def forward(self, input):
                return center_slice_helper(input, torch.tensor(input.shape[1] - 1))

        x = torch.randn(3, 4)
        self.run_test(CenterCrop(), x)

    @skipIfNoLapack
    @skipIfUnsupportedMinOpsetVersion(11)
    def test_logdet(self):
        class LogDet(torch.nn.Module):
            def forward(self, x):
                return torch.logdet(x)

        x = torch.randn(2, 3, 5, 5)
        self.run_test(LogDet(), x)

    def _dispatch_rnn_test(self, name, *args, **kwargs):
        if name == 'elman':
            self._elman_rnn_test(*args, **kwargs)
        if name == 'lstm':
            self._lstm_test(*args, **kwargs)
        if name == 'gru':
            self._gru_test(*args, **kwargs)

    def _elman_rnn_test(self, layers, nonlinearity, bidirectional,
                        initial_state, packed_sequence, dropout):
        batch_first = True if packed_sequence == 2 else False
        model = torch.nn.RNN(RNN_INPUT_SIZE, RNN_HIDDEN_SIZE, layers, nonlinearity=nonlinearity,
                             bidirectional=bidirectional, dropout=dropout, batch_first=batch_first)

        if packed_sequence == 1:
            model = RnnModelWithPackedSequence(model, False)
        if packed_sequence == 2:
            model = RnnModelWithPackedSequence(model, True)

        def make_input(batch_size):
            seq_lengths = np.random.randint(1, RNN_SEQUENCE_LENGTH + 1, size=batch_size)
            seq_lengths = list(reversed(sorted(map(int, seq_lengths))))
            inputs = [torch.randn(l, RNN_INPUT_SIZE) for l in seq_lengths]
            inputs = rnn_utils.pad_sequence(inputs, batch_first=batch_first)
            inputs = [inputs]

            directions = 2 if bidirectional else 1

            if initial_state:
                h0 = torch.randn(directions * layers, batch_size, RNN_HIDDEN_SIZE)
                inputs.append(h0)
            if packed_sequence != 0:
                inputs.append(torch.IntTensor(seq_lengths))
            if len(inputs) == 1:
                input = inputs[0]
            else:
                input = tuple(inputs)
            return input

        input = make_input(RNN_BATCH_SIZE)
        self.run_test(model, input, batch_size=RNN_BATCH_SIZE)

        # test that the model still runs with a different batch size
        other_input = make_input(RNN_BATCH_SIZE + 1)
        self.run_test(model, other_input, batch_size=RNN_BATCH_SIZE + 1)

    def _lstm_test(self, layers, bidirectional, initial_state,
                   packed_sequence, dropout):
        batch_first = True if packed_sequence == 2 else False
        model = LstmFlatteningResult(
            RNN_INPUT_SIZE, RNN_HIDDEN_SIZE, layers,
            bidirectional=bidirectional, dropout=dropout, batch_first=batch_first)
        if packed_sequence == 1:
            model = RnnModelWithPackedSequence(model, False)
        if packed_sequence == 2:
            model = RnnModelWithPackedSequence(model, True)

        def make_input(batch_size):
            seq_lengths = np.random.randint(1, RNN_SEQUENCE_LENGTH + 1, size=batch_size)
            seq_lengths = list(reversed(sorted(map(int, seq_lengths))))
            inputs = [torch.randn(l, RNN_INPUT_SIZE) for l in seq_lengths]
            inputs = rnn_utils.pad_sequence(inputs, batch_first=batch_first)
            inputs = [inputs]

            directions = 2 if bidirectional else 1

            if initial_state:
                h0 = torch.randn(directions * layers, batch_size, RNN_HIDDEN_SIZE)
                c0 = torch.randn(directions * layers, batch_size, RNN_HIDDEN_SIZE)
                inputs.append((h0, c0))
            if packed_sequence != 0:
                inputs.append(torch.IntTensor(seq_lengths))
            if len(inputs) == 1:
                input = inputs[0]
            else:
                input = tuple(inputs)
            return input

        input = make_input(RNN_BATCH_SIZE)
        self.run_test(model, input, batch_size=RNN_BATCH_SIZE)

        # test that the model still runs with a different batch size
        other_input = make_input(RNN_BATCH_SIZE + 1)
        self.run_test(model, other_input, batch_size=RNN_BATCH_SIZE + 1)

    def _gru_test(self, layers, bidirectional, initial_state,
                  packed_sequence, dropout):
        batch_first = True if packed_sequence == 2 else False
        model = torch.nn.GRU(RNN_INPUT_SIZE, RNN_HIDDEN_SIZE, layers, bidirectional=bidirectional, dropout=dropout,
                             batch_first=batch_first)
        if packed_sequence == 1:
            model = RnnModelWithPackedSequence(model, False)
        if packed_sequence == 2:
            model = RnnModelWithPackedSequence(model, True)

        def make_input(batch_size):
            seq_lengths = np.random.randint(1, RNN_SEQUENCE_LENGTH + 1, size=batch_size)
            seq_lengths = list(reversed(sorted(map(int, seq_lengths))))
            inputs = [torch.randn(l, RNN_INPUT_SIZE) for l in seq_lengths]
            inputs = rnn_utils.pad_sequence(inputs, batch_first=batch_first)
            inputs = [inputs]

            directions = 2 if bidirectional else 1

            if initial_state:
                h0 = torch.randn(directions * layers, batch_size, RNN_HIDDEN_SIZE)
                inputs.append(h0)
            if packed_sequence != 0:
                inputs.append(torch.IntTensor(seq_lengths))
            if len(inputs) == 1:
                input = inputs[0]
            else:
                input = tuple(inputs)
            return input

        input = make_input(RNN_BATCH_SIZE)
        self.run_test(model, input, batch_size=RNN_BATCH_SIZE)

        # test that the model still runs with a different batch size
        other_input = make_input(RNN_BATCH_SIZE + 1)
        self.run_test(model, other_input, batch_size=RNN_BATCH_SIZE + 1)


def make_test(name, base, layer, bidirectional, initial_state,
              variable_length, dropout,
              **extra_kwargs):
    test_name = str('_'.join([
        'test', name, layer[1],
        bidirectional[1], initial_state[1],
        variable_length[1], dropout[1]
    ]))

    # Cannot export with older opsets because of 'ConstantFill' op
    # ConstantFill was a temp op removed at opset 8. This is no longer supported by onnxruntime
    @skipIfUnsupportedMinOpsetVersion(9)
    def f(self):
        self._dispatch_rnn_test(
            base,
            layers=layer[0],
            bidirectional=bidirectional[0],
            initial_state=initial_state[0],
            packed_sequence=variable_length[0],
            dropout=dropout[0],
            **extra_kwargs)

    f.__name__ = test_name
    setattr(TestONNXRuntime, f.__name__, f)


def setup_rnn_tests():
    layers_opts = [
        (1, 'unilayer'),
        (3, 'trilayer')
    ]
    bidirectional_opts = [
        (False, 'forward'),
        (True, 'bidirectional')
    ]
    initial_state_opts = [
        (True, 'with_initial_state'),
        (False, 'no_initial_state')
    ]
    variable_length_opts = [
        (0, 'without_sequence_lengths'),
        (1, 'with_variable_length_sequences'),
        (2, 'with_batch_first_sequence_lengths')
    ]
    dropout_opts = [
        (0.2, 'with_dropout'),
        (0.0, 'without_dropout')
    ]
    test_count = 0
    for (layer, bidirectional, initial_state, variable_length, dropout) in \
        itertools.product(
            layers_opts,
            bidirectional_opts,
            initial_state_opts,
            variable_length_opts,
            dropout_opts,
    ):

        for base, name, extra_kwargs in (
                ('elman', 'elman_relu', {'nonlinearity': u'relu'}),
                ('elman', 'elman_tanh', {'nonlinearity': u'tanh'}),
                ('lstm', 'lstm', {}),
                ('gru', 'gru', {})
        ):
            # This is a hack to skip elman_rnn bidirectional tests for now
            # TODO: Revert this once elman_rnn bidirectional issue is fixed
            if base == 'elman' and bidirectional[1] == 'bidirectional':
                continue
            make_test(name, base, layer, bidirectional, initial_state,
                      variable_length, dropout,
                      **extra_kwargs)
            test_count += 1

    # sanity check that a representative example does exist
    TestONNXRuntime.test_gru_trilayer_forward_with_initial_state_without_sequence_lengths_with_dropout

    # make sure no one accidentally disables all the tests without
    # noticing
    # assert test_count == 192, test_count
    # TODO: Revert this once elman_rnn bidirectional issue is fixed
    if test_count != 144:
        raise ValueError('Expected 144 tests but found {}'.format(test_count))


setup_rnn_tests()


# opset 7 tests
TestONNXRuntime_opset7 = type(str("TestONNXRuntime_opset7"),
                              (unittest.TestCase,),
                              dict(TestONNXRuntime.__dict__, opset_version=7))

# opset 8 tests
TestONNXRuntime_opset8 = type(str("TestONNXRuntime_opset8"),
                              (unittest.TestCase,),
                              dict(TestONNXRuntime.__dict__, opset_version=8))


# opset 10 tests
TestONNXRuntime_opset10 = type(str("TestONNXRuntime_opset10"),
                               (unittest.TestCase,),
                               dict(TestONNXRuntime.__dict__, opset_version=10))

# opset 11 tests
TestONNXRuntime_opset11 = type(str("TestONNXRuntime_opset11"),
                               (unittest.TestCase,),
                               dict(TestONNXRuntime.__dict__, opset_version=11))


# opset 9 tests, with keep_initializers_as_inputs=False for 
# IR version 4 style export.
TestONNXRuntime_opset9_IRv4 = type(str("TestONNXRuntime_opset9_IRv4"),
                                   (unittest.TestCase,),
                                   dict(TestONNXRuntime.__dict__,
                                   keep_initializers_as_inputs=False))


# opset 10 tests, with keep_initializers_as_inputs=False for 
# IR version 4 style export.
TestONNXRuntime_opset10_IRv4 = type(str("TestONNXRuntime_opset10_IRv4"),
                                    (unittest.TestCase,),
                                    dict(TestONNXRuntime.__dict__, opset_version=10,
                                    keep_initializers_as_inputs=False))


# opset 11 tests, with keep_initializers_as_inputs=False for 
# IR version 4 style export.
TestONNXRuntime_opset11_IRv4 = type(str("TestONNXRuntime_opset11_IRv4"),
                                    (unittest.TestCase,),
                                    dict(TestONNXRuntime.__dict__, opset_version=11,
                                    keep_initializers_as_inputs=False))


if __name__ == '__main__':
    unittest.main()
