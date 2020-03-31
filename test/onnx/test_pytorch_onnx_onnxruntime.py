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
from test_pytorch_common import (skipIfUnsupportedMinOpsetVersion, enableScriptTest,
                                 skipIfNoLapack)
from test_pytorch_common import BATCH_SIZE
from test_pytorch_common import RNN_BATCH_SIZE, RNN_SEQUENCE_LENGTH, RNN_INPUT_SIZE, RNN_HIDDEN_SIZE
import model_defs.word_language_model as word_language_model
import torchvision
import onnx


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

        # if additional test inputs are provided run the onnx
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
        self.is_script_test_enabled = False

    def run_test(self, model, input, rtol=1e-3, atol=1e-7, do_constant_folding=True,
                 batch_size=2, use_gpu=True, dynamic_axes=None, test_with_inputs=None,
                 input_names=None, output_names=None, fixed_batch_size=False):
        def _run_test(m):
            return run_model_test(self, m, batch_size=batch_size,
                                  input=input, use_gpu=use_gpu, rtol=rtol, atol=atol,
                                  do_constant_folding=do_constant_folding,
                                  dynamic_axes=dynamic_axes, test_with_inputs=test_with_inputs,
                                  input_names=input_names, output_names=output_names,
                                  fixed_batch_size=fixed_batch_size)
        if self.is_script_test_enabled:
            script_model = torch.jit.script(model)
            _run_test(script_model)
        _run_test(model)

    def run_model_test_with_external_data(self, model, input, rtol=0.001, atol=1e-7,
                                          example_outputs=None, do_constant_folding=True,
                                          dynamic_axes=None, input_names=None, output_names=None,
                                          ort_optim_on=True):
        import os
        import tempfile

        model.eval()
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
            with tempfile.TemporaryDirectory() as tmpdirname:
                model_file_name = os.path.join(tmpdirname, 'model.onnx')
                input_copy = copy.deepcopy(input)
                torch.onnx.export(model, input_copy, model_file_name,
                                  opset_version=self.opset_version,
                                  example_outputs=output,
                                  verbose=False,
                                  do_constant_folding=do_constant_folding,
                                  keep_initializers_as_inputs=self.keep_initializers_as_inputs,
                                  dynamic_axes=dynamic_axes,
                                  input_names=input_names, output_names=output_names,
                                  use_external_data_format=True)
                # compute onnxruntime output prediction
                ort_sess_opt = onnxruntime.SessionOptions()
                ort_sess_opt.graph_optimization_level = \
                    onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED if ort_optim_on else \
                    onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
                ort_sess = onnxruntime.InferenceSession(model_file_name, sess_options=ort_sess_opt)
                input_copy = copy.deepcopy(input)
                ort_test_with_input(ort_sess, input_copy, output, rtol, atol)


    @skipIfUnsupportedMinOpsetVersion(9)  # Because external data format was released with Opset 9.
    def test_embedding_model_with_external_data(self):
        class LargeModel(torch.nn.Module):
            def __init__(self):
                super(LargeModel, self).__init__()
                dim = 15
                n = 4 * 100
                self.emb = torch.nn.Embedding(n, dim)
                self.lin1 = torch.nn.Linear(dim, 1)
                self.seq = torch.nn.Sequential(
                    self.emb,
                    self.lin1,
                )

            def forward(self, input):
                return self.seq(input)

        model = LargeModel()
        x = torch.tensor([2], dtype=torch.long)
        self.run_model_test_with_external_data(model, x)

    @skipIfUnsupportedMinOpsetVersion(9)  # Because external data format was released with Opset 9.
    def test_mobilenet_v2_with_external_data(self):
        model = torchvision.models.mobilenet_v2(pretrained=True)
        x = torch.randn(2, 3, 224, 224, requires_grad=True)
        # We are turning off Onnx Runtime optimization off in this test,
        # because external data format is not supported to in ORT optimizer.
        # Once that support is added, we can set ort_optim_on=True (default).
        self.run_model_test_with_external_data(model, x, rtol=1e-3, atol=1e-5,
                                               ort_optim_on=False)

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

    def test_reshape_constant_fold(self):
        class Reshape(torch.nn.Module):
            def __init__(self, ):
                super(Reshape, self).__init__()
                self.register_buffer("weight", torch.ones(5))

            def forward(self, x):
                scale_1 = self.weight.reshape(1, -1, 1, 1)
                return x * scale_1

        x = torch.randn(4, 5)
        self.run_test(Reshape(), (x,), rtol=1e-3, atol=1e-5)

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

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_faster_rcnn(self):
        model = torchvision.models.detection.faster_rcnn.fasterrcnn_resnet50_fpn(pretrained=True, min_size=200,
                                                                                 max_size=300)
        model.eval()
        x = torch.randn(2, 3, 200, 300, requires_grad=True)
        self.run_test(model, (x,), rtol=1e-3, atol=1e-5)

    def get_image_from_url(self, url):
        import sys
        import os
        if sys.version_info < (3,):
            from urlparse import urlsplit
            import urllib2
            request = urllib2
        else:
            from urllib.parse import urlsplit
            from urllib import request
        from PIL import Image
        from torchvision import transforms
        from torch._utils_internal import get_writable_path

        filename = os.path.basename(urlsplit(url)[2])
        data_dir = get_writable_path(os.path.join(os.path.dirname(__file__)))
        path = os.path.join(data_dir, filename)
        data = request.urlopen(url, timeout=15).read()
        with open(path, 'wb') as f:
            f.write(data)
        image = Image.open(path).convert("RGB")
        image = image.resize((300, 200), Image.BILINEAR)
        to_tensor = transforms.ToTensor()
        return to_tensor(image)

    def get_test_images(self):
        image_url = "http://farm3.staticflickr.com/2469/3915380994_2e611b1779_z.jpg"
        image = self.get_image_from_url(url=image_url)
        images = [image]
        return images

    @skipIfUnsupportedMinOpsetVersion(11)
    @unittest.skip("disabled due to removal of aten::__interpolate")
    def test_mask_rcnn(self):
        model = torchvision.models.detection.mask_rcnn.maskrcnn_resnet50_fpn(pretrained=True, min_size=200,
                                                                             max_size=300)
        images = self.get_test_images()
        self.run_test(model, (images,), rtol=1e-3, atol=1e-5)

    @skipIfUnsupportedMinOpsetVersion(11)
    @unittest.skip("Disabled w removal of aten::__interpolate")
    def test_keypoint_rcnn(self):
        class KeyPointRCNN(torch.nn.Module):
            def __init__(self):
                super(KeyPointRCNN, self).__init__()
                self.model = torchvision.models.detection.keypoint_rcnn.keypointrcnn_resnet50_fpn(pretrained=True,
                                                                                                  min_size=200,
                                                                                                  max_size=300)

            def forward(self, images):
                output = self.model(images)
                # TODO: The keypoints_scores require the use of Argmax that is updated in ONNX.
                #       For now we are testing all the output of KeypointRCNN except keypoints_scores.
                #       Enable When Argmax is updated in ONNX Runtime.
                return output[0]['boxes'], output[0]['labels'], output[0]['scores'], output[0]['keypoints']
        images = self.get_test_images()
        self.run_test(KeyPointRCNN(), (images,), rtol=1e-3, atol=1e-5)

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

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_index_mask(self):
        self._test_index_generic(lambda input: input[torch.tensor([0, 1, 0], dtype=torch.uint8)])
        self._test_index_generic(lambda input: input[torch.tensor([0, 1, 0], dtype=torch.bool)])

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

    def test_maxpool_default_stride(self):
        class MaxPoolModel(torch.nn.Module):
            def forward(self, x):
                return torch.nn.functional.max_pool2d(x, 2)

        model = MaxPoolModel()
        x = torch.randn(10, 20, 16, 50)
        self.run_test(model, x)

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

    def test_avgpool_default_stride(self):
        class AvgPoolModel(torch.nn.Module):
            def forward(self, x):
                return torch.nn.functional.avg_pool2d(x, 2)

        model = AvgPoolModel()
        x = torch.randn(10, 20, 16, 50)
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

    @enableScriptTest()
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

    def test_floor_div(self):
        class FloorDivModule(torch.nn.Module):
            def forward(self, x, y):
                return x // 3, x // 2., \
                    x.to(dtype=torch.float64) // 3, x.to(dtype=torch.float64) // 2., \
                    x.to(dtype=torch.int64) // 3, x.to(dtype=torch.int64) // 2., \
                    x // (y + 1.).to(dtype=torch.int64), x // y, \
                    x.to(dtype=torch.float64) // y.to(dtype=torch.int64), x.to(dtype=torch.float64) // y.to(dtype=torch.float64), \
                    x.to(dtype=torch.int64) // y.to(dtype=torch.int64), x.to(dtype=torch.int64) // y

        x = torch.randn(2, 3, 4)
        y = torch.arange(1, 2 * 3 * 4 + 1).reshape(2, 3, 4)
        self.run_test(FloorDivModule(), (x, y))

    def test_floor_div_script(self):
        class FloorDivModule(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x, y):
                return x // 3, x // 2., x // y

        x = torch.randn(2, 3, 4)
        y = torch.randn(2, 3, 4)
        self.run_test(FloorDivModule(), (x, y))

    def test_true_div(self):
        class TrueDivModule(torch.nn.Module):
            def forward(self, x, y):
                return torch.true_divide(x, y)

        x = torch.randn(2, 3, 4).to(torch.int)
        y = torch.arange(1, 2 * 3 * 4 + 1).reshape(2, 3, 4).to(torch.int)
        self.run_test(TrueDivModule(), (x, y))
        self.run_test(TrueDivModule(), (x.float(), y))
        self.run_test(TrueDivModule(), (x.to(torch.short), y.to(torch.short)))

    # Note: true_divide cannot (generally) be exported via scripting
    # since its type promotion logic is dependent on knowing the scalar types
    # of the input tensors. That is, the ONNX graph is dependent on the
    # data type of the inputs. This makes it appropriate for tracing only.
    def test_true_div_trace(self):
        class TrueDivModule(torch.nn.Module):
            def forward(self, x, y):
                return torch.true_divide(x, y)

        x = torch.randn(2, 3, 4).to(torch.int)
        y = torch.arange(1, 2 * 3 * 4 + 1).reshape(2, 3, 4).to(torch.int)

        prev_default = torch.get_default_dtype()

        torch.set_default_dtype(torch.float)
        self.run_test(torch.jit.trace(TrueDivModule(), (x, y)), (x, y))

        torch.set_default_dtype(torch.double)
        self.run_test(torch.jit.trace(TrueDivModule(), (x, y)), (x, y))

        torch.set_default_dtype(prev_default)

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
        self.run_test(torch.jit.script(ArangeModel()), x,
                      test_with_inputs=[y], input_names=['input_1'],
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

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_index_put(self):
        class IndexPutModel(torch.nn.Module):
            def forward(self, x, ind, update):
                x[ind] = update
                return x

        x = torch.randn(3, 4)
        ind = torch.tensor([1], dtype=torch.long)
        update = torch.ones(4)
        self.run_test(IndexPutModel(), (x, ind, update))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_index_put_accumulate(self):
        class IndexPutModel(torch.nn.Module):
            def forward(self, x, ind, update):
                return x.index_put((ind, ), update, accumulate=True)

        x = torch.randn(3, 4)
        ind = torch.tensor([2], dtype=torch.long)
        update = torch.ones(4)
        self.run_test(IndexPutModel(), (x, ind, update))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_index_put_slice_index(self):
        class IndexPutModel(torch.nn.Module):
            def forward(self, x, update):
                x[1:2, 1:3, torch.tensor([1])] += update
                return x

        x = torch.randn(3, 4, 5)
        update = torch.tensor([10, 15]).view(1, 2, 1)
        self.run_test(IndexPutModel(), (x, update))

        class IndexPutModel2(torch.nn.Module):
            def forward(self, x, update):
                x[torch.tensor([0, 2]), torch.tensor([1, 2])] += update
                return x

        x = torch.randn(3, 4, 5)
        update = torch.randn(2, 5)
        self.run_test(IndexPutModel2(), (x, update))

        class IndexPutModel3(torch.nn.Module):
            def forward(self, x, update):
                x[torch.tensor([0, 2]), 1:2] += update
                return x

        x = torch.randn(3, 4, 5)
        update = torch.tensor([10, 15]).view(2, 1, 1)
        self.run_test(IndexPutModel3(), (x, update))

        class IndexPutModel4(torch.nn.Module):
            def forward(self, x, update):
                x[torch.tensor([0, 2]), 2] += update
                return x

        x = torch.randn(3, 4, 5)
        update = torch.tensor([10, 15]).view(2, 1)
        self.run_test(IndexPutModel4(), (x, update))

        class IndexPutModel5(torch.nn.Module):
            def forward(self, x, update):
                x[1:3, torch.tensor([0, 2]), 2] += update
                return x

        x = torch.randn(3, 4, 5)
        update = torch.tensor([10, 15]).view(2, 1)
        self.run_test(IndexPutModel5(), (x, update))

        class IndexPutModel6(torch.nn.Module):
            def forward(self, x, update):
                x[1:3, 0] = update
                return x

        x = torch.randn(3, 4, 5)
        update = torch.arange(2 * 5).to(torch.float).view(2, 5)
        self.run_test(IndexPutModel6(), (x, update))

        class IndexPutModel7(torch.nn.Module):
            def forward(self, x, update):
                x[1:, 0] = update
                return x

        x = torch.randn(3, 4, 5)
        update = torch.arange(2 * 5).to(torch.float).view(2, 5)
        self.run_test(IndexPutModel7(), (x, update))

        class IndexPutModel8(torch.nn.Module):
            def forward(self, x, update):
                x[:3, 0] = update
                return x

        x = torch.randn(3, 4, 5)
        update = torch.arange(3 * 5).to(torch.float).view(3, 5)
        self.run_test(IndexPutModel8(), (x, update))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_index_put_ellipsis(self):
        class IndexPutModel(torch.nn.Module):
            def forward(self, x, update):
                x[..., torch.tensor([2, 1, 3]), 2:4] += update
                return x

        x = torch.randn(3, 4, 5, 6, 7)
        update = torch.randn(3, 1, 1, 3, 2)
        self.run_test(IndexPutModel(), (x, update))

        class IndexPutModel2(torch.nn.Module):
            def forward(self, x, update):
                x[2, ..., torch.tensor([2, 1, 3]), 2:4] += update
                return x

        x = torch.randn(3, 4, 5, 6, 7)
        update = torch.randn(4, 1, 3, 2)
        self.run_test(IndexPutModel2(), (x, update))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_copy_(self):
        class CopyModel(torch.nn.Module):
            def forward(self, x, data):
                x[1:3] = data
                return x

        x = torch.randn(3, 4)
        update = torch.randn(2, 4)
        self.run_test(CopyModel(), (x, update))

        # mixed slice and select
        class CopyModel2(torch.nn.Module):
            def forward(self, x, data):
                x[1:3, 0] = data
                return x

        x = torch.randn(3, 4)
        update = torch.tensor([0], dtype=torch.float32)
        self.run_test(CopyModel2(), (x, update))

        update = torch.tensor([2, 3], dtype=torch.float32)
        self.run_test(CopyModel2(), (x, update))

        update = torch.randn(2)
        self.run_test(CopyModel2(), (x, update))

        class CopyModel3(torch.nn.Module):
            def forward(self, x, data):
                x[1, 1:3] = data
                return x

        x = torch.randn(3, 4)
        update = torch.tensor([0], dtype=torch.float32)
        self.run_test(CopyModel3(), (x, update))

        update = torch.tensor([2, 3], dtype=torch.float32)
        self.run_test(CopyModel3(), (x, update))

        update = torch.randn(2)
        self.run_test(CopyModel3(), (x, update))

        update = torch.randn(1, 2)
        self.run_test(CopyModel3(), (x, update))

        class CopyModel4(torch.nn.Module):
            def forward(self, x, ind, data):
                x[ind] = data
                return x

        x = torch.randn(3, 4)
        ind = torch.tensor(2)
        data = torch.randn(4)
        self.run_test(CopyModel4(), (x, ind, data))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_copy_ellipsis(self):
        class CopyModel(torch.nn.Module):
            def forward(self, x, update):
                x[..., 1] = update
                return x

        x = torch.randn(2, 3, 4)
        update = torch.ones(1)
        self.run_test(CopyModel(), (x, update))

        x = torch.randn(2, 3, 4, 5, 6)
        update = torch.ones(1)
        self.run_test(CopyModel(), (x, update))

        class CopyModel2(torch.nn.Module):
            def forward(self, x, update):
                x[2, ..., 1:3] = update
                return x

        x = torch.randn(3, 4, 5, 6)
        update = torch.ones(1)
        self.run_test(CopyModel2(), (x, update))

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
                return torch.mul(x, (torch.randn(2, 3, 4) + x).size(0))

        x = torch.randn(2, 3, 4)
        self.run_test(RandN(), x)

        class Rand(torch.nn.Module):
            def forward(self, x):
                return torch.mul(x, (torch.rand(2, 3, 4) + x).size(0))

        x = torch.randn(2, 3, 4)
        self.run_test(Rand(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_random_dynamic_size(self):
        class RandN(torch.nn.Module):
            def forward(self, x):
                return torch.mul(x, torch.randn(x.size()).size(1))

        x = torch.randn(2, 3, 4)
        self.run_test(RandN(), x)

        class Rand(torch.nn.Module):
            def forward(self, x):
                return torch.mul(x, torch.rand(x.size()).size(1))

        x = torch.randn(2, 3, 4)
        self.run_test(Rand(), x)

    def test_random_like(self):
        class RandNLike(torch.nn.Module):
            def forward(self, x):
                return torch.mul(x, torch.randn_like(x).size(0))

        x = torch.randn(2, 3, 4)
        self.run_test(RandNLike(), x)
        self.run_test(torch.jit.script(RandNLike()), x)

        class RandLike(torch.nn.Module):
            def forward(self, x):
                return torch.mul(x, torch.rand_like(x).size(0))

        x = torch.randn(2, 3, 4)
        self.run_test(RandLike(), x)
        self.run_test(torch.jit.script(RandLike()), x)

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

    def _interpolate(self, x, mode, use_size, is_upsample, align_corners=False):
        class MyModel(torch.nn.Module):
            def forward(self, x):
                scale = 2.3 if is_upsample else 0.5
                if len(x.size()) == 3:
                    scale_array = 2.3
                if len(x.size()) == 4:
                    scale_array = [2.3, 5.1]
                if len(x.size()) == 5:
                    scale_array = [3.3, 2.3, 5.1]
                if use_size:
                    size_array = [int(float(v) * scale) for v in x.size()[2:]]
                    if align_corners:
                        return torch.nn.functional.interpolate(x, mode=mode, size=size_array[0], align_corners=True), \
                            torch.nn.functional.interpolate(x, mode=mode, size=size_array, align_corners=True)
                    return torch.nn.functional.interpolate(x, mode=mode, size=size_array[0]), \
                        torch.nn.functional.interpolate(x, mode=mode, size=size_array)
                if align_corners:
                    return torch.nn.functional.interpolate(x, mode=mode, scale_factor=scale,
                                                           align_corners=True, recompute_scale_factor=False), \
                        torch.nn.functional.interpolate(x, mode=mode, scale_factor=scale_array,
                                                        align_corners=True, recompute_scale_factor=False)
                return torch.nn.functional.interpolate(x, mode=mode,
                                                       scale_factor=scale, recompute_scale_factor=False), \
                    torch.nn.functional.interpolate(x, mode=mode,
                                                    scale_factor=scale_array, recompute_scale_factor=False)

        self.run_test(MyModel(), x)

    def _interpolate_script(self, x, mode, use_size, is_upsample, align_corners=False):
        # test disabled
        return 

        class MyModel(torch.jit.ScriptModule):
            __constants__ = ['mode', 'use_size', 'is_upsample', 'size', 'scale', 'size_array', 'scale_array', 'align_corners']

            def __init__(self, mode, use_size, is_upsample, align_corners):
                super(MyModel, self).__init__()
                self.mode = mode
                self.use_size = use_size
                self.is_upsample = is_upsample
                self.align_corners = align_corners
                self.scale = 2.0 if self.is_upsample else 0.5
                self.size = 24 if self.is_upsample else 2
                if x.dim() == 3:
                    self.scale_array = [2.3]
                    self.size_array = [16]
                elif x.dim() == 4:
                    self.scale_array = [2.3, 3.1]
                    self.size_array = [16, 32]
                else:
                    self.scale_array = [2.3, 3.1, 4.6]
                    self.size_array = [16, 32, 64]

            @torch.jit.script_method
            def forward(self, x):
                if self.use_size:
                    if self.align_corners:
                        return torch.nn.functional.interpolate(x, mode=self.mode, size=self.size, align_corners=True), \
                            torch.nn.functional.interpolate(x, mode=self.mode, size=self.size_array, align_corners=True)
                    return torch.nn.functional.interpolate(x, mode=self.mode, size=self.size), \
                        torch.nn.functional.interpolate(x, mode=self.mode, size=self.size_array)
                if self.align_corners:
                    return torch.nn.functional.interpolate(x, mode=self.mode,
                                                           scale_factor=self.scale, recompute_scale_factor=False), \
                        torch.nn.functional.interpolate(x, mode=self.mode,
                                                        scale_factor=self.scale_array, recompute_scale_factor=False)
                return torch.nn.functional.interpolate(x, mode=self.mode,
                                                       scale_factor=self.scale, recompute_scale_factor=False), \
                    torch.nn.functional.interpolate(x, mode=self.mode,
                                                    scale_factor=self.scale_array, recompute_scale_factor=False)

        model = MyModel(mode, use_size, is_upsample, align_corners)
        self.run_test(model, x, atol=1e-6)

    def _interpolate_tests(self, is_upsample):
        # - cubic mode is not supported for opsets below 11;
        # - linear mode does not match for opsets below 11;
        modes = ["nearest", "linear", "bicubic"]
        if self.opset_version < 11:
            modes = ["nearest"]
        x = [torch.randn(1, 2, 6, requires_grad=True),
             torch.randn(1, 2, 4, 6, requires_grad=True),
             torch.randn(1, 2, 4, 4, 6, requires_grad=True)]

        for mode in modes:
            for xi in x:
                mode_i = mode
                # TODO: enable bicubic downsample when ORT precision loss fixed
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
                # test with align_corners if supported
                if mode != 'nearest':
                    self._interpolate(xi, mode_i, True, is_upsample, True)
                    self._interpolate_script(xi, mode_i, True, is_upsample, True)
                # the following cases, require dynamic sizes/scales,
                # which which is not supported for opset_version < 9
                if self.opset_version >= 9:
                    self._interpolate_script(xi, mode_i, True, is_upsample)
                    self._interpolate(xi, mode_i, False, is_upsample)
                    # test with align_corners if supported
                    if mode != 'nearest':
                        self._interpolate(xi, mode_i, False, is_upsample, True)
                        self._interpolate_script(xi, mode_i, False, is_upsample, True)
                    self._interpolate_script(xi, mode_i, False, is_upsample)

    def test_interpolate_upsample(self):
        self._interpolate_tests(True)

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_interpolate_downsample(self):
        self._interpolate_tests(False)

    @skipIfUnsupportedMinOpsetVersion(11)
    @unittest.skip("Interpolate script NYI")
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

    def test_take(self):
        class TakeModel(torch.nn.Module):
            def forward(self, x, y):
                return torch.take(x, y)

        x = torch.randn(6, 4, 3, 3)
        y = torch.tensor([4, 1, 7, 15, 63])
        self.run_test(TakeModel(), (x, y))

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
    def test_one_hot(self):
        class OneHot(torch.nn.Module):
            def __init__(self, num_classes):
                super().__init__()
                self.num_classes = num_classes

            def forward(self, x):
                return torch.nn.functional.one_hot(x, self.num_classes)

        x = torch.arange(10)
        self.run_test(OneHot(15), (x))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_gather(self):
        class GatherModel(torch.nn.Module):
            def forward(self, input, indices):
                return input.gather(1, indices)

        input = torch.tensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
        indices = torch.tensor([[1, 0], [0, 1], [0, 1]], dtype=torch.int64)
        self.run_test(GatherModel(), input=(input, indices))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_expand(self):
        class ExpandModel(torch.nn.Module):
            def forward(self, input):
                return input.expand(2, 3, -1)

        input = torch.randn(2, 1, 4)
        self.run_test(ExpandModel(), input=(input))

        class ExpandInferDimModel(torch.nn.Module):
            def forward(self, input):
                return input.expand(-1, input.size(0))

        input = torch.randn(3, 1)
        self.run_test(ExpandInferDimModel(), input=(input))

        class ExpandTensorSizeModel(torch.nn.Module):
            def forward(self, input, size):
                return input.expand(size)

        input = torch.randn(3,)
        size = torch.tensor([-1])
        self.run_test(ExpandTensorSizeModel(), input=(input, size))

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

        model = torch.nn.utils.weight_norm(torch.nn.Conv1d(3, 6, 3), name='weight')
        x = torch.randn(3, 3, 5, requires_grad=True)
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

    def test_flatten2d_neg(self):
        class FlattenModel(torch.nn.Module):
            def forward(self, x):
                return torch.flatten(x, 1, -1), torch.flatten(x, 0, -2), torch.flatten(x, 1, -2)

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
    def test_len(self):
        class LenModel(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, input):
                return len(input.unbind()) + input

        x = torch.randn(4, 5)
        self.run_test(LenModel(), x, input_names=['input'], dynamic_axes={'input': {0: 'seq'}},
                      test_with_inputs=(torch.randn(5, 5),))

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
    def test_split_size_as_list(self):
        class SplitModel(torch.nn.Module):
            def forward(self, input):
                out = []
                split_sizes = [input.shape[0] - 1, 1]
                for ob in input.split(split_sizes):
                    out.append(ob)
                return torch.cat(out, dim=0)

        x = torch.randn(5, 4, 3)
        self.run_test(SplitModel(), x)

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

    def test_concat(self):
        class ConcatModel(torch.nn.Module):
            def forward(self, x, y, z):
                return torch.cat((x, y, z))

        x = torch.randn(3, 4, 5)
        y = torch.randn(1, 4, 5)
        z = torch.randn(2, 4, 5)
        self.run_test(ConcatModel(), (x, y, z))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_concat_dynamic(self):
        class ConcatDynamicModel(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x):
                return torch.cat(x.unbind())

        x = torch.randn(4, 5, 6)
        self.run_test(ConcatDynamicModel(), x)

    def test_stack(self):
        class StackModel(torch.nn.Module):
            def forward(self, x, y, z):
                return torch.stack((x, y, z), 1)

        x = torch.randn(3, 4, 5)
        y = torch.randn(3, 4, 5)
        z = torch.randn(3, 4, 5)
        self.run_test(StackModel(), (x, y, z))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_stack_dynamic(self):
        class StackDynamicModel(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x):
                return torch.stack(x.unbind(), 1)

        x = torch.randn(4, 5, 6)
        self.run_test(StackDynamicModel(), x)

    def test_loop_dynamic(self):
        class LoopModel(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x):
                for i in range(x.size(2)):
                    x = x + i
                return x

        model = LoopModel()
        inputs = torch.zeros(1, 2, 3, dtype=torch.long)
        self.run_test(model, inputs)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_loop_nested(self):
        class NestedLoopsModel(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x):
                for i in range(5):
                    a = 0
                    while a < 4:
                        a += 1
                    x = x + a
                return x

        model = NestedLoopsModel()
        inputs = torch.zeros(1, 2, 3, dtype=torch.long)
        self.run_test(model, inputs)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_loop_with_list(self):
        class ListLoopModel(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x):
                res = []
                res1 = []
                arr = x.split([3, 4, 1, 1, 2, 3, 2], 0)
                res2 = torch.zeros(3, 4, dtype=torch.long)
                for i in range(len(arr)):
                    res = res.append(arr[i].sum(0, False))
                    res1 = res1.append(arr[-1 - i].sum(0, False))
                    res2 += 1
                return torch.stack(res), torch.stack(res1), res2

        model = ListLoopModel()
        inputs = torch.randn(16)
        self.run_test(model, inputs)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_list(self):
        class ListModel(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x):
                tensors = x.unbind()
                res = []
                res.append(tensors[0])
                res.append(tensors[1])
                res.pop(1)

                res.insert(0, tensors[1])
                res.append(tensors[2])
                return torch.ones(len(res))

        model = ListModel()
        inputs = torch.randn(16, 1)
        self.run_test(model, inputs)

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
    def test_new_zero(self):
        class Zero_(torch.nn.Module):
            def forward(self, x):
                return x.new_zeros(x.shape[2:])

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
            def forward(self, x):
                out = []
                for i in range(-2, 2):
                    out.append(torch.sort(x, dim=i, descending=True))
                return out

        x = torch.randn(3, 4)
        self.run_test(SortModel(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_sort_ascending(self):
        class SortModel(torch.nn.Module):
            def forward(self, x):
                out = []
                for i in range(-2, 2):
                    out.append(torch.sort(x, dim=i, descending=False))
                return out

        x = torch.randn(3, 4)
        self.run_test(SortModel(), x)

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

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_im2col(self):
        class Unfold(torch.nn.Module):
            def forward(self, input):
                return torch.nn.functional.unfold(input, kernel_size=(10, 15), dilation=2, padding=5, stride=3), \
                    torch.nn.functional.unfold(input, kernel_size=(2, 2), dilation=1, padding=0, stride=3), \
                    torch.nn.functional.unfold(input, kernel_size=(1, 1), dilation=5, padding=2, stride=3)

        x = torch.rand(1, 1, 200, 100)
        self.run_test(Unfold(), x)

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

    def test_dim(self):
        class DimModel(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, input):
                out = input * 2
                out *= out.dim()
                return out
        empty_input = torch.randn(0, requires_grad=True)
        multi_dim_input = torch.randn(1, 2, 3, requires_grad=True)
        self.run_test(DimModel(), empty_input)
        self.run_test(DimModel(), multi_dim_input)

    @unittest.skip("Enable this once einsum supported in ORT")
    @skipIfUnsupportedMinOpsetVersion(12)
    def test_einsum(self):
        class EinsumModelBatchDiagonal(torch.nn.Module):
            def forward(self, *tensor_list):
                eqn = '...ii ->...i'
                return torch.einsum(eqn, *tensor_list)

        x = torch.randn(3, 5, 5)
        self.run_test(EinsumModelBatchDiagonal(), input=(x,))

        class EinsumModelBatchMatmul(torch.nn.Module):
            def forward(self, *tensor_list):
                eqn = 'bij, bjk -> bik'
                return torch.einsum(eqn, *tensor_list)

        x = torch.randn(5, 2, 3)
        y = torch.randn(5, 3, 4)
        self.run_test(EinsumModelBatchMatmul(), input=(x, y))

        class EinsumModelInnerProd(torch.nn.Module):
            def forward(self, *tensor_list):
                eqn = 'i,i'
                return torch.einsum(eqn, *tensor_list)

        x = torch.randn(5)
        y = torch.randn(5)
        self.run_test(EinsumModelInnerProd(), input=(x, y))

        class EinsumModelTranspose(torch.nn.Module):
            def forward(self, *tensor_list):
                eqn = 'ij->ji'
                return torch.einsum(eqn, *tensor_list)

        x = torch.randn(3, 4)
        self.run_test(EinsumModelTranspose(), input=(x,))

    def test_empty_branch(self):
        class EmptyBranchModel(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, input):
                out = input + 1
                if out.dim() > 2:
                    if out.dim() > 3:
                        out += 3
                    else:
                        pass
                else:
                    pass
                return out
        x = torch.randn(1, 2, 3, requires_grad=True)
        self.run_test(EmptyBranchModel(), x)

    @unittest.skip("Enable this once ORT version is updated")
    @skipIfUnsupportedMinOpsetVersion(12)
    def test_nllloss(self):
        class NLLModel(torch.nn.Module):
            def __init__(self):
                super(NLLModel, self).__init__()
                self.loss = torch.nn.NLLLoss(reduction='none')
                self.m = torch.nn.LogSoftmax(dim=1)

            def forward(self, input, target):
                output = self.loss(self.m(2 * input), target)
                return output

        N, C = 5, 4
        input = torch.randn(N, 16)
        target = torch.empty(N, dtype=torch.long).random_(0, C)
        self.run_test(NLLModel(), (input, target))

    @unittest.skip("Enable this once ORT version is updated")
    @skipIfUnsupportedMinOpsetVersion(12)
    def test_nllloss_2d_none(self):
        class NLLModel(torch.nn.Module):
            def __init__(self):
                super(NLLModel, self).__init__()
                self.loss = torch.nn.NLLLoss(reduction='none')
                self.conv = torch.nn.Conv2d(16, C, (3, 3))
                self.m = torch.nn.LogSoftmax(dim=1)

            def forward(self, input, target):
                output = self.loss(self.m(self.conv(input)), target)
                return output

        N, C = 5, 4
        input = torch.randn(N, 16, 10, 10)
        target = torch.empty(N, 8, 8, dtype=torch.long).random_(0, C)
        self.run_test(NLLModel(), (input, target))

    @unittest.skip("Enable this once ORT version is updated")
    @skipIfUnsupportedMinOpsetVersion(12)
    def test_nllloss_2d_mean(self):
        class NLLModel(torch.nn.Module):
            def __init__(self):
                super(NLLModel, self).__init__()
                self.loss = torch.nn.NLLLoss(reduction='mean')
                self.conv = torch.nn.Conv2d(16, C, (3, 3))
                self.m = torch.nn.LogSoftmax(dim=1)

            def forward(self, input, target):
                output = self.loss(self.m(self.conv(input)), target)
                return output

        N, C = 5, 4
        input = torch.randn(N, 16, 10, 10)
        target = torch.empty(N, 8, 8, dtype=torch.long).random_(0, C)
        self.run_test(NLLModel(), (input, target))

    @unittest.skip("Enable this once ORT version is updated")
    @skipIfUnsupportedMinOpsetVersion(12)
    def test_nllloss_2d_sum(self):
        class NLLModel(torch.nn.Module):
            def __init__(self):
                super(NLLModel, self).__init__()
                self.loss = torch.nn.NLLLoss(reduction='sum')
                self.conv = torch.nn.Conv2d(16, C, (3, 3))
                self.m = torch.nn.LogSoftmax(dim=1)

            def forward(self, input, target):
                output = self.loss(self.m(self.conv(input)), target)
                return output

        N, C = 5, 4
        input = torch.randn(N, 16, 10, 10)
        target = torch.empty(N, 8, 8, dtype=torch.long).random_(0, C)
        self.run_test(NLLModel(), (input, target))

    @unittest.skip("Enable this once ORT version is updated")
    @skipIfUnsupportedMinOpsetVersion(12)
    def test_nllloss_2d_mean_weights(self):
        class NLLModel(torch.nn.Module):
            def __init__(self):
                super(NLLModel, self).__init__()
                self.loss = torch.nn.NLLLoss(reduction='mean', weight=torch.randn(C))
                self.conv = torch.nn.Conv2d(16, C, (3, 3))
                self.m = torch.nn.LogSoftmax(dim=1)

            def forward(self, input, target):
                output = self.loss(self.m(self.conv(input)), target)
                return output

        N, C = 5, 4
        input = torch.randn(N, 16, 10, 10)
        target = torch.empty(N, 8, 8, dtype=torch.long).random_(0, C)
        self.run_test(NLLModel(), (input, target))

    @unittest.skip("Enable this once ORT version is updated")
    @skipIfUnsupportedMinOpsetVersion(12)
    def test_nllloss_2d_mean_ignore_index(self):
        class NLLModel(torch.nn.Module):
            def __init__(self):
                super(NLLModel, self).__init__()
                self.loss = torch.nn.NLLLoss(reduction='mean', ignore_index=1)
                self.conv = torch.nn.Conv2d(16, C, (3, 3))
                self.m = torch.nn.LogSoftmax(dim=1)

            def forward(self, input, target):
                output = self.loss(self.m(self.conv(input)), target)
                return output

        N, C = 5, 4
        input = torch.randn(N, 16, 10, 10)
        target = torch.empty(N, 8, 8, dtype=torch.long).random_(0, C)
        self.run_test(NLLModel(), (input, target))

    @unittest.skip("Enable this once ORT version is updated")
    @skipIfUnsupportedMinOpsetVersion(12)
    def test_nllloss_2d_mean_ignore_index_weights(self):
        class NLLModel(torch.nn.Module):
            def __init__(self):
                super(NLLModel, self).__init__()
                self.loss = torch.nn.NLLLoss(reduction='mean', weight=torch.randn(C), ignore_index=1)
                self.conv = torch.nn.Conv2d(16, C, (3, 3))
                self.m = torch.nn.LogSoftmax(dim=1)

            def forward(self, input, target):
                output = self.loss(self.m(self.conv(input)), target)
                return output

        N, C = 5, 4
        input = torch.randn(N, 16, 10, 10)
        target = torch.empty(N, 8, 8, dtype=torch.long).random_(0, C)
        self.run_test(NLLModel(), (input, target))

    def test_torch_mm(self):
        class M(torch.nn.Module):
            def forward(self, mat1, mat2):
                mm = torch.mm(mat1, mat2)
                return mm

        mat1 = torch.randn(2, 3)
        mat2 = torch.randn(3, 3)
        self.run_test(M(), input=(mat1, mat2))

    def test_onnx_proto_checker(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()

            def forward(self, x):
                return 2 * x
        x = torch.randn(1, 2, 3, requires_grad=True)
        f = io.BytesIO()
        torch.onnx._export(Model(), x, f)
        model = onnx.load(f)
        model.ir_version = 0

        def check_proto():
            torch._C._check_onnx_proto(model.SerializeToString())
        self.assertRaises(RuntimeError, check_proto)

    def test_split_tensor_scalar(self):
        class SplitModel(torch.nn.Module):
            def forward(self, x):
                return torch.split(x, x.size(1))
        x = torch.randn(1, 2, 3, requires_grad=True)
        self.run_test(SplitModel(), x)

    def test_split_tensor_multi(self):
        class SplitModel(torch.nn.Module):
            def forward(self, x):
                return torch.split(x, torch.ones(3))
        x = torch.randn(1, 2, 3, requires_grad=True)

        def run_model():
            SplitModel(x)
        self.assertRaises(TypeError, run_model)

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

# opset 12 tests
TestONNXRuntime_opset12 = type(str("TestONNXRuntime_opset12"),
                               (unittest.TestCase,),
                               dict(TestONNXRuntime.__dict__, opset_version=12))

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

# opset 12 tests, with keep_initializers_as_inputs=False for
# IR version 4 style export.
TestONNXRuntime_opset12_IRv4 = type(str("TestONNXRuntime_opset12_IRv4"),
                                    (unittest.TestCase,),
                                    dict(TestONNXRuntime.__dict__, opset_version=12,
                                    keep_initializers_as_inputs=False))


if __name__ == '__main__':
    unittest.main()
