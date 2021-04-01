import unittest
import onnxruntime  # noqa
import torch

import numpy as np
import io
import itertools
import copy
import os
import random

from torch.nn.utils import rnn as rnn_utils
from model_defs.lstm_flattening_result import (LstmFlatteningResultWithSeqLength,
                                               LstmFlatteningResultWithoutSeqLength)
from model_defs.rnn_model_with_packed_sequence import (RnnModelWithPackedSequence,
                                                       RnnModelWithPackedSequenceWithState,
                                                       RnnModelWithPackedSequenceWithoutState)
from test_pytorch_common import (skipIfUnsupportedMinOpsetVersion, skipIfUnsupportedOpsetVersion,
                                 skipIfNoLapack, disableScriptTest, skipIfONNXShapeInference,
                                 skipIfUnsupportedMaxOpsetVersion)
from test_pytorch_common import BATCH_SIZE
from test_pytorch_common import RNN_BATCH_SIZE, RNN_SEQUENCE_LENGTH, RNN_INPUT_SIZE, RNN_HIDDEN_SIZE
from typing import List, Tuple, Optional, Dict
import model_defs.word_language_model as word_language_model

import onnx

import torchvision
from torchvision import ops
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
from collections import OrderedDict

from torch.nn.utils.rnn import PackedSequence

def to_numpy(tensor):
    if tensor.requires_grad:
        return tensor.detach().cpu().numpy()
    else:
        return tensor.cpu().numpy()

def convert_to_onnx(model, input=None, opset_version=9, example_outputs=None,
                    do_constant_folding=True, keep_initializers_as_inputs=True,
                    dynamic_axes=None, input_names=None, output_names=None,
                    fixed_batch_size=False, training=None,
                    onnx_shape_inference=False):
    # export the model to ONNX
    f = io.BytesIO()
    input_copy = copy.deepcopy(input)
    torch.onnx._export(model, input_copy, f,
                       opset_version=opset_version,
                       example_outputs=example_outputs,
                       do_constant_folding=do_constant_folding,
                       keep_initializers_as_inputs=keep_initializers_as_inputs,
                       dynamic_axes=dynamic_axes,
                       input_names=input_names, output_names=output_names,
                       fixed_batch_size=fixed_batch_size, training=training,
                       onnx_shape_inference=onnx_shape_inference)

    # compute onnxruntime output prediction
    ort_sess = onnxruntime.InferenceSession(f.getvalue())
    return ort_sess


def inline_flatten_list(inputs, res_list):
    for i in inputs:
        res_list.append(i) if not isinstance(i, (list, tuple)) else inline_flatten_list(i, res_list)
    return res_list


def run_ort(ort_sess, input):
    input_copy = copy.deepcopy(input)
    input, _ = torch.jit._flatten(input_copy)
    inputs = [to_numpy(inp) for inp in input]

    ort_inputs = dict((ort_sess.get_inputs()[i].name, input) for i, input in enumerate(inputs))
    ort_outs = ort_sess.run(None, ort_inputs)
    return inline_flatten_list(ort_outs, [])


def ort_compare_with_pytorch(ort_outs, output, rtol, atol):
    output, _ = torch.jit._flatten(output)
    outputs = [to_numpy(outp) for outp in output]

    # compare onnxruntime and PyTorch results
    assert len(outputs) == len(ort_outs), "number of outputs differ"

    # compare onnxruntime and PyTorch results
    [np.testing.assert_allclose(out, ort_out, rtol=rtol, atol=atol) for out, ort_out in zip(outputs, ort_outs)]


def run_model_test(self, model, batch_size=2, state_dict=None,
                   input=None, use_gpu=True, rtol=0.001, atol=1e-7,
                   example_outputs=None, do_constant_folding=True,
                   dynamic_axes=None, test_with_inputs=None,
                   input_names=None, output_names=None,
                   fixed_batch_size=False, dict_check=True,
                   training=None):
    model.eval()
    if input is None:
        input = torch.randn(batch_size, 3, 224, 224, requires_grad=True)
    with torch.no_grad():
        if isinstance(input, torch.Tensor):
            input = (input,)
        # In-place operators will update input tensor data as well.
        # Thus inputs are replicated before every forward call.
        if isinstance(input, dict):
            input = (input,)
        input_args = copy.deepcopy(input)
        input_kwargs = {}
        if dict_check and isinstance(input_args[-1], dict):
            input_kwargs = input_args[-1]
            input_args = input_args[:-1]
        try:
            model_copy = copy.deepcopy(model)
            output = model_copy(*input_args, **input_kwargs)
        except Exception:
            output = model(*input_args, **input_kwargs)
        if isinstance(output, torch.Tensor):
            output = (output,)

        if not dict_check and isinstance(input[-1], dict):
            input = input + ({},)

        ort_sess = convert_to_onnx(model, input=input, opset_version=self.opset_version,
                                   example_outputs=output, do_constant_folding=do_constant_folding,
                                   keep_initializers_as_inputs=self.keep_initializers_as_inputs,
                                   dynamic_axes=dynamic_axes, input_names=input_names,
                                   output_names=output_names, fixed_batch_size=fixed_batch_size, training=training,
                                   onnx_shape_inference=self.onnx_shape_inference)
        # compute onnxruntime output prediction
        ort_outs = run_ort(ort_sess, input)
        ort_compare_with_pytorch(ort_outs, output, rtol, atol)


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
                ort_outs = run_ort(ort_sess, test_input)
                ort_compare_with_pytorch(ort_outs, output, rtol, atol)

def _init_test_generalized_rcnn_transform():
    min_size = 100
    max_size = 200
    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]
    transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)
    return transform

def _init_test_rpn():
    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
    out_channels = 256
    rpn_head = RPNHead(out_channels, rpn_anchor_generator.num_anchors_per_location()[0])
    rpn_fg_iou_thresh = 0.7
    rpn_bg_iou_thresh = 0.3
    rpn_batch_size_per_image = 256
    rpn_positive_fraction = 0.5
    rpn_pre_nms_top_n = dict(training=2000, testing=1000)
    rpn_post_nms_top_n = dict(training=2000, testing=1000)
    rpn_nms_thresh = 0.7
    rpn_score_thresh = 0.0

    rpn = RegionProposalNetwork(
        rpn_anchor_generator, rpn_head,
        rpn_fg_iou_thresh, rpn_bg_iou_thresh,
        rpn_batch_size_per_image, rpn_positive_fraction,
        rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh,
        score_thresh=rpn_score_thresh)
    return rpn

def _init_test_roi_heads_faster_rcnn():
    out_channels = 256
    num_classes = 91

    box_fg_iou_thresh = 0.5
    box_bg_iou_thresh = 0.5
    box_batch_size_per_image = 512
    box_positive_fraction = 0.25
    bbox_reg_weights = None
    box_score_thresh = 0.05
    box_nms_thresh = 0.5
    box_detections_per_img = 100

    box_roi_pool = ops.MultiScaleRoIAlign(
        featmap_names=['0', '1', '2', '3'],
        output_size=7,
        sampling_ratio=2)

    resolution = box_roi_pool.output_size[0]
    representation_size = 1024
    box_head = TwoMLPHead(
        out_channels * resolution ** 2,
        representation_size)

    representation_size = 1024
    box_predictor = FastRCNNPredictor(
        representation_size,
        num_classes)

    roi_heads = RoIHeads(
        box_roi_pool, box_head, box_predictor,
        box_fg_iou_thresh, box_bg_iou_thresh,
        box_batch_size_per_image, box_positive_fraction,
        bbox_reg_weights,
        box_score_thresh, box_nms_thresh, box_detections_per_img)
    return roi_heads

def set_rng_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

class TestONNXRuntime(unittest.TestCase):
    from torch.onnx.symbolic_helper import _export_onnx_opset_version
    opset_version = _export_onnx_opset_version
    keep_initializers_as_inputs = True  # For IR version 3 type export.
    onnx_shape_inference = True

    def setUp(self):
        torch.manual_seed(0)
        onnxruntime.set_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
        np.random.seed(seed=0)
        os.environ['ALLOW_RELEASED_ONNX_OPSET_ONLY'] = '0'
        self.is_script_test_enabled = True

    def run_test(self, model, input, rtol=1e-3, atol=1e-7, do_constant_folding=True,
                 batch_size=2, use_gpu=True, dynamic_axes=None, test_with_inputs=None,
                 input_names=None, output_names=None, fixed_batch_size=False, dict_check=True,
                 training=None):
        def _run_test(m):
            return run_model_test(self, m, batch_size=batch_size,
                                  input=input, use_gpu=use_gpu, rtol=rtol, atol=atol,
                                  do_constant_folding=do_constant_folding,
                                  dynamic_axes=dynamic_axes, test_with_inputs=test_with_inputs,
                                  input_names=input_names, output_names=output_names,
                                  fixed_batch_size=fixed_batch_size, dict_check=dict_check,
                                  training=training)
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
                ort_outs = run_ort(ort_sess, input_copy)
                ort_compare_with_pytorch(ort_outs, output, rtol, atol)


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

    @skipIfUnsupportedMinOpsetVersion(9)  # Because external data format was released with Opset 9.
    def test_attribute_with_external_data(self):
        class LargeModel(torch.nn.Module):
            def forward(self, x):
                return x + torch.ones(2, 1024)

        x = torch.randn(2, 1)
        self.run_model_test_with_external_data(LargeModel(), x)

    @skipIfUnsupportedMinOpsetVersion(9)  # Because external data format was released with Opset 9.
    @unittest.skip("Enable this once large model with subgraph is supported in ORT")
    def test_subgraph_with_external_data(self):
        class LargeModel(torch.nn.Module):
            def forward(self, x):
                for i in range(x.size(0)):
                    x = x + torch.ones(2, 1024)
                return x

        x = torch.randn(2, 1)
        self.run_model_test_with_external_data(torch.jit.script(LargeModel()), x)

    def test_fuse_conv_bn1d(self):
        class Fuse(torch.nn.Module):
            def __init__(self):
                super(Fuse, self).__init__()
                self.conv = torch.nn.Conv1d(16, 33, 3, stride=2)
                self.bn = torch.nn.BatchNorm1d(33)

            def forward(self, x):
                out = self.conv(x)
                return self.bn(out)

        model = Fuse()
        x = torch.randn(20, 16, 50, requires_grad=True)
        self.run_test(model, (x,))

    def test_fuse_conv_bn2d(self):
        class Fuse(torch.nn.Module):
            def __init__(self):
                super(Fuse, self).__init__()
                self.conv = torch.nn.Conv2d(3, 2, kernel_size=1, stride=2, padding=3, bias=False)
                self.bn = torch.nn.BatchNorm2d(2)

            def forward(self, x):
                out = self.conv(x)
                return self.bn(out)

        model = Fuse()
        x = torch.randn(2, 3, 2, 2, requires_grad=True)
        self.run_test(model, (x,))

    def test_fuse_conv_bn3d(self):
        class Fuse(torch.nn.Module):
            def __init__(self):
                super(Fuse, self).__init__()
                self.conv = torch.nn.Conv3d(3, 2, (3, 5, 2), stride=(2, 1, 1), padding=(3, 2, 0), bias=False)
                self.bn = torch.nn.BatchNorm3d(2)

            def forward(self, x):
                out = self.conv(x)
                return self.bn(out)

        model = Fuse()
        x = torch.randn(2, 3, 10, 50, 100, requires_grad=True)
        self.run_test(model, (x,), rtol=1e-3, atol=1e-6)

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

    def get_image_from_url(self, url, size=(300, 200)):
        import os
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

        image = image.resize(size, Image.BILINEAR)

        to_tensor = transforms.ToTensor()
        return to_tensor(image)

    def get_test_images(self):
        image_url = "http://farm3.staticflickr.com/2469/3915380994_2e611b1779_z.jpg"
        image = self.get_image_from_url(url=image_url, size=(100, 320))

        image_url2 = "https://pytorch.org/tutorials/_static/img/tv_tutorial/tv_image05.png"
        image2 = self.get_image_from_url(url=image_url2, size=(250, 380))

        return [image], [image2]

    @skipIfUnsupportedMinOpsetVersion(11)
    @disableScriptTest()  # Faster RCNN model is not scriptable
    def test_faster_rcnn(self):
        model = torchvision.models.detection.faster_rcnn.fasterrcnn_resnet50_fpn(pretrained=True, min_size=200,
                                                                                 max_size=300)
        model.eval()
        x = torch.randn(2, 3, 200, 300, requires_grad=True)
        self.run_test(model, (x,), rtol=1e-3, atol=1e-5)
        self.run_test(model, (x,), input_names=["images_tensors"], output_names=["outputs"],
                      dynamic_axes={"images_tensors": [0, 1, 2, 3], "outputs": [0, 1, 2, 3]}, rtol=1e-3, atol=1e-5)
        dummy_image = [torch.ones(3, 100, 100) * 0.3]
        images, test_images = self.get_test_images()
        self.run_test(model, (images,), test_with_inputs=[(images,), (test_images,), (dummy_image,)],
                      input_names=["images_tensors"], output_names=["outputs"],
                      dynamic_axes={"images_tensors": [0, 1, 2], "outputs": [0, 1, 2]}, rtol=1e-3, atol=1e-5)
        self.run_test(model, (dummy_image,), test_with_inputs=[(dummy_image,), (images,)],
                      input_names=["images_tensors"], output_names=["outputs"],
                      dynamic_axes={"images_tensors": [0, 1, 2], "outputs": [0, 1, 2]}, rtol=1e-3, atol=1e-5)

    def test_paste_mask_in_image(self):
        # disable profiling
        torch._C._jit_set_profiling_executor(False)
        torch._C._jit_set_profiling_mode(False)

        masks = torch.rand(10, 1, 26, 26)
        boxes = torch.rand(10, 4)
        boxes[:, 2:] += torch.rand(10, 2)
        boxes *= 50
        o_im_s = (100, 100)
        from torchvision.models.detection.roi_heads import paste_masks_in_image
        out = paste_masks_in_image(masks, boxes, o_im_s)
        jit_trace = torch.jit.trace(paste_masks_in_image,
                                    (masks, boxes,
                                     [torch.tensor(o_im_s[0]),
                                      torch.tensor(o_im_s[1])]))
        out_trace = jit_trace(masks, boxes, [torch.tensor(o_im_s[0]), torch.tensor(o_im_s[1])])

        assert torch.all(out.eq(out_trace))

        masks2 = torch.rand(20, 1, 26, 26)
        boxes2 = torch.rand(20, 4)
        boxes2[:, 2:] += torch.rand(20, 2)
        boxes2 *= 100
        o_im_s2 = (200, 200)
        from torchvision.models.detection.roi_heads import paste_masks_in_image
        out2 = paste_masks_in_image(masks2, boxes2, o_im_s2)
        out_trace2 = jit_trace(masks2, boxes2, [torch.tensor(o_im_s2[0]), torch.tensor(o_im_s2[1])])

        assert torch.all(out2.eq(out_trace2))

    @skipIfUnsupportedMinOpsetVersion(11)
    @disableScriptTest()
    def test_mask_rcnn(self):
        model = torchvision.models.detection.mask_rcnn.maskrcnn_resnet50_fpn(pretrained=True, min_size=200,
                                                                             max_size=300)
        images, test_images = self.get_test_images()
        self.run_test(model, (images,), rtol=1e-3, atol=1e-5)
        self.run_test(model, (images,), input_names=["images_tensors"], output_names=["boxes", "labels", "scores", "masks"],
                      dynamic_axes={"images_tensors": [0, 1, 2], "boxes": [0, 1], "labels": [0],
                                    "scores": [0], "masks": [0, 1, 2]}, rtol=1e-3, atol=1e-5)
        dummy_image = [torch.ones(3, 100, 100) * 0.3]
        self.run_test(model, (images,), test_with_inputs=[(images,), (test_images,), (dummy_image,)],
                      input_names=["images_tensors"], output_names=["boxes", "labels", "scores", "masks"],
                      dynamic_axes={"images_tensors": [0, 1, 2], "boxes": [0, 1], "labels": [0],
                                    "scores": [0], "masks": [0, 1, 2]}, rtol=1e-3, atol=1e-5)
        self.run_test(model, (dummy_image,), test_with_inputs=[(dummy_image,), (images,)],
                      input_names=["images_tensors"], output_names=["boxes", "labels", "scores", "masks"],
                      dynamic_axes={"images_tensors": [0, 1, 2], "boxes": [0, 1], "labels": [0],
                                    "scores": [0], "masks": [0, 1, 2]}, rtol=1e-3, atol=1e-5)

    def test_heatmaps_to_keypoints(self):
        # disable profiling
        torch._C._jit_set_profiling_executor(False)
        torch._C._jit_set_profiling_mode(False)

        maps = torch.rand(10, 1, 26, 26)
        rois = torch.rand(10, 4)
        from torchvision.models.detection.roi_heads import heatmaps_to_keypoints
        out = heatmaps_to_keypoints(maps, rois)
        jit_trace = torch.jit.trace(heatmaps_to_keypoints, (maps, rois))
        out_trace = jit_trace(maps, rois)

        assert torch.all(out[0].eq(out_trace[0]))
        assert torch.all(out[1].eq(out_trace[1]))

        maps2 = torch.rand(20, 2, 21, 21)
        rois2 = torch.rand(20, 4)
        from torchvision.models.detection.roi_heads import heatmaps_to_keypoints
        out2 = heatmaps_to_keypoints(maps2, rois2)
        out_trace2 = jit_trace(maps2, rois2)

        assert torch.all(out2[0].eq(out_trace2[0]))
        assert torch.all(out2[1].eq(out_trace2[1]))

    @skipIfUnsupportedMinOpsetVersion(11)
    @disableScriptTest()
    def test_keypoint_rcnn(self):
        model = torchvision.models.detection.keypoint_rcnn.keypointrcnn_resnet50_fpn(pretrained=True, min_size=200,
                                                                                     max_size=300)
        images, test_images = self.get_test_images()
        self.run_test(model, (images,), rtol=1e-3, atol=1e-5)
        self.run_test(model, (images,), input_names=["images_tensors"],
                      output_names=["outputs1", "outputs2", "outputs3", "outputs4"],
                      dynamic_axes={"images_tensors": [0, 1, 2]},
                      rtol=1e-3, atol=1e-5)
        dummy_images = [torch.ones(3, 100, 100) * 0.3]
        self.run_test(model, (images,), test_with_inputs=[(images,), (test_images,), (dummy_images,)],
                      input_names=["images_tensors"], output_names=["outputs1", "outputs2", "outputs3", "outputs4"],
                      dynamic_axes={"images_tensors": [0, 1, 2]},
                      rtol=5e-3, atol=1e-5)
        self.run_test(model, (dummy_images,), test_with_inputs=[(dummy_images,), (test_images,)],
                      input_names=["images_tensors"], output_names=["outputs1", "outputs2", "outputs3", "outputs4"],
                      dynamic_axes={"images_tensors": [0, 1, 2]},
                      rtol=5e-3, atol=1e-5)

    @skipIfUnsupportedMinOpsetVersion(11)
    @disableScriptTest()
    def test_shufflenet_v2_dynamic_axes(self):
        model = torchvision.models.shufflenet_v2_x0_5(pretrained=True)
        dummy_input = torch.randn(1, 3, 224, 224, requires_grad=True)
        test_inputs = torch.randn(3, 3, 224, 224, requires_grad=True)
        self.run_test(model, (dummy_input,), test_with_inputs=[(dummy_input,), (test_inputs,)],
                      input_names=["input_images"], output_names=["outputs"],
                      dynamic_axes={"input_images": {0: 'batch_size'}, "output": {0: 'batch_size'}},
                      rtol=1e-3, atol=1e-5)

    @disableScriptTest()
    def test_word_language_model_RNN_TANH(self):
        self.run_word_language_model("RNN_TANH")

    @disableScriptTest()
    def test_word_language_model_RNN_RELU(self):
        self.run_word_language_model("RNN_RELU")

    @disableScriptTest()
    def test_word_language_model_LSTM(self):
        self.run_word_language_model("LSTM")

    @disableScriptTest()
    def test_word_language_model_GRU(self):
        self.run_word_language_model("GRU")

    def test_index_1d(self):
        class MyModel(torch.nn.Module):
            def forward(self, input):
                return input[0]

        m1 = torch.randn(3, 4, 5, 6, 7)
        self.run_test(MyModel(), m1)

    def test_index_2d_1dimslice(self):
        class MyModel(torch.nn.Module):
            def forward(self, input):
                return input[0:1, :]

        m1 = torch.randn(3, 4, 5, 6, 7)
        self.run_test(MyModel(), m1)

    def test_index_2d_sliceint(self):
        class MyModel(torch.nn.Module):
            def forward(self, input):
                return input[1, :]

        m1 = torch.randn(3, 4, 5, 6, 7)
        self.run_test(MyModel(), m1)

    def test_index_2d_neg_slice(self):
        class MyModel(torch.nn.Module):
            def forward(self, input):
                return input[0:-1, :]

        m1 = torch.randn(3, 4, 5, 6, 7)
        self.run_test(MyModel(), m1)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_index_mask(self):
        class MyModel(torch.nn.Module):
            def forward(self, input):
                return input[torch.tensor([0, 1, 0], dtype=torch.uint8)]

        m1 = torch.randn(3, 4, 5, 6, 7)
        self.run_test(MyModel(), m1)

        class MyModel(torch.nn.Module):
            def forward(self, input):
                return input[torch.tensor([0, 1, 0], dtype=torch.bool)]

        m1 = torch.randn(3, 4, 5, 6, 7)
        self.run_test(MyModel(), m1)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_data(self):
        class Data(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x):
                return x.new_zeros(x.data.size())

        x = torch.randn(3, 4)
        self.run_test(Data(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    @disableScriptTest()  # Need type inference
    def test_index_mask_nd(self):
        class MyModel(torch.nn.Module):
            def forward(self, input):
                return input[input > 0]

        m1 = torch.randn(3, 4, 5, 6, 7)
        self.run_test(MyModel(), m1)

    @disableScriptTest()
    def test_dict(self):
        class MyModel(torch.nn.Module):
            def forward(self, x_in):
                x_out = {}
                x_out["test_key_out"] = torch.add(x_in[list(x_in.keys())[0]], list(x_in.keys())[0])
                return x_out

        x = {torch.tensor(1.): torch.randn(1, 2, 3)}
        self.run_test(MyModel(), (x, {}))

    @disableScriptTest()
    def test_dict_str(self):
        class MyModel(torch.nn.Module):
            def forward(self, x_in):
                x_out = {}
                x_out["test_key_out"] = torch.add(x_in["test_key_in"], 2.)
                return x_out

        x = {"test_key_in": torch.randn(1, 2, 3)}
        self.run_test(MyModel(), (x, {}))

    @disableScriptTest()
    def test_dict_output(self):
        class DictModelOutput(OrderedDict):
            tensor_out: torch.Tensor
            tuple_out: Optional[Tuple[torch.Tensor]] = None
            list_out: Optional[List[torch.Tensor]] = None

        class MyModel(torch.nn.Module):
            def forward(self, a, b, c, d):
                return DictModelOutput(
                    tensor_out=a,
                    tuple_out=(b, c),
                    list_out=[d],
                )

        a = torch.randn(2, 3)
        b = torch.randn(2, 3)
        c = torch.randn(2, 3)
        d = torch.randn(2, 3)
        self.run_test(MyModel(), (a, b, c, d))

    def test_tuple_output(self):
        class MyModel(torch.nn.Module):
            def forward(self, a, b, c, d):
                return a, (b, c), d

        a = torch.randn(2, 3)
        b = torch.randn(2, 3)
        c = torch.randn(2, 3)
        d = torch.randn(2, 3)
        self.run_test(MyModel(), (a, b, c, d))

    def test_nested_tuple_output(self):
        class MyModel(torch.nn.Module):
            def forward(self, a, b, c, d):
                return a, ((b,), (c, d))

        a = torch.randn(2, 3)
        b = torch.randn(2, 3)
        c = torch.randn(2, 3)
        d = torch.randn(2, 3)
        self.run_test(MyModel(), (a, b, c, d))

    @disableScriptTest()
    def test_optional_inputs_with_no_optionals(self):
        class NoOptionalModel(torch.nn.Module):
            def forward(self, input):
                return input

        # Without empty optional arguments dictionary
        x = torch.randn(2, 3)
        self.run_test(NoOptionalModel(), (x,))
        # With empty optional arguments dictionary
        y = torch.randn(2, 3)
        self.run_test(NoOptionalModel(), (y, {}))

    @disableScriptTest()
    def test_optional_inputs_with_mixed_optionals(self):
        class MixedModel(torch.nn.Module):
            def forward(self, x, y=None, z=None):
                if y is not None:
                    return x + y
                if z is not None:
                    return x + z
                return x

        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        z = torch.randn(2, 3)
        # Without optional arguments dictionary
        self.run_test(MixedModel(), (x, y, None))
        self.run_test(MixedModel(), (x, None, z))
        # With optional arguments dictionary
        self.run_test(MixedModel(), (x, {'y': y, 'z': None}))
        self.run_test(MixedModel(), (x, {'y': None, 'z': z}))
        self.run_test(MixedModel(), (x, {'z': z}))
        self.run_test(MixedModel(), (x, {'y': y}))

    @disableScriptTest()
    def test_optional_inputs_with_all_optionals(self):
        class AllOptionalModel(torch.nn.Module):
            def forward(self, y=None, z=None):
                if y is not None:
                    return y
                if z is not None:
                    return z

        y = torch.randn(2, 3)
        # Without optional arguments dictionary
        self.run_test(AllOptionalModel(), (y, None))
        # With optional arguments dictionary
        self.run_test(AllOptionalModel(), {'y': y, 'z': None})

    @disableScriptTest()
    def test_input_names_with_optional_args(self):
        class NoOptionalModel(torch.nn.Module):
            def forward(self, input):
                return input

        # Without empty optional arguments dictionary
        x = torch.randn(2, 3)
        self.run_test(NoOptionalModel(), (x,), input_names=['input_x'])
        # With empty optional arguments dictionary
        y = torch.randn(2, 3)
        self.run_test(NoOptionalModel(), (y, {}))

        class MixedModel(torch.nn.Module):
            def forward(self, x, y=None, z=None):
                if y is not None:
                    return x + y
                if z is not None:
                    return x + z
                return x

        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        z = torch.randn(2, 3)
        # Without optional arguments dictionary
        self.run_test(MixedModel(), (x, y, None), input_names=['input_x', 'input_y'])
        self.run_test(MixedModel(), (x, None, z), input_names=['input_x', 'input_z'])

        # With optional arguments dictionary
        self.run_test(MixedModel(), (x, {'y': y, 'z': None}), input_names=['input_x', 'input_y'])
        self.run_test(MixedModel(), (x, {'y': None, 'z': z}), input_names=['input_x', 'input_z'])

        class AllOptionalModel(torch.nn.Module):
            def forward(self, y=None, z=None):
                if y is not None:
                    return y
                if z is not None:
                    return z

        y = torch.randn(2, 3)
        z = torch.randn(2, 3)
        # Without optional arguments dictionary
        self.run_test(AllOptionalModel(), (y, None), input_names=['input_y'])
        self.run_test(AllOptionalModel(), (None, z), input_names=['input_z'])
        # With optional arguments dictionary
        self.run_test(AllOptionalModel(), {'y': y, 'z': None}, input_names=['input_y'])
        self.run_test(AllOptionalModel(), {'y': None, 'z': z}, input_names=['input_z'])

    def test_input_as_output(self):
        class Model(torch.nn.Module):
            def forward(self, x, y):
                return x, y

        x = torch.randn(2, 3)
        y = torch.randn(3, 4)
        self.run_test(Model(), (x, y), input_names=['x', 'y'], output_names=['x_out', 'y_out'])

    @disableScriptTest()
    def test_none_as_input(self):
        class Model(torch.nn.Module):
            def forward(self, x, y):
                if y is not None:
                    return x + y
                return x

        x = torch.randn(2, 3)
        self.run_test(Model(), (x, None))

    @disableScriptTest()
    def test_none_as_tuple_input(self):
        class Model(torch.nn.Module):
            def forward(self, x, y):
                if y[0] is not None:
                    return x + y[0]
                if y[1] is not None:
                    return x + y[1]
                return x

        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        self.run_test(Model(), (x, (None, y)))

    @disableScriptTest()
    def test_none_as_named_input(self):
        class Model(torch.nn.Module):
            def forward(self, x, y=None, z=None):
                if y is not None:
                    return x + y
                if z is not None:
                    return x + z
                return x

        x = torch.randn(2, 3)
        z = torch.randn(2, 3)
        self.run_test(Model(), (x, None, z))

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

    def test_tensor(self):
        class ScalarInputModel(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, input):
                return torch.tensor(input.shape[1])

        x = torch.randn(3, 4)
        self.run_test(ScalarInputModel(), x)

        class TensorInputModel(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, input):
                return torch.tensor([input.shape[0], input.shape[1]])

        x = torch.randn(3, 4)
        self.run_test(TensorInputModel(), x)

        class FloatInputModel(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, input):
                return torch.tensor([float(input)])

        x = torch.randn(1)
        self.run_test(FloatInputModel(), x)

        class InputWithDtypeModel(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, input):
                return torch.tensor(input.shape[1], dtype=torch.long)

        x = torch.randn(3, 4)
        self.run_test(InputWithDtypeModel(), x)

        class MixedInputModel(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, input):
                return torch.tensor([input.shape[0], int(input)])

        x = torch.randn(1)
        self.run_test(MixedInputModel(), x)

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

    def test_hardswish(self):
        model = torch.nn.Hardswish()

        x = torch.rand(3, 3).to(dtype=torch.float32)
        self.run_test(model, x)

        # Testing edge cases
        x = torch.tensor(3).to(dtype=torch.float32)
        self.run_test(model, x)
        x = torch.tensor(-3).to(dtype=torch.float32)
        self.run_test(model, x)

    def test_hardswish_script(self):
        class MyModel(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x):
                return torch.nn.functional.hardswish(x)

        x = torch.rand(3, 3).to(dtype=torch.float32)
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

        x1 = torch.randn(20, 16, 50)
        x2 = torch.randn(20, 16, 50, 100)
        x3 = torch.randn(20, 16, 10, 50, 100)

        self.run_test(TraceModel(), (x1, x2, x3), atol=10e-5)

    def test_conv_shape_inference(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super(Model, self).__init__()
                self.conv2 = torch.nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))

            def forward(self, input):
                return self.conv2(input) + 2

        x = torch.randn(20, 16, 50, 100)
        self.run_test(Model(), x, atol=10e-5,
                      input_names=['x'],
                      dynamic_axes={'x': [0]})

    def test_conv_transpose(self):
        class TraceModel(torch.nn.Module):
            def __init__(self):
                super(TraceModel, self).__init__()
                self.conv1 = torch.nn.ConvTranspose1d(16, 33, 3, stride=2)
                self.conv2 = torch.nn.ConvTranspose2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
                self.conv3 = torch.nn.ConvTranspose3d(16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(4, 2, 0))

            def forward(self, input1, input2, input3):
                return self.conv1(input1), self.conv2(input2), self.conv3(input3)

        x1 = torch.randn(20, 16, 50)
        x2 = torch.randn(20, 16, 50, 100)
        x3 = torch.randn(20, 16, 10, 50, 100)

        self.run_test(TraceModel(), (x1, x2, x3), atol=10e-5)

    # Conversion of Transpose depends on input shape to be known.
    # The following test only works when onnx shape inference is enabled.
    @skipIfONNXShapeInference(False)
    def test_transpose_infer_shape(self):
        class TransposeModule(torch.jit.ScriptModule):
            def __init__(self):
                super(TransposeModule, self).__init__()
                self.conv = torch.nn.Conv2d(3, 1, 3, stride=2)

            @torch.jit.script_method
            def forward(self, x):
                x = self.conv(x)
                return x.transpose(0, 1)

        x = torch.randn(32, 3, 64, 64)
        y = torch.randn(16, 3, 8, 64)
        self.run_test(TransposeModule(), x, input_names=['x'],
                      dynamic_axes={'x': [0, 2]},
                      test_with_inputs=[y])

    def squeeze_model_tests(self, d, x1, x2):
        class Squeeze(torch.nn.Module):
            def __init__(self, d):
                super(Squeeze, self).__init__()
                self.d = d

            def forward(self, x):
                if self.d is not None:
                    return torch.squeeze(x, dim=self.d)
                else:
                    return torch.squeeze(x)

        x2 = [] if x2 is None else [x2]
        if len(x2) > 0:
            self.run_test(Squeeze(d), x1,
                          input_names=['input'], dynamic_axes={'input': {0: '0', 1: '1', 2: '2'}},
                          test_with_inputs=x2)
        else:
            self.run_test(Squeeze(d), x1)

    def test_squeeze_without_no_op(self):
        x = torch.randn(2, 1, 4)
        self.squeeze_model_tests(1, x, None)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_squeeze_dynamic(self):
        x_squeeze = torch.randn(2, 1, 4)
        x_noop = torch.randn(2, 2, 3)
        self.squeeze_model_tests(1, x_squeeze, x_noop)

    def test_squeeze_neg_without_no_op(self):
        x = torch.randn(2, 1, 4)
        self.squeeze_model_tests(-2, x, None)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_squeeze_neg(self):
        x_squeeze = torch.randn(2, 1, 4)
        x_noop = torch.randn(2, 2, 3)
        self.squeeze_model_tests(-2, x_squeeze, x_noop)

    def test_squeeze_all_dims(self):
        x_squeeze = torch.randn(2, 1, 4)
        x_noop = torch.randn(2, 2, 3)
        self.squeeze_model_tests(None, x_squeeze, x_noop)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_squeeze_no_op(self):
        x_noop = torch.randn(2, 1, 4)
        x_squeeze = torch.randn(2, 2, 1)
        self.squeeze_model_tests(2, x_noop, x_squeeze)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_squeeze_runtime_dim(self):
        class Squeeze(torch.nn.Module):
            def forward(self, d1, d2):
                t = torch.zeros(d1[0], d2[0])
                return t.squeeze(0)

        d1 = torch.tensor([1])
        d3 = torch.tensor([3])
        d4 = torch.tensor([4])
        self.run_test(Squeeze(), (d1, d4), test_with_inputs=[(d3, d4)])
        self.run_test(Squeeze(), (d3, d4), test_with_inputs=[(d1, d3)])

    def test_squeeze(self):
        class Squeeze(torch.nn.Module):
            def forward(self, x):
                return torch.squeeze(x, dim=-2)

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
        y = torch.randn(32, 16, 50, requires_grad=True)
        self.run_test(model, x, input_names=['x'],
                      dynamic_axes={'x' : [0]},
                      test_with_inputs=[y])

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
    @disableScriptTest()  # Functional module not scriptable
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
        y = torch.randn(32, 8, 50, 44, 31)
        self.run_test(model, x, input_names=['x'],
                      dynamic_axes={'x' : [0, 1]},
                      test_with_inputs=[y])

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_floating_point(self):
        class FloatingPoint(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x):
                if x.is_floating_point():
                    return x.new_zeros(x.shape)
                return x.new_zeros(x.shape)

        x = torch.randn(2, 3, 4)
        self.run_test(FloatingPoint(), x)

        class FloatingPoint(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x):
                if x.size(0) > 1:
                    a = x + 2
                    if a.is_floating_point():
                        return x + 1
                    return x + 1
                return x

        x = torch.randn(2, 3, 4)
        self.run_test(FloatingPoint(), x)

    # Operator rank mismatch between outputs of two branches for opsets below 11.
    @skipIfUnsupportedMinOpsetVersion(11)
    @skipIfONNXShapeInference(False)
    def test_floating_point_infer_dtype(self):
        class FloatingPoint(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x):
                if x.size(0) > 1:
                    a = x + 2
                    if a.is_floating_point():
                        return x.new_zeros(x.shape[1:])
                    return x.new_zeros(x.shape)
                return x

        x = torch.randn(2, 3, 4)
        self.run_test(FloatingPoint(), x)

        class FloatingPoint(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x):
                if x.size(0) > 1:
                    a = x + 2
                    if a.is_floating_point():
                        return x + 1
                    return x
                return x

        x = torch.randn(2, 3, 4).to(torch.int32)
        self.run_test(FloatingPoint(), x)

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

    def test_arithmetic_prim_long(self):
        class ArithmeticModule(torch.nn.Module):
            def forward(self, x, y: int):
                x = x + y
                x = x - y
                x = x * (y * 3)
                x = x / (y * 4)
                return x

        x = torch.randn(2, 3, 4)
        y = 2
        self.run_test(ArithmeticModule(), (x, y))

        class ArithmeticModule(torch.nn.Module):
            def forward(self, x):
                x = x + 2
                x = x - 3
                return x.shape[0]

        x = torch.randn(2, 3, 4)
        self.run_test(ArithmeticModule(), x)

    def test_arithmetic_prim_float(self):
        class ArithmeticModule(torch.nn.Module):
            def forward(self, x, y: float):
                x = x + y
                x = x - y
                x = x * (y * 3)
                x = x / (y * 4)
                return x

        x = torch.randn(2, 3, 4)
        y = 2.5
        self.run_test(ArithmeticModule(), (x, y))

        class ArithmeticModule(torch.nn.Module):
            def forward(self, x):
                x = x + 2
                x = x - 3
                return x.shape[1] / 2

        x = torch.randn(2, 3, 4)
        self.run_test(ArithmeticModule(), x)

    def test_arithmetic_prim_bool(self):
        class ArithmeticModule(torch.nn.Module):
            def forward(self, x, y: int, z: bool, t: float):
                x = x + y
                x = x - y
                if z:
                    x = x * (y * 3)
                    x = x / (y * 4)
                return x / t, z

        x = torch.randn(2, 3, 4)
        y = 2
        z = False
        t = 2.5
        self.run_test(ArithmeticModule(), (x, y, z, t))

        class ArithmeticModule(torch.nn.Module):
            def forward(self, x: float, y: float):
                return x == y

        x = 3
        y = 2
        self.run_test(ArithmeticModule(), (x, y))

    # In scripting the first transpose node do not carry shape and dtype info.
    # The following test only works when onnx shape inference is enabled.
    @skipIfONNXShapeInference(False)
    def test_arithmetic_infer_dtype(self):
        class ArithmeticModule(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x):
                x = x.t()
                x = x + 2
                x = x - 4
                x = x * 6
                x = x / 8
                return x

        x = torch.randn(2, 3)
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

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_floordiv(self):
        class FloordivModule(torch.nn.Module):
            def forward(self, x):
                return x.new_zeros(x.size(2) // x.size(1))

        x = torch.randn(2, 3, 4)
        self.run_test(FloordivModule(), (x,))

    def test_div(self):
        class DivModule(torch.nn.Module):
            def forward(self, x, y):
                return x / y, torch.true_divide(x, y)

        x = torch.randn(2, 3, 4).to(torch.int)
        y = torch.arange(1, 2 * 3 * 4 + 1).reshape(2, 3, 4).to(torch.int)
        self.run_test(DivModule(), (x, y))
        self.run_test(DivModule(), (x.float(), y.float()))

    # Note: div cannot (generally) be exported via scripting
    # since its type promotion logic is dependent on knowing the scalar types
    # of the input tensors. That is, the ONNX graph is dependent on the
    # data type of the inputs. This makes it appropriate for tracing only.
    def test_div_promotion_trace(self):
        class DivModule(torch.nn.Module):
            def forward(self, x, y):
                return x / y, torch.true_divide(x, y)

        x = torch.randn(2, 3, 4).to(torch.int)
        y = torch.arange(1, 2 * 3 * 4 + 1).reshape(2, 3, 4).to(torch.int)

        prev_default = torch.get_default_dtype()

        torch.set_default_dtype(torch.float)
        self.run_test(torch.jit.trace(DivModule(), (x, y)), (x, y))

        torch.set_default_dtype(torch.double)
        self.run_test(torch.jit.trace(DivModule(), (x, y)), (x, y))

        torch.set_default_dtype(prev_default)

    # In scripting x, y do not carry shape and dtype info.
    # The following test only works when onnx shape inference is enabled.
    @skipIfONNXShapeInference(False)
    def test_div_promotion_script(self):
        class DivModule(torch.nn.Module):
            def forward(self, x, y):
                # Add transpose to hide shape/type information
                # Otherwise shape and type are still avaiable from input.
                x = x.transpose(1, 2)
                y = y.transpose(1, 2)
                return x / y, torch.true_divide(x, y)

        x = torch.randn(2, 3, 4).to(torch.int)
        y = torch.arange(1, 2 * 3 * 4 + 1).reshape(2, 3, 4).to(torch.int)

        prev_default = torch.get_default_dtype()

        # 1. x,y are int, and output is float.
        #    This can be handled by the default case, where both are cast to float.
        #    It works even if type of x, y are unknown.
        torch.set_default_dtype(torch.float)
        self.run_test(torch.jit.script(DivModule()), (x, y))

        # 2. x,y are int, and output is double.
        #    This can be handled by the default case, where both are cast to double.
        #    It works even if type of x, y are unknown.
        torch.set_default_dtype(torch.double)
        self.run_test(torch.jit.script(DivModule()), (x, y))

        # 3. x is int, y is double, and output is double.
        #    This can only be handled when both type of x and y are known.
        torch.set_default_dtype(prev_default)
        x = torch.randn(2, 3, 4).to(torch.int)
        y = torch.arange(1, 2 * 3 * 4 + 1).reshape(2, 3, 4).to(torch.double)
        self.run_test(torch.jit.script(DivModule()), (x, y))

    def test_div_rounding_mode(self):
        class TrueDivModule(torch.nn.Module):
            def forward(self, x, y):
                return (x.div(y, rounding_mode='true'),
                        torch.div(x, y, rounding_mode='true'))

        class TruncDivModule(torch.nn.Module):
            def forward(self, x, y):
                return (x.div(y, rounding_mode='trunc'),
                        torch.div(x, y, rounding_mode='trunc'))

        class FloorDivModule(torch.nn.Module):
            def forward(self, x, y):
                return (x.div(y, rounding_mode='floor'),
                        torch.div(x, y, rounding_mode='floor'))

        modules = [TrueDivModule(), TruncDivModule()]
        if self.opset_version >= 9:
            modules.append(FloorDivModule())

        x = (torch.randn(2, 3, 4) * 100).to(torch.int)
        y = torch.arange(1, 2 * 3 * 4 + 1).reshape(2, 3, 4).to(torch.int)

        for module in modules:
            self.run_test(module, (x, y))
            self.run_test(torch.jit.trace(module, (x, y)), (x, y))
            self.run_test(torch.jit.script(module), (x, y))

        x = torch.randn(2, 3, 4)
        y = torch.rand(2, 3, 4) * 10.0 + 0.1

        for module in modules:
            self.run_test(module, (x, y))
            self.run_test(torch.jit.trace(module, (x, y)), (x, y))
            self.run_test(torch.jit.script(module), (x, y))

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

    def test_slice_neg_large_negone(self):
        class NegSlice(torch.nn.Module):
            def forward(self, x):
                return x[:, :, :, :, -1]

        x = torch.randn(3, 4, 5, 6, 7)
        self.run_test(NegSlice(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_slice_with_input_index(self):
        class InputIndexSlice(torch.nn.Module):
            def forward(self, x, y):
                x[:y.size(0), 0, :] = y
                return x

        x = torch.zeros((56, 6, 256))
        y = torch.rand((22, 256))
        self.run_test(InputIndexSlice(), (x, y))

    @skipIfUnsupportedMinOpsetVersion(10)
    @disableScriptTest()  # scripting tuple/list append
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
                return x[1:x.size(1)]

        x = torch.rand(1, 2)
        self.run_test(DynamicSliceModel(), x)

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_slice_dynamic_shape_script(self):
        class DynamicSliceModel(torch.nn.Module):
            def forward(self, x):
                return x.new_zeros(x.shape[1:x.size(2)])

        x = torch.rand(1, 2, 3, 4)
        self.run_test(DynamicSliceModel(), x)

    @skipIfUnsupportedMinOpsetVersion(10)
    @disableScriptTest()   # scripting tuple/list append
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

    def test_square(self):
        class Square(torch.nn.Module):
            def forward(self, x):
                return torch.square(x)

        x = torch.randn(2, 3, 4)
        self.run_test(Square(), x)

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

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_dynamic_arange_out(self):
        class ArangeOutModel(torch.nn.Module):
            def forward(self, end):
                out_t = torch.tensor([1], dtype=torch.int64)
                return torch.arange(end, out=out_t)

        x = torch.tensor(8)
        self.run_test(ArangeOutModel(), (x))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_dynamic_arange_start_out(self):
        class ArangeStartOutModel(torch.nn.Module):
            def forward(self, start, end):
                out_t = torch.tensor([1], dtype=torch.int64)
                return torch.arange(start.size(0), end, out=out_t)

        x = torch.randn(2, 3, 4)
        y = torch.tensor(8)
        self.run_test(ArangeStartOutModel(), (x, y))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_arange(self):
        class ArangeModel(torch.nn.Module):
            def forward(self, start, end):
                return torch.arange(start.size(0), end, 1.5, dtype=torch.int64)

        x = torch.randn(2, 3, 4)
        y = torch.tensor(8.5, dtype=torch.float)
        self.run_test(ArangeModel(), (x, y))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_arange_out(self):
        class ArangeOutModel(torch.nn.Module):
            def forward(self, end):
                out_t = torch.tensor([1], dtype=torch.float)
                return torch.arange(end, out=out_t)

        x = torch.tensor(8.5, dtype=torch.float)
        self.run_test(ArangeOutModel(), (x))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_arange_start_out(self):
        class ArangeStartOutModel(torch.nn.Module):
            def forward(self, start, end):
                out_t = torch.tensor([1], dtype=torch.float)
                return torch.arange(start.size(0), end, out=out_t)

        x = torch.randn(2, 3, 4)
        y = torch.tensor(8.5, dtype=torch.float)
        self.run_test(ArangeStartOutModel(), (x, y))

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
                return torch.arange(input.size(0)), torch.arange(input.size(-1)), torch.ones(input.shape)

        x = torch.randn(5, 3, 2)
        self.run_test(SizeModel(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    @disableScriptTest()  # x.stride() not scriptable
    def test_as_strided(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                chunk_size = list(x.size())
                chunk_size[1] = chunk_size[1] * 2 - 1
                chunk_stride = list(x.stride())
                chunk_stride[1] = chunk_stride[1] // 2
                return x.as_strided((3, 3, 3), (1, 4, 2), storage_offset=2), x.as_strided(chunk_size, chunk_stride)

        x = torch.randn(5, 8, 7)
        self.run_test(Model(), x)

    @disableScriptTest()  # Ellipses followed by tensor indexing not scriptable
    def test_tensor_index_advanced_indexing_ellipsis(self):
        class MyModel(torch.nn.Module):
            def forward(self, input):
                return input[..., torch.tensor([2, 1]), torch.tensor([0, 3])]

        m1 = torch.randn(3, 4, 5, 6, 7)
        self.run_test(MyModel(), (m1,))

    def test_tensor_index_advanced_indexing(self):
        class MyModel(torch.nn.Module):
            def forward(self, input):
                return input[:, torch.tensor([[0, 2], [1, 1]]), :, torch.tensor([2, 1]), torch.tensor([0, 3])]

        m1 = torch.randn(3, 4, 5, 6, 7)
        self.run_test(MyModel(), (m1,))

        class MyModel(torch.nn.Module):
            def forward(self, input):
                return input[:, torch.tensor([0, 2]), None, 2:4, torch.tensor([[1, 3], [4, 0]])]

        self.run_test(MyModel(), (m1,))

        class MyModel(torch.nn.Module):
            def forward(self, input):
                return input[:, torch.tensor([0, 2]), torch.tensor([1]), 2:4, torch.tensor([[1], [4]])]

        self.run_test(MyModel(), (m1,))

    def test_tensor_index_advanced_indexing_consecutive(self):
        class MyModel(torch.nn.Module):
            def forward(self, input):
                return input[:, torch.tensor([0, 2]), torch.tensor([[1, 3], [4, 0]]), None]

        m1 = torch.randn(3, 4, 5, 6, 7)
        self.run_test(MyModel(), (m1,))

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
    def test_index_put_singular(self):
        class IndexPutBoolModel(torch.nn.Module):
            def forward(self, mask, indices):
                mask[indices] = True
                return mask

        mask = torch.zeros(100, dtype=torch.bool)
        indices = (torch.rand(25) * mask.shape[0]).to(torch.int64)
        self.run_test(IndexPutBoolModel(), (mask, indices))

        class IndexPutFloatModel(torch.nn.Module):
            def forward(self, mask, indices):
                mask[indices] = torch.tensor(5.5)
                return mask

        mask = torch.rand(100, dtype=torch.float)
        indices = (torch.rand(50) * mask.shape[0]).to(torch.int64)
        self.run_test(IndexPutFloatModel(), (mask, indices))

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

        class IndexPutModel9(torch.nn.Module):
            def forward(self, poses):
                w = 32
                x = poses[:, :, 0] - (w - 1) // 2
                boxes = torch.zeros([poses.shape[0], 17, 4])
                boxes[:, :, 0] = x
                return boxes

        x = torch.zeros([2, 17, 3], dtype=torch.int64)
        self.run_test(IndexPutModel9(), (x,))

    @skipIfUnsupportedMinOpsetVersion(11)
    @disableScriptTest()  # Ellipses followed by tensor indexing not scriptable
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
    def test_index_put_loop(self):
        @torch.jit.script
        def ngram_attention_bias(sequence_length: int, ngram: int, device: torch.device, dtype: torch.dtype):
            bias = torch.ones((ngram, sequence_length), device=device, dtype=dtype) * float("-inf")
            for stream_idx in range(ngram):
                for i in range(sequence_length):
                    bias = bias * 2
                    bias[stream_idx, i] = 5
                    bias = bias * 5
                    bias[0, 0] = 5

            for stream_idx in range(ngram):
                for i in range(sequence_length):
                    bias[stream_idx, i] = 5
                    bias[0, i] = 5
            return bias

        class ScriptModel(torch.nn.Module):
            def __init__(self):
                super(ScriptModel, self).__init__()
                self.ngram = 2
                self.max_target_positions = 512

            def forward(self, hidden_states):
                seq_length, batch_size = hidden_states.shape[:2]
                predict_causal_mask = ngram_attention_bias(
                    self.max_target_positions, self.ngram, hidden_states.device, hidden_states.dtype
                )
                predict_causal_mask = predict_causal_mask[:, :seq_length]
                return predict_causal_mask

        x = torch.randn(6, 2)
        y = torch.randn(4, 1)
        self.run_test(ScriptModel(), x, input_names=['x'],
                      dynamic_axes={'x': {0: 'seq_length', 1: 'batch_size'}}, test_with_inputs=[y])

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

        class CopyModel4(torch.nn.Module):
            def forward(self, x, ind, data):
                x[ind] = data
                return x

        x = torch.randn(3, 4)
        ind = torch.tensor(2)
        data = torch.randn(4)
        self.run_test(CopyModel4(), (x, ind, data))

        class CopyModel5(torch.nn.Module):
            def forward(self, x, mask):
                if mask is not None:
                    x.copy_(mask)
                    return x

        x = torch.randn(3, 4)
        mask = torch.randn(3, 1)
        self.run_test(CopyModel5(), (x, mask))

    @skipIfUnsupportedMinOpsetVersion(11)
    @disableScriptTest()  # Model not scriptable (output with shape doesn't match the broadcast shape)
    def test_copy_tracing(self):
        class CopyModel(torch.nn.Module):
            def forward(self, x, data):
                x[1, 1:3] = data
                return x

        x = torch.randn(3, 4)
        update = torch.randn(1, 2)
        self.run_test(CopyModel(), (x, update))

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

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_copy_ellipsis_script(self):
        class CopyModel(torch.nn.Module):
            def forward(self, x, update):
                # Insert reshape node to ensure no shape/type info for
                # x in scripting, without onnx shape inference.
                x = x.reshape(4, 3, 5, 6)
                x[2, ..., 1:3] = update
                return x

        x = torch.randn(3, 4, 5, 6)

        update = torch.ones(1)
        self.run_test(CopyModel(), (x, update))

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
                # the following cases, require dynamic sizes/scales,
                # which which is not supported for opset_version < 9
                if self.opset_version >= 9:
                    self._interpolate(xi, mode_i, True, is_upsample)
                    # test with align_corners if supported
                    if mode != 'nearest':
                        self._interpolate(xi, mode_i, False, is_upsample, True)
                    self._interpolate(xi, mode_i, False, is_upsample)

    # ONNX export failed on interpolate scripting because dynamic size not supported for opsets below 9.
    @skipIfUnsupportedMinOpsetVersion(9)
    def test_interpolate_upsample(self):
        self._interpolate_tests(True)

    @skipIfUnsupportedMaxOpsetVersion(8)
    @disableScriptTest()
    def test_interpolate_upsample_trace(self):
        self._interpolate_tests(True)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_interpolate_function_substitution(self):
        class ScriptModel(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x):
                return torch.nn.functional.interpolate(x, mode="nearest", scale_factor=2.)

        class ScriptModule(torch.jit.ScriptModule):
            def __init__(self):
                super(ScriptModule, self).__init__()
                self.submodule = ScriptModel()

            @torch.jit.script_method
            def forward(self, input):
                return self.submodule(input)

        x = torch.randn(1, 2, 4, 4, 6)
        self.run_test(ScriptModule(), (x,))

        @torch.jit.script
        def script_method(x):
            return torch.nn.functional.interpolate(x, mode="nearest", scale_factor=2.)

        class TracingModule(torch.nn.Module):
            def forward(self, x):
                return script_method(x)

        self.run_test(TracingModule(), (x,))

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

    # scripting will throw the OnnxRuntimeError
    @disableScriptTest()
    def test_interpolate_adaptive_pooling_error(self):
        x = torch.randn(1, 2, 6, requires_grad=True)
        with self.assertRaises(RuntimeError) as cm:
            self._interpolate(x, "area", True, True)

        with self.assertRaises(RuntimeError) as cm:
            self._interpolate(x, "area", False, True)

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

    @disableScriptTest()
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

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_listunpack(self):
        class ListUnpack(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x):
                a, b = x.shape
                return x.new_zeros((a, b))

        x = torch.randn(2, 3)
        self.run_test(ListUnpack(), x)

        class ListUnpackSlice(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x):
                a, b = x.shape[2:]
                return x.new_zeros((a, b))

        x = torch.randn(2, 3, 4, 5)
        self.run_test(ListUnpackSlice(), x)

    def test_pow(self):
        class PowModule(torch.nn.Module):
            def forward(self, x, y):
                return x.pow(y)

        x = torch.randn(2, 3, 4)
        y = torch.randn(2, 3, 4)
        self.run_test(PowModule(), (x, y))

        x = torch.randint(10, (2, 3, 4))
        y = torch.randint(10, (2, 3, 4)).to(dtype=torch.int32)
        self.run_test(PowModule(), (x, y))

        x = torch.randint(10, (2, 3, 4))
        y = torch.randint(10, (2, 3, 4))
        self.run_test(PowModule(), (x, y))

        x = torch.randn(2, 3, 4).to(dtype=torch.float64)
        y = torch.randint(10, (2, 3, 4))
        self.run_test(PowModule(), (x, y))

        class PowModule2(torch.nn.Module):
            def forward(self, x):
                return torch.pow(2, x)

        x = torch.randn(1, 10)
        self.run_test(PowModule2(), (x,))

        x = torch.randint(10, (2, 3, 4))
        self.run_test(PowModule2(), (x,))

        x = torch.randn(1, 10).to(dtype=torch.float64)
        self.run_test(PowModule2(), (x,))

        class PowModule3(torch.nn.Module):
            def forward(self, x, y):
                return y[torch.pow(2, x)]

        x = torch.randint(5, (2, 3, 4))
        y = torch.rand(100)
        self.run_test(PowModule3(), (x, y))

    def test_std(self):
        class StandardDeviation(torch.nn.Module):
            def forward(self, input):
                return torch.std(input, unbiased=False)

        x = torch.randn(2, 3, 4)
        model = StandardDeviation()
        self.run_test(model, x)

        class StandardDeviationUnbiased(torch.nn.Module):
            def forward(self, input):
                return torch.std(input, unbiased=True)

        model = StandardDeviationUnbiased()
        self.run_test(model, x)

    def test_std_along_dims(self):
        class StandardDeviation(torch.nn.Module):
            def forward(self, input):
                return torch.std(input, dim=(0, 1), unbiased=False)

        x = torch.randn(2, 3, 4)
        model = StandardDeviation()
        self.run_test(model, x)

        class StandardDeviationUnbiased(torch.nn.Module):
            def forward(self, input):
                return torch.std(input, dim=(0, 1), unbiased=True)

        x = torch.randn(2, 3, 4)
        model = StandardDeviationUnbiased()
        self.run_test(model, x)

    def test_std_keepdim(self):
        class StandardDeviation(torch.nn.Module):
            def forward(self, input):
                return torch.std(input, dim=(0, 1), unbiased=False, keepdim=True)

        x = torch.randn(2, 3, 4)
        model = StandardDeviation()
        self.run_test(model, x)

        class StandardDeviationUnbiased(torch.nn.Module):
            def forward(self, input):
                return torch.std(input, dim=(0, 1), unbiased=True, keepdim=True)

        x = torch.randn(2, 3, 4)
        model = StandardDeviationUnbiased()
        self.run_test(model, x)

    def test_var(self):
        class Variance(torch.nn.Module):
            def forward(self, input):
                return torch.var(input, unbiased=False)

        x = torch.randn(2, 3, 4)
        model = Variance()
        self.run_test(model, x)

        class VarianceUnbiased(torch.nn.Module):
            def forward(self, input):
                return torch.var(input, unbiased=True)

        model = VarianceUnbiased()
        self.run_test(model, x)

        class VarianceSqrt(torch.nn.Module):
            def forward(self, input):
                y = torch.var(input, 1)
                return torch.sqrt(y + 1e-8)

        x = torch.randn(1, 2, 3, 300, 300)
        model = VarianceSqrt()
        self.run_test(model, x)

    def test_var_along_dims(self):
        class Variance(torch.nn.Module):
            def forward(self, input):
                return torch.var(input, dim=(0, 1), unbiased=False)

        x = torch.randn(2, 3, 4)
        model = Variance()
        self.run_test(model, x)

        class VarianceUnbiased(torch.nn.Module):
            def forward(self, input):
                return torch.var(input, dim=(0, 1), unbiased=True)

        x = torch.randn(2, 3, 4)
        model = VarianceUnbiased()
        self.run_test(model, x)

    def test_var_keepdim(self):
        class Variance(torch.nn.Module):
            def forward(self, input):
                return torch.var(input, dim=(0, 1), unbiased=False, keepdim=True)

        x = torch.randn(2, 3, 4)
        model = Variance()
        self.run_test(model, x)

        class VarianceUnbiased(torch.nn.Module):
            def forward(self, input):
                return torch.var(input, dim=(0, 1), unbiased=True, keepdim=True)

        x = torch.randn(2, 3, 4)
        model = VarianceUnbiased()
        self.run_test(model, x)

    def test_var_mean(self):
        class Variance(torch.nn.Module):
            def forward(self, input):
                return torch.var_mean(input, unbiased=False)

        x = torch.randn(2, 3, 4)
        model = Variance()
        self.run_test(model, x)

        class VarianceUnbiased(torch.nn.Module):
            def forward(self, input):
                return torch.var_mean(input, unbiased=True)

        model = VarianceUnbiased()
        self.run_test(model, x)

    def test_var_mean_along_dims(self):
        class Variance(torch.nn.Module):
            def forward(self, input):
                return torch.var_mean(input, dim=(0, 1), unbiased=False)

        x = torch.randn(2, 3, 4)
        model = Variance()
        self.run_test(model, x)

        class VarianceUnbiased(torch.nn.Module):
            def forward(self, input):
                return torch.var_mean(input, dim=(0, 1), unbiased=True)

        x = torch.randn(2, 3, 4)
        model = VarianceUnbiased()
        self.run_test(model, x)

    def test_var_mean_mixed_dims(self):
        class ReverseDims(torch.nn.Module):
            def forward(self, input):
                return torch.var_mean(input, dim=(2, 1), unbiased=False)

        x = torch.randn(2, 3, 4)
        model = ReverseDims()
        self.run_test(model, x)

        class SkipDims(torch.nn.Module):
            def forward(self, input):
                return torch.var_mean(input, dim=(0, 2), unbiased=False)

        x = torch.randn(2, 3, 4)
        model = SkipDims()
        self.run_test(model, x)

        class NonZeroDims(torch.nn.Module):
            def forward(self, input):
                return torch.var_mean(input, dim=(1, 2), unbiased=False)

        x = torch.randn(2, 3, 4)
        model = NonZeroDims()
        self.run_test(model, x)

    def test_var_mean_keepdim(self):
        class Variance(torch.nn.Module):
            def forward(self, input):
                return torch.var_mean(input, dim=(0, 1), unbiased=False, keepdim=True)

        x = torch.randn(2, 3, 4)
        model = Variance()
        self.run_test(model, x)

        class VarianceUnbiased(torch.nn.Module):
            def forward(self, input):
                return torch.var_mean(input, dim=(0, 1), unbiased=True, keepdim=True)

        x = torch.randn(2, 3, 4)
        model = VarianceUnbiased()
        self.run_test(model, x)

    def test_std_mean(self):
        class StandardDeviation(torch.nn.Module):
            def forward(self, input):
                return torch.std_mean(input, unbiased=False)

        x = torch.randn(2, 3, 4)
        model = StandardDeviation()
        self.run_test(model, x)

        class StandardDeviationUnbiased(torch.nn.Module):
            def forward(self, input):
                return torch.std_mean(input, unbiased=True)

        model = StandardDeviationUnbiased()
        self.run_test(model, x)

    def test_std_mean_along_dims(self):
        class StandardDeviation(torch.nn.Module):
            def forward(self, input):
                return torch.std_mean(input, dim=(0, 1), unbiased=False)

        x = torch.randn(2, 3, 4)
        model = StandardDeviation()
        self.run_test(model, x)

        class VarianceUnbiased(torch.nn.Module):
            def forward(self, input):
                return torch.std_mean(input, dim=(0, 1), unbiased=True)

        x = torch.randn(2, 3, 4)
        model = VarianceUnbiased()
        self.run_test(model, x)

    def test_std_mean_keepdim(self):
        class StandardDeviation(torch.nn.Module):
            def forward(self, input):
                return torch.std_mean(input, dim=(0, 1), unbiased=False, keepdim=True)

        x = torch.randn(2, 3, 4)
        model = StandardDeviation()
        self.run_test(model, x)

        class StandardDeviationUnbiased(torch.nn.Module):
            def forward(self, input):
                return torch.std_mean(input, dim=(0, 1), unbiased=True, keepdim=True)

        x = torch.randn(2, 3, 4)
        model = StandardDeviationUnbiased()
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

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_narrow_dynamic(self):
        class NarrowModel(torch.nn.Module):
            def forward(self, input):
                return torch.narrow(input, 0, 0, input.shape[0] - 1)

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

    def test_select(self):
        class Select(torch.nn.Module):
            def forward(self, x):
                return x[:, 1]

        x = torch.randn(3, 4)
        self.run_test(Select(), x)

    def test_select_negative_index(self):
        class Select(torch.nn.Module):
            def forward(self, x):
                return x[:, -1]

        x = torch.randn(3, 4)
        self.run_test(Select(), x)

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
                # When sorted=False, order of elements in the outout tensors
                # are not expected to match between PyTorch and ORT
                topk_unsorted = torch.topk(x, k, largest=False, sorted=False)
                topk_sorted = torch.topk(x, k, largest=False, sorted=True)
                return topk_sorted, torch.sort(topk_unsorted.values).values

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

    @skipIfUnsupportedOpsetVersion([7])
    def test_normalize(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.nn.functional.normalize(x)

        x = torch.randn(3, 3)
        self.run_test(Model(), x)

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

    def test_batchnorm1d_norunningstats(self):
        x = torch.randn(10, 10)
        model = torch.nn.BatchNorm1d(10, track_running_stats=False)
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

    def test_batchnorm2d_norunningstats(self):
        x = torch.randn(10, 3, 128, 128)
        model = torch.nn.BatchNorm2d(3, track_running_stats=False)
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
    def test_scatter_with_scalar(self):
        class ScatterModel(torch.nn.Module):
            def forward(self, input, indices):
                values = 1.0
                return input.scatter(1, indices, values)

        input = torch.tensor([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]], dtype=torch.float64)
        indices = torch.tensor([[1, 0], [0, 1], [0, 1]], dtype=torch.int64)
        self.run_test(ScatterModel(), input=(input, indices))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_scatter_with_scalar_different_types(self):
        # Tests the case when scalar src (updates values) type is different
        # from self type. Happens only with scalar src - PyTorch does not
        # allow this when src is a tensor.
        class ScatterModel(torch.nn.Module):
            def forward(self, input, indices):
                values = 1.0
                return input.scatter(1, indices, values)

        input = torch.tensor([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]], dtype=torch.float32)
        indices = torch.tensor([[1, 0], [0, 1], [0, 1]], dtype=torch.int64)
        self.run_test(ScatterModel(), input=(input, indices))

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

        @torch.jit.script
        def scatter_sum(src: torch.Tensor, index: torch.Tensor):
            size = src.size()
            out = torch.zeros(size, dtype=src.dtype)
            return out.scatter_add_(1, index, src)

        class ScatterModel(torch.nn.Module):
            def forward(self, src, index):
                return scatter_sum(src, index)

        src = torch.rand(3, 2)
        index = torch.tensor([[0, 1], [0, 1], [0, 1]], dtype=torch.int64)
        self.run_test(ScatterModel(), (src, index))

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

    @disableScriptTest()  # RuntimeError: Python type cannot be used as a value
    @skipIfUnsupportedMinOpsetVersion(11)
    def test_gather_constant_fold(self):
        class GatherModule(torch.nn.Module):
            def __init__(self):
                super(GatherModule, self).__init__()
                self.register_buffer("weight", torch.ones(5))

            def forward(self, x):
                # shape is of rank 0
                shape = self.weight.shape[0]
                m = 5 - shape
                return x.clamp(min=m)

        x = torch.randn(1)
        self.run_test(GatherModule(), (x,))

        class GatherModule(torch.nn.Module):
            def __init__(self):
                super(GatherModule, self).__init__()
                self.register_buffer("weight", torch.ones(2))

            def forward(self, x):
                # shape is of rank 0
                shape = self.weight.shape[0]
                pad = [1, shape, shape, shape]
                zero_pad = torch.nn.ZeroPad2d(pad)
                return zero_pad(x)

        x = torch.randn(1, 3, 2)
        self.run_test(GatherModule(), (x,))

    @skipIfUnsupportedOpsetVersion([13])
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
        size = torch.tensor(-1)
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

    def _test_reduced_ops(self, op):
        class ReducedOpModule(torch.nn.Module):
            def forward(self, input):
                return op(input, dim=-1)

        if op != torch.mean:  # torch.mean only supports float types
            x = torch.randint(10, (4, 4), dtype=torch.uint8)
            self.run_test(ReducedOpModule(), x)

            x = torch.randint(10, (4, 4), dtype=torch.int8)
            self.run_test(ReducedOpModule(), x)

            x = torch.randint(10, (4, 4), dtype=torch.int16)
            self.run_test(ReducedOpModule(), x)

            x = torch.randint(10, (4, 4), dtype=torch.int32)
            self.run_test(ReducedOpModule(), x)

            x = torch.randint(10, (4, 4), dtype=torch.int64)
            self.run_test(ReducedOpModule(), x)

        # torch.mean only supports float types
        # ORT does not support double ReduceProd for double
        if op != torch.prod and op != torch.mean:
            x = torch.randn(4, 5, dtype=torch.double)
            self.run_test(ReducedOpModule(), x)

        if op != torch.prod:  # torch.prod not implemented for Half
            x = torch.randn(4, 4, dtype=torch.half)
            self.run_test(ReducedOpModule(), x)

        x = torch.randn(4, 5, dtype=torch.float)
        self.run_test(ReducedOpModule(), x)

    def test_reduced_sum(self):
        return self._test_reduced_ops(op=torch.sum)

    def test_reduced_mean(self):
        return self._test_reduced_ops(op=torch.mean)

    def test_reduced_prod(self):
        return self._test_reduced_ops(op=torch.prod)

    def test_reduced_min_max(self):
        class ReducedMinMaxModule(torch.nn.Module):
            def forward(self, input):
                return torch.min(input, dim=-1)[0], torch.max(input, dim=0)[0]
        x = torch.randint(10, (4, 4), dtype=torch.int32)
        self.run_test(ReducedMinMaxModule(), x)

        x = torch.randint(10, (4, 4), dtype=torch.int64)
        self.run_test(ReducedMinMaxModule(), x)

        x = torch.randn(4, 5, dtype=torch.float)
        self.run_test(ReducedMinMaxModule(), x)

    def test_reduce_log_sum_exp(self):
        class ReduceLogSumExpModel(torch.nn.Module):
            def forward(self, input):
                a = torch.logsumexp(input, dim=0)
                b = torch.logsumexp(input, dim=(0, 1))
                return a + b

        x = torch.randn(4, 4, requires_grad=True)
        self.run_test(ReduceLogSumExpModel(), x)

    def test_softmax(self):
        for i in range(-4, 3):
            model = torch.nn.Softmax(dim=i)
            input = torch.randn(3, 4, 5, 6)
            self.run_test(model, input)

            class SoftmaxUnknownRank(torch.nn.Module):
                def __init__(self, i):
                    super().__init__()
                    self.softmax = torch.nn.Softmax(dim=i)

                def forward(self, x):
                    return self.softmax(x.reshape(3, 4, 5, 6))

            model = torch.jit.script(SoftmaxUnknownRank(i))
            self.run_test(model, input)

    def test_softmax_large_values(self):
        input = torch.tensor([[-1e12, -1e12, -1e12], [1e12, 0.0, -5.0], [3.0, 4.0, 5.0]])
        for i in range(-2, 1):
            model = torch.nn.Softmax(dim=i)
            self.run_test(model, input)

            class SoftmaxUnknownRank(torch.nn.Module):
                def __init__(self, i):
                    super().__init__()
                    self.softmax = torch.nn.Softmax(dim=i)

                def forward(self, x):
                    return self.softmax(x.reshape(3, 3))

            model = torch.jit.script(SoftmaxUnknownRank(i))
            self.run_test(model, input)

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

    def test_logsoftmax_dtype(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                return torch.nn.functional.log_softmax(x, dim=1, dtype=torch.float64)

        x = torch.randn(3, 4, 5, requires_grad=True)
        self.run_test(Model(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    @disableScriptTest()  # scripting prim_dtype
    def test_lstm_no_hidden(self):
        class LSTMModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.rnn = torch.nn.LSTM(input_size=16, hidden_size=16)

            def forward(self, x):
                return self.rnn(x)

        input = torch.randn((10, 16, 16))
        self.run_test(LSTMModel(), (input,))

    @skipIfUnsupportedMinOpsetVersion(9)
    @disableScriptTest()  # scripting prim_dtype
    def test_lstm_proj_no_hidden(self):
        class LSTMModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.rnn = torch.nn.LSTM(input_size=16, hidden_size=16, proj_size=8)

            def forward(self, x):
                return self.rnn(x)

        input = torch.randn((10, 16, 16))
        with self.assertRaises(RuntimeError):
            self.run_test(LSTMModel(), (input,))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_lstm(self):
        class LSTMModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.rnn = torch.nn.LSTM(RNN_INPUT_SIZE, RNN_HIDDEN_SIZE, 1, bidirectional=False)

            def forward(self, x, h0, c0):
                return self.rnn(x, (h0, c0))

        input = torch.randn(RNN_SEQUENCE_LENGTH, BATCH_SIZE, RNN_INPUT_SIZE)
        h0 = torch.randn(1, BATCH_SIZE, RNN_HIDDEN_SIZE)
        c0 = torch.randn(1, BATCH_SIZE, RNN_HIDDEN_SIZE)
        self.run_test(LSTMModel(), (input, h0, c0))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_lstm_default_init_state(self):
        class LSTMModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.rnn = torch.nn.LSTM(RNN_INPUT_SIZE, RNN_HIDDEN_SIZE, 1, bidirectional=False)

            def forward(self, x):
                return self.rnn(x)

        input = torch.randn(RNN_SEQUENCE_LENGTH, BATCH_SIZE, RNN_INPUT_SIZE)
        self.run_test(LSTMModel(), input)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_lstm_fixed_batch_size(self):
        class LSTMModel(torch.nn.Module):
            def __init__(self):
                super(LSTMModel, self).__init__()
                self.lstm = torch.nn.LSTM(RNN_INPUT_SIZE, RNN_HIDDEN_SIZE, 1, bidirectional=False)
                self.RNN_HIDDEN_SIZE = RNN_HIDDEN_SIZE

            def forward(self, input):
                batch_size = input.size()[1]
                h0 = torch.ones([1, batch_size, self.RNN_HIDDEN_SIZE])
                c0 = torch.ones([1, batch_size, self.RNN_HIDDEN_SIZE])
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
                self.lstm = torch.nn.LSTM(RNN_INPUT_SIZE, RNN_HIDDEN_SIZE, 1, bidirectional=False)
                self.RNN_HIDDEN_SIZE = RNN_HIDDEN_SIZE

            def forward(self, input):
                batch_size = input.size()[1]
                h0 = torch.ones([1, batch_size, self.RNN_HIDDEN_SIZE])
                c0 = torch.ones([1, batch_size, self.RNN_HIDDEN_SIZE])
                return self.lstm(input, (h0, c0))

        model = LSTMModel()
        input = torch.randn(RNN_SEQUENCE_LENGTH, 1, RNN_INPUT_SIZE)
        # verify with different input of different batch size
        input2 = torch.randn(RNN_SEQUENCE_LENGTH, BATCH_SIZE, RNN_INPUT_SIZE)
        self.run_test(model, input, input_names=["input.1"], dynamic_axes={'input.1' : {0 : 'seq', 1 : 'batch'}},
                      test_with_inputs=[input2])

    def test_lstm_constant_folding(self):
        class LstmNet(torch.nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, bidirectional):
                super(LstmNet, self).__init__()
                self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, bidirectional=bidirectional)

            def forward(self, input, initial_state: Tuple[torch.Tensor, torch.Tensor]):
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

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_lstm_no_bias(self):
        class LstmNet(torch.nn.Module):
            def __init__(self, num_layers, bidirectional):
                super(LstmNet, self).__init__()
                self.lstm = torch.nn.LSTM(RNN_INPUT_SIZE, RNN_HIDDEN_SIZE, num_layers, bias=False, bidirectional=bidirectional)

            def forward(self, input, initial_state: Tuple[torch.Tensor, torch.Tensor]):
                return self.lstm(input, initial_state)

        def get_LstmNet_model_and_inputs(num_layers, bidirectional):
            input = torch.randn(RNN_SEQUENCE_LENGTH, BATCH_SIZE, RNN_INPUT_SIZE)
            num_directions = 2 if bidirectional else 1
            model = LstmNet(num_layers, bidirectional)
            h0 = torch.randn(num_layers * num_directions, BATCH_SIZE, RNN_HIDDEN_SIZE)
            c0 = torch.randn(num_layers * num_directions, BATCH_SIZE, RNN_HIDDEN_SIZE)
            return model, (input, (h0, c0))

        num_layers = [1, 1, 2, 3]
        bidirectional = [True, False, True, False]
        models_and_inputs = [get_LstmNet_model_and_inputs(n, b) for n, b in zip(num_layers, bidirectional)]
        for model, input in models_and_inputs:
            self.run_test(model, input)

    @disableScriptTest()
    def test_rnn_no_bias(self):
        def make_model(layers, packed_sequence):
            batch_first = True if packed_sequence == 2 else False
            model = torch.nn.RNN(RNN_INPUT_SIZE, RNN_HIDDEN_SIZE, layers, bidirectional=False,
                                 batch_first=batch_first, bias=False)

            if packed_sequence == 1:
                model = RnnModelWithPackedSequence(model, False)
            if packed_sequence == 2:
                model = RnnModelWithPackedSequence(model, True)
            return model

        def make_input(batch_size, layers, packed_sequence):
            batch_first = True if packed_sequence == 2 else False
            seq_lengths = np.random.randint(1, RNN_SEQUENCE_LENGTH + 1, size=batch_size)
            seq_lengths = list(reversed(sorted(map(int, seq_lengths))))
            inputs = [torch.randn(l, RNN_INPUT_SIZE) for l in seq_lengths]
            inputs = rnn_utils.pad_sequence(inputs, batch_first=batch_first)
            inputs = [inputs]

            h0 = torch.randn(layers, batch_size, RNN_HIDDEN_SIZE)
            inputs.append(h0)
            if packed_sequence != 0:
                inputs.append(torch.IntTensor(seq_lengths))
            if len(inputs) == 1:
                input = inputs[0]
            else:
                input = tuple(inputs)
            return input

        layers = [1, 3, 1, 3, 1, 3]
        packed_sequence = [0, 0, 1, 1, 2, 2]
        models = [make_model(l, p) for l, p in zip(layers, packed_sequence)]
        inputs = [make_input(RNN_BATCH_SIZE, l, p) for l, p in zip(layers, packed_sequence)]

        for model, input in zip(models, inputs):
            self.run_test(model, input, batch_size=RNN_BATCH_SIZE)

    def test_gru_no_bias(self):
        class GruNet(torch.nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, bidirectional):
                super(GruNet, self).__init__()
                self.mygru = torch.nn.GRU(input_size, hidden_size, num_layers, bidirectional=bidirectional, bias=False)

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

        input_size = [7, 5]
        hidden_size = [3, 4]
        num_layers = [2, 3]
        batch_size = [3, 4]
        seq_len = [5, 7]
        bidirectional = [True, False]
        models_and_inputs = [get_GruNet_model_and_inputs(i, h, n, b, s, bi)
                             for i, h, n, b, s, bi in zip(input_size, hidden_size, num_layers, batch_size, seq_len, bidirectional)]
        for model, input in models_and_inputs:
            self.run_test(model, input, do_constant_folding=True)

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

    def _test_compare_ops(self, model, num_inputs):
        x_float = torch.randn(1, 2, 3, 4, requires_grad=True)
        x_int = torch.randint(10, (3, 4), dtype=torch.int32)
        if num_inputs > 1:
            y_float = torch.randn(1, 2, 3, 4, requires_grad=True)
            y_int = torch.randint(10, (3, 4), dtype=torch.int32)
            self.run_test(model, (x_float, y_float))
            self.run_test(model, (x_float, y_int))
            self.run_test(model, (x_int, y_float))
            self.run_test(model, (x_int, y_int))
        else:
            self.run_test(model, x_float)
            self.run_test(model, x_int)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_logical_and(self):
        class AndModel(torch.nn.Module):
            def forward(self, x, y):
                return torch.logical_and(x, y)

        x = torch.randint(0, 2, (5, 5), dtype=torch.bool)
        y = torch.randint(0, 2, (5, 5), dtype=torch.bool)
        self.run_test(AndModel(), input=(x, y))

        x = torch.randint(10, (5, 5), dtype=torch.int32)
        y = torch.randint(10, (5, 5), dtype=torch.int32)
        self.run_test(AndModel(), input=(x, y))

        x = torch.randint(10, (5, 5), dtype=torch.double)
        y = torch.randint(10, (5, 5), dtype=torch.double)
        self.run_test(AndModel(), input=(x, y))

        x = torch.randint(10, (2, 3, 5), dtype=torch.float32)
        y = torch.randint(10, (2, 3, 5), dtype=torch.long)
        self.run_test(AndModel(), input=(x, y))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_logical_or(self):
        class OrModel(torch.nn.Module):
            def forward(self, x, y):
                return torch.logical_or(x, y)

        x = torch.randint(0, 2, (5, 5), dtype=torch.bool)
        y = torch.randint(0, 2, (5, 5), dtype=torch.bool)
        self.run_test(OrModel(), input=(x, y))

        x = torch.randint(10, (5, 5), dtype=torch.int32)
        y = torch.randint(10, (5, 5), dtype=torch.int32)
        self.run_test(OrModel(), input=(x, y))

        x = torch.randint(10, (5, 5), dtype=torch.double)
        y = torch.randint(10, (5, 5), dtype=torch.double)
        self.run_test(OrModel(), input=(x, y))

        x = torch.randint(10, (2, 3, 5), dtype=torch.float32)
        y = torch.randint(10, (2, 3, 5), dtype=torch.long)
        self.run_test(OrModel(), input=(x, y))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_logical_xor(self):
        class XorModel(torch.nn.Module):
            def forward(self, x, y):
                return torch.logical_xor(x, y)

        x = torch.randint(0, 2, (5, 5), dtype=torch.bool)
        y = torch.randint(0, 2, (5, 5), dtype=torch.bool)
        self.run_test(XorModel(), input=(x, y))

        x = torch.randint(10, (5, 5), dtype=torch.int32)
        y = torch.randint(10, (5, 5), dtype=torch.int32)
        self.run_test(XorModel(), input=(x, y))

        x = torch.randint(10, (5, 5), dtype=torch.double)
        y = torch.randint(10, (5, 5), dtype=torch.double)
        self.run_test(XorModel(), input=(x, y))

        x = torch.randint(10, (2, 3, 5), dtype=torch.float32)
        y = torch.randint(10, (2, 3, 5), dtype=torch.long)
        self.run_test(XorModel(), input=(x, y))

    def test_gt(self):
        class GreaterModel(torch.nn.Module):
            def forward(self, input, other):
                return input > other
        self._test_compare_ops(GreaterModel(), 2)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_ge(self):
        class GreaterOrEqualModel(torch.nn.Module):
            def forward(self, input, other):
                return input >= other
        self._test_compare_ops(GreaterOrEqualModel(), 2)

    def test_gt_scalar(self):
        class GreaterModel(torch.nn.Module):
            def forward(self, input):
                return input > 1
        self._test_compare_ops(GreaterModel(), 1)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_ge_scalar(self):
        class GreaterOrEqualModel(torch.nn.Module):
            def forward(self, input):
                return input >= 1
        self._test_compare_ops(GreaterOrEqualModel(), 1)

    def test_lt(self):
        class LessModel(torch.nn.Module):
            def forward(self, input, other):
                return input > other
        self._test_compare_ops(LessModel(), 2)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_le(self):
        class LessOrEqualModel(torch.nn.Module):
            def forward(self, input, other):
                return input <= other
        self._test_compare_ops(LessOrEqualModel(), 2)

    def test_lt_scalar(self):
        class LessModel(torch.nn.Module):
            def forward(self, input):
                return input < 1
        self._test_compare_ops(LessModel(), 1)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_le_scalar(self):
        class LessOrEqualModel(torch.nn.Module):
            def forward(self, input):
                return input <= 1
        self._test_compare_ops(LessOrEqualModel(), 1)

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

    def _argmin_argmax_model(self, input):
        class ArgminArgmaxModel(torch.nn.Module):
            def forward(self, input):
                return torch.argmin(input), \
                    torch.argmax(input), \
                    torch.argmin(input, keepdim=True), \
                    torch.argmax(input, keepdim=True)

        self.run_test(ArgminArgmaxModel(), input)

    def test_argmin_argmax(self):
        input = torch.randn(7, 3, 5)
        self._argmin_argmax_model(input)

    # Argmin and Argmax with "select_last_index" is not supprted before opset 12
    # "select_last_index" was added in opset 12 to deal with corner case where the
    # same value appears multiple times in the tensor
    @skipIfUnsupportedMinOpsetVersion(12)
    def test_argmin_argmax_select_last_index(self):
        input = torch.tensor([[1., 2., 3.],
                             [1., 1., 2.]])
        self._argmin_argmax_model(input)

        input = torch.ones(7, 3, 5)
        self._argmin_argmax_model(input)

    def test_repeat(self):
        class RepeatModel(torch.nn.Module):
            def forward(self, x, y):
                x2 = x.repeat(y.shape[0], 1)
                y1 = y.view(-1, 1)
                return x2 + y1

        x = torch.tensor([1, 2, 3])
        y = torch.tensor([4, 5, 8, 9])
        self.run_test(RepeatModel(), (x, y))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_repeat_interleave(self):
        class FlattenModel(torch.nn.Module):
            def forward(self, x):
                return x.repeat_interleave(2)

        x = torch.tensor([1, 2, 3])
        self.run_test(FlattenModel(), (x,))

        class DimsModel(torch.nn.Module):
            def forward(self, x):
                return x.repeat_interleave(4, dim=1)

        x = torch.tensor([[1, 2], [3, 4]])
        self.run_test(DimsModel(), (x,))

        class DimsModel2(torch.nn.Module):
            def forward(self, x):
                repeats = torch.tensor([4])
                return torch.repeat_interleave(x, repeats, dim=1)

        x = torch.tensor([[1, 2], [3, 4]])
        self.run_test(DimsModel2(), (x,))

        class RepeatsDimsModel(torch.nn.Module):
            def forward(self, x):
                repeats = torch.tensor([1, 2])
                return torch.repeat_interleave(x, repeats, dim=0)

        x = torch.tensor([[1, 2], [3, 4]])
        self.run_test(RepeatsDimsModel(), (x,))

        class RepeatsDimsModel2(torch.nn.Module):
            def forward(self, x):
                repeats = torch.tensor([1, 2])
                return torch.repeat_interleave(x, repeats, dim=1)

        x = torch.tensor([[1, 2], [3, 4]])
        self.run_test(RepeatsDimsModel2(), (x,))

    def test_view(self):
        class ViewModel(torch.nn.Module):
            def forward(self, input):
                return input.view(4, 24)

        x = torch.randint(10, (4, 2, 3, 4), dtype=torch.int32)
        self.run_test(ViewModel(), x)

    def test_view_dynamic(self):
        class ViewModel(torch.nn.Module):
            def forward(self, input, other):
                return input.view(other.shape)

        x = torch.randn(2, 3, 4)
        shape = torch.randn(6, 4)
        self.run_test(ViewModel(), (x, shape))

    def test_view_dynamic_zero_dim(self):
        class ViewModel(torch.nn.Module):
            def forward(self, input):
                input = input.view(-1, 2)
                return input.view(1, -1)

        x = torch.ones(2)
        another_x = torch.empty((0,))
        self.run_test(ViewModel(), x, test_with_inputs=[another_x],
                      input_names=['input_1'], dynamic_axes={'input_1': [0, ]})

    def test_view_as(self):
        class ViewModel(torch.nn.Module):
            def forward(self, input, other):
                return input.view_as(other)

        x = torch.randn(2, 3, 4)
        y = torch.randn(6, 4)
        self.run_test(ViewModel(), (x, y))

    def test_linear(self):
        class LinearModel(torch.nn.Module):
            def __init__(self):
                super(LinearModel, self).__init__()
                self.fc = torch.nn.Linear(16, 16)

            def forward(self, x):
                out = self.fc(x)
                out = self.fc(out)
                return out

        x = torch.randn(3, 16)
        self.run_test(LinearModel(), (x,))

        class LinearModel(torch.nn.Module):
            def forward(self, input, weight, bias):
                return torch.nn.functional.linear(input, weight, bias)

        # input of rank 2
        x = torch.randn(2, 2)
        y = torch.randn(2, 2)
        z = torch.randn(1)
        self.run_test(LinearModel(), (x, y, z))

        # input of rank 3
        x = torch.randn(3, 3, 3)
        y = torch.randn(3, 3)
        z = torch.randn(1)
        self.run_test(LinearModel(), (x, y, z))

    @disableScriptTest()
    def test_weight_norm(self):
        # addmm for 3-d inputs converts to onnx::MatMul
        model = torch.nn.utils.weight_norm(torch.nn.Linear(5, 10), dim=1)
        x = torch.randn(3, 4, 5, requires_grad=True)
        self.run_test(model, x)

        # addmm for 2-d inputs converts to onnx::Gemm
        model = torch.nn.utils.weight_norm(torch.nn.Linear(5, 10), dim=1)
        x = torch.randn(4, 5, requires_grad=True)
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

    @disableScriptTest()
    def test_weight_norm_nodim(self):
        # addmm for 3-d inputs converts to onnx::MatMul
        model = torch.nn.utils.weight_norm(torch.nn.Linear(5, 10), dim=None)
        x = torch.randn(3, 4, 5, requires_grad=True)
        self.run_test(model, x)

        # addmm for 2-d inputs converts to onnx::Gemm
        model = torch.nn.utils.weight_norm(torch.nn.Linear(5, 10), dim=None)
        x = torch.randn(4, 5, requires_grad=True)
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

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_flatten_dynamic_axes(self):
        class MyModule(torch.nn.Module):
            def forward(self, x):
                return torch.flatten(x, start_dim=2, end_dim=3)

        batch_size = 3
        x = torch.randn(batch_size, 5, 4, 5)
        y = torch.randn(5, 5, 4, 5)
        model = MyModule()
        self.run_test(model, x, test_with_inputs=[y],
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={'input' : {0 : 'batch_size'},
                                    'output' : {0 : 'batch_size'}})

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

    @disableScriptTest()  # torch.nonzero(x, as_tuple=True) is not scriptable.
    @skipIfUnsupportedMinOpsetVersion(9)
    def test_nonzero(self):
        class NonzeroModel(torch.nn.Module):
            def forward(self, x):
                return x.nonzero(), x.nonzero(as_tuple=True)

        x = torch.randn(60).index_fill_(0, torch.randint(0, 60, (20,)), 0).view(3, 4, 5)
        self.run_test(NonzeroModel(), (x,))

    def test_unbind(self):
        class UnbindModel(torch.nn.Module):
            def forward(self, input):
                _, out, _ = input.unbind()
                return out

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

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_len_list(self):
        class LenListModel(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, input):
                return torch.ones(len(input.shape))

        x = torch.randn(4, 5)
        self.run_test(LenListModel(), x)

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

    @disableScriptTest()  # scripting tests run for opsets > 11. See: test_split_script
    def test_split(self):
        class SplitModel(torch.nn.Module):
            def forward(self, input):
                return input.split([2, 1, 2]), input.split([3, 2])[0]

        x = torch.randn(5, 4, 3)
        self.run_test(SplitModel(), x)

        class SplitModel2(torch.nn.Module):
            def forward(self, input):
                return input.split([2, 1, 1], -2), input.split([2, 2], -2)[-1]

        x = torch.randn(5, 4, 3)
        self.run_test(SplitModel2(), x)

        class SplitModel3(torch.nn.Module):
            def forward(self, input):
                return input.split([2, 1, 2])

        x = torch.randn(5, 4, 3)
        self.run_test(SplitModel3(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_split_script(self):
        class SplitModel(torch.nn.Module):
            def forward(self, input):
                return input.split([2, 1, 2]), input.split([3, 2])[0]

        x = torch.randn(5, 4, 3)
        self.run_test(SplitModel(), x)

        class SplitModel2(torch.nn.Module):
            def forward(self, input):
                return input.split([2, 1, 1], -2), input.split([2, 2], -2)[-1]

        x = torch.randn(5, 4, 3)
        self.run_test(SplitModel2(), x)

        class SplitModel3(torch.nn.Module):
            def forward(self, input):
                return input.split([2, 1, 2])

        x = torch.randn(5, 4, 3)
        self.run_test(SplitModel3(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    @disableScriptTest()
    def test_split_size_as_list(self):
        class SplitModel(torch.nn.Module):
            def forward(self, input, split_sizes: List[int]):
                out = []
                split_list: List[torch.Tensor] = input.split(split_sizes)

                for ob in split_list:
                    out.append(ob)
                return torch.cat(out, dim=0)

        x = torch.randn(6, 4, 3)
        split_sizes = [torch.tensor(2), torch.tensor(4)]
        self.run_test(SplitModel(), (x, split_sizes))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_split_size_with_slice(self):
        class SplitModule(torch.nn.Module):
            def forward(self, x, y, t):
                splits = (x.size(1), y.size(1))
                out, out2 = torch.split(t, splits, dim=1)
                return out, out2

        x = torch.randn(2, 3)
        y = torch.randn(2, 4)
        t = torch.randn(2, 7)
        self.run_test(SplitModule(), (x, y, t))

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

    @skipIfUnsupportedMinOpsetVersion(11)
    @disableScriptTest()
    def test_chunk(self):
        class ChunkModel(torch.nn.Module):
            def __init__(self, dim=1):
                super(ChunkModel, self).__init__()
                self.dim = dim

            def forward(self, x):
                return torch.chunk(x, 3, dim=self.dim)

        model = ChunkModel()
        model.eval()
        model_neg_dim = ChunkModel(-1)
        model_neg_dim.eval()
        x = torch.randn(1, 18)

        for dim_size_ in range(13, 16):
            y = torch.randn(1, dim_size_)
            self.run_test(model, x, test_with_inputs=[y],
                          input_names=['x'],
                          dynamic_axes={'x': {0: 'batch_size', 1: 'dims'}})

            self.run_test(model_neg_dim, x, test_with_inputs=[y],
                          input_names=['x'],
                          dynamic_axes={'x': {0: 'batch_size', 1: 'dims'}})

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
                res3 = []
                res4 = []
                for i in range(len(arr)):
                    res.append(arr[i].sum(0, False))
                    res1.append(arr[-1 - i].sum(0, False))
                    res2 += 1
                    res3 = res3 + [arr[i].sum(0, False)]
                    res4 += [arr[-1 - i].sum(0, False)]
                return res, res1, res2, torch.stack(res3), torch.stack(res4)

        model = ListLoopModel()
        inputs = torch.randn(16)
        self.run_test(model, inputs)

    @skipIfONNXShapeInference(False)
    @skipIfUnsupportedMinOpsetVersion(11)
    def test_loop_transpose(self):
        class LoopModel(torch.nn.Module):
            def forward(self, x):
                res = torch.zeros_like(x[0])
                for i in range(x.size(0)):
                    res += x[0].transpose(0, 1)
                return res

        model = torch.jit.script(LoopModel())
        x = torch.randn(5, 3, 3)
        self.run_test(model, x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_loop_multi_dim(self):
        class LoopMultiDimModel(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x, y):
                for x_ in torch.flip(x.narrow(0, 0, 7), [0]):
                    y = x_[0][y]
                return y

        model = LoopMultiDimModel()
        x = torch.randint(0, 5, (8, 1, 17), dtype=torch.long)
        y = torch.ones(1, dtype=torch.long)
        self.run_test(model, (x, y))

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
                res += [tensors[3], tensors[4]]
                res = res + [tensors[5]]
                return torch.ones(len(res))

        model = ListModel()
        inputs = torch.randn(16, 1)
        self.run_test(model, inputs)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_list_append(self):
        class ListModel(torch.nn.Module):
            def forward(self, x, y):
                res = []
                for i in range(x.size(0)):
                    res += [torch.matmul(x[i], y)]
                return res

        model = torch.jit.script(ListModel())
        x = torch.randn(16, 3, 4)
        y = torch.randn(4, 5)
        self.run_test(model, (x, y))

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_list_append_nested(self):
        class ListModel(torch.nn.Module):
            def forward(self, x, y):
                res = []
                for i in range(x.size(0)):
                    for j in range(x.size(1)):
                        res += [torch.matmul(x[i][j], y)]
                return res

        model = torch.jit.script(ListModel())
        x = torch.randn(4, 4, 3, 4)
        y = torch.randn(4, 5)
        self.run_test(model, (x, y))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_list_pop(self):
        class ListModel(torch.nn.Module):
            def forward(self, x, y):
                res = []
                for i in range(x.size(0)):
                    res += [torch.matmul(x[i], y)]
                res.pop()
                return res

        model = torch.jit.script(ListModel())
        x = torch.randn(16, 3, 4)
        y = torch.randn(4, 5)
        self.run_test(model, (x, y))

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_list_pop_nested(self):
        class ListModel(torch.nn.Module):
            def forward(self, x, y):
                res = []
                for i in range(x.size(0)):
                    for j in range(x.size(1)):
                        res += [torch.matmul(x[i][j], y)]
                        res.pop()
                    res += [torch.matmul(x[i][0], y)]
                return res

        model = torch.jit.script(ListModel())
        x = torch.randn(4, 4, 3, 4)
        y = torch.randn(4, 5)
        self.run_test(model, (x, y))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_list_del(self):
        class ListModel(torch.nn.Module):
            def forward(self, x, y):
                res = []
                for i in range(x.size(0)):
                    res += [torch.matmul(x[i], y)]
                del res[2]
                return res

        model = torch.jit.script(ListModel())
        x = torch.randn(16, 3, 4)
        y = torch.randn(4, 5)
        self.run_test(model, (x, y))

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_list_del_nested(self):
        class ListModel(torch.nn.Module):
            def forward(self, x, y):
                res = []
                for i in range(x.size(0)):
                    for j in range(x.size(1)):
                        res += [torch.matmul(x[i][j], y)]
                        del res[i]
                    res += [torch.matmul(x[i][0], y)]
                return res

        model = torch.jit.script(ListModel())
        x = torch.randn(4, 4, 3, 4)
        y = torch.randn(4, 5)
        self.run_test(model, (x, y))

    @unittest.skip("Enable this once remove is supported by pytorch")
    @skipIfUnsupportedMinOpsetVersion(11)
    def test_list_remove(self):
        class ListModel(torch.nn.Module):
            def forward(self, x, y):
                res = []
                for i in range(x.size(0)):
                    res += [torch.matmul(x[i], y)]
                # The following fails with pytorch
                # RuntimeError: Boolean value of Tensor with more than one value is ambiguous
                res.remove(res[2])
                return res

        model = torch.jit.script(ListModel())
        x = torch.randn(16, 3, 4)
        y = torch.randn(4, 5)
        self.run_test(model, (x, y))

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
    def test_eye(self):
        class TensorFactory(torch.nn.Module):
            def forward(self, x):
                return torch.eye(x.size()[1], 3), torch.eye(4, 4, dtype=torch.long), \
                    torch.eye(x.size()[1], 2, dtype=torch.long), torch.eye(x.shape[0]), \
                    torch.eye(x.shape[0], dtype=torch.float64)

        x = torch.randn(2, 3, 4)
        another_x = torch.randn(5, 6, 7)
        self.run_test(TensorFactory(), x, test_with_inputs=[another_x],
                      input_names=['input_1'], dynamic_axes={'input_1': [0, 1, 2]})

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_inplace_zero(self):
        class Zero_(torch.nn.Module):
            def forward(self, x):
                return x.zero_(), x

        x = torch.randn(2, 3, 4)
        self.run_test(Zero_(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_new_zeros(self):
        class Zero_(torch.nn.Module):
            def forward(self, x):
                return x.new_zeros(x.shape[1:2]), x.new_zeros(x.shape[2:], dtype=torch.long)

        x = torch.randn(2, 3, 4)
        self.run_test(Zero_(), x)

    @skipIfONNXShapeInference(True)
    @skipIfUnsupportedMinOpsetVersion(9)
    def test_tolist(self):
        class List(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, input):
                cur_shape = torch._shape_as_tensor(input)
                final_shape: List[int] = cur_shape.tolist()
                pad_tensor = torch.zeros([1, 2] + final_shape)
                return pad_tensor

        x = torch.randn(2, 3)
        self.run_test(List(), (x,))

    @skipIfUnsupportedMinOpsetVersion(9)
    @disableScriptTest()
    def test_list_pass(self):
        class Slice(torch.nn.Module):
            def forward(self, x, y):
                return x.new_zeros(x.shape[2:] + y.shape[1:])

        x = torch.randn(2, 3, 4, 5)
        y = torch.randn(1, 2, 3, 4)
        self.run_test(Slice(), (x, y))

        class Size(torch.nn.Module):
            def forward(self, x, y):
                return x.new_zeros(x.shape + y.shape)

        x = torch.randn(2, 3, 4)
        y = torch.randn(1, 2, 3)
        self.run_test(Size(), (x, y))

        class Array(torch.nn.Module):
            def forward(self, x, y):
                arr1 = [x.shape[0], x.shape[1], 2]
                arr2 = [y.shape[0], y.shape[1]]
                return x.new_zeros(arr1 + arr2)

        x = torch.randn(2, 3, 4)
        y = torch.randn(1, 2, 3)
        self.run_test(Array(), (x, y))

        class List(torch.nn.Module):
            def forward(self, x, y):
                l1 = list(x.shape)
                l2 = list(y.shape)
                return x.new_zeros(l1 + l2)

        x = torch.randn(2, 3, 4)
        y = torch.randn(1, 2, 3)
        self.run_test(List(), (x, y))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_new_empty(self):
        class Emtpy(torch.nn.Module):
            def forward(self, x):
                return x.new_empty(x.shape[0]).fill_(0), x.new_empty(x.shape[0], dtype=torch.long) * 0

        x = torch.randn(2, 3, 4)
        self.run_test(Emtpy(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_new_full(self):
        class Full(torch.nn.Module):
            def forward(self, x):
                return x.new_full(x.shape[1:2], 5), x.new_full(x.shape[0:1], 1.3, dtype=torch.long)

        x = torch.randn(2, 3, 4)
        self.run_test(Full(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_inplace_list(self):
        class Arithmetic(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x, y):
                return torch.cat([x.add_(3), y.fill_(0)])

        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        self.run_test(Arithmetic(), (x, y))

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

    def test_inplace_arithmetic_half(self):
        class InplaceAddModel(torch.nn.Module):
            def forward(self, x, y):
                return x.add_(y)

        class InplaceMulModel(torch.nn.Module):
            def forward(self, x, y):
                return x.mul_(y)

        x = torch.randn(2, 2, dtype=torch.half)
        y = torch.randn(2, 2, dtype=torch.float)
        self.run_test(InplaceAddModel(), (x, y), rtol=1e-2, atol=1e-2)
        self.run_test(InplaceMulModel(), (x, y), rtol=1e-2, atol=1e-2)

    @disableScriptTest()  # Sort with dynamic dim not supported in ONNX
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
    @disableScriptTest()  # Sort with dynamic dim not supported in ONNX
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

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_masked_fill_inplace(self):

        class MaskedFillModel(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x):
                mask = torch.tensor([[0, 0, 1], [1, 1, 0]], dtype=torch.uint8)
                x.masked_fill_(mask, 2)
                return x

        x = torch.zeros(4, 2, 3, requires_grad=True)
        self.run_test(MaskedFillModel(), x)

        class MaskedFillModel2(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x):
                x.masked_fill_(x > 3, -1)
                return x

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

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_index_put_to_masked_fill(self):
        class MaskedFillModel(torch.nn.Module):
            def forward(self, input_mask, some_const):
                mask = input_mask.clone()
                mask[mask != some_const] = 1
                mask[mask == some_const] = 0
                return mask

        mask = torch.randn(2, 2, 2, requires_grad=True)
        constant = torch.tensor(5, dtype=torch.float)
        self.run_test(MaskedFillModel(), (mask, constant))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_index_put_to_masked_scatter(self):
        class MaskedScatterModel(torch.nn.Module):
            def forward(self, input_mask, some_const):
                mask = input_mask.clone()
                mask[mask != some_const] = torch.ones(8)
                return mask

        mask = torch.randn(2, 2, 2, requires_grad=True)
        constant = torch.tensor(5, dtype=torch.float)
        self.run_test(MaskedScatterModel(), (mask, constant))

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
                return x.size(0) * 2 * x, 2 - x

        x = torch.ones(2, 3, dtype=torch.float32)
        self.run_test(ArithmeticModel(), x)

        class ReciprocalModel(torch.nn.Module):
            def forward(self, x):
                return torch.reciprocal(x)

        x = torch.tensor([2.0, 4.0], dtype=torch.double)
        self.run_test(ReciprocalModel(), x)

        class ComparisonModel(torch.nn.Module):
            def forward(self, x, y):
                a = torch.tensor([12.0])
                return x.lt(1.5) & y.le(2) & x.le(1), x.gt(y), x.lt(y), a.ge(x.size(0))

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
        x = torch.tensor(12.)
        self.run_test(FullModel(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_full_like(self):
        class FullLikeModel(torch.nn.Module):
            def forward(self, x):
                return torch.full_like(x, 4)

        x = torch.tensor(12)
        self.run_test(FullLikeModel(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_full_like_value(self):
        class FullLikeModel(torch.nn.Module):
            def forward(self, x, y):
                out = y + 2
                return torch.full_like(x, out)

        x = torch.tensor(12)
        y = torch.tensor(2)
        self.run_test(FullLikeModel(), (x, y))

    def test_l1_norm(self):
        class NormModel(torch.nn.Module):
            def forward(self, x):
                return torch.norm(x, p=1, dim=-1, keepdim=False)

        x = torch.randn(4, 2, 3, requires_grad=True)
        self.run_test(NormModel(), x)

    def test_l2_norm(self):
        class NormModel(torch.nn.Module):
            def forward(self, x):
                return torch.norm(x, p=2, dim=-2, keepdim=False)

        x = torch.randn(4, 2, 3, requires_grad=True)
        self.run_test(NormModel(), x)

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
        y = torch.randn(2, 1, 3, requires_grad=True)
        self.run_test(UnfoldModel(), x,
                      dynamic_axes={'x': [0, 1]},
                      input_names=['x'],
                      test_with_inputs=[y])

    @skipIfONNXShapeInference(False)
    def test_unfold_infer_shape(self):
        class UnfoldModule(torch.jit.ScriptModule):
            def __init__(self):
                super(UnfoldModule, self).__init__()
                self.conv = torch.nn.Conv1d(3, 1, 3, stride=2)

            @torch.jit.script_method
            def forward(self, x):
                x = self.conv(x)
                return x.unfold(dimension=2, size=2, step=2)

        x = torch.randn(32, 3, 64)
        self.run_test(UnfoldModule(), x)

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_unfold_dynamic_inputs(self):
        class UnfoldModel(torch.nn.Module):
            def forward(self, x):
                return x.unfold(dimension=2, size=x.shape[1], step=x.shape[1] - 1)

        x = torch.randn(4, 2, 4, requires_grad=True)
        self.run_test(UnfoldModel(), x)

        class UnfoldModel(torch.nn.Module):
            def forward(self, x):
                return x.unfold(dimension=2, size=x.shape[1], step=1)

        x = torch.randn(4, 2, 4, requires_grad=True)
        self.run_test(UnfoldModel(), x)

    def test_prelu(self):
        class PReluModel(torch.nn.Module):
            def __init__(self):
                super(PReluModel, self).__init__()
                self.prelu = torch.nn.PReLU()

            def forward(self, x):
                return self.prelu(x)

        x = torch.randn(2, 3, 4)
        y = torch.randn(2, 4, 5)
        self.run_test(PReluModel(), x, input_names=['x'],
                      dynamic_axes={'x': [1, 2]},
                      test_with_inputs=[y])

    def test_silu(self):
        class SiLUModel(torch.nn.Module):
            def __init__(self):
                super(SiLUModel, self).__init__()
                self.silu = torch.nn.SiLU()

            def forward(self, x):
                return self.silu(x)

        x = torch.randn(2, 3, 4)
        self.run_test(SiLUModel(), (x))

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
    def test_glu(self):
        class GluModel(torch.nn.Module):
            def forward(self, x):
                return torch.nn.functional.glu(x)

        x = torch.randn(2, 4, 5, 6, requires_grad=True)
        self.run_test(GluModel(), x)

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

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_cumsum_with_cast(self):
        class CumSum(torch.nn.Module):
            def forward(self, input):
                return torch.cumsum(input, dim=0, dtype=torch.float32)

        model = CumSum()
        x = torch.tensor([2, 3, 4], dtype=torch.int32)
        self.run_test(model, x)
        x = torch.tensor([False, True, True])
        self.run_test(model, x)

    @disableScriptTest()  # error in propagate as assign input shape
    @skipIfUnsupportedMinOpsetVersion(10)
    def test_embedding_bag(self):
        model = torch.nn.EmbeddingBag(10, 5, mode='sum', scale_grad_by_freq=True)
        input = torch.randint(10, (7,))
        offset = torch.tensor([0, 2, 5, 6])
        self.run_test(model, (input, offset))

        model = torch.nn.EmbeddingBag(10, 5, mode='sum', include_last_offset=True)
        input = torch.randint(10, (7,))
        offset = torch.tensor([0, 2, 5, 6])
        self.run_test(model, (input, offset))

        model = torch.nn.EmbeddingBag(10, 5, mode='max')
        input = torch.randint(10, (7, 5))
        self.run_test(model, (input))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_embedding_bag_1d_per_sample_weights(self):
        class EmbeddingModel(torch.nn.Module):
            def forward(self, embedding_matrix, input, offset, weights):
                return torch.nn.functional.embedding_bag(input, embedding_matrix, offsets=offset,
                                                         mode='sum', per_sample_weights=weights)

        model = EmbeddingModel()
        x = torch.randint(7, (6,))
        w = torch.randn(6, )
        offset = torch.tensor([0, 2, 5])
        embedding_matrix = torch.rand(10, 15)
        self.run_test(model, (embedding_matrix, x, offset, w))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_embedding_bag_2d_per_sample_weights(self):
        class EmbeddingModel(torch.nn.Module):
            def forward(self, embedding_matrix, input, weights):
                return torch.nn.functional.embedding_bag(input, embedding_matrix,
                                                         mode='sum', per_sample_weights=weights)

        embedding_matrix = torch.rand(10, 15)
        model = EmbeddingModel()
        x = torch.randint(7, (2, 3))
        w = torch.randn(2, 3)
        self.run_test(model, (embedding_matrix, x, w))

    @disableScriptTest()  # scripting prim::Uninitialized, prim::dtype, prim::unchecked_cast
    @skipIfUnsupportedMinOpsetVersion(11)
    @unittest.skip("Due to ONNX Loop shape inference issue.")
    def test_embedding_bag_dynamic_input(self):
        class EmbeddingModel1D(torch.nn.Module):
            def forward(self, embedding_matrix, input, weights, offsets):
                return torch.nn.functional.embedding_bag(input, embedding_matrix, offsets=offsets,
                                                         mode='sum', per_sample_weights=weights)

        model = EmbeddingModel1D()
        x = torch.randint(7, (6,))
        w = torch.randn(6, )
        offsets = torch.tensor([0, 2, 5], dtype=torch.long)
        embedding_matrix = torch.rand(10, 15)
        x2 = torch.randint(7, (2,))
        w2 = torch.randn(2, )
        embedding_matrix2 = torch.rand(12, 25)
        offsets2 = torch.tensor([0, ], dtype=torch.long)
        self.run_test(model, (embedding_matrix, x, w, offsets),
                      test_with_inputs=[(embedding_matrix2, x2, w2, offsets2)],
                      input_names=['embedding_matrix', 'x', 'offsets', 'w'],
                      dynamic_axes={'embedding_matrix': [0, 1], 'x': [0], 'offsets': [0], 'w': [0]})

        class EmbeddingModel2D(torch.nn.Module):
            def forward(self, embedding_matrix, input, weights):
                return torch.nn.functional.embedding_bag(input, embedding_matrix,
                                                         mode='sum', per_sample_weights=weights)

        model = EmbeddingModel2D()
        x = torch.randint(7, (2, 3))
        w = torch.randn(2, 3)
        embedding_matrix = torch.rand(10, 15)
        x2 = torch.randint(7, (3, 5))
        w2 = torch.randn(3, 5)
        embedding_matrix2 = torch.rand(12, 25)
        self.run_test(model, (embedding_matrix, x, w),
                      test_with_inputs=[(embedding_matrix2, x2, w2)],
                      input_names=['embedding_matrix', 'x', 'w'],
                      dynamic_axes={'embedding_matrix': [0, 1], 'x': [0, 1], 'w': [0, 1]})

    @skipIfUnsupportedMinOpsetVersion(8)
    def test_meshgrid(self):
        class Meshgrid(torch.nn.Module):
            def forward(self, x, y, z):
                output1, output2, output3 = torch.meshgrid(x, y, z)
                return output1, output2, output3

        x = torch.randn(3, requires_grad=True)
        y = torch.zeros(4, requires_grad=True)
        z = torch.randn(5, requires_grad=True)
        self.run_test(Meshgrid(), (x, y, z))

    @skipIfUnsupportedMinOpsetVersion(8)
    def test_meshgrid_scalar(self):
        class Meshgrid(torch.nn.Module):
            def forward(self, x, y, z):
                output1, output2, output3 = torch.meshgrid(x, y, z)
                return output1, output2, output3

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

    def test_numel(self):
        class MyModule(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, input):
                return input.numel() * input

        x = torch.randn(2, 3, 5)
        model = MyModule()
        self.run_test(model, (x,))

    def test_numel_empty(self):
        class MyModule(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, input):
                return input.numel() * input

        x = torch.randn(0)
        model = MyModule()
        self.run_test(model, (x,))

    def test_dtype(self):
        class MyModel(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, input, other):
                return input.to(dtype=other.dtype) + other

        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        self.run_test(MyModel(), (x, y))

    def test_dtype_eq(self):
        class MyModel(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, input, other):
                if input.dtype == other.dtype:
                    return input + other
                return input

        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        self.run_test(MyModel(), (x, y))

    def test_cast_to(self):
        class MyModule(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, input, other):
                return input.to(other) + other

        x = torch.randn(2, 3, 4)
        y = torch.tensor([1], dtype=torch.int64)
        model = MyModule()
        self.run_test(model, (x, y))

    def test_cast_to_bool(self):
        class MyModule(torch.nn.Module):
            def forward(self, input, other):
                return torch.cat((input.to(other), other), 0)

        x = torch.randn(2, 3, 4)
        y = torch.zeros([2, 3, 4], dtype=torch.bool)
        model = MyModule()
        self.run_test(model, (x, y))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_ones_bool(self):
        class MyModule(torch.nn.Module):
            def forward(self, input):
                true = torch.ones(input.shape, dtype=torch.bool)
                return input.to(true) & true

        x = torch.randn(2, 3, 4)
        model = MyModule()
        self.run_test(model, x)

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
    @disableScriptTest()  # Functional module not scriptable
    def test_pad_types(self):
        # Test for different pad integer types
        class Pad(torch.nn.Module):
            def forward(self, x, pad: List[int]):
                return torch.nn.functional.pad(x, pad)

        x = torch.randn(2, 2, 4, 4)
        y = pad = (torch.tensor(2, dtype=torch.int32), torch.tensor(4, dtype=torch.int32))
        self.run_test(Pad(), (x, y))

        y = pad = (torch.tensor(2, dtype=torch.int64), torch.tensor(4, dtype=torch.int64))
        self.run_test(Pad(), (x, y))

    @skipIfUnsupportedMaxOpsetVersion(10)
    def test_unsupported_pad(self):
        class Pad(torch.nn.Module):
            def forward(self, x, pad):
                return torch.nn.functional.pad(x, pad)

        def run():
            x = torch.randn(2, 2, 4, 4)
            y = pad = (torch.tensor(2, dtype=torch.int32), torch.tensor(4, dtype=torch.int32))
            p = Pad()
            f = io.BytesIO()
            torch.onnx._export(p, (x, y), f)

        with self.assertRaises(RuntimeError) as cm:
            run()

        the_exception = cm.exception
        self.assertEqual('Unsupported: ONNX export of Pad in opset 9. The sizes of the padding must be constant. ' +
                         'Please try opset version 11.', the_exception.args[0])

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_if_fold(self):
        class IfFoldModel(torch.nn.Module):
            def forward(self, y):
                if y.dim() == 2:
                    y = y + 4
                    y = y + 2
                else:
                    y = y - 1
                return y
        x = torch.ones((3, 4), dtype=torch.int)
        self.run_test(IfFoldModel(), x)

        class IfFoldModel(torch.nn.Module):
            def forward(self, y):
                if y.numel() > 1:
                    y = y + 4
                else:
                    y = y + 2
                return y

        x = torch.ones((3, 4), dtype=torch.int)
        self.run_test(IfFoldModel(), x)

        class IfFoldModel(torch.nn.Module):
            def forward(self, y):
                if y.dim() != 3:
                    y = y + 4
                    y = y + 2
                else:
                    return y
                return y

        x = torch.ones((3, 4), dtype=torch.int)
        self.run_test(IfFoldModel(), x)

        class IfFoldModel(torch.nn.Module):
            def forward(self, y):
                if y.dim() >= 1:
                    y = y + 4
                else:
                    y = y - 1
                return y

        x = torch.ones((3, 4), dtype=torch.int)
        self.run_test(IfFoldModel(), x)

        class IfFoldModel(torch.nn.Module):
            def forward(self, y):
                if y.dim() <= 1:
                    y = y + 4
                else:
                    y = y + 2
                return y

        x = torch.ones((3, 4), dtype=torch.int)
        self.run_test(IfFoldModel(), x)

        class IfFoldModel(torch.nn.Module):
            def forward(self, y):
                if y.dim() < 3 and y.dtype == torch.int:
                    y = y + 4
                    y = y + 2
                else:
                    return y
                return y

        x = torch.ones((3, 4), dtype=torch.int)
        self.run_test(IfFoldModel(), x)

        class IfFoldModel(torch.nn.Module):
            def forward(self, y):
                if y.dim() == 3 and y.dtype == torch.int:
                    y = y + 4
                    y = y + 2
                else:
                    y = y + 1
                return y

        x = torch.ones((3, 4), dtype=torch.int)
        self.run_test(IfFoldModel(), x)

        class IfFoldModel(torch.nn.Module):
            def forward(self, y):
                if y.numel() != 0 and y.dim() == 2:
                    y = y + 4
                    y = y + 2
                else:
                    return y
                return y

        x = torch.ones((3, 4), dtype=torch.int)
        self.run_test(IfFoldModel(), x)

        class IfFoldModel(torch.nn.Module):
            def forward(self, x, y):
                if x.numel() == y.numel():
                    y = x + y
                else:
                    y = y - x
                return y

        x = torch.ones((3, 4), dtype=torch.int)
        y = torch.ones((3, 4), dtype=torch.int)
        self.run_test(IfFoldModel(), (x, y))

        class IfFoldModel(torch.nn.Module):
            def forward(self, x, y):
                if x.numel() != y.numel():
                    y = x + y
                else:
                    y = y - x
                return y

        x = torch.ones((3, 4), dtype=torch.int)
        y = torch.ones((3, 4), dtype=torch.int)
        self.run_test(IfFoldModel(), (x, y))

    @skipIfUnsupportedMinOpsetVersion(11)
    @skipIfONNXShapeInference(False)
    def test_uninitialized(self):
        class UninitializedModel(torch.nn.Module):
            def forward(self, y):
                if y.shape[1] < 5:
                    if y.size(0) == 1:
                        y = y + 4
                    else:
                        return y
                return y

        x = torch.ones((3, 4), dtype=torch.int)
        self.run_test(UninitializedModel(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    @skipIfONNXShapeInference(False)
    def test_uninitialized_dynamic(self):
        class UninitializedModel(torch.nn.Module):
            def forward(self, y):
                if y.shape[1] < 5:
                    if y.size(0) == 1:
                        y = y + 4
                    else:
                        return y
                return y

        x = torch.ones((3, 4), dtype=torch.int)
        y = torch.ones((6, 7), dtype=torch.int)
        self.run_test(UninitializedModel(), x, test_with_inputs=[y],
                      input_names=['input_1'],
                      dynamic_axes={'input_1': [0, 1]})

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

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_outer(self):
        class Outer(torch.nn.Module):
            def forward(self, x, y):
                return torch.outer(x, y)

        x = torch.arange(1, 5)
        y = torch.arange(1, 4)
        self.run_test(Outer(), input=(x, y))

        x = torch.arange(1, 6).to(dtype=torch.float32)
        y = torch.arange(1, 4).to(dtype=torch.long)
        self.run_test(Outer(), input=(x, y))

        x = torch.arange(2, 5).to(dtype=torch.float32)
        y = torch.arange(2, 4).to(dtype=torch.float64)
        self.run_test(Outer(), input=(x, y))

        x = torch.arange(3, 6).to(dtype=torch.int32)
        y = torch.arange(4, 7).to(dtype=torch.long)
        self.run_test(Outer(), input=(x, y))

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_einsum(self):
        class EinsumModelBatchDiagonal(torch.nn.Module):
            def forward(self, x):
                eqn = '...ii ->...i'
                return torch.einsum(eqn, x)

        x = torch.randn(3, 5, 5)
        self.run_test(EinsumModelBatchDiagonal(), input=(x,))

        class EinsumModelBatchMatmul(torch.nn.Module):
            def forward(self, x, y):
                eqn = 'bij, bjk -> bik'
                return torch.einsum(eqn, x, y)

        x = torch.randn(5, 2, 3)
        y = torch.randn(5, 3, 4)
        self.run_test(EinsumModelBatchMatmul(), input=(x, y))

        class EinsumModelInnerProd(torch.nn.Module):
            def forward(self, x, y):
                eqn = 'i,i'
                return torch.einsum(eqn, x, y)

        x = torch.randn(5)
        y = torch.randn(5)
        self.run_test(EinsumModelInnerProd(), input=(x, y))

        class EinsumModelTranspose(torch.nn.Module):
            def forward(self, x):
                eqn = 'ij->ji'
                return torch.einsum(eqn, x)

        x = torch.randn(3, 4)
        self.run_test(EinsumModelTranspose(), input=(x,))

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_crossentropyloss(self):
        for ignore_index in [-100, 1]:
            x = torch.randn(3, 5)
            y = torch.empty(3, dtype=torch.long).random_(5)
            y[y == 1] = ignore_index

            self._crossentropyloss(x, y, ignore_index)

            x = torch.randn(3, 5, 2)
            y = torch.empty(3, 2, dtype=torch.long).random_(5)
            y[y == 1] = ignore_index
            self._crossentropyloss(x, y, ignore_index)

            x = torch.randn(3, 5, 2, 7)
            y = torch.empty(3, 2, 7, dtype=torch.long).random_(5)
            y[y == 1] = ignore_index
            self._crossentropyloss(x, y, ignore_index)

    def _crossentropyloss(self, x, y, ignore_index):
        class CrossEntropyLossNone(torch.nn.Module):
            def __init__(self, ignore_index):
                super(CrossEntropyLossNone, self).__init__()
                if ignore_index == -100:
                    self.loss = torch.nn.CrossEntropyLoss(reduction='none')
                else:
                    self.loss = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=ignore_index)

            def forward(self, input, target):
                return self.loss(input, target)

        self.run_test(CrossEntropyLossNone(ignore_index), input=(x, y))

        class CrossEntropyLossNoneWeight(torch.nn.Module):
            def __init__(self, ignore_index):
                super(CrossEntropyLossNoneWeight, self).__init__()
                if ignore_index == -100:
                    self.loss = torch.nn.CrossEntropyLoss(reduction='none', weight=torch.randn(5))
                else:
                    self.loss = torch.nn.CrossEntropyLoss(reduction='none', weight=torch.randn(5), ignore_index=ignore_index)

            def forward(self, input, target):
                return self.loss(input, target)

        self.run_test(CrossEntropyLossNoneWeight(ignore_index), input=(x, y))

        class CrossEntropyLossSum(torch.nn.Module):
            def __init__(self, ignore_index):
                super(CrossEntropyLossSum, self).__init__()
                if ignore_index == -100:
                    self.loss = torch.nn.CrossEntropyLoss(reduction='sum')
                else:
                    self.loss = torch.nn.CrossEntropyLoss(reduction='sum', ignore_index=ignore_index)

            def forward(self, input, target):
                return self.loss(input, target)

        self.run_test(CrossEntropyLossSum(ignore_index), input=(x, y))

        class CrossEntropyLossSumWeight(torch.nn.Module):
            def __init__(self, ignore_index):
                super(CrossEntropyLossSumWeight, self).__init__()
                if ignore_index == -100:
                    self.loss = torch.nn.CrossEntropyLoss(reduction='sum', weight=torch.randn(5))
                else:
                    self.loss = torch.nn.CrossEntropyLoss(reduction='sum', weight=torch.randn(5), ignore_index=ignore_index)

            def forward(self, input, target):
                return self.loss(input, target)

        self.run_test(CrossEntropyLossSumWeight(ignore_index), input=(x, y))

        class CrossEntropyLossMean(torch.nn.Module):
            def __init__(self, ignore_index):
                super(CrossEntropyLossMean, self).__init__()
                if ignore_index == -100:
                    self.loss = torch.nn.CrossEntropyLoss()
                else:
                    self.loss = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)

            def forward(self, input, target):
                return self.loss(input, target)

        self.run_test(CrossEntropyLossMean(ignore_index), input=(x, y))

        class CrossEntropyLossMeanWeight(torch.nn.Module):
            def __init__(self, ignore_index):
                super(CrossEntropyLossMeanWeight, self).__init__()
                if ignore_index == -100:
                    self.loss = torch.nn.CrossEntropyLoss(weight=torch.randn(5))
                else:
                    self.loss = torch.nn.CrossEntropyLoss(weight=torch.randn(5), ignore_index=ignore_index)

            def forward(self, input, target):
                return self.loss(input, target)

        self.run_test(CrossEntropyLossMeanWeight(ignore_index), input=(x, y))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_kldiv_loss(self):

        x = torch.randn(5)
        y = torch.randn(5)
        self._kldiv_loss(x, y)

        x = torch.randn(2, 3, 5)
        y = torch.randn(2, 3, 5)
        self._kldiv_loss(x, y)

        x = torch.randn(2, 3, 5, 7)
        y = torch.randn(2, 3, 5, 7)
        self._kldiv_loss(x, y)

    def _kldiv_loss(self, x, y):
        class KLDivLossNone(torch.nn.Module):
            def __init__(self):
                super(KLDivLossNone, self).__init__()
                self.loss = torch.nn.KLDivLoss(reduction='none', log_target=True)

            def forward(self, input, target):
                return self.loss(input, target)

        self.run_test(KLDivLossNone(), input=(x, y))

        class KLDivLossMean(torch.nn.Module):
            def __init__(self):
                super(KLDivLossMean, self).__init__()
                self.loss = torch.nn.KLDivLoss(reduction='mean', log_target=False)

            def forward(self, input, target):
                return self.loss(input, target)

        self.run_test(KLDivLossMean(), input=(x, y))

        class KLDivLossSum(torch.nn.Module):
            def __init__(self):
                super(KLDivLossSum, self).__init__()
                self.loss = torch.nn.KLDivLoss(reduction='sum', log_target=True)

            def forward(self, input, target):
                return self.loss(input, target)

        self.run_test(KLDivLossSum(), input=(x, y))

        class KLDivLossBatchMean(torch.nn.Module):
            def __init__(self):
                super(KLDivLossBatchMean, self).__init__()
                self.loss = torch.nn.KLDivLoss(reduction='batchmean', log_target=False)

            def forward(self, input, target):
                return self.loss(input, target)

        self.run_test(KLDivLossBatchMean(), input=(x, y))

        class KLDivLossMiniBatchMean(torch.nn.Module):
            def __init__(self):
                super(KLDivLossMiniBatchMean, self).__init__()
                self.loss = torch.nn.KLDivLoss(reduction='batchmean', size_average=False, log_target=True)

            def forward(self, input, target):
                return self.loss(input, target)

        self.run_test(KLDivLossMiniBatchMean(), input=(x, y))

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

        # using test data containing default ignore_index=-100
        target[target == 1] = -100
        self.run_test(NLLModel(), (input, target))

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

        # using test data containing default ignore_index=-100
        target[target == 1] = -100
        self.run_test(NLLModel(), (input, target))

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

        # using test data containing default ignore_index=-100
        target[target == 1] = -100
        self.run_test(NLLModel(), (input, target))

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

        # using test data containing default ignore_index=-100
        target[target == 1] = -100
        self.run_test(NLLModel(), (input, target))

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

        # using test data containing default ignore_index=-100
        target[target == 1] = -100
        self.run_test(NLLModel(), (input, target))

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

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_binary_cross_entropy_with_logits(self):
        x = torch.randn(5)
        y = torch.empty(5).random_(2)
        self._bce_logits(x, y)

        x = torch.randn(3, 4)
        y = torch.empty(3, 4).random_(2)
        weight = torch.tensor([3])
        self._bce_logits_wegiht(x, y, weight)

        x = torch.randn(3, 2, 4)
        y = torch.empty(3, 2, 4).random_(2)
        pos_weight = torch.empty([2, 4]).random_(2)
        self._bce_logits_posweight(x, y, pos_weight)

        x = torch.randn(3, 3, 4)
        y = torch.empty(3, 3, 4).random_(2)
        weight = torch.tensor([3])
        pos_weight = torch.empty([3, 4]).random_(2)
        self._bce_logits_loss_weight_posweight(x, y, weight, pos_weight)

    def _bce_logits(self, x, y):
        class BCEWithLogitsLossNone(torch.nn.Module):
            def forward(self, input, target):
                return torch.nn.functional.binary_cross_entropy_with_logits(input, target, reduction='none')

        self.run_test(BCEWithLogitsLossNone(), input=(x, y))

        class BCEWithLogitsLossMean(torch.nn.Module):
            def forward(self, input, target):
                return torch.nn.functional.binary_cross_entropy_with_logits(input, target, reduction='mean')

        self.run_test(BCEWithLogitsLossMean(), input=(x, y))

        class BCEWithLogitsLossSum(torch.nn.Module):
            def forward(self, input, target):
                return torch.nn.functional.binary_cross_entropy_with_logits(input, target, reduction='sum')

        self.run_test(BCEWithLogitsLossSum(), input=(x, y))

    def _bce_logits_wegiht(self, x, y, weight):
        class BCEWithLogitsLossWegihtNone(torch.nn.Module):
            def forward(self, input, target, weight):
                return torch.nn.functional.binary_cross_entropy_with_logits(input, target, weight=weight, reduction='none')
        self.run_test(BCEWithLogitsLossWegihtNone(), input=(x, y, weight))

        class BCEWithLogitsLossWegihtMean(torch.nn.Module):
            def forward(self, input, target, weight):
                return torch.nn.functional.binary_cross_entropy_with_logits(input, target, weight=weight, reduction='mean')

        self.run_test(BCEWithLogitsLossWegihtMean(), input=(x, y, weight))

        class BCEWithLogitsLossWegihtSum(torch.nn.Module):
            def forward(self, input, target, weight):
                return torch.nn.functional.binary_cross_entropy_with_logits(input, target, weight=weight, reduction='sum')

        self.run_test(BCEWithLogitsLossWegihtSum(), input=(x, y, weight))

    def _bce_logits_posweight(self, x, y, pos_weight):
        class BCEWithLogitsLossPosWegihtNone(torch.nn.Module):
            def forward(self, input, target, pos_weight):
                return torch.nn.functional.binary_cross_entropy_with_logits(input, target, pos_weight=pos_weight, reduction='none')
        self.run_test(BCEWithLogitsLossPosWegihtNone(), input=(x, y, pos_weight))

        class BCEWithLogitsLossPosWegihtMean(torch.nn.Module):
            def forward(self, input, target, pos_weight):
                return torch.nn.functional.binary_cross_entropy_with_logits(input, target, pos_weight=pos_weight, reduction='mean')

        self.run_test(BCEWithLogitsLossPosWegihtMean(), input=(x, y, pos_weight))

        class BCEWithLogitsLossPosWegihtSum(torch.nn.Module):
            def forward(self, input, target, pos_weight):
                return torch.nn.functional.binary_cross_entropy_with_logits(input, target, pos_weight=pos_weight, reduction='sum')

        self.run_test(BCEWithLogitsLossPosWegihtSum(), input=(x, y, pos_weight))

    def _bce_logits_loss_weight_posweight(self, x, y, weight, pos_weight):
        class BCEWithLogitsLossWeightPosweightNone(torch.nn.Module):
            def forward(self, input, target, weight, pos_weight):
                return torch.nn.functional.binary_cross_entropy_with_logits(input, target, weight=weight,
                                                                            pos_weight=pos_weight, reduction='none')

        self.run_test(BCEWithLogitsLossWeightPosweightNone(), input=(x, y, weight, pos_weight))

        class BCEWithLogitsLossWeightPosweightMean(torch.nn.Module):
            def forward(self, input, target, weight, pos_weight):
                return torch.nn.functional.binary_cross_entropy_with_logits(input, target, weight=weight,
                                                                            pos_weight=pos_weight, reduction='mean')

        self.run_test(BCEWithLogitsLossWeightPosweightMean(), input=(x, y, weight, pos_weight))

        class BCEWithLogitsLossWeightPosweightSum(torch.nn.Module):
            def forward(self, input, target, weight, pos_weight):
                return torch.nn.functional.binary_cross_entropy_with_logits(input, target, weight=weight,
                                                                            pos_weight=pos_weight, reduction='sum')

        self.run_test(BCEWithLogitsLossWeightPosweightSum(), input=(x, y, weight, pos_weight))


    def test_torch_mm(self):
        class M(torch.nn.Module):
            def forward(self, mat1, mat2):
                mm = torch.mm(mat1, mat2)
                return mm

        mat1 = torch.randn(2, 3)
        mat2 = torch.randn(3, 3)
        self.run_test(M(), input=(mat1, mat2))

    @skipIfUnsupportedMinOpsetVersion(9)  # Because where op is not supported for opset < 9.
    def test_where_with_bool_tensor(self):
        class M(torch.nn.Module):
            def forward(self, mat1, mat2):
                out = torch.where(mat1 > 0, mat1, mat2)
                return out

        mat1 = torch.randn(2, 3)
        mat2 = torch.ones(2, 3)
        self.run_test(M(), input=(mat1, mat2))

    @skipIfUnsupportedMinOpsetVersion(9)  # Because where op is not supported for opset < 9.
    def test_where_with_byte_tensor(self):
        class M(torch.nn.Module):
            def forward(self, cond, mat1, mat2):
                out = torch.where(cond, mat1, mat2)
                return out

        cond = torch.ones(2, 3, dtype=torch.uint8)
        cond[1, 2] = 0
        mat1 = torch.randn(2, 3)
        mat2 = torch.ones(2, 3)
        self.run_test(M(), input=(cond, mat1, mat2))

    @skipIfUnsupportedMinOpsetVersion(10)  # ONNX IsInf op is added in opset 10.
    def test_isinf(self):
        class M(torch.nn.Module):
            def forward(self, x):
                return x.isinf()

        x = torch.tensor([[1, 2, float('inf')], [2, float('nan'), float('inf')]])
        self.run_test(M(), (x, ))

    @skipIfUnsupportedMinOpsetVersion(9)  # ONNX IsNaN op is added in opset 9.
    def test_isnan(self):
        class M(torch.nn.Module):
            def forward(self, x):
                return x.isnan()

        x = torch.tensor([[1, 2, float('inf')], [2, float('nan'), float('inf')]])
        self.run_test(M(), (x, ))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_any(self):
        class M(torch.nn.Module):
            def forward(self, x):
                return x.any()

        x = torch.tensor([[True, False], [False, False]])
        self.run_test(M(), (x, ))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_all(self):
        class M(torch.nn.Module):
            def forward(self, x):
                return x.all()

        x = torch.tensor([[True, False], [False, False]])
        self.run_test(M(), (x, ))

    def test_dropout(self):
        class M(torch.nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.dropout = torch.nn.Dropout(0.3)

            def forward(self, x):
                dropout = self.dropout(x)
                return dropout

        x = torch.randn(10, 3, 53)
        self.run_test(M(), (x))

    def test_shape_constant_fold(self):
        class ShapeModule(torch.nn.Module):
            def __init__(self):
                super(ShapeModule, self).__init__()
                self.register_buffer("weight", torch.ones(5))

            def forward(self, x):
                shape = self.weight.shape[0]
                return x + shape

        x = torch.randn(2, 5)
        self.run_test(ShapeModule(), (x,), rtol=1e-3, atol=1e-5)

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_celu(self):
        class Celu(torch.nn.Module):
            def __init__(self):
                super(Celu, self).__init__()
                self.celu = torch.nn.CELU(alpha=1.0)

            def forward(self, input):
                return self.celu(input)

        input = torch.randn(2)
        self.run_test(Celu(), (input,))

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_celu_default(self):
        class Celu(torch.nn.Module):
            def __init__(self):
                super(Celu, self).__init__()
                self.celu = torch.nn.CELU()

            def forward(self, input):
                return self.celu(input)

        input = torch.randn(2)
        self.run_test(Celu(), (input,))

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_celu_alpha(self):
        class Celu(torch.nn.Module):
            def __init__(self):
                super(Celu, self).__init__()
                self.celu = torch.nn.CELU(alpha=2.)

            def forward(self, input):
                return self.celu(input)

        input = torch.randn(2)
        self.run_test(Celu(), (input,))

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_celu_cast(self):
        class Celu(torch.nn.Module):
            def __init__(self):
                super(Celu, self).__init__()
                self.celu = torch.nn.CELU()

            def forward(self, input):
                return self.celu(input)

        input = torch.randn(2, 5, 7, dtype=torch.float64)
        self.run_test(Celu(), (input,))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_where(self):
        class Model(torch.nn.Module):
            def forward(self, cond, input, other):
                return torch.where(cond, input, other)

        x = torch.randint(0, 1, (2, 3, 4), dtype=torch.bool)
        y = torch.randn(2, 1, 4)
        z = torch.ones(2, 3, 1)
        self.run_test(Model(), (x, y, z))

    @skipIfUnsupportedMinOpsetVersion(9)
    @disableScriptTest()  # scripting tests run for opsets > 11. See: test_where_condition_script
    def test_where_condition(self):
        class Model1(torch.nn.Module):
            def forward(self, input):
                return torch.stack(torch.where(input > 0.5), dim=1)

        x = torch.randint(0, 2, (2, 3, 4), dtype=bool)
        self.run_test(Model1(), (x))

        class Model2(torch.nn.Module):
            def forward(self, input, other):
                return torch.stack(torch.where(input > other), dim=1)

        x = torch.randint(0, 1, (2, 3, 4), dtype=bool)
        y = torch.randint(1, 2, (2, 3, 4), dtype=bool)
        self.run_test(Model2(), (x, y))

    @skipIfUnsupportedOpsetVersion([13])
    @skipIfUnsupportedMinOpsetVersion(11)
    def test_where_condition_script(self):
        class Model1(torch.nn.Module):
            def forward(self, input):
                return torch.stack(torch.where(input > 0.5), dim=1)

        x = torch.randint(0, 2, (2, 3, 4), dtype=bool)
        self.run_test(Model1(), (x))

        class Model2(torch.nn.Module):
            def forward(self, input, other):
                return torch.stack(torch.where(input > other), dim=1)

        x = torch.randint(0, 1, (2, 3, 4), dtype=bool)
        y = torch.randint(1, 2, (2, 3, 4), dtype=bool)
        self.run_test(Model2(), (x, y))

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

    @disableScriptTest()
    def test_derive_index(self):
        class MyModule(torch.nn.Module):
            def forward(self, x: torch.Tensor):
                j = []
                for idx in range(len(x) - 1, -len(x), -2):
                    y = x[idx]
                    j += [x * y]
                return j

        x = torch.randn(5, 13)
        self.run_test(MyModule(), x)

        class MyModule(torch.nn.Module):
            def forward(self, x: torch.Tensor):
                j = []
                for idx in range(-len(x), len(x) - 1, 2):
                    y = x[idx]
                    j += [x * y]
                return j

        x = torch.randn(5, 13)
        self.run_test(MyModule(), x)

        class MyModule(torch.nn.Module):
            def forward(self, x: torch.Tensor):
                j = []
                for idx in range(len(x) - 1, -len(x), -3):
                    y = x[idx]
                    j += [x * y]
                return j

        self.run_test(MyModule(), x)

        class MyModule(torch.nn.Module):
            def forward(self, x: torch.Tensor):
                j = []
                for idx in range(-len(x), len(x) - 1, 3):
                    y = x[idx]
                    j += [x * y]
                return j

        self.run_test(MyModule(), x)

    @skipIfONNXShapeInference(False)
    @skipIfUnsupportedMinOpsetVersion(11)
    def test_if_transpose(self):
        class IfModel(torch.nn.Module):
            def forward(self, x):
                x = x.transpose(0, 1)
                if x.size(0) == 2:
                    return x.transpose(0, 1)
                else:
                    return x

        x = torch.randn(2, 3)
        self.run_test(torch.jit.script(IfModel()), x,
                      output_names=['output_1'],
                      dynamic_axes={'output_1': [0, 1]})

    @skipIfONNXShapeInference(False)
    @skipIfUnsupportedMinOpsetVersion(13)
    def test_if_list(self):
        class IfModel(torch.nn.Module):
            def forward(self, x, y, cond):
                res = []
                if cond:
                    res = res + [x]
                else:
                    res = res + [y]
                return res

        x = torch.randn(2, 3)
        y = torch.randn(3, 3)
        cond = torch.tensor(1, dtype=torch.bool)
        self.run_test(torch.jit.script(IfModel()), (x, y, cond))

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_if_view(self):
        class IfModel(torch.nn.Module):
            def forward(self, x, y, cond):
                bs, seq = y.shape[:2]
                if cond:
                    res = x.view(bs, seq, -1)
                else:
                    res = y
                return res.transpose(1, 2)

        x = torch.randn(2, 16, 2, 2)
        y = torch.randn(2, 16, 8)
        cond = torch.tensor(1, dtype=torch.bool)
        self.run_test(torch.jit.script(IfModel()), (x, y, cond),
                      output_names=['output_1'],
                      dynamic_axes={'output_1': [1]})

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

    @disableScriptTest()  # dtype mismatch
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

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_embedding(self):
        class EmbedModel(torch.nn.Module):
            def forward(self, input, emb):
                return torch.nn.functional.embedding(input, emb, padding_idx=1)

        model = EmbedModel()
        x = torch.randint(4, (4,))
        x[2] = x[0] = 1
        embedding_matrix = torch.rand(10, 3)
        self.run_test(model, (x, embedding_matrix))

        x = torch.randint(4, (4, 3, 2))
        x[2] = 1
        x[0][1] = 1
        self.run_test(model, (x, embedding_matrix))
        self.run_test(model, (x, embedding_matrix), training=torch.onnx.TrainingMode.TRAINING)

        class EmbedModelWithoutPaddingIdx(torch.nn.Module):
            def forward(self, input, emb):
                return torch.nn.functional.embedding(input, emb)

        model = EmbedModelWithoutPaddingIdx()
        x = torch.randint(4, (4, 3, 2))
        self.run_test(model, (x, embedding_matrix))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_embedding_module(self):
        class EmbedModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = torch.nn.Embedding(4, 3, padding_idx=1)
                self.emb2 = torch.nn.Embedding(4, 3, padding_idx=1)
                with torch.no_grad():
                    self.emb2.weight[1] = torch.ones(3)

            def forward(self, input):
                return self.emb(input), self.emb2(input)

        model = EmbedModel()
        x = torch.randint(4, (4,))
        x[2] = x[0] = 1
        self.run_test(model, (x,))

        x = torch.randint(4, (4, 3, 2))
        x[2] = 1
        x[0][1] = 1
        self.run_test(model, (x,))

        class EmbedModelWithoutPaddingIdx(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = torch.nn.Embedding(4, 3)

            def forward(self, input):
                return self.emb(input)

        model = EmbedModelWithoutPaddingIdx()
        x = torch.randint(4, (4, 3, 2))
        self.run_test(model, (x,))

    def _dispatch_rnn_test(self, name, *args, **kwargs):
        if name == 'elman':
            self._elman_rnn_test(*args, **kwargs)
        if name == 'lstm':
            self._lstm_test(*args, **kwargs)
        if name == 'gru':
            self._gru_test(*args, **kwargs)

    def _elman_rnn_test(self, layers, nonlinearity, bidirectional,
                        initial_state, packed_sequence, dropout):

        class ElmanWithStateModel(torch.nn.Module):
            def __init__(self, layers, nonlinearity, bidirect, dropout, batch_first):
                super(ElmanWithStateModel, self).__init__()

                self.batch_first = batch_first
                self.inner_model = torch.nn.RNN(RNN_INPUT_SIZE, RNN_HIDDEN_SIZE, layers, nonlinearity=nonlinearity,
                                                bidirectional=bidirectional, dropout=dropout, batch_first=batch_first)

            def forward(self, input: PackedSequence, hx=None):
                return self.inner_model(input, hx)

        class ElmanWithoutStateModel(torch.nn.Module):
            def __init__(self, layers, nonlinearity, bidirect, dropout, batch_first):
                super(ElmanWithoutStateModel, self).__init__()
                self.batch_first = batch_first
                self.inner_model = torch.nn.RNN(RNN_INPUT_SIZE, RNN_HIDDEN_SIZE, layers, nonlinearity=nonlinearity,
                                                bidirectional=bidirectional, dropout=dropout, batch_first=batch_first)

            def forward(self, input: PackedSequence):
                return self.inner_model(input)

        batch_first = True if packed_sequence == 2 else False

        if initial_state:
            model = ElmanWithStateModel(layers=layers, bidirect=bidirectional, nonlinearity=nonlinearity,
                                        dropout=dropout, batch_first=batch_first)

            if packed_sequence == 1:
                model = RnnModelWithPackedSequenceWithState(model, False)
            if packed_sequence == 2:
                model = RnnModelWithPackedSequenceWithState(model, True)
        else:
            model = ElmanWithStateModel(layers=layers, bidirect=bidirectional,
                                        nonlinearity=nonlinearity, dropout=dropout,
                                        batch_first=batch_first)

            if packed_sequence == 1:
                model = RnnModelWithPackedSequenceWithoutState(model, False)
            if packed_sequence == 2:
                model = RnnModelWithPackedSequenceWithoutState(model, True)

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

        if packed_sequence == 0:
            model = LstmFlatteningResultWithoutSeqLength(RNN_INPUT_SIZE, RNN_HIDDEN_SIZE, layers,
                                                         bidirectional, dropout, batch_first)
        else:
            model = LstmFlatteningResultWithSeqLength(RNN_INPUT_SIZE, RNN_HIDDEN_SIZE, layers,
                                                      bidirectional, dropout, batch_first)
            if initial_state:
                if packed_sequence == 1:
                    model = RnnModelWithPackedSequenceWithState(model, False)
                if packed_sequence == 2:
                    model = RnnModelWithPackedSequenceWithState(model, True)
            else:
                if packed_sequence == 1:
                    model = RnnModelWithPackedSequenceWithoutState(model, False)
                if packed_sequence == 2:
                    model = RnnModelWithPackedSequenceWithoutState(model, True)

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

        class GRUWithStateModel(torch.nn.Module):
            def __init__(self, layers, bidirect, dropout, batch_first):
                super(GRUWithStateModel, self).__init__()

                self.batch_first = batch_first
                self.inner_model = torch.nn.GRU(RNN_INPUT_SIZE, RNN_HIDDEN_SIZE, num_layers=layers,
                                                bidirectional=bidirectional, dropout=dropout,
                                                batch_first=batch_first)

            def forward(self, input: PackedSequence, hx):
                return self.inner_model(input, hx)

        class GRUWithoutStateModel(torch.nn.Module):
            def __init__(self, layers, bidirect, dropout, batch_first):
                super(GRUWithoutStateModel, self).__init__()
                self.batch_first = batch_first
                self.inner_model = torch.nn.GRU(RNN_INPUT_SIZE, RNN_HIDDEN_SIZE, num_layers=layers,
                                                bidirectional=bidirectional, dropout=dropout,
                                                batch_first=batch_first)

            def forward(self, input: PackedSequence):
                return self.inner_model(input)

        class GRUNoSeqLengthWithoutStateModel(torch.nn.Module):
            def __init__(self, layers, bidirect, dropout, batch_first):
                super(GRUNoSeqLengthWithoutStateModel, self).__init__()
                self.batch_first = batch_first
                self.inner_model = torch.nn.GRU(RNN_INPUT_SIZE, RNN_HIDDEN_SIZE, num_layers=layers,
                                                bidirectional=bidirectional, dropout=dropout,
                                                batch_first=batch_first)

            def forward(self, input):
                return self.inner_model(input)

        class GRUNoSeqLengthWithStateModel(torch.nn.Module):
            def __init__(self, layers, bidirect, dropout, batch_first):
                super(GRUNoSeqLengthWithStateModel, self).__init__()
                self.batch_first = batch_first
                self.inner_model = torch.nn.GRU(RNN_INPUT_SIZE, RNN_HIDDEN_SIZE, num_layers=layers,
                                                bidirectional=bidirectional, dropout=dropout,
                                                batch_first=batch_first)

            def forward(self, input, hx):
                return self.inner_model(input, hx)

        batch_first = True if packed_sequence == 2 else False

        if packed_sequence == 0:
            if initial_state:
                model = GRUNoSeqLengthWithStateModel(layers=layers, bidirect=bidirectional,
                                                     dropout=dropout, batch_first=batch_first)
            else:
                model = GRUNoSeqLengthWithoutStateModel(layers=layers, bidirect=bidirectional,
                                                        dropout=dropout, batch_first=batch_first)
        else:
            if initial_state:
                model = GRUWithStateModel(layers=layers, bidirect=bidirectional, dropout=dropout,
                                          batch_first=batch_first)
                if packed_sequence == 1:
                    model = RnnModelWithPackedSequenceWithState(model, False)
                if packed_sequence == 2:
                    model = RnnModelWithPackedSequenceWithState(model, True)
            else:
                model = GRUWithoutStateModel(layers=layers, bidirect=bidirectional, dropout=dropout,
                                             batch_first=batch_first)
                if packed_sequence == 1:
                    model = RnnModelWithPackedSequenceWithoutState(model, False)
                if packed_sequence == 2:
                    model = RnnModelWithPackedSequenceWithoutState(model, True)

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

    @disableScriptTest()  # TODO: RuntimeError: Exporting the operator __is_ to ONNX is not supported
    def test_transformer_encoder(self):
        from torch.nn import TransformerEncoderLayer, TransformerEncoder

        class MyModule(torch.nn.Module):
            def __init__(self, ninp, nhead, nhid, dropout, nlayers):
                super(MyModule, self).__init__()
                encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
                self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

            def forward(self, input):
                return self.transformer_encoder(input)

        x = torch.rand(10, 32, 512)
        self.run_test(MyModule(512, 8, 2048 , 0., 3), (x,), atol=1e-6)

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_fake_quantize_per_tensor(self):
        class FakeQuantizePerTensorModel(torch.nn.Module):
            def forward(self, input):
                scale = 1. / 127
                zero_point = 0
                quant_min = -128
                quant_max = 127
                return torch.fake_quantize_per_tensor_affine(input, scale, zero_point, quant_min, quant_max)

        x = torch.randn(6, 4, 3, 3)
        self.run_test(FakeQuantizePerTensorModel(), (x))

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_fake_quantize_per_channel(self):
        class FakeQuantizePerChannelModel(torch.nn.Module):
            def forward(self, input):
                amax = torch.ones(4)
                scale = amax / 127.
                zero_point = torch.zeros_like(amax, dtype=torch.long)
                # Quantize twice to test differnet branches
                y = torch.fake_quantize_per_channel_affine(input, scale, zero_point, 1, 0, 255)
                return torch.fake_quantize_per_channel_affine(y, scale, zero_point, 1, -128, 127)

        x = torch.randn(6, 4, 3, 3)
        self.run_test(FakeQuantizePerChannelModel(), (x))

    def test_batchnorm_training(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()
                self.bn = torch.nn.BatchNorm2d(3, affine=True)

            def forward(self, x):
                bn = self.bn(x)
                return bn

        model = MyModule()
        x = torch.randn(10, 3, 128, 128)

        model.train()
        out = model(x)

        # state after 1 train epoch
        running_mean = model.bn.running_mean
        running_var = model.bn.running_var
        saved_mean = x.mean((0, 2, 3))
        saved_var = x.var((0, 2, 3))

        pytorch_out = [out.detach().numpy(),
                       running_mean.cpu().numpy(), running_var.cpu().numpy(),
                       saved_mean.cpu().numpy(), saved_var.cpu().numpy()]

        model_export = MyModule()
        f = io.BytesIO()

        ort_sess = convert_to_onnx(model_export, input=(x,), opset_version=self.opset_version,
                                   training=torch.onnx.TrainingMode.TRAINING)
        ort_outs = run_ort(ort_sess, input=(x,))
        [np.testing.assert_allclose(p_out, ort_out, atol=10e-3, rtol=10e-3) for p_out, ort_out in zip(pytorch_out, ort_outs)]

        model_export = torch.jit.script(MyModule())
        ort_sess = convert_to_onnx(model_export, input=(x,), opset_version=self.opset_version,
                                   example_outputs=out,
                                   training=torch.onnx.TrainingMode.TRAINING,
                                   onnx_shape_inference=True)
        ort_outs = run_ort(ort_sess, input=(x,))
        [np.testing.assert_allclose(p_out, ort_out, atol=10e-3, rtol=10e-3) for p_out, ort_out in
         zip(pytorch_out, ort_outs)]

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_dropout_training(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()
                self.dropout = torch.nn.Dropout(0.4)

            def forward(self, x):
                dropout = self.dropout(x)
                return dropout

        model = MyModule()
        x = torch.randn(10)

        model.train()

        ort_sess = convert_to_onnx(model, input=(x,), opset_version=self.opset_version,
                                   training=torch.onnx.TrainingMode.TRAINING)
        ort_outs = run_ort(ort_sess, input=(x,))
        assert not torch.all(torch.eq(x, torch.from_numpy(ort_outs[0])))

        script_model = torch.jit.script(model)
        output = model(x)
        ort_sess = convert_to_onnx(script_model, input=(x,), opset_version=self.opset_version,
                                   example_outputs=output,
                                   training=torch.onnx.TrainingMode.TRAINING)
        ort_outs = run_ort(ort_sess, input=(x,))
        assert not torch.all(torch.eq(x, torch.from_numpy(ort_outs[0])))

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_dropout_training_zero(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()
                self.dropout = torch.nn.Dropout(0.5)

            def forward(self, x):
                dropout = self.dropout(x)
                return dropout

        model = MyModule()

        # ensure there are no zeros in the input
        x = torch.randn(10, 3, 128, 128)
        y = x.numpy()
        y_mask = np.where(y == 0, 1, y)
        input = torch.from_numpy(y_mask)
        nb_elements = torch.numel(input)

        model.train()

        ort_sess = convert_to_onnx(model, input=(x,), opset_version=self.opset_version,
                                   training=torch.onnx.TrainingMode.TRAINING)
        ort_outs = run_ort(ort_sess, input=(x,))

        y = model(input)
        output = y.cpu().numpy()
        ort_mask = np.where(ort_outs[0] != 0, 1, 0)
        pyt_mask = np.where(output != 0, 1, 0)

        ratio_pytorch = np.sum(pyt_mask) / nb_elements
        ratio_ort = np.sum(ort_mask) / nb_elements

        np.testing.assert_allclose(ratio_pytorch, ratio_ort, rtol=0.01, atol=0.01)

        script_model = torch.jit.script(model)
        y = model(input)
        output = y.cpu().numpy()
        ort_sess = convert_to_onnx(script_model, input=(x,), opset_version=self.opset_version,
                                   example_outputs=y,
                                   training=torch.onnx.TrainingMode.TRAINING)
        ort_outs = run_ort(ort_sess, input=(x,))
        ort_mask = np.where(ort_outs[0] != 0, 1, 0)
        pyt_mask = np.where(output != 0, 1, 0)

        ratio_pytorch = np.sum(pyt_mask) / nb_elements
        ratio_ort = np.sum(ort_mask) / nb_elements

        np.testing.assert_allclose(ratio_pytorch, ratio_ort, rtol=0.01, atol=0.01)

    def test_conv_bn(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()
                self.conv = torch.nn.Conv2d(3, 16, kernel_size=1, stride=2, padding=3, bias=True)
                self.bn = torch.nn.BatchNorm2d(16, affine=True)

            def forward(self, x):
                x = self.conv(x)
                bn = self.bn(x)
                return bn

        model = MyModule()
        x = torch.randn(10, 3, 128, 128)
        ort_sess1 = convert_to_onnx(model, input=(x,), opset_version=self.opset_version,
                                    training=torch.onnx.TrainingMode.TRAINING)
        ort_outs1 = run_ort(ort_sess1, input=(x,))
        ort_sess2 = convert_to_onnx(model, input=(x,), opset_version=self.opset_version,
                                    training=torch.onnx.TrainingMode.EVAL)
        ort_outs2 = run_ort(ort_sess2, input=(x,))
        [np.testing.assert_allclose(ort_out1, ort_out2, atol=1e-7, rtol=0.001) for ort_out1, ort_out2 in
         zip(ort_outs1, ort_outs2)]

        script_model = torch.jit.script(model)
        outputs = model(x)
        ort_sess1 = convert_to_onnx(script_model, input=(x,), opset_version=self.opset_version,
                                    example_outputs=outputs,
                                    training=torch.onnx.TrainingMode.TRAINING)
        ort_outs1 = run_ort(ort_sess1, input=(x,))
        ort_sess2 = convert_to_onnx(script_model, input=(x,), opset_version=self.opset_version,
                                    example_outputs=outputs,
                                    training=torch.onnx.TrainingMode.EVAL)
        ort_outs2 = run_ort(ort_sess2, input=(x,))
        [np.testing.assert_allclose(ort_out1, ort_out2, atol=1e-7, rtol=0.001) for ort_out1, ort_out2 in
         zip(ort_outs1, ort_outs2)]

    def test_multiple_conv_bn(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()
                self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
                self.conv2 = torch.nn.Conv2d(64, 2, kernel_size=1, stride=1, padding=0, bias=False)
                self.conv3 = torch.nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1, bias=False)
                self.bn = torch.nn.BatchNorm2d(64)
                self.bn2 = torch.nn.BatchNorm2d(2)
                self.relu = torch.nn.ReLU(inplace=True)
                self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            def forward(self, x):
                x = self.conv1(x)
                x = self.bn(x)
                x = self.relu(x)
                x = self.maxpool(x)
                x = self.conv2(x)
                x = self.bn2(x)
                x = self.relu(x)
                x = self.conv3(x)
                x = self.bn2(x)
                x = self.relu(x)
                return x

        model = MyModule()
        x = torch.randn(2, 3, 224, 224)
        ort_sess1 = convert_to_onnx(model, input=(x,), opset_version=self.opset_version,
                                    training=torch.onnx.TrainingMode.TRAINING)
        ort_outs1 = run_ort(ort_sess1, input=(x,))
        ort_sess2 = convert_to_onnx(model, input=(x,), opset_version=self.opset_version,
                                    training=torch.onnx.TrainingMode.EVAL)
        ort_outs2 = run_ort(ort_sess2, input=(x,))
        [np.testing.assert_allclose(ort_out1, ort_out2, atol=1e-7, rtol=0.001) for ort_out1, ort_out2 in
         zip(ort_outs1, ort_outs2)]

    def test_script_custom_class_error(self):
        class BoxCoder(object):
            def __init__(self, bbox_xform_clip: float):
                # type: (float) -> None
                self.bbox_xform_clip = bbox_xform_clip

            def decode(self, rel_codes, boxes):
                # type: (Tensor, List[Tensor]) -> Tensor
                boxes = torch.cat(boxes, dim=0)
                pred_ctr_x = torch.clamp(rel_codes[:, 0::4], max=self.bbox_xform_clip) * boxes[:, 2]
                return pred_ctr_x

        class MyModule(torch.nn.Module):
            __annotations__ = {
                'box_coder': BoxCoder,
            }

            def __init__(self):
                super(MyModule, self).__init__()
                self.box_coder = BoxCoder(1.4)

            def forward(self, box_regression: torch.Tensor, proposals: List[torch.Tensor]):
                return self.box_coder.decode(box_regression, proposals)

        model = torch.jit.script(MyModule())
        box_regression = torch.randn([4, 4])
        proposal = [torch.randn(2, 4), torch.randn(2, 4)]
        outputs = model(box_regression, proposal)

        with self.assertRaises(RuntimeError) as cm:
            convert_to_onnx(model, input=(box_regression, proposal),
                            example_outputs=outputs)

    def test_initializer_sequence(self):
        class MyModule(torch.nn.Module):
            def __init__(self, input_size, hidden_size, num_classes):
                super(MyModule, self).__init__()
                self.fc1 = torch.nn.Linear(input_size, hidden_size)
                self.relu = torch.nn.ReLU()
                self.fc2 = torch.nn.Linear(hidden_size, num_classes)

            def forward(self, x):
                out = self.fc1(x)
                out = self.relu(out)
                out = self.fc2(out)
                return out

        test_model = MyModule(3, 4, 10)
        state_dict_list = [k for (k, v) in test_model.state_dict().items()]
        named_params_list = [k for (k, v) in test_model.named_parameters()]

        x = torch.randn(32, 3)
        f = io.BytesIO()
        torch.onnx._export(test_model, (x,), f, _retain_param_name=True, do_constant_folding=False)
        loaded_model = onnx.load_from_string(f.getvalue())

        actual_list = [p.name for p in loaded_model.graph.initializer]
        assert actual_list == state_dict_list, \
            "Initializers' sequence is not as same as state_dict(). Expected: (" \
            + ', '.join(state_dict_list) + "). Actual:(" + ', '.join(actual_list) + ")."
        assert actual_list == named_params_list, \
            "Initializers' sequence is not as same as named_parameters(). Expected: (" \
            + ', '.join(named_params_list) + "). Actual:(" + ', '.join(actual_list) + ")."

    def test_initializer_sequence_script_model(self):
        def list_is_expected(short_list, long_list) -> bool:
            if (len(short_list) > len(long_list)):
                return False

            for i in range(len(short_list)):
                if (short_list[i] not in long_list[i]):
                    return False

            return True

        def loop(x, y):
            for i in range(int(y)):
                x = x + i
            return x

        class MyModule(torch.nn.Module):
            def __init__(self, input_size, hidden_size, num_classes):
                super(MyModule, self).__init__()
                self.fc1 = torch.nn.Linear(input_size, hidden_size)
                self.relu = torch.nn.ReLU()
                self.fc2 = torch.nn.Linear(hidden_size, num_classes)

            def forward(self, x, y):
                x = loop(x, y)
                out = self.fc1(x)
                out = self.relu(out)
                out = self.fc2(out)
                return out

        test_model = torch.jit.script(MyModule(3, 4, 10))
        state_dict_list = [k for (k, v) in test_model.state_dict().items()]
        named_params_list = [k for (k, v) in test_model.named_parameters()]

        x = torch.ones(2, 3, dtype=torch.float)
        y = torch.tensor(5, dtype=torch.long)
        example_output = (test_model(x, y),)
        f = io.BytesIO()

        torch.onnx.export(test_model, (x, y), f, example_outputs=example_output, _retain_param_name=True, do_constant_folding=False)
        loaded_model = onnx.load_from_string(f.getvalue())

        actual_list = [p.name for p in loaded_model.graph.initializer]
        assert list_is_expected(state_dict_list, actual_list), \
            "ScriptModel - Initializers' sequence is not as same as state_dict(). Expected: (" \
            + ', '.join(state_dict_list) + "). Actual:(" + ', '.join(actual_list) + ")."
        assert list_is_expected(named_params_list, actual_list), \
            "ScriptModel - Initializers' sequence is not as same as named_parameters(). Expected: (" \
            + ', '.join(named_params_list) + "). Actual:(" + ', '.join(actual_list) + ")."

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_nms(self):
        boxes = torch.rand(5, 4)
        boxes[:, 2:] += torch.rand(5, 2)
        scores = torch.randn(5)

        class Module(torch.nn.Module):
            def forward(self, boxes, scores):
                return ops.nms(boxes, scores, 0.5)

        self.run_test(Module(), (boxes, scores))

    @skipIfUnsupportedMinOpsetVersion(11)
    @disableScriptTest()
    def test_clip_boxes_to_image(self):
        boxes = torch.randn(5, 4) * 500
        boxes[:, 2:] += boxes[:, :2]
        size = torch.randn(200, 300)

        size_2 = torch.randn(300, 400)

        class Module(torch.nn.Module):
            def forward(self, boxes, size):
                shape = (size.shape[0], size.shape[1])
                return ops.boxes.clip_boxes_to_image(boxes, shape)

        self.run_test(Module(), (boxes, size),
                      input_names=["boxes", "size"],
                      dynamic_axes={"size": [0, 1]},
                      test_with_inputs=[(boxes, size), (boxes, size_2)])

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_roi_align(self):
        x = torch.rand(1, 1, 10, 10, dtype=torch.float32)
        single_roi = torch.tensor([[0, 0, 0, 4, 4]], dtype=torch.float32)
        model = ops.RoIAlign((5, 5), 1., 2)
        self.run_test(model, (x, single_roi))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_roi_align_aligned(self):
        x = torch.rand(1, 1, 10, 10, dtype=torch.float32)
        single_roi = torch.tensor([[0, 1.5, 1.5, 3, 3]], dtype=torch.float32)
        model1 = ops.RoIAlign((5, 5), 1., 2, aligned=True)
        self.run_test(model1, (x, single_roi))

        x = torch.rand(1, 1, 10, 10, dtype=torch.float32)
        single_roi = torch.tensor([[0, 0.2, 0.3, 4.5, 3.5]], dtype=torch.float32)
        model2 = ops.RoIAlign((5, 5), 0.5, 3, aligned=True)
        self.run_test(model2, (x, single_roi))

        x = torch.rand(1, 1, 10, 10, dtype=torch.float32)
        single_roi = torch.tensor([[0, 0.2, 0.3, 4.5, 3.5]], dtype=torch.float32)
        model3 = ops.RoIAlign((5, 5), 1.8, 2, aligned=True)
        self.run_test(model3, (x, single_roi))

        x = torch.rand(1, 1, 10, 10, dtype=torch.float32)
        single_roi = torch.tensor([[0, 0.2, 0.3, 4.5, 3.5]], dtype=torch.float32)
        model4 = ops.RoIAlign((2, 2), 2.5, 0, aligned=True)
        self.run_test(model4, (x, single_roi))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_roi_pool(self):
        x = torch.rand(1, 1, 10, 10, dtype=torch.float32)
        rois = torch.tensor([[0, 0, 0, 4, 4]], dtype=torch.float32)
        pool_h = 5
        pool_w = 5
        model = ops.RoIPool((pool_h, pool_w), 2.)
        self.run_test(model, (x, rois))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_resize_images(self):
        class TransformModule(torch.nn.Module):
            def __init__(self):
                super(TransformModule, self).__init__()
                self.transform = _init_test_generalized_rcnn_transform()

            def forward(self, images):
                return self.transform.resize(images, None)[0]

        input = torch.rand(3, 10, 20)
        input_test = torch.rand(3, 100, 150)
        self.run_test(TransformModule(), (input,),
                      input_names=["input1"], dynamic_axes={"input1": [0, 1, 2]},
                      test_with_inputs=[(input,), (input_test,)])

    @skipIfUnsupportedMinOpsetVersion(11)
    @disableScriptTest()
    def test_transform_images(self):

        class TransformModule(torch.nn.Module):
            def __init__(self):
                super(TransformModule, self).__init__()
                self.transform = _init_test_generalized_rcnn_transform()

            def forward(self, images: List[torch.Tensor]):
                return self.transform(images)[0].tensors

        input = torch.rand(3, 100, 200), torch.rand(3, 200, 200)
        input_test = torch.rand(3, 100, 200), torch.rand(3, 200, 200)
        self.run_test(TransformModule(), (input,), test_with_inputs=[(input,), (input_test,)])

    def get_features(self, images):
        s0, s1 = images.shape[-2:]
        features = [
            ('0', torch.rand(2, 256, s0 // 4, s1 // 4)),
            ('1', torch.rand(2, 256, s0 // 8, s1 // 8)),
            ('2', torch.rand(2, 256, s0 // 16, s1 // 16)),
            ('3', torch.rand(2, 256, s0 // 32, s1 // 32)),
            ('4', torch.rand(2, 256, s0 // 64, s1 // 64)),
        ]
        features = OrderedDict(features)
        return features

    @skipIfUnsupportedMinOpsetVersion(11)
    @disableScriptTest()
    def test_rpn(self):
        set_rng_seed(0)

        class RPNModule(torch.nn.Module):
            def __init__(self):
                super(RPNModule, self).__init__()
                self.rpn = _init_test_rpn()

            def forward(self, images, features: Dict[str, torch.Tensor]):
                images_m = ImageList(images, [(i.shape[-1], i.shape[-2]) for i in images])
                return self.rpn(images_m, features)

        images = torch.rand(2, 3, 150, 150)
        features = self.get_features(images)
        images2 = torch.rand(2, 3, 80, 80)
        test_features = self.get_features(images2)

        model = RPNModule()
        model.eval()
        model(images, features)
        self.run_test(model, (images, features),
                      input_names=["input1", "input2", "input3", "input4", "input5", "input6"],
                      dynamic_axes={"input1": [0, 1, 2, 3], "input2": [0, 1, 2, 3],
                                    "input3": [0, 1, 2, 3], "input4": [0, 1, 2, 3],
                                    "input5": [0, 1, 2, 3], "input6": [0, 1, 2, 3]},
                      test_with_inputs=[(images, features), (images2, test_features)],
                      dict_check=False)

    @skipIfUnsupportedMinOpsetVersion(11)
    @disableScriptTest()
    def test_multi_scale_roi_align(self):

        class TransformModule(torch.nn.Module):
            def __init__(self):
                super(TransformModule, self).__init__()
                self.model = ops.MultiScaleRoIAlign(['feat1', 'feat2'], 3, 2)
                self.image_sizes = [(512, 512)]

            def forward(self, input, boxes):
                # type: (Dict[str, torch.Tensor], List[torch.Tensor]) -> torch.Tensor
                return self.model(input, boxes, self.image_sizes)

        i = OrderedDict()
        i['feat1'] = torch.rand(1, 5, 64, 64)
        i['feat2'] = torch.rand(1, 5, 16, 16)
        boxes = torch.rand(6, 4) * 256
        boxes[:, 2:] += boxes[:, :2]

        i1 = OrderedDict()
        i1['feat1'] = torch.rand(1, 5, 64, 64)
        i1['feat2'] = torch.rand(1, 5, 16, 16)
        boxes1 = torch.rand(6, 4) * 256
        boxes1[:, 2:] += boxes1[:, :2]

        self.run_test(TransformModule(), (i, [boxes],), test_with_inputs=[(i, [boxes],), (i1, [boxes1],)])

    @skipIfUnsupportedMinOpsetVersion(11)
    @disableScriptTest()
    def test_roi_heads(self):
        class RoiHeadsModule(torch.nn.Module):
            def __init__(self):
                super(RoiHeadsModule, self).__init__()
                self.transform = _init_test_generalized_rcnn_transform()
                self.rpn = _init_test_rpn()
                self.roi_heads = _init_test_roi_heads_faster_rcnn()

            def forward(self, images, features: Dict[str, torch.Tensor]):
                original_image_sizes = [(img.shape[-1], img.shape[-2]) for img in images]

                images_m = ImageList(images, [(i.shape[-1], i.shape[-2]) for i in images])
                proposals, _ = self.rpn(images_m, features)
                detections, _ = self.roi_heads(features, proposals, images_m.image_sizes)
                detections = self.transform.postprocess(detections,
                                                        images_m.image_sizes,
                                                        original_image_sizes)
                return detections

        images = torch.rand(2, 3, 100, 100)
        features = self.get_features(images)
        images2 = torch.rand(2, 3, 150, 150)
        test_features = self.get_features(images2)

        model = RoiHeadsModule()
        model.eval()
        model(images, features)

        self.run_test(model, (images, features),
                      input_names=["input1", "input2", "input3", "input4", "input5", "input6"],
                      dynamic_axes={"input1": [0, 1, 2, 3], "input2": [0, 1, 2, 3], "input3": [0, 1, 2, 3],
                                    "input4": [0, 1, 2, 3], "input5": [0, 1, 2, 3], "input6": [0, 1, 2, 3]},
                      test_with_inputs=[(images, features), (images2, test_features)],
                      dict_check=False)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_set_attr_modules(self):
        class InnerModule2(torch.nn.Module):
            def __init__(self, embedding_dim):
                super().__init__()
                self.weights = InnerModule2.get_embedding(embedding_dim)
                self.register_buffer("_float_tensor", torch.FloatTensor(1))
                self.const = 2

            @staticmethod
            def get_embedding(embedding_dim: int):
                emb = 4 / ((embedding_dim // 2) - 1)
                emb = torch.exp(torch.arange((embedding_dim // 2), dtype=torch.float) * -emb)
                return emb

            def forward(self, input, incremental_state: Optional[torch.Tensor] = None):
                bsz, seq_len = input.shape[0], input.shape[1]
                self.const = 3
                if self.weights is None:
                    self.weights = InnerModule.get_embedding(self.embedding_dim)
                self.weights = self.weights.to(self._float_tensor)
                self.weights = self.weights * self.const
                if incremental_state is not None:
                    pos = seq_len
                    return self.weights[1 + pos, :].expand(bsz, 1, -1)
                return (
                    self.weights.index_select(0, torch.ones((bsz * seq_len), dtype=torch.int64)).view(bsz, seq_len, -1)
                )

        class InnerModule(torch.nn.Module):
            def __init__(self, embedding_dim):
                super().__init__()
                self.weights = InnerModule.get_embedding(embedding_dim)
                self.module = InnerModule2(embedding_dim=8)

            @staticmethod
            def get_embedding(embedding_dim: int):
                emb = 4 / ((embedding_dim // 2) - 1)
                emb = torch.exp(torch.arange((embedding_dim // 2), dtype=torch.float) * -emb)
                return emb

            def forward(self, x):
                return self.module(x) + self.weights

        class Module(torch.nn.Module):
            def __init__(self):
                super(Module, self).__init__()
                self.module = InnerModule(embedding_dim=8)

            def forward(self, x):
                return self.module(x)

        x = torch.randn(3, 256)
        self.run_test(Module(), (x, ))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_set_attr_modules_2(self):
        class InnerModule(torch.nn.Module):
            def __init__(self, embedding_dim):
                super().__init__()
                self.embedding_dim = embedding_dim
                self.const = 2.5
                self.weights = InnerModule.get_embedding(self.embedding_dim)
                self.register_buffer("_float_tensor", torch.FloatTensor(1))

            @staticmethod
            def get_embedding(embedding_dim: int):
                emb = 4 / ((embedding_dim // 2) - 1)
                emb = torch.exp(torch.arange((embedding_dim // 2), dtype=torch.float) * -emb)
                return emb

            def forward(self, input, incremental_state: Optional[torch.Tensor] = None):
                bsz, seq_len = input.shape[0], input.shape[1]
                self.const = 1.5
                self.weights = InnerModule.get_embedding(self.embedding_dim)
                return (
                    self.weights.index_select(0, torch.ones((bsz * seq_len), dtype=torch.int64)).view(bsz, seq_len, -1)
                ) * self.const

        class Module(torch.nn.Module):
            def __init__(self):
                super(Module, self).__init__()
                self.module = InnerModule(embedding_dim=8)

            def forward(self, x):
                return self.module(x)

        x = torch.randn(3, 256)
        self.run_test(Module(), (x, ))

    def test_set_attr(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()
                self.conv = torch.nn.Conv1d(3, 10, 2)
                self.b = False

            def forward(self, box_regression, weight):
                self.b = True
                self.conv.weight = weight
                w = torch.softmax(self.conv.weight, dim=0)
                self.conv.weight = w + w
                if self.b:
                    return box_regression + self.conv.weight
                else:
                    return box_regression - self.conv.weight

        model = torch.jit.script(MyModule())
        weight = torch.ones(3, 2)
        box_regression = torch.randn(3, 2)
        self.run_test(model, (box_regression, weight))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_set_attr_2(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()
                self.conv = torch.nn.Conv1d(10, 3, 3)
                self.conv.bias = torch.nn.Parameter(torch.zeros(3, 10, 3))

            def set_cell_anchors(self, anchors):
                if self.conv.bias is not None:
                    b = self.conv.bias
                    assert b is not None
                    self.conv.bias = anchors + b
                elif self.conv.weight is not None:
                    self.conv.weight = torch.randn(3, 10)
                    self.conv.bias = self.conv.weight[:]

            def forward(self, anchors) -> Optional[torch.Tensor]:
                self.set_cell_anchors(anchors)
                return self.conv.bias

        model = torch.jit.script(MyModule())
        anchors = torch.ones(3, 10, 3)
        self.run_test(model, (anchors))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_set_attr_3(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()
                self.conv = torch.nn.Conv1d(10, 3, 3)
                self.conv.weight = torch.nn.Parameter(torch.zeros(3, 10))
                self.conv.bias = torch.nn.Parameter(torch.zeros(3, 10, 3))

            def set_cell_anchors(self, anchors, boxes):
                self.conv.weight = torch.ones(3, 10)
                if self.conv.bias is not None:
                    self.conv.bias = torch.randn(3, 10, 3)
                    self.conv.weight = anchors + self.conv.weight
                    boxes[:] = torch.zeros(2, 3)

            def forward(self, anchors) -> Tuple[torch.Tensor, torch.Tensor]:
                boxes = torch.ones(2, 2, 3)
                self.set_cell_anchors(anchors, boxes)
                if self.conv.bias is not None:
                    return self.conv.weight, boxes
                return anchors, boxes

        model = torch.jit.script(MyModule())
        anchors = torch.rand(3, 10)
        self.run_test(model, (anchors))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_set_attr_4(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()
                self.conv = torch.nn.Conv1d(10, 3, 3)
                self.conv.bias = torch.nn.Parameter(torch.zeros(3, 10, 3))

            def set_cell_anchors(self, anchors):
                self.conv.weight = torch.zeros(10, 3)
                if self.conv.bias is not None:
                    w = self.conv.bias
                    assert w is not None
                    self.conv.bias = anchors + w
                else:
                    self.conv.bias = torch.ones(3, 10, 3)

            def forward(self, feature_maps, anchors) -> Tuple[torch.Tensor, torch.Tensor]:
                self.set_cell_anchors(anchors)
                result = []
                if self.conv.bias is not None:
                    a = self.conv.bias
                    assert a is not None
                    result += [a]
                result += [feature_maps]
                return result[0], result[1]

        model = torch.jit.script(MyModule())
        x = torch.rand(5, 11, 30)
        anchors = torch.ones(3, 10, 3)
        self.run_test(model, (x, anchors))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_set_attr_in_loop(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super(MyModule, self).__init__()
                self.conv = torch.nn.Conv1d(10, 3, 3)
                self.conv.weight = torch.nn.Parameter(torch.zeros(3, 10))
                self.conv.bias = torch.nn.Parameter(torch.zeros(3, 10, 3))

            def set_cell_anchors(self, anchors, boxes):
                self.conv.weight = torch.randn(3, 10)
                for i in range(self.conv.weight.size(0)):
                    for j in range(10):
                        self.conv.bias = torch.randn(3, 10, 3)
                        self.conv.weight = anchors * i
                        boxes[j] += torch.ones(3, 3)

            def forward(self, anchors) -> Tuple[torch.Tensor, torch.Tensor]:
                boxes = torch.ones(10, 3, 3)
                self.set_cell_anchors(anchors, boxes)
                if self.conv.bias is not None:
                    return self.conv.weight, boxes
                return anchors, boxes

        model = torch.jit.script(MyModule())
        anchors = torch.rand(10)
        self.run_test(model, anchors)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_index_put_if(self):
        @torch.jit.script
        def check_init(input_data, hidden_size, prev_state):
            # type: (torch.Tensor, int, torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
            batch_size = input_data.size(0)
            spatial_size_0 = input_data.size(2)
            spatial_size_1 = input_data.size(3)
            # generate empty prev_state, if None is provided
            state_size = (2, batch_size, hidden_size, spatial_size_0, spatial_size_1)
            state = torch.zeros(state_size, device=input_data.device)
            state_copy = torch.zeros(state_size, device=input_data.device)
            if prev_state.size(0) == 0:
                state[:] = torch.zeros(batch_size, hidden_size, spatial_size_0, spatial_size_1) + state[:]
                state_copy[:] = torch.ones(batch_size, hidden_size, spatial_size_0, spatial_size_1) * 2
                state_copy[:] = torch.zeros(batch_size, hidden_size, spatial_size_0, spatial_size_1) * 2
            else:
                state[:] = torch.ones(batch_size, hidden_size, spatial_size_0, spatial_size_1) * 4
            return state, state_copy

        class Example(torch.nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.hidden_size = hidden_size

            def forward(self, input_data, prev_state):
                prev_state = check_init(input_data, self.hidden_size, prev_state)
                return prev_state[0], prev_state[1]

        model = Example(10)
        random_data = torch.rand((1, 5, 30, 30))
        empty_tensor = torch.tensor([], dtype=torch.float).view(0, 0, 0, 0, 0)
        self.run_test(model, (random_data, empty_tensor))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_index_put_if_2(self):
        @torch.jit.script
        def check_init(input_data, hidden_size, prev_state):
            # type: (torch.Tensor, int, torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
            batch_size = input_data.size(0)
            spatial_size_0 = input_data.size(2)
            spatial_size_1 = input_data.size(3)
            # generate empty prev_state, if None is provided
            state_size = (2, batch_size, hidden_size, spatial_size_0, spatial_size_1)
            state = torch.zeros(state_size, device=input_data.device)
            state_copy = torch.zeros(state_size, device=input_data.device)
            if prev_state.size(0) == 0:
                for i in range(2):
                    state[:] = torch.ones(batch_size, hidden_size, spatial_size_0, spatial_size_1) * i
                    state_copy[:] = torch.ones(batch_size, hidden_size, spatial_size_0, spatial_size_1) * i
            elif prev_state.size(0) == 1:
                s = state[:]
                state[:] = prev_state + s
            elif prev_state.size(0) == 2:
                state[:] = torch.ones(batch_size, hidden_size, spatial_size_0, spatial_size_1) * 4
            return state, state_copy

        class Example(torch.nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.hidden_size = hidden_size

            def forward(self, input_data, prev_state):
                prev_state = check_init(input_data, self.hidden_size, prev_state)
                return prev_state[0], prev_state[1]

        model = Example(10)
        random_data = torch.rand((1, 5, 30, 30))
        empty_tensor = torch.tensor([], dtype=torch.float).view(0, 0, 0, 0, 0)
        self.run_test(model, (random_data, empty_tensor))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_index_put_if_3(self):
        @torch.jit.script
        def check_init(input_data, hidden_size, prev_state):
            # type: (torch.Tensor, int, torch.Tensor) -> torch.Tensor
            batch_size = input_data.size(0)
            spatial_size_0 = input_data.size(2)
            spatial_size_1 = input_data.size(3)
            # generate empty prev_state, if None is provided
            state_size = (2, batch_size, hidden_size, spatial_size_0, spatial_size_1)
            state = torch.zeros(state_size, device=input_data.device)
            if prev_state.size(0) < 2:
                state = state * 3
                if prev_state.size(0) == 0:
                    state[:] = torch.ones(batch_size, hidden_size, spatial_size_0, spatial_size_1) * 3
                else:
                    state = state + 2

            return state

        class Example(torch.nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.hidden_size = hidden_size

            def forward(self, input_data, prev_state):
                prev_state = check_init(input_data, self.hidden_size, prev_state)
                return prev_state

        model = Example(4)
        random_data = torch.rand((1, 5, 4, 4))
        empty_tensor = torch.tensor([], dtype=torch.float).view(0, 0, 0, 0, 0)
        self.run_test(model, (random_data, empty_tensor))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_index_put_if_4(self):
        @torch.jit.script
        def check_init(input_data, hidden_size, prev_state):
            # type: (torch.Tensor, int, torch.Tensor) -> torch.Tensor
            batch_size = input_data.size(0)
            spatial_size_0 = input_data.size(2)
            spatial_size_1 = input_data.size(3)
            # generate empty prev_state, if None is provided
            state_size = (2, batch_size, hidden_size, spatial_size_0, spatial_size_1)
            state = torch.zeros(state_size, device=input_data.device)
            if prev_state.size(0) == 0:
                state = state + 3
                state[:] = torch.ones(batch_size, hidden_size, spatial_size_0, spatial_size_1) * 3
                state = state + 3
                state[:] = torch.ones(batch_size, hidden_size, spatial_size_0, spatial_size_1) * 4
            else:
                state = state + 2
            return state

        class Example(torch.nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.hidden_size = hidden_size

            def forward(self, input_data, prev_state):
                prev_state = check_init(input_data, self.hidden_size, prev_state)
                return prev_state

        model = Example(4)
        random_data = torch.rand((1, 5, 4, 4))
        empty_tensor = torch.tensor([], dtype=torch.float).view(0, 0, 0, 0, 0)
        self.run_test(model, (random_data, empty_tensor))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_list_append_in_block(self):
        class ListModel(torch.nn.Module):
            def forward(self, x, y):
                res = []
                for i in range(x.size(0)):
                    res.append(torch.matmul(x[i], y))
                return res

        model = torch.jit.script(ListModel())
        x = torch.randn(16, 3, 4)
        y = torch.randn(4, 5)
        self.run_test(model, (x, y))

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_list_append_in_nested_block(self):
        class ListModel(torch.nn.Module):
            def forward(self, x, y):
                res = []
                for i in range(x.size(0)):
                    for j in range(x.size(1)):
                        res.append(torch.matmul(x[i][j], y))
                return res

        model = torch.jit.script(ListModel())
        x = torch.randn(4, 4, 3, 4)
        y = torch.randn(4, 5)
        self.run_test(model, (x, y))

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_list_pop_in_block(self):
        class ListModel(torch.nn.Module):
            def forward(self, x, y):
                res = []
                elem = torch.matmul(x[0], y)
                for i in range(x.size(0)):
                    res.append(torch.matmul(x[i], y))
                for i in range(x.size(0)):
                    elem = res.pop()
                for i in range(x.size(0)):
                    res.append(torch.matmul(x[i], y))
                    elem = res.pop()
                return res.append(elem)

        model = torch.jit.script(ListModel())
        x = torch.randn(16, 3, 4)
        y = torch.randn(4, 5)
        self.run_test(model, (x, y))


    @skipIfUnsupportedMinOpsetVersion(13)
    def test_list_del_in_block(self):
        class ListModel(torch.nn.Module):
            def forward(self, x, y):
                res = []
                elem = torch.matmul(x[0], y)
                for i in range(x.size(0)):
                    res.append(torch.matmul(x[i], y))
                for i in range(x.size(0)):
                    del res[0]
                for i in range(x.size(0)):
                    res.append(torch.matmul(x[i], y))
                    del res[0]
                return res.append(elem)

        model = torch.jit.script(ListModel())
        x = torch.randn(16, 3, 4)
        y = torch.randn(4, 5)
        self.run_test(model, (x, y))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_index_put_inplace_ops(self):
        @torch.jit.script
        def check_init(input_data, hidden_size):
            # type: (torch.Tensor, int) -> torch.Tensor
            batch_size = input_data.size(0)
            spatial_size_0 = input_data.size(2)
            spatial_size_1 = input_data.size(3)
            # generate empty prev_state, if None is provided
            state_size = (2, batch_size, hidden_size, spatial_size_0, spatial_size_1)
            state = torch.zeros(state_size, device=input_data.device)
            if input_data.size(0) == 1:
                state[1] += torch.ones(batch_size, hidden_size, spatial_size_0, spatial_size_1) * 2
                state[1] /= torch.ones(batch_size, hidden_size, spatial_size_0, spatial_size_1) * 3
            for i in range(input_data.size(0)):
                state[1] += torch.ones(batch_size, hidden_size, spatial_size_0, spatial_size_1)
                state[1] /= torch.ones(batch_size, hidden_size, spatial_size_0, spatial_size_1) * i
            return state

        class Example(torch.nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.hidden_size = hidden_size

            def forward(self, input_data):
                state = check_init(input_data, self.hidden_size)
                return state

        model = Example(10)
        random_data = torch.rand((1, 5, 30, 30))
        self.run_test(model, (random_data))

    @disableScriptTest()
    def test_unsafe_chunk(self):
        class ChunkModel(torch.nn.Module):
            def forward(self, x):
                return torch.unsafe_chunk(x, 3, dim=1)

        model = ChunkModel()
        model.eval()
        x = torch.randn(1, 18)
        self.run_test(model, x, input_names=['x'])

    def test_symbolic_shape_inference(self):
        # ConstantOfShape is tested in test_embedding_bag
        # Tile is tested in test_repeat
        # test Shape, Reshape, Transpose, Gather
        class ShapeModel(torch.nn.Module):
            def forward(self, x, y):
                shape = x.size()[:3] + (-1,)  # shape [4], ('batch', 3, 4, -1)
                y = y.reshape(shape)  # batch, 3, 4, 10/batch
                return y.transpose(1, 2)

        model = ShapeModel()
        model.eval()
        x = torch.ones(2, 3, 4, 5)
        y = torch.ones(3, 4, 5, 2)
        self.run_test(model, (x, y))

        class ViewModel(torch.nn.Module):
            def forward(self, x):
                return x.view(-1)

        model = ViewModel()
        model.eval()
        x = torch.tensor(2.)
        self.run_test(model, (x,))

        # test prim::ListConstruct for Reshape input 1
        class ViewModel_2(torch.nn.Module):
            def forward(self, x):
                N, C, H, W = x.shape[0], x.shape[2], x.shape[3], x.shape[4]
                x1 = x.view(N, -1, C, H, W)
                x2 = x1.permute(0, 3, 4, 1, 2)
                return x2.reshape(N, -1, C)

        model = ViewModel_2()
        model.eval()
        x = torch.ones(2, 3, 4, 5, 6)
        self.run_test(model, x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_symbolic_shape_inference_arange(self):
        # test Range
        class ArangeModel(torch.nn.Module):
            def forward(self, signal):
                frame_step = 2
                outer_dimensions = signal.size()[:-2]
                frames, frame_length = signal.size()[-2:]

                subframe_length = signal.size()[0]
                subframe_step = frame_step // subframe_length
                subframes_per_frame = frame_length // subframe_length
                output_size = frame_step * (frames - 1) + frame_length
                output_subframes = output_size // subframe_length

                frame = torch.arange(0, output_subframes)
                return frame

        model = ArangeModel()
        model.eval()
        M, C, K, N = 1, 2, 3, 4
        x = torch.randint(5, (M, C, K, N))
        y = torch.randint(5, (M, C + 1, K + 1, N + 1))
        self.run_test(model, x)
        self.run_test(model, x, input_names=['x'],
                      dynamic_axes={'x' : [0, 1, 2, 3]}, test_with_inputs=[(x,), (y,)])

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_symbolic_shape_inference_box(self):
        # test NonZero
        class BoxModel(torch.nn.Module):
            def forward(self, boxes):
                min_size = 1e-2
                ws, hs = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
                keep = (ws >= min_size) & (hs >= min_size)
                keep = torch.where(keep)[0]
                return keep

        model = BoxModel()
        model.eval()
        x = torch.ones(2, 4)
        y = torch.ones(3, 5)
        self.run_test(model, x)
        self.run_test(model, x, input_names=['x'],
                      dynamic_axes={'x' : [0, 1]}, test_with_inputs=[(x,), (y,)])

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_symbolic_shape_inference_box_if(self):
        # test If
        class BoxIfModel(torch.nn.Module):
            def forward(self, boxes, scores):
                score_thresh = 0.0
                inds = torch.where(scores > score_thresh)[0]
                boxes_1 = boxes[inds]
                if boxes_1.numel() > 3:
                    return boxes_1
                else:
                    return boxes_1 * 2

        model = BoxIfModel()
        model.eval()
        boxes = torch.ones(2, 4)
        scores = torch.ones(1, 4)
        self.run_test(model, (boxes, scores))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_symbolic_shape_inference_arange_2(self):
        # test Range
        class ArangeModel(torch.nn.Module):
            def forward(self, start):
                return torch.arange(start.size(0), 8.5, 1.5, dtype=torch.int64)
        x = torch.randn(2, 3, 4)
        self.run_test(ArangeModel(), (x,))

        class ArangeModel2(torch.nn.Module):
            def forward(self, start):
                return torch.arange(start.size(0), 8.5, 1.5, dtype=torch.double)
        x = torch.randn(2, 3, 4)
        self.run_test(ArangeModel2(), (x,))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_symbolic_shape_inference_nonzero(self):
        class OneLikeModel(torch.nn.Module):
            def forward(self, x):
                ones = torch.ones_like(x, dtype=torch.float, layout=torch.strided, device=torch.device('cpu'))
                return torch.nonzero(ones)

        x = torch.randn(2)
        self.run_test(OneLikeModel(), x)
        x = torch.randn(2, 3, 4)
        self.run_test(OneLikeModel(), x)

        class ZeroLikeModel(torch.nn.Module):
            def forward(self, x):
                zeros = torch.zeros_like(x, dtype=torch.float, layout=torch.strided, device=torch.device('cpu'))
                return torch.nonzero(zeros)

        x = torch.randn(2)
        self.run_test(ZeroLikeModel(), x)
        x = torch.randn(2, 3, 4)
        self.run_test(ZeroLikeModel(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_symbolic_shape_inference_expand_1(self):
        class ExpandModel(torch.nn.Module):
            def forward(self, x):
                return x.expand(4, 6, 2)
        x = torch.randn(6, 1, requires_grad=True)
        self.run_test(ExpandModel(), (x,))

    @skipIfUnsupportedMinOpsetVersion(9)
    @disableScriptTest()  # Test code not scriptable
    def test_symbolic_shape_inference_expand_2(self):
        class M(torch.nn.Module):
            def forward(self, x):
                input_shape = x.size()
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length)
                causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
                return causal_mask.transpose(0, 1)
        x = torch.randn(3, 16)
        self.run_test(M(), (x,))

    @skipIfUnsupportedMinOpsetVersion(10)
    @disableScriptTest()  # Test code not scriptable
    def test_symbolic_shape_inference_slice(self):
        class M(torch.nn.Module):
            def forward(self, x, position_bias):
                input_shape = x.size()
                batch_size, seq_length = input_shape
                position_bias = position_bias[:, :, -seq_length:, :]
                return position_bias.transpose(0, 1)
        x = torch.randn(3, 16)
        position_bias = torch.randn(1, 3, 20, 8)
        self.run_test(M(), (x, position_bias))

    def test_symbolic_shape_inference_slice_2(self):
        class M(torch.nn.Module):
            def forward(self, position_bias):
                position_bias = position_bias[:, :, -2:, :]
                return position_bias.transpose(0, 1)
        position_bias = torch.randn(1, 3, 20, 8)
        self.run_test(M(), (position_bias,))

    @skipIfUnsupportedMinOpsetVersion(9)
    @disableScriptTest()
    def test_symbolic_shape_inference_time(self):
        input = torch.randn(RNN_SEQUENCE_LENGTH, BATCH_SIZE, RNN_INPUT_SIZE)
        h0 = torch.randn(1, BATCH_SIZE, RNN_HIDDEN_SIZE)
        c0 = torch.randn(1, BATCH_SIZE, RNN_HIDDEN_SIZE)
        model_lstm = torch.nn.LSTM(RNN_INPUT_SIZE, RNN_HIDDEN_SIZE, 1, bidirectional=False)
        self.run_test(model_lstm, (input, (h0, c0)), input_names=['x', 'y'],
                      dynamic_axes={'x' : [0, 1]})
        model_gru = torch.nn.GRU(RNN_INPUT_SIZE, RNN_HIDDEN_SIZE, 1, bidirectional=False, bias=False)
        self.run_test(model_gru, (input, h0), input_names=['x', 'y'],
                      dynamic_axes={'x' : [0, 1]})
        model_rnn = torch.nn.RNN(RNN_INPUT_SIZE, RNN_HIDDEN_SIZE, 1, bidirectional=False, bias=False)
        self.run_test(model_rnn, (input, h0), input_names=['x', 'y'],
                      dynamic_axes={'x' : [0, 1]})

    def test_symbolic_shape_inference_dynamic_axes(self):
        class M(torch.nn.Module):
            def forward(self, input_ids):
                input_shape = input_ids.size()
                input_ids = input_ids.view(-1, input_shape[-1])
                return input_ids.transpose(0, 1)
        x = torch.randn(3, 16)
        self.run_test(M(), (x,), input_names=['input_ids'],
                      dynamic_axes={'input_ids': {0: 'batch', 1: 'sequence'}})

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
    # There are still some issues prevent us from enabling script test for these scenarios:
    # test_gru_*:
    #   Operator aten::as_tensor is not supported by exporter yet.
    #       - https://msdata.visualstudio.com/Vienna/_workitems/edit/1055382
    #   Operator aten::_pack_padded_sequence is not supported by exporter yet.
    #       - https://msdata.visualstudio.com/Vienna/_workitems/edit/1055384
    @disableScriptTest()
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
                dropout_opts,):

        for base, name, extra_kwargs in (
                ('elman', 'elman_relu', {'nonlinearity': u'relu'}),
                ('elman', 'elman_tanh', {'nonlinearity': u'tanh'}),
                ('lstm', 'lstm', {}),
                ('gru', 'gru', {})
        ):
            make_test(name, base, layer, bidirectional, initial_state,
                      variable_length, dropout,
                      **extra_kwargs)
            test_count += 1

    # sanity check that a representative example does exist
    TestONNXRuntime.test_gru_trilayer_forward_with_initial_state_without_sequence_lengths_with_dropout

    # make sure no one accidentally disables all the tests without
    # noticing
    if test_count != 192:
        raise ValueError('Expected 192 tests but found {}'.format(test_count))

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

# opset 13 tests
TestONNXRuntime_opset13 = type(str("TestONNXRuntime_opset13"),
                               (unittest.TestCase,),
                               dict(TestONNXRuntime.__dict__, opset_version=13,
                                    keep_initializers_as_inputs=False,
                                    onnx_shape_inference=True))

if __name__ == '__main__':
    unittest.main()
