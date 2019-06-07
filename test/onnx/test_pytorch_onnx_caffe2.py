from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import sys
import unittest

import torch.onnx
import torch.onnx.operators
from torch.autograd import Variable, function
from debug_embed_params import run_embed_params
import io

import caffe2.python.onnx.backend as c2

import verify

skip = unittest.skip


def skipIfEmbed(func):
    def wrapper(testCaffe2Backend):
        if testCaffe2Backend.embed_params:
            raise unittest.SkipTest("Skip embed_params verify test")
        return func(testCaffe2Backend)
    return wrapper


# def import_model(proto, input, workspace=None, use_gpu=True):
#    model_def = onnx.ModelProto.FromString(proto)
#    onnx.checker.check_model(model_def)
#
#    if workspace is None:
#        workspace = {}
#    if isinstance(input, tuple):
#        for i in range(len(input)):
#            workspace[model_def.graph.input[i]] = input[i]
#    else:
#        workspace[model_def.graph.input[0]] = input
#
#    caffe2_out_workspace = c2.run_model(
#        init_graph=None,
#        predict_graph=graph_def,
#        inputs=workspace,
#        use_gpu=use_gpu)
#    caffe2_out = caffe2_out_workspace[0]
#    return caffe2_out


def do_export(model, inputs, *args, **kwargs):
    f = io.BytesIO()
    out = torch.onnx._export(model, inputs, f, *args, **kwargs)
    if isinstance(model, torch.jit.ScriptModule):
        # Special case for common case of passing a single Tensor
        if isinstance(inputs, torch.Tensor):
            inputs = (inputs,)
        out = model(*inputs)
    return f.getvalue(), out


torch.set_default_tensor_type('torch.FloatTensor')
try:
    import torch
except ImportError:
    print('Cannot import torch, hence caffe2-torch test will not run.')
    sys.exit(0)


BATCH_SIZE = 2

RNN_BATCH_SIZE = 7
RNN_SEQUENCE_LENGTH = 11
RNN_INPUT_SIZE = 5
RNN_HIDDEN_SIZE = 3

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
    'dcgan_b': 'https://s3.amazonaws.com/pytorch/test_data/export/netG_bedroom_epoch_1-0649e76b.pth',
    'dcgan_f': 'https://s3.amazonaws.com/pytorch/test_data/export/netG_faces_epoch_49-d86035a6.pth',
    'densenet121': 'https://download.pytorch.org/models/densenet121-d66d3027.pth',
    'inception_v3_google': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'srresNet': 'https://s3.amazonaws.com/pytorch/demos/srresnet-e10b2039.pth',
    'super_resolution': 'https://s3.amazonaws.com/pytorch/test_data/export/superres_epoch100-44c6958e.pth',
    'squeezenet1_0': 'https://download.pytorch.org/models/squeezenet1_0-a815701f.pth',
    'squeezenet1_1': 'https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}


def setUp():
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    np.random.seed(seed=0)


def convert_cuda(model, input):
    cuda_model = model.cuda()
    # input might be nested - we want to move everything to GPU
    cuda_input = function._nested_map(
        lambda o: isinstance(o, Variable) or torch.is_tensor(o),
        lambda o: o.cuda())(input)
    return cuda_model, cuda_input

def run_debug_test(testCaffe2Backend, model, train, batch_size, state_dict=None,
                   input=None, use_gpu=True, example_outputs=None,
                   opset_version=None):
    """
    # TODO: remove this from the final release version
    This test is for our debugging only for the case where
    embed_params=False
    """
    if not isinstance(model, torch.jit.ScriptModule):
        model.train(train)
    if state_dict is not None:
        model.load_state_dict(state_dict)

    # Either user specified input or random (deterministic) input
    if input is None:
        input = torch.randn(batch_size, 3, 224, 224, requires_grad=True)
    if use_gpu:
        model, input = convert_cuda(model, input)

    onnxir, torch_out = do_export(model, input, export_params=testCaffe2Backend.embed_params,
                                  verbose=False, example_outputs=example_outputs,
                                  do_constant_folding=False,
                                  opset_version=opset_version)
    if isinstance(torch_out, torch.autograd.Variable):
        torch_out = (torch_out,)

    caffe2_out = run_embed_params(onnxir, model, input, state_dict, use_gpu)
    for _, (x, y) in enumerate(zip(torch_out, caffe2_out)):
        np.testing.assert_almost_equal(x.data.cpu().numpy(), y, decimal=3)

def run_actual_test(testCaffe2Backend, model, train, batch_size, state_dict=None,
                    input=None, use_gpu=True, rtol=0.001, atol=1e-7,
                    example_outputs=None, do_constant_folding=False,
                    opset_version=None):
    """
    This is what the user facing version will look like
    """
    # set the training/test mode for the model
    if not isinstance(model, torch.jit.ScriptModule):
        model.train(train)
    # use the pre-trained model params if available
    if state_dict is not None:
        model.load_state_dict(state_dict)

    # Either user specified input or random (deterministic) input
    if input is None:
        input = torch.randn(batch_size, 3, 224, 224, requires_grad=True)
    # GPU-ize the model, if requested
    if use_gpu:
        model, input = convert_cuda(model, input)

    # Verify the model runs the same in Caffe2
    verify.verify(model, input, c2, rtol=rtol, atol=atol,
                  example_outputs=example_outputs,
                  do_constant_folding=do_constant_folding,
                  opset_version=opset_version)

def run_model_test(testCaffe2Backend, model, train, batch_size, state_dict=None,
                   input=None, use_gpu=True, rtol=0.001, atol=1e-7,
                   example_outputs=None, do_constant_folding=True,
                   opset_version=None):
    use_gpu_ = torch.cuda.is_available() and use_gpu
    # NOTE: do_constant_folding is turned on only when model has
    # parameters embedded (which are needed for constant folding),
    # i.e. for self.embed_params=True case. self.embed_params=True
    # for the TestCaffe2BackendEmbed class defined at the bottom.
    if testCaffe2Backend.embed_params:
        run_actual_test(testCaffe2Backend, model, train, batch_size, state_dict, input,
                        use_gpu=use_gpu_, rtol=rtol, atol=atol,
                        example_outputs=example_outputs,
                        do_constant_folding=do_constant_folding,
                        opset_version=opset_version)
    else:
        run_debug_test(testCaffe2Backend, model, train, batch_size, state_dict, input,
                       use_gpu=use_gpu_, example_outputs=example_outputs,
                       opset_version=opset_version)


if __name__ == '__main__':
    from test_pytorch_onnx_caffe2_opset9 import TestCaffe2Backend_opset9
    from test_pytorch_onnx_caffe2_opset9 import TestCaffe2BackendEmbed_opset9 
    from test_pytorch_onnx_caffe2_opset10 import TestCaffe2Backend_opset10
    unittest.main()
