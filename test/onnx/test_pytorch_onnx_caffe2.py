from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from functools import wraps
import numpy as np
import sys
import unittest
import itertools

import torch.onnx
import torch.onnx.operators
from torch import nn
from torch.autograd import Variable, function
import torch.utils.model_zoo as model_zoo
from torch.nn.utils import rnn as rnn_utils
from debug_embed_params import run_embed_params
import io

# Import various models for testing
from torchvision.models.alexnet import alexnet
from torchvision.models.inception import inception_v3
from torchvision.models.densenet import densenet121
from torchvision.models.resnet import resnet50
from torchvision.models.vgg import vgg16, vgg16_bn, vgg19, vgg19_bn

from model_defs.squeezenet import SqueezeNet
from model_defs.super_resolution import SuperResolutionNet
from model_defs.srresnet import SRResNet
import model_defs.dcgan as dcgan
import model_defs.word_language_model as word_language_model
from model_defs.mnist import MNIST
from model_defs.lstm_flattening_result import LstmFlatteningResult
from model_defs.rnn_model_with_packed_sequence import RnnModelWithPackedSequence

import onnx
import caffe2.python.onnx.backend as c2

from test_pytorch_common import skipIfTravis, skipIfNoLapack, skipIfNoCuda
import verify

skip = unittest.skip


def skipIfEmbed(func):
    def wrapper(self):
        if self.embed_params:
            raise unittest.SkipTest("Skip embed_params verify test")
        return func(self)
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


class TestCaffe2Backend(unittest.TestCase):
    embed_params = False

    def setUp(self):
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
        np.random.seed(seed=0)

    def convert_cuda(self, model, input):
        cuda_model = model.cuda()
        # input might be nested - we want to move everything to GPU
        cuda_input = function._nested_map(
            lambda o: isinstance(o, Variable) or torch.is_tensor(o),
            lambda o: o.cuda())(input)
        return cuda_model, cuda_input

    def run_debug_test(self, model, train, batch_size, state_dict=None,
                       input=None, use_gpu=True, example_outputs=None):
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
            model, input = self.convert_cuda(model, input)

        onnxir, torch_out = do_export(model, input, export_params=self.embed_params, verbose=False,
                                      example_outputs=example_outputs)
        if isinstance(torch_out, torch.autograd.Variable):
            torch_out = (torch_out,)

        caffe2_out = run_embed_params(onnxir, model, input, state_dict, use_gpu)
        for i, (x, y) in enumerate(zip(torch_out, caffe2_out)):
            np.testing.assert_almost_equal(x.data.cpu().numpy(), y, decimal=3)

    def run_actual_test(self, model, train, batch_size, state_dict=None,
                        input=None, use_gpu=True, rtol=0.001, atol=1e-7,
                        example_outputs=None):
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
            model, input = self.convert_cuda(model, input)

        # Verify the model runs the same in Caffe2
        verify.verify(model, input, c2, rtol=rtol, atol=atol)

    def run_model_test(self, model, train, batch_size, state_dict=None,
                       input=None, use_gpu=True, rtol=0.001, atol=1e-7,
                       example_outputs=None):
        use_gpu_ = torch.cuda.is_available() and use_gpu
        if self.embed_params:
            self.run_actual_test(model, train, batch_size, state_dict, input,
                                 use_gpu=use_gpu_, rtol=rtol, atol=atol,
                                 example_outputs=example_outputs)
        else:
            self.run_debug_test(model, train, batch_size, state_dict, input,
                                use_gpu=use_gpu_, example_outputs=example_outputs)

    def test_linear(self):
        model = nn.Linear(1, 1)
        input = torch.randn(1, 1, requires_grad=True)
        self.run_model_test(model, train=False, batch_size=0, input=input)

    def test_lstm_cell(self):
        model = nn.LSTMCell(RNN_INPUT_SIZE, RNN_HIDDEN_SIZE)
        input = torch.randn(BATCH_SIZE, RNN_INPUT_SIZE)
        h0 = torch.randn(BATCH_SIZE, RNN_HIDDEN_SIZE)
        c0 = torch.randn(BATCH_SIZE, RNN_HIDDEN_SIZE)
        self.run_model_test(model, train=False, batch_size=BATCH_SIZE, input=(input, (h0, c0)), use_gpu=False)

    def test_gru_cell(self):
        model = nn.GRUCell(RNN_INPUT_SIZE, RNN_HIDDEN_SIZE)
        input = torch.randn(BATCH_SIZE, RNN_INPUT_SIZE)
        h0 = torch.randn(BATCH_SIZE, RNN_HIDDEN_SIZE)
        self.run_model_test(model, train=False, batch_size=BATCH_SIZE, input=(input, h0), use_gpu=False)

    def _dispatch_rnn_test(self, name, *args, **kwargs):
        if name == 'elman':
            self._elman_rnn_test(*args, **kwargs)
        if name == 'lstm':
            self._lstm_test(*args, **kwargs)
        if name == 'gru':
            self._gru_test(*args, **kwargs)

    def _elman_rnn_test(self, layers, nonlinearity, bidirectional,
                        initial_state, packed_sequence, dropout):
        model = nn.RNN(RNN_INPUT_SIZE, RNN_HIDDEN_SIZE,
                       layers,
                       nonlinearity=nonlinearity,
                       bidirectional=bidirectional,
                       dropout=dropout)

        if packed_sequence == 1:
            model = RnnModelWithPackedSequence(model, False)
        if packed_sequence == 2:
            model = RnnModelWithPackedSequence(model, True)

        def make_input(batch_size):
            seq_lengths = np.random.randint(1, RNN_SEQUENCE_LENGTH + 1, size=batch_size)
            seq_lengths = list(reversed(sorted(map(int, seq_lengths))))
            inputs = [torch.randn(l, RNN_INPUT_SIZE) for l in seq_lengths]
            inputs = rnn_utils.pad_sequence(inputs)
            if packed_sequence == 2:
                inputs = inputs.transpose(0, 1)
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
        self.run_model_test(model, train=False, batch_size=RNN_BATCH_SIZE, input=input, use_gpu=False, atol=1e-7)

        # test that the model still runs with a different batch size
        onnxir, _ = do_export(model, input)
        other_input = make_input(RNN_BATCH_SIZE + 1)
        _ = run_embed_params(onnxir, model, other_input, use_gpu=False)

    def _lstm_test(self, layers, bidirectional, initial_state,
                   packed_sequence, dropout):
        model = LstmFlatteningResult(
            RNN_INPUT_SIZE, RNN_HIDDEN_SIZE, layers,
            bidirectional=bidirectional, dropout=dropout)
        if packed_sequence == 1:
            model = RnnModelWithPackedSequence(model, False)
        if packed_sequence == 2:
            model = RnnModelWithPackedSequence(model, True)

        def make_input(batch_size):
            seq_lengths = np.random.randint(1, RNN_SEQUENCE_LENGTH + 1, size=batch_size)
            seq_lengths = list(reversed(sorted(map(int, seq_lengths))))
            inputs = [torch.randn(l, RNN_INPUT_SIZE) for l in seq_lengths]
            inputs = rnn_utils.pad_sequence(inputs)
            if packed_sequence == 2:
                inputs = inputs.transpose(0, 1)
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
        self.run_model_test(model, train=False, batch_size=RNN_BATCH_SIZE, input=input, use_gpu=False)

        # test that the model still runs with a different batch size
        onnxir, _ = do_export(model, input)
        other_input = make_input(RNN_BATCH_SIZE + 1)
        _ = run_embed_params(onnxir, model, other_input, use_gpu=False)

    def _gru_test(self, layers, bidirectional, initial_state,
                  packed_sequence, dropout):
        model = nn.GRU(RNN_INPUT_SIZE, RNN_HIDDEN_SIZE, layers,
                       bidirectional=bidirectional, dropout=dropout)
        if packed_sequence == 1:
            model = RnnModelWithPackedSequence(model, False)
        if packed_sequence == 2:
            model = RnnModelWithPackedSequence(model, True)

        def make_input(batch_size):
            seq_lengths = np.random.randint(1, RNN_SEQUENCE_LENGTH + 1, size=batch_size)
            seq_lengths = list(reversed(sorted(map(int, seq_lengths))))
            inputs = [torch.randn(l, RNN_INPUT_SIZE) for l in seq_lengths]
            inputs = rnn_utils.pad_sequence(inputs)
            if packed_sequence == 2:
                inputs = inputs.transpose(0, 1)
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
        self.run_model_test(model, train=False, batch_size=RNN_BATCH_SIZE, input=input, use_gpu=False)

        # test that the model still runs with a different batch size
        onnxir, _ = do_export(model, input)
        other_input = make_input(RNN_BATCH_SIZE + 1)
        _ = run_embed_params(onnxir, model, other_input, use_gpu=False)

    def test_rnn_init_predict_split(self):
        model = nn.LSTM(RNN_INPUT_SIZE, RNN_HIDDEN_SIZE, 3, bidirectional=True)
        seq_lengths = np.random.randint(1, RNN_SEQUENCE_LENGTH + 1, size=7)
        seq_lengths = list(reversed(sorted(map(int, seq_lengths))))
        input = [torch.randn(l, RNN_INPUT_SIZE) for l in seq_lengths]
        input = rnn_utils.pad_sequence(input)

        # Test that we are correctly splitting between init and
        # predict net. When we embed parameters, there should be more
        # ops in the init net.
        mp = onnx.ModelProto.FromString(do_export(model, input, export_params=self.embed_params)[0])
        prepared = c2.prepare(mp, device='CPU')
        if self.embed_params:
            assert len(prepared.init_net.op) == 875
            assert len(prepared.predict_net.op) == 130
        else:
            assert len(prepared.init_net.op) == 8
            assert len(prepared.predict_net.op) == 997

    def test_alexnet(self):
        state_dict = model_zoo.load_url(model_urls['alexnet'], progress=False)
        self.run_model_test(alexnet(), train=False, batch_size=BATCH_SIZE,
                            state_dict=state_dict, atol=1e-3)

    @skipIfNoCuda
    def test_dcgan(self):
        # dcgan is flaky on some seeds, see:
        # https://github.com/ProjectToffee/onnx/pull/70
        torch.manual_seed(1)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(1)

        netD = dcgan._netD(1)
        netD.apply(dcgan.weights_init)
        input = torch.randn(BATCH_SIZE, 3, dcgan.imgsz, dcgan.imgsz)
        self.run_model_test(netD, train=False, batch_size=BATCH_SIZE,
                            input=input)

        netG = dcgan._netG(1)
        netG.apply(dcgan.weights_init)
        state_dict = model_zoo.load_url(model_urls['dcgan_b'], progress=False)
        # state_dict = model_zoo.load_url(model_urls['dcgan_f'], progress=False)
        noise = torch.randn(BATCH_SIZE, dcgan.nz, 1, 1).normal_(0, 1)
        self.run_model_test(netG, train=False, batch_size=BATCH_SIZE,
                            input=noise, state_dict=state_dict, rtol=1e-2, atol=1e-6)

    @unittest.skipIf(not torch.cuda.is_available(),
                     "model on net has cuda in it, awaiting fix")
    def test_densenet(self):
        state_dict = model_zoo.load_url(model_urls['densenet121'], progress=False)
        self.run_model_test(densenet121(), train=False, batch_size=BATCH_SIZE,
                            state_dict=state_dict, atol=1e-7)

    @skip("doesn't match exactly...")
    # TODO: figure out the numerical instabilities
    def test_inception(self):
        x = torch.randn(BATCH_SIZE, 3, 299, 299, requires_grad=True)
        # state_dict = model_zoo.load_url(model_urls['inception_v3_google'], progress=False)
        state_dict = None
        self.run_model_test(inception_v3(), train=False, batch_size=BATCH_SIZE,
                            state_dict=state_dict, input=x)

    def test_resnet(self):
        state_dict = model_zoo.load_url(model_urls['resnet50'], progress=False)
        self.run_model_test(resnet50(), train=False, batch_size=BATCH_SIZE,
                            state_dict=state_dict, atol=1e-6)

    def test_squeezenet(self):
        sqnet_v1_1 = SqueezeNet(version=1.1)
        state_dict = model_zoo.load_url(model_urls['squeezenet1_1'], progress=False)
        # state_dict = model_zoo.load_url(model_urls['squeezenet1_0'], progress=False)
        self.run_model_test(sqnet_v1_1, train=False, batch_size=BATCH_SIZE,
                            state_dict=state_dict)

    # @skip('takes long to run, LAPACK needed for gpu')
    @skipIfNoLapack
    @unittest.skip("This model takes too much memory")
    def test_srresnet(self):
        super_resolution_net = SRResNet(
            rescale_factor=4, n_filters=64, n_blocks=8)
        state_dict = model_zoo.load_url(model_urls['srresNet'], progress=False)
        x = torch.randn(1, 3, 224, 224, requires_grad=True)
        self.run_model_test(super_resolution_net, train=False,
                            batch_size=1, state_dict=state_dict,
                            input=x, use_gpu=False)

    @skipIfTravis
    @skipIfNoLapack
    @skipIfNoCuda
    def test_super_resolution(self):
        super_resolution_net = SuperResolutionNet(upscale_factor=3)
        state_dict = model_zoo.load_url(model_urls['super_resolution'], progress=False)
        x = torch.randn(1, 1, 224, 224, requires_grad=True)
        self.run_model_test(super_resolution_net, train=False,
                            batch_size=BATCH_SIZE, state_dict=state_dict,
                            input=x, use_gpu=False, atol=1e-6)

    @unittest.skip("This model takes too much memory")
    def test_vgg16(self):
        state_dict = model_zoo.load_url(model_urls['vgg16'], progress=False)
        self.run_model_test(vgg16(), train=False, batch_size=BATCH_SIZE,
                            state_dict=state_dict)

    @skip("disable to run tests faster...")
    def test_vgg16_bn(self):
        self.run_model_test(vgg16_bn(), train=False,
                            batch_size=BATCH_SIZE)

    @skip("disable to run tests faster...")
    def test_vgg19(self):
        state_dict = model_zoo.load_url(model_urls['vgg19'], progress=False)
        self.run_model_test(vgg19(), train=False, batch_size=BATCH_SIZE,
                            state_dict=state_dict)

    @skip("disable to run tests faster...")
    def test_vgg19_bn(self):
        self.run_model_test(vgg19_bn(), train=False,
                            batch_size=BATCH_SIZE)

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
        self.run_model_test(model, train=False, input=(x, model.hidden),
                            batch_size=batchsize, use_gpu=False)

    def test_word_language_model_RNN_TANH(self):
        self.run_word_language_model("RNN_TANH")

    def test_word_language_model_RNN_RELU(self):
        self.run_word_language_model("RNN_RELU")

    def test_word_language_model_LSTM(self):
        self.run_word_language_model("LSTM")

    def test_word_language_model_GRU(self):
        self.run_word_language_model("GRU")

    def test_batchnorm1d_special(self):
        c = torch.randn(BATCH_SIZE, 224)
        model = nn.BatchNorm1d(224)
        self.run_model_test(model, train=True, input=c, batch_size=BATCH_SIZE)

    def test_batchnorm2d_noaffine(self):
        c = torch.randn(128, 128, 1, 1)
        model = nn.BatchNorm2d(128, affine=False)
        self.run_model_test(model, train=False, input=c, batch_size=BATCH_SIZE)

    def test_constant(self):
        c = torch.randn(BATCH_SIZE, 3, 224, 224)

        class MyModel(torch.nn.Module):
            def __init__(self):
                super(MyModel, self).__init__()

            def forward(self, input):
                return input + c.type_as(input)

        self.run_model_test(MyModel(), train=False, batch_size=BATCH_SIZE)

    def test_consumed_bn(self):
        underlying = nn.BatchNorm2d(3)
        self.run_model_test(underlying, train=True, batch_size=BATCH_SIZE)

    def _test_index_generic(self, fn):
        class MyModel(torch.nn.Module):
            def __init__(self):
                super(MyModel, self).__init__()

            def forward(self, input):
                return fn(input)

        m1 = torch.randn(3, 4)
        self.run_model_test(MyModel(), input=m1, train=False, batch_size=BATCH_SIZE)

    def test_index_1d(self):
        self._test_index_generic(lambda input: input[0])

    def test_index_2d_1dimslice(self):
        self._test_index_generic(lambda input: input[0:1, :])

    def test_index_2d_sliceint(self):
        self._test_index_generic(lambda input: input[1, :])

    def test_index_2d_neg_slice(self):
        self._test_index_generic(lambda input: input[0:-1, :])

    # TODO: Slicing along two dimensions is currently unsupported by the caffe2
    # backend. Revisit if this becomes supported in the future.
    """
    def test_index_2d_2dimslice(self):
        self._test_index_generic(lambda input: input[0:1, 0:1])
    """
    """
    def test_index_2d_neg_slice2dim(self):
        self._test_index_generic(lambda input: input[0:-1, 0:-1])
    """

    def test_chunk(self):
        class MyModel(torch.nn.Module):
            def __init__(self):
                super(MyModel, self).__init__()

            def forward(self, input):
                # TODO: Why index? This returns a tuple and test runner doesn't
                # support tuple comparison.
                return input.chunk(8, dim=2)[-1]
        self.run_model_test(MyModel(), train=False, batch_size=BATCH_SIZE)

    def test_sqrt(self):
        class MyModel(torch.nn.Module):
            def __init__(self):
                super(MyModel, self).__init__()

            def forward(self, input):
                return input.sqrt()
        input = torch.empty(BATCH_SIZE, 10, 10).uniform_(4, 9)
        self.run_model_test(MyModel(), train=False, input=input, batch_size=BATCH_SIZE)

    def test_log(self):
        class MyModel(torch.nn.Module):
            def __init__(self):
                super(MyModel, self).__init__()

            def forward(self, input):
                return input.log()
        input = torch.empty(BATCH_SIZE, 10, 10).uniform_(4, 9)
        self.run_model_test(MyModel(), train=False, input=input, batch_size=BATCH_SIZE)

    def test_trigonometry(self):
        def test_func(name):
            class MyModel(torch.nn.Module):
                def __init__(self):
                    super(MyModel, self).__init__()

                def forward(self, input):
                    return getattr(input, name)()
            input = torch.empty(BATCH_SIZE, 10, 10).uniform_()
            self.run_model_test(MyModel(), train=False, input=input, batch_size=BATCH_SIZE)

        test_func('cos')
        test_func('sin')
        test_func('tan')
        test_func('acos')
        test_func('asin')
        test_func('atan')

    def test_addconstant(self):
        class MyModel(torch.nn.Module):
            def __init__(self):
                super(MyModel, self).__init__()

            def forward(self, input):
                # TODO: Why index? This returns a tuple and test runner doesn't
                # support tuple comparison.
                return input + 1
        self.run_model_test(MyModel(), train=False, batch_size=BATCH_SIZE)

    def test_subconstant(self):
        class MyModel(torch.nn.Module):
            def __init__(self):
                super(MyModel, self).__init__()

            def forward(self, input):
                # TODO: Why index? This returns a tuple and test runner doesn't
                # support tuple comparison.
                return input - 1
        self.run_model_test(MyModel(), train=False, batch_size=BATCH_SIZE)

    def test_embedding(self):
        model = nn.Embedding(10, 3, padding_idx=-1)
        input = torch.LongTensor(list(range(10))[::-1])
        self.run_model_test(model, train=False, input=input, batch_size=BATCH_SIZE)

    def test_constantpad2d(self):
        model = nn.ConstantPad2d((1, 2, 3, 4), 3.5)
        self.run_model_test(model, train=False, batch_size=BATCH_SIZE)

    def test_reflectionpad2d(self):
        model = nn.ReflectionPad2d((1, 2, 3, 4))
        self.run_model_test(model, train=False, batch_size=BATCH_SIZE)

    def test_replicationpad2d(self):
        model = nn.ReplicationPad2d((1, 2, 3, 4))
        self.run_model_test(model, train=False, batch_size=BATCH_SIZE)

    def test_maxpool2d(self):
        model = nn.MaxPool2d(5, padding=(1, 2))
        self.run_model_test(model, train=False, batch_size=BATCH_SIZE)

    def test_maxpool2d_single_padding(self):
        model = nn.MaxPool2d(5, padding=2)
        self.run_model_test(model, train=False, batch_size=BATCH_SIZE)

    @unittest.skip("C2 and PyTorch have small difference in padding implementation")
    def test_avgpool2d(self):
        model = nn.AvgPool2d(5, padding=(2))
        self.run_model_test(model, train=False, batch_size=BATCH_SIZE)

    def test_avgpool2d_no_padding(self):
        model = nn.AvgPool2d(5)
        self.run_model_test(model, train=False, batch_size=BATCH_SIZE)

    def test_weight_norm(self):
        model = nn.utils.weight_norm(nn.Conv1d(1, 1, 3))
        input = torch.randn(1, 1, 5, requires_grad=True)
        self.run_model_test(
            model, train=True, batch_size=0, input=input, use_gpu=False
        )

    def test_mnist(self):
        model = MNIST()
        input = torch.randn(BATCH_SIZE, 1, 28, 28)
        state_dict = None
        # TODO: test with state_dict
        self.run_model_test(model, train=False, input=input, batch_size=BATCH_SIZE,
                            state_dict=state_dict)

    def test_mm(self):
        class MyModel(torch.nn.Module):
            def __init__(self):
                super(MyModel, self).__init__()

            def forward(self, m1, m2):
                return torch.mm(m1, m2)
        m1 = torch.randn(3, 4)
        m2 = torch.randn(4, 5)
        self.run_model_test(MyModel(), train=False, input=(m1, m2), batch_size=BATCH_SIZE, use_gpu=False)

    def test_addmm(self):
        class MyModel(torch.nn.Module):
            def __init__(self):
                super(MyModel, self).__init__()

            def forward(self, ma, m1, m2):
                return torch.addmm(ma, m1, m2)
        ma = torch.randn(5)
        m1 = torch.randn(3, 4)
        m2 = torch.randn(4, 5)
        self.run_model_test(MyModel(), train=False, input=(ma, m1, m2), batch_size=BATCH_SIZE, use_gpu=False)

    # test for a pytorch optimization pass, see https://github.com/pytorch/pytorch/pull/7872
    def test_consecutive_transposes(self):
        class MyModel(torch.nn.Module):
            def __init__(self):
                super(MyModel, self).__init__()

            def forward(self, x):
                return x.transpose(1, 2).transpose(2, 3)
        x = torch.randn(5, 6, 7, 8)
        self.run_model_test(MyModel(), train=False, input=x, batch_size=BATCH_SIZE, use_gpu=False)

    def test_sum(self):
        shape = (3, 4, 5)
        for params in [{}] + [{'dim': i} for i in range(len(shape))]:
            class MyModel(torch.nn.Module):
                def __init__(self):
                    super(MyModel, self).__init__()

                def forward(self, x):
                    return torch.sum(x, **params)
            x = torch.randn(*shape)
            self.run_model_test(MyModel(), train=False, input=(x), batch_size=BATCH_SIZE, use_gpu=False)

    def test_cumsum(self):
        shape = (3, 4, 5)
        for params in [{'dim': i} for i in range(len(shape))]:
            class MyModel(torch.nn.Module):
                def __init__(self):
                    super(MyModel, self).__init__()

                def forward(self, x):
                    return torch.cumsum(x, **params)
            x = torch.randn(*shape)
            self.run_model_test(MyModel(), train=False, input=(x), batch_size=BATCH_SIZE, use_gpu=False)

    def test_layer_norm(self):
        shape = (20, 5, 10, 10)

        class MyModel(torch.nn.Module):
            def __init__(self):
                super(MyModel, self).__init__()
                self.ln = torch.nn.LayerNorm([5, 10, 10])

            def forward(self, x):
                return self.ln(x)

        x = torch.randn(*shape)
        self.run_model_test(MyModel(), train=False, input=(x,), batch_size=BATCH_SIZE, use_gpu=False)

    def test_repeat(self):
        class MyModel(torch.nn.Module):
            def __init__(self):
                super(MyModel, self).__init__()

            def forward(self, x):
                return x.repeat(1, 2, 3, 4)

        x = torch.randn(4, 3, 2, 1, requires_grad=True)
        self.run_model_test(MyModel(), train=False, input=(x), batch_size=BATCH_SIZE, use_gpu=False)

    def test_upsample(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        model = nn.Upsample(scale_factor=2, mode='nearest')
        self.run_model_test(model, train=False, input=(x),
                            batch_size=BATCH_SIZE, use_gpu=False)

    def test_repeat_dim_overflow(self):
        class MyModel(torch.nn.Module):
            def __init__(self):
                super(MyModel, self).__init__()

            def forward(self, x):
                return x.repeat(1, 2, 3, 4)

        x = torch.randn(1, 2, requires_grad=True)
        self.run_model_test(MyModel(), train=False, input=(x), batch_size=BATCH_SIZE, use_gpu=False)

    def test_repeat_dynamic(self):
        class MyModel(torch.nn.Module):
            def __init__(self):
                super(MyModel, self).__init__()

            def forward(self, x, y):
                return x.repeat(y.size()[0] / 2, y.size()[1] * 2)

        x = torch.randn(1, 2, requires_grad=True)
        y = torch.randn(2, 4, requires_grad=True)
        self.run_model_test(MyModel(), train=False, input=(x, y), batch_size=BATCH_SIZE, use_gpu=False)

    def test_mean(self):
        shape = (3, 4, 5)
        for params in [{}] + [{'dim': i} for i in range(len(shape))]:
            class MyModel(torch.nn.Module):
                def __init__(self):
                    super(MyModel, self).__init__()

                def forward(self, x):
                    return torch.mean(x, **params)
            x = torch.randn(*shape)
            self.run_model_test(MyModel(), train=False, input=(x), batch_size=BATCH_SIZE, use_gpu=False)

    # TODO: Add test cases for prod once Caffe2 has support for ReduceProd
    def test_softmax(self):
        for i in range(7)[2:]:
            model = nn.Softmax(dim=i - 1)
            dims = [2] * (i - 2) + [3, 4]
            input = torch.ones(*dims, requires_grad=True)
            self.run_model_test(model, train=False, batch_size=BATCH_SIZE, input=input)

    def test_logsoftmax(self):
        for i in range(7)[2:]:
            model = nn.LogSoftmax(dim=i - 1)
            dims = [2] * (i - 2) + [3, 4]
            input = torch.ones(*dims, requires_grad=True)
            self.run_model_test(model, train=False, batch_size=BATCH_SIZE, input=input)

    def test_randn(self):
        x = torch.randn(1, 2, 3, 4)

        class MyModule(torch.nn.Module):
            def forward(self, x):
                return (torch.randn(1, 2, 3, 4) + x).shape
        self.run_model_test(MyModule(), train=False, input=(x),
                            batch_size=BATCH_SIZE, use_gpu=False)

    def test_convtranspose(self):
        model = nn.ConvTranspose2d(3, 3, 3, stride=3, bias=False, padding=1, output_padding=2)
        self.run_model_test(model, train=False, batch_size=BATCH_SIZE, atol=1e-7)

    def test_unsqueeze(self):
        shape = (3, 4, 5)
        for dim in range(len(shape) + 1):
            class MyModel(torch.nn.Module):
                def __init__(self):
                    super(MyModel, self).__init__()

                def forward(self, x):
                    return x.unsqueeze(dim)
            x = torch.randn(*shape)
            self.run_model_test(MyModel(), train=False, input=(x), batch_size=BATCH_SIZE, atol=1e-7)

    # NB: InstanceNorm model includes unused weights, so skip this in TestCaffe2BackendEmbed
    # TODO: We should have another pass to eliminate the unused initializers in ONNX models.
    @skipIfEmbed
    def test_instance_norm(self):
        underlying = nn.InstanceNorm2d(3)
        self.run_model_test(underlying, train=False, batch_size=BATCH_SIZE)

    def test_pixel_shuffle(self):
        underlying = nn.PixelShuffle(4)
        shape = (1, 64, 5, 5)
        input = Variable(torch.randn(*shape),
                         requires_grad=True)
        self.run_model_test(underlying, train=False, input=(input),
                            batch_size=BATCH_SIZE)

    def test_dynamic_sizes(self):
        class MyModel(torch.nn.Module):
            def __init__(self):
                super(MyModel, self).__init__()

            def forward(self, x):
                shape = torch.onnx.operators.shape_as_tensor(x)
                new_shape = torch.cat((torch.LongTensor([-1]), shape[0].view(1)))
                return torch.onnx.operators.reshape_from_tensor_shape(x, new_shape)
        x = torch.randn(3, 5, 7)
        self.run_model_test(MyModel(), train=False, input=x, batch_size=BATCH_SIZE, use_gpu=False)

    def test_advanced_broadcast(self):
        class MyModel(torch.nn.Module):
            def __init__(self):
                super(MyModel, self).__init__()

            def forward(self, x, y):
                return torch.mul(x, y)
        x = torch.randn(1, 5, 10)
        y = torch.randn(1, 5, 1)
        self.run_model_test(MyModel(), train=False, input=(x, y), batch_size=BATCH_SIZE, use_gpu=False)

    def test_int8_export(self):
        class MyModel(torch.nn.Module):
            def __init__(self):
                super(MyModel, self).__init__()
                self.param = torch.ByteTensor(3, 4).random_()

            def forward(self, x):
                return x * self.param.float()

        import io
        f = io.BytesIO()
        from torch.onnx import ExportTypes
        torch.onnx._export(MyModel(), (torch.rand(3, 4),), f, verbose=True, export_type=ExportTypes.ZIP_ARCHIVE)

        X = np.random.rand(3, 4).astype(np.float32)

        f.seek(0)
        import caffe2.python.onnx.backend as c2
        model = c2.prepare_zip_archive(f)
        model.run(X)

    def test_neg_slice(self):
        class NegSlice(torch.nn.Module):
            def forward(self, x):
                return x[-1, :, :]

        x = torch.randn(3, 4, 5)
        self.run_model_test(NegSlice(), train=False, input=(x,), batch_size=BATCH_SIZE, use_gpu=False)

    def test_neg_slice_large(self):
        class NegSlice(torch.nn.Module):
            def forward(self, x):
                return x[:, :, :, :, -3]

        x = torch.randn(3, 4, 5, 6, 7)
        self.run_model_test(NegSlice(), train=False, input=(x,), batch_size=BATCH_SIZE, use_gpu=False)

    @unittest.skip('https://github.com/pytorch/pytorch/issues/10984')
    def test_neg_slice_large_negone(self):
        class NegSlice(torch.nn.Module):
            def forward(self, x):
                return x[:, :, :, :, -1]

        x = torch.randn(3, 4, 5, 6, 7)
        self.run_model_test(NegSlice(), train=False, input=(x,), batch_size=BATCH_SIZE, use_gpu=False)

    def test_dynamic_slice(self):
        class DynamicSliceExportMod(torch.nn.Module):
            def forward(self, x):
                results = []
                for i in range(4):
                    results.append(x[:x.size(0) - i, i:x.size(2), i:3])
                return tuple(results)

        x = torch.rand(5, 5, 5)
        self.run_model_test(DynamicSliceExportMod(), train=False, input=(x,), batch_size=BATCH_SIZE, use_gpu=False)

    def test_dynamic_slice_to_the_end(self):
        class DynamicSliceExportMod(torch.nn.Module):
            def forward(self, x):
                results = []
                for i in range(4):
                    results.append(x[:, i:, x.size(2) - 5])
                return tuple(results)

        x = torch.rand(5, 5, 5)
        self.run_model_test(DynamicSliceExportMod(), train=False, input=(x,), batch_size=BATCH_SIZE, use_gpu=False)

    def test_tensor_factories(self):
        class TensorFactory(torch.nn.Module):
            def forward(self, x):
                return torch.zeros(x.size()) + torch.ones(x.size())

        x = torch.randn(2, 3, 4)
        self.run_model_test(TensorFactory(), train=False, input=(x,), batch_size=BATCH_SIZE, use_gpu=False)

    def test_where_functional(self):
        class WhereFunctional(torch.nn.Module):
            def forward(self, x):
                return torch.where(x > 2.0, x, torch.neg(x))

        x = torch.randn(3, 4)
        self.run_model_test(WhereFunctional(), train=False, input=(x,), batch_size=BATCH_SIZE, use_gpu=False)

    def test_where_method(self):
        class WhereMethod(torch.nn.Module):
            def forward(self, x):
                return x.where(x > 2.0, torch.neg(x))

        x = torch.randn(3, 4)
        self.run_model_test(WhereMethod(), train=False, input=(x,), batch_size=BATCH_SIZE, use_gpu=False)

    def test_data_dependent_zeros_factory(self):
        class ZerosFactory(torch.nn.Module):
            def forward(self, input):
                return torch.cat([input, torch.zeros(input.size(0), 1).type_as(input)], dim=1)

        x = torch.zeros(3, 4)
        self.run_model_test(ZerosFactory(), train=False, input=(x,), batch_size=BATCH_SIZE, use_gpu=False)

    def test_implicit_expand(self):
        class ImplicitExpandExportMod(torch.nn.Module):
            def forward(self, x):
                return x + 1

        x = torch.randn(3, 4)
        self.run_model_test(ImplicitExpandExportMod(), train=False, input=(x,), batch_size=BATCH_SIZE, use_gpu=False)

    def test_reduce_sum(self):
        class ReduceSumNegativeIndices(torch.nn.Module):
            def forward(self, x):
                return x.sum(-1)

        x = torch.randn(2, 3, 4)
        self.run_model_test(ReduceSumNegativeIndices(), train=False, input=(x,), batch_size=BATCH_SIZE, use_gpu=False)


# a bit of metaprogramming to set up all the rnn tests


def make_test(name, base, layer, bidirectional, initial_state,
              variable_length, dropout,
              **extra_kwargs):
    test_name = str('_'.join([
        'test', name, layer[1],
        bidirectional[1], initial_state[1],
        variable_length[1], dropout[1]
    ]))

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
    setattr(TestCaffe2Backend, f.__name__, f)


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
            make_test(name, base, layer, bidirectional, initial_state,
                      variable_length, dropout,
                      **extra_kwargs)
            test_count += 1

    # sanity check that a representative example does exist
    TestCaffe2Backend.test_gru_trilayer_forward_with_initial_state_without_sequence_lengths_with_dropout

    # make sure no one accidentally disables all the tests without
    # noticing
    assert test_count == 192, test_count
setup_rnn_tests()

# add the same test suite as above, but switch embed_params=False
# to embed_params=True
TestCaffe2BackendEmbed = type(str("TestCaffe2BackendEmbed"),
                              (unittest.TestCase,),
                              dict(TestCaffe2Backend.__dict__, embed_params=True))


if __name__ == '__main__':
    unittest.main()
