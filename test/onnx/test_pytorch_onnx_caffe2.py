# Owner(s): ["module: onnx"]

import io
import itertools
import sys
import unittest
from typing import Tuple

import caffe2.python.onnx.backend as c2

import model_defs.dcgan as dcgan
import model_defs.word_language_model as word_language_model
import numpy as np
import onnx
import torch.onnx
import torch.onnx.operators
import torch.utils.model_zoo as model_zoo
import verify
from caffe2.python.operator_test.torch_integration_test import (
    create_bbox_transform_inputs,
    generate_rois_rotated,
)
from debug_embed_params import run_embed_params
from model_defs.lstm_flattening_result import LstmFlatteningResult
from model_defs.mnist import MNIST
from model_defs.rnn_model_with_packed_sequence import RnnModelWithPackedSequence
from model_defs.squeezenet import SqueezeNet
from model_defs.srresnet import SRResNet
from model_defs.super_resolution import SuperResolutionNet
from pytorch_test_common import (
    BATCH_SIZE,
    RNN_BATCH_SIZE,
    RNN_HIDDEN_SIZE,
    RNN_INPUT_SIZE,
    RNN_SEQUENCE_LENGTH,
    skipIfNoCuda,
    skipIfTravis,
    skipIfUnsupportedMinOpsetVersion,
    skipIfUnsupportedOpsetVersion,
)
from torch import nn
from torch.autograd import function, Variable
from torch.nn.utils import rnn as rnn_utils
from torch.onnx import ExportTypes
from torch.testing._internal import common_utils
from torch.testing._internal.common_utils import skipIfNoLapack

# Import various models for testing
from torchvision.models.alexnet import alexnet
from torchvision.models.densenet import densenet121
from torchvision.models.inception import inception_v3
from torchvision.models.resnet import resnet50
from torchvision.models.vgg import vgg16, vgg16_bn, vgg19, vgg19_bn

skip = unittest.skip


def skipIfEmbed(func):
    def wrapper(self):
        if self.embed_params:
            raise unittest.SkipTest("Skip embed_params verify test")
        return func(self)

    return wrapper


def skipIfNoEmbed(func):
    def wrapper(self):
        if not self.embed_params:
            raise unittest.SkipTest("Skip debug embed_params test")
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
    if isinstance(model, torch.jit.ScriptModule):
        # Special case for common case of passing a single Tensor
        if isinstance(inputs, torch.Tensor):
            inputs = (inputs,)
        out = model(*inputs)
    return f.getvalue(), out


torch.set_default_tensor_type("torch.FloatTensor")
try:
    import torch
except ImportError:
    print("Cannot import torch, hence caffe2-torch test will not run.")
    sys.exit(0)


model_urls = {
    "alexnet": "https://s3.amazonaws.com/download.caffe2.ai/test_data/alexnet-owt-4df8aa71.pth",
    "dcgan_b": "https://s3.amazonaws.com/pytorch/test_data/export/netG_bedroom_epoch_1-0649e76b.pth",
    "dcgan_f": "https://s3.amazonaws.com/pytorch/test_data/export/netG_faces_epoch_49-d86035a6.pth",
    "densenet121": "https://s3.amazonaws.com/download.caffe2.ai/test_data/densenet121-d66d3027.pth",
    "inception_v3_google": "https://s3.amazonaws.com/download.caffe2.ai/test_data/inception_v3_google-1a9a5a14.pth",
    "resnet50": "https://s3.amazonaws.com/download.caffe2.ai/test_data/resnet50-19c8e357.pth",
    "srresNet": "https://s3.amazonaws.com/pytorch/demos/srresnet-e10b2039.pth",
    "super_resolution": "https://s3.amazonaws.com/pytorch/test_data/export/superres_epoch100-44c6958e.pth",
    "squeezenet1_0": "https://s3.amazonaws.com/download.caffe2.ai/test_data/squeezenet1_0-a815701f.pth",
    "squeezenet1_1": "https://s3.amazonaws.com/download.caffe2.ai/test_data/squeezenet1_1-f364aa15.pth",
    "vgg16": "https://s3.amazonaws.com/download.caffe2.ai/test_data/vgg16-397923af.pth",
    "vgg19": "https://s3.amazonaws.com/download.caffe2.ai/test_data/vgg19-dcbb9e9d.pth",
}


class TestCaffe2Backend_opset9(common_utils.TestCase):
    opset_version = 9
    embed_params = False

    def setUp(self):
        # the following should ideally be super().setUp(), https://github.com/pytorch/pytorch/issues/79630
        common_utils.TestCase.setUp(self)
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
        np.random.seed(seed=0)

    def convert_cuda(self, model, input):
        cuda_model = model.cuda()
        # input might be nested - we want to move everything to GPU
        cuda_input = function._nested_map(
            lambda o: isinstance(o, Variable) or isinstance(o, torch.Tensor),
            lambda o: o.cuda(),
        )(input)
        return cuda_model, cuda_input

    def run_debug_test(
        self,
        model,
        train,
        batch_size,
        state_dict=None,
        input=None,
        use_gpu=True,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
    ):
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

        onnxir, torch_out = do_export(
            model,
            input,
            export_params=self.embed_params,
            verbose=False,
            do_constant_folding=False,
            opset_version=self.opset_version,
            keep_initializers_as_inputs=True,
            add_node_names=False,
            operator_export_type=operator_export_type,
        )
        if isinstance(torch_out, torch.autograd.Variable):
            torch_out = (torch_out,)

        caffe2_out = run_embed_params(onnxir, model, input, state_dict, use_gpu)
        for _, (x, y) in enumerate(zip(torch_out, caffe2_out)):
            np.testing.assert_almost_equal(x.data.cpu().numpy(), y, decimal=3)

    def run_actual_test(
        self,
        model,
        train,
        batch_size,
        state_dict=None,
        input=None,
        use_gpu=True,
        rtol=0.001,
        atol=1e-7,
        do_constant_folding=True,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
        input_names=None,
        dynamic_axes=None,
        remained_onnx_input_idx=None,
    ):
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
        verify.verify(
            model,
            input,
            c2,
            rtol=rtol,
            atol=atol,
            do_constant_folding=do_constant_folding,
            opset_version=self.opset_version,
            keep_initializers_as_inputs=True,
            operator_export_type=operator_export_type,
            input_names=input_names,
            dynamic_axes=dynamic_axes,
            remained_onnx_input_idx=remained_onnx_input_idx,
        )

    def run_model_test(
        self,
        model,
        train,
        batch_size,
        state_dict=None,
        input=None,
        use_gpu=True,
        rtol=0.001,
        atol=1e-7,
        do_constant_folding=True,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
        input_names=None,
        dynamic_axes=None,
        remained_onnx_input_idx=None,
    ):
        use_gpu_ = torch.cuda.is_available() and use_gpu
        # NOTE: do_constant_folding is turned on only when model has
        # parameters embedded (which are needed for constant folding),
        # i.e. for self.embed_params=True case. self.embed_params=True
        # for the TestCaffe2BackendEmbed class defined at the bottom.
        if self.embed_params:
            self.run_actual_test(
                model,
                train,
                batch_size,
                state_dict,
                input,
                use_gpu=use_gpu_,
                rtol=rtol,
                atol=atol,
                do_constant_folding=do_constant_folding,
                operator_export_type=operator_export_type,
                input_names=input_names,
                dynamic_axes=dynamic_axes,
                remained_onnx_input_idx=remained_onnx_input_idx,
            )
        else:
            self.run_debug_test(
                model,
                train,
                batch_size,
                state_dict,
                input,
                use_gpu=use_gpu_,
                operator_export_type=operator_export_type,
            )

    def test_linear(self):
        class MyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.many_fc = nn.Sequential(
                    nn.Linear(4, 5, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Linear(5, 6, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Linear(6, 7, bias=True),
                )

            def forward(self, input):
                return self.many_fc(input)

        model = MyModel()
        input = torch.randn(3, 4, requires_grad=True)
        self.run_model_test(model, train=False, batch_size=0, input=input)

    def test_onnx_export_with_parameter_renaming(self):
        class SimpleFcNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(5, 10)

            def forward(self, input):
                return self.fc1(input)

        model = SimpleFcNet()
        input = torch.randn(7, 5)
        output = model(input)

        f = io.BytesIO()
        # Note that the export call explicitly sets the names of not just the input,
        # but also the parameters. This test checks that the model can be loaded and
        # executed in Caffe2 backend correctly.
        torch.onnx._export(
            model,
            input,
            f,
            verbose=True,
            export_type=ExportTypes.ZIP_ARCHIVE,
            input_names=["input1", "parameter1", "parameter2"],
            keep_initializers_as_inputs=True,
        )

        f.seek(0)
        model_c2 = c2.prepare_zip_archive(f)
        result = model_c2.run(input.numpy())
        np.testing.assert_almost_equal(output.data.cpu().numpy(), result[0], decimal=3)

    def test_onnx_export_param_name_duplication(self):
        class SimpleFcNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(5, 10)

            def forward(self, input):
                return self.fc1(input)

        model = SimpleFcNet()
        input = torch.randn(7, 5)
        output = model(input)

        f = io.BytesIO()
        # The export call explicitly sets the names of the input, and the first parameter.
        # But note that the target first parameter name is the same as the second parameter name.
        # This test checks that given this edge condition, the model can be loaded and executed
        # in Caffe2 backend correctly.
        torch.onnx._export(
            model,
            input,
            f,
            verbose=True,
            export_type=ExportTypes.ZIP_ARCHIVE,
            input_names=["input1", "fc1.bias"],
            keep_initializers_as_inputs=True,
        )

        f.seek(0)
        model_c2 = c2.prepare_zip_archive(f)
        result = model_c2.run(input.numpy())
        np.testing.assert_almost_equal(output.data.cpu().numpy(), result[0], decimal=3)

    def test_lstm_cell(self):
        model = nn.LSTMCell(RNN_INPUT_SIZE, RNN_HIDDEN_SIZE)
        input = torch.randn(BATCH_SIZE, RNN_INPUT_SIZE)
        h0 = torch.randn(BATCH_SIZE, RNN_HIDDEN_SIZE)
        c0 = torch.randn(BATCH_SIZE, RNN_HIDDEN_SIZE)
        self.run_model_test(
            model,
            train=False,
            batch_size=BATCH_SIZE,
            input=(input, (h0, c0)),
            use_gpu=False,
        )

    def test_gru_cell(self):
        model = nn.GRUCell(RNN_INPUT_SIZE, RNN_HIDDEN_SIZE)
        input = torch.randn(BATCH_SIZE, RNN_INPUT_SIZE)
        h0 = torch.randn(BATCH_SIZE, RNN_HIDDEN_SIZE)
        self.run_model_test(
            model, train=False, batch_size=BATCH_SIZE, input=(input, h0), use_gpu=False
        )

    def _dispatch_rnn_test(self, name, *args, **kwargs):
        if name == "elman":
            self._elman_rnn_test(*args, **kwargs)
        if name == "lstm":
            self._lstm_test(*args, **kwargs)
        if name == "gru":
            self._gru_test(*args, **kwargs)

    def _elman_rnn_test(
        self,
        layers,
        nonlinearity,
        bidirectional,
        initial_state,
        packed_sequence,
        dropout,
    ):
        batch_first = True if packed_sequence == 2 else False
        model = nn.RNN(
            RNN_INPUT_SIZE,
            RNN_HIDDEN_SIZE,
            layers,
            nonlinearity=nonlinearity,
            bidirectional=bidirectional,
            dropout=dropout,
            batch_first=batch_first,
        )

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
        self.run_model_test(
            model,
            train=False,
            batch_size=RNN_BATCH_SIZE,
            input=input,
            use_gpu=False,
            atol=1e-7,
        )

        # test that the model still runs with a different batch size
        # (save the model with a batch_size of 1 with rnn with a variable batch size,
        # otherwise expand will fail)
        variable_batch_size_init_input = make_input(1)
        # Constant folding works when model has parameters embedded. For this case, we need to disable it
        onnxir, _ = do_export(
            model,
            variable_batch_size_init_input,
            keep_initializers_as_inputs=True,
            do_constant_folding=False,
        )
        other_input = make_input(RNN_BATCH_SIZE + 1)
        _ = run_embed_params(onnxir, model, other_input, use_gpu=False)

    def _lstm_test(
        self, layers, bidirectional, initial_state, packed_sequence, dropout
    ):
        batch_first = True if packed_sequence == 2 else False
        model = LstmFlatteningResult(
            RNN_INPUT_SIZE,
            RNN_HIDDEN_SIZE,
            layers,
            bidirectional=bidirectional,
            dropout=dropout,
            batch_first=batch_first,
        )
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
        self.run_model_test(
            model, train=False, batch_size=RNN_BATCH_SIZE, input=input, use_gpu=False
        )

        # test that the model still runs with a different batch size
        # (save the model with a batch_size of 1 with rnn with a variable batch size,
        # otherwise expand will fail)
        variable_batch_size_init_input = make_input(1)
        # Constant folding works when model has parameters embedded. For this case, we need to disable it
        onnxir, _ = do_export(
            model,
            variable_batch_size_init_input,
            keep_initializers_as_inputs=True,
            do_constant_folding=False,
        )
        other_input = make_input(RNN_BATCH_SIZE + 1)
        _ = run_embed_params(onnxir, model, other_input, use_gpu=False)

    def _gru_test(self, layers, bidirectional, initial_state, packed_sequence, dropout):
        batch_first = True if packed_sequence == 2 else False
        model = nn.GRU(
            RNN_INPUT_SIZE,
            RNN_HIDDEN_SIZE,
            layers,
            bidirectional=bidirectional,
            dropout=dropout,
            batch_first=batch_first,
        )
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
        self.run_model_test(
            model, train=False, batch_size=RNN_BATCH_SIZE, input=input, use_gpu=False
        )

        # test that the model still runs with a different batch size
        # (save the model with a batch_size of 1 with rnn with a variable batch size,
        # otherwise expand will fail)
        variable_batch_size_init_input = make_input(1)
        # Constant folding works when model has parameters embedded. For this case, we need to disable it
        onnxir, _ = do_export(
            model,
            variable_batch_size_init_input,
            keep_initializers_as_inputs=True,
            do_constant_folding=False,
        )
        other_input = make_input(RNN_BATCH_SIZE + 1)
        _ = run_embed_params(onnxir, model, other_input, use_gpu=False)

    @unittest.skip("Disabled due to onnx optimizer deprecation")
    def test_rnn_init_predict_split(self):
        model = nn.LSTM(RNN_INPUT_SIZE, RNN_HIDDEN_SIZE, 3, bidirectional=True)
        seq_lengths = np.random.randint(1, RNN_SEQUENCE_LENGTH + 1, size=7)
        seq_lengths = list(reversed(sorted(map(int, seq_lengths))))
        input = [torch.randn(l, RNN_INPUT_SIZE) for l in seq_lengths]
        input = rnn_utils.pad_sequence(input)

        # Test that we are correctly splitting between init and
        # predict net. When we embed parameters, there should be more
        # ops in the init net.
        mp = onnx.ModelProto.FromString(
            do_export(
                model,
                input,
                export_params=self.embed_params,
                keep_initializers_as_inputs=True,
                do_constant_folding=False,
            )[0]
        )
        prepared = c2.prepare(mp, device="CPU")
        if self.embed_params:
            assert len(prepared.init_net.op) == 950
            assert len(prepared.predict_net.op) == 101
        else:
            assert len(prepared.init_net.op) == 83
            assert len(prepared.predict_net.op) == 968

    def test_alexnet(self):
        state_dict = model_zoo.load_url(model_urls["alexnet"], progress=False)
        self.run_model_test(
            alexnet(),
            train=False,
            batch_size=BATCH_SIZE,
            state_dict=state_dict,
            atol=1e-3,
        )

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
        self.run_model_test(netD, train=False, batch_size=BATCH_SIZE, input=input)

        netG = dcgan._netG(1)
        netG.apply(dcgan.weights_init)
        state_dict = model_zoo.load_url(model_urls["dcgan_b"], progress=False)
        # state_dict = model_zoo.load_url(model_urls["dcgan_f"], progress=False)
        noise = torch.randn(BATCH_SIZE, dcgan.nz, 1, 1).normal_(0, 1)
        self.run_model_test(
            netG,
            train=False,
            batch_size=BATCH_SIZE,
            input=noise,
            state_dict=state_dict,
            rtol=1e-2,
            atol=1e-6,
        )

    @unittest.skipIf(
        not torch.cuda.is_available(), "model on net has cuda in it, awaiting fix"
    )
    def test_densenet(self):
        state_dict = model_zoo.load_url(model_urls["densenet121"], progress=False)
        self.run_model_test(
            densenet121(),
            train=False,
            batch_size=BATCH_SIZE,
            state_dict=state_dict,
            atol=1e-7,
        )

    @skip("doesn't match exactly...")
    # TODO: figure out the numerical instabilities
    def test_inception(self):
        x = torch.randn(BATCH_SIZE, 3, 299, 299, requires_grad=True)
        # state_dict = model_zoo.load_url(model_urls["inception_v3_google"], progress=False)
        state_dict = None
        self.run_model_test(
            inception_v3(),
            train=False,
            batch_size=BATCH_SIZE,
            state_dict=state_dict,
            input=x,
        )

    @skipIfNoEmbed
    def test_resnet(self):
        state_dict = model_zoo.load_url(model_urls["resnet50"], progress=False)
        self.run_model_test(
            resnet50(),
            train=False,
            batch_size=BATCH_SIZE,
            state_dict=state_dict,
            atol=1e-5,
        )

    def test_squeezenet(self):
        sqnet_v1_1 = SqueezeNet(version=1.1)
        state_dict = model_zoo.load_url(model_urls["squeezenet1_1"], progress=False)
        # state_dict = model_zoo.load_url(model_urls["squeezenet1_0"], progress=False)
        self.run_model_test(
            sqnet_v1_1, train=False, batch_size=BATCH_SIZE, state_dict=state_dict
        )

    # @skip("takes long to run, LAPACK needed for gpu")
    @skipIfNoLapack
    @unittest.skip("This model takes too much memory")
    def test_srresnet(self):
        super_resolution_net = SRResNet(rescale_factor=4, n_filters=64, n_blocks=8)
        state_dict = model_zoo.load_url(model_urls["srresNet"], progress=False)
        x = torch.randn(1, 3, 224, 224, requires_grad=True)
        self.run_model_test(
            super_resolution_net,
            train=False,
            batch_size=1,
            state_dict=state_dict,
            input=x,
            use_gpu=False,
        )

    @skipIfTravis
    @skipIfNoLapack
    @skipIfNoCuda
    def test_super_resolution(self):
        super_resolution_net = SuperResolutionNet(upscale_factor=3)
        state_dict = model_zoo.load_url(model_urls["super_resolution"], progress=False)
        x = torch.randn(1, 1, 224, 224, requires_grad=True)
        self.run_model_test(
            super_resolution_net,
            train=False,
            batch_size=BATCH_SIZE,
            state_dict=state_dict,
            input=x,
            use_gpu=False,
            atol=1e-6,
        )

    @unittest.skip("This model takes too much memory")
    def test_vgg16(self):
        state_dict = model_zoo.load_url(model_urls["vgg16"], progress=False)
        self.run_model_test(
            vgg16(), train=False, batch_size=BATCH_SIZE, state_dict=state_dict
        )

    @skip("disable to run tests faster...")
    def test_vgg16_bn(self):
        self.run_model_test(vgg16_bn(), train=False, batch_size=BATCH_SIZE)

    @skip("disable to run tests faster...")
    def test_vgg19(self):
        state_dict = model_zoo.load_url(model_urls["vgg19"], progress=False)
        self.run_model_test(
            vgg19(), train=False, batch_size=BATCH_SIZE, state_dict=state_dict
        )

    @skip("disable to run tests faster...")
    def test_vgg19_bn(self):
        self.run_model_test(vgg19_bn(), train=False, batch_size=BATCH_SIZE)

    def run_word_language_model(self, model_name):
        ntokens = 50
        emsize = 5
        nhid = 5
        nlayers = 5
        dropout = 0.2
        tied = False
        batchsize = 5
        model = word_language_model.RNNModel(
            model_name, ntokens, emsize, nhid, nlayers, dropout, tied, batchsize
        )
        x = torch.arange(0, ntokens).long().view(-1, batchsize)
        # Only support CPU version, since tracer is not working in GPU RNN.
        self.run_model_test(
            model,
            train=False,
            input=(x, model.hidden),
            batch_size=batchsize,
            use_gpu=False,
        )

    @unittest.skip("Disabled due to onnx optimizer deprecation")
    @skipIfUnsupportedOpsetVersion([10])
    def test_word_language_model_RNN_TANH(self):
        self.run_word_language_model("RNN_TANH")

    @unittest.skip("Disabled due to onnx optimizer deprecation")
    @skipIfUnsupportedOpsetVersion([10])
    def test_word_language_model_RNN_RELU(self):
        self.run_word_language_model("RNN_RELU")

    @unittest.skip("Disabled due to onnx optimizer deprecation")
    @skipIfUnsupportedOpsetVersion([10])
    def test_word_language_model_LSTM(self):
        self.run_word_language_model("LSTM")

    @unittest.skip("Disabled due to onnx optimizer deprecation")
    @skipIfUnsupportedOpsetVersion([10])
    def test_word_language_model_GRU(self):
        self.run_word_language_model("GRU")

    def test_batchnorm1d_special(self):
        c = torch.randn(BATCH_SIZE, 224)
        model = nn.BatchNorm1d(224)
        self.run_model_test(model, train=True, input=c, batch_size=BATCH_SIZE)

    def test_batchnorm1d(self):
        c = torch.randn(BATCH_SIZE, 224, 224)
        model = nn.BatchNorm1d(224)
        self.run_model_test(model, train=True, input=c, batch_size=BATCH_SIZE)

    def test_batchnorm1d_noaffine(self):
        c = torch.randn(BATCH_SIZE, 224)
        model = nn.BatchNorm1d(224, affine=False)
        self.run_model_test(model, train=False, input=c, batch_size=BATCH_SIZE)

    def test_batchnorm2d_noaffine(self):
        c = torch.randn(128, 128, 1, 1)
        model = nn.BatchNorm2d(128, affine=False)
        self.run_model_test(model, train=False, input=c, batch_size=BATCH_SIZE)

    def test_batchnorm3d_noaffine(self):
        c = torch.randn(128, 128, 1, 1, 1)
        model = nn.BatchNorm3d(128, affine=False)
        self.run_model_test(model, train=False, input=c, batch_size=BATCH_SIZE)

    def test_constant(self):
        c = torch.randn(BATCH_SIZE, 3, 224, 224)

        class MyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input):
                return input + c.type_as(input)

        self.run_model_test(MyModel(), train=False, batch_size=BATCH_SIZE)

    def test_consumed_bn(self):
        underlying = nn.BatchNorm2d(3)
        self.run_model_test(underlying, train=True, batch_size=BATCH_SIZE)

    def _test_index_generic(self, fn):
        class MyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input):
                return fn(input)

        m1 = torch.randn(3, 4, 5, 6, 7)
        self.run_model_test(MyModel(), input=m1, train=False, batch_size=BATCH_SIZE)

    def test_index_1d(self):
        self._test_index_generic(lambda input: input[0])

    @skipIfUnsupportedOpsetVersion([10])
    def test_index_2d_1dimslice(self):
        self._test_index_generic(lambda input: input[0:1, :])

    @skipIfUnsupportedOpsetVersion([10])
    def test_index_2d_sliceint(self):
        self._test_index_generic(lambda input: input[1, :])

    @skipIfUnsupportedOpsetVersion([10])
    def test_index_2d_neg_slice(self):
        self._test_index_generic(lambda input: input[0:-1, :])

    @skipIfUnsupportedOpsetVersion([10])
    def test_index_2d_2dimslice(self):
        self._test_index_generic(lambda input: input[0:1, 0:1])

    @skipIfUnsupportedOpsetVersion([10])
    def test_index_2d_neg_slice2dim(self):
        self._test_index_generic(lambda input: input[0:-1, 0:-1])

    def test_tensor_index_1d(self):
        self._test_index_generic(lambda input: input[torch.tensor([0, 2])])

    def test_tensor_index_2d_1dconstant(self):
        self._test_index_generic(lambda input: input[1, torch.tensor([0, 2])])

    @skipIfUnsupportedOpsetVersion([10])
    def test_tensor_index_2d_1dslice(self):
        self._test_index_generic(lambda input: input[torch.tensor([0, 2]), 0:1])

    @skipIfUnsupportedOpsetVersion([10])
    def test_tensor_index_2d_1dslice_first(self):
        self._test_index_generic(lambda input: input[1:3, torch.tensor([0, 2])])

    def test_tensor_index_newaxis(self):
        self._test_index_generic(lambda input: input[None, torch.tensor([0, 2])])

    def test_tensor_index_advanced_indexing(self):
        self._test_index_generic(
            lambda input: input[
                :,
                torch.tensor([[0, 2], [1, 1]]),
                :,
                torch.tensor([2, 1]),
                torch.tensor([0, 3]),
            ]
        )

    @skipIfUnsupportedOpsetVersion([10])
    def test_tensor_index_advanced_indexing_with_slice(self):
        self._test_index_generic(
            lambda input: input[
                :, torch.tensor([0, 2]), None, 2:4, torch.tensor([[1, 3], [4, 0]])
            ]
        )
        self._test_index_generic(
            lambda input: input[
                :,
                torch.tensor([0, 2]),
                torch.tensor([1]),
                2:4,
                torch.tensor([[1], [4]]),
            ]
        )

    def test_tensor_index_advanced_indexing_consecutive(self):
        self._test_index_generic(
            lambda input: input[
                :, torch.tensor([0, 2]), torch.tensor([[1, 3], [4, 0]]), None
            ]
        )

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_tensor_index_advanced_indexing_masked(self):
        self._test_index_generic(
            lambda input: input[
                :,
                torch.tensor([1, 0, 1, 0], dtype=torch.uint8),
                torch.tensor([[1, 3], [4, 0]]),
                None,
            ]
        )

    def test_chunk(self):
        class MyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input):
                # TODO: Why index? This returns a tuple and test runner doesn't
                # support tuple comparison.
                return input.chunk(8, dim=2)[-1]

        self.run_model_test(MyModel(), train=False, batch_size=BATCH_SIZE)

    def test_sqrt(self):
        class MyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input):
                return input.sqrt()

        input = torch.empty(BATCH_SIZE, 10, 10).uniform_(4, 9)
        self.run_model_test(MyModel(), train=False, input=input, batch_size=BATCH_SIZE)

    def test_rsqrt(self):
        class MyModel(torch.nn.Module):
            def forward(self, input):
                return input.rsqrt()

        input = torch.randn(4, 2, 3, requires_grad=True)
        self.run_model_test(MyModel(), train=False, input=input, batch_size=BATCH_SIZE)

    def test_log(self):
        class MyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input):
                return input.log()

        input = torch.empty(BATCH_SIZE, 10, 10).uniform_(4, 9)
        self.run_model_test(MyModel(), train=False, input=input, batch_size=BATCH_SIZE)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_erf(self):
        class MyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input):
                return input.erf()

        input = torch.empty(BATCH_SIZE, 10, 10).uniform_(4, 9)
        self.run_model_test(MyModel(), train=False, input=input, batch_size=BATCH_SIZE)

    def test_trigonometry(self):
        def test_func(name):
            class MyModel(torch.nn.Module):
                def __init__(self):
                    super().__init__()

                def forward(self, input):
                    return getattr(input, name)()

            input = torch.empty(BATCH_SIZE, 10, 10).uniform_()
            self.run_model_test(
                MyModel(), train=False, input=input, batch_size=BATCH_SIZE
            )

        test_func("cos")
        test_func("sin")
        test_func("tan")
        test_func("acos")
        test_func("asin")
        test_func("atan")

    def test_addconstant(self):
        class MyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input):
                # TODO: Why index? This returns a tuple and test runner doesn't
                # support tuple comparison.
                return input + 1

        self.run_model_test(MyModel(), train=False, batch_size=BATCH_SIZE)

    def test_subconstant(self):
        class MyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input):
                # TODO: Why index? This returns a tuple and test runner doesn't
                # support tuple comparison.
                return input - 1

        self.run_model_test(MyModel(), train=False, batch_size=BATCH_SIZE)

    def test_arithmetic(self):
        class ArithmeticModule(torch.nn.Module):
            def forward(self, x):
                x = x + 2
                x = x - 4
                x = x * 6
                x = x / 8
                return x

        x = torch.randn(2, 3, 4)
        self.run_model_test(
            ArithmeticModule(), input=x, train=False, batch_size=BATCH_SIZE
        )

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

    @skipIfUnsupportedOpsetVersion([10])
    def test_maxpool1d_ceil(self):
        model = nn.MaxPool1d(3, 2, ceil_mode=True)
        x = torch.randn(20, 16, 50, requires_grad=True)
        self.run_model_test(model, train=False, input=x, batch_size=BATCH_SIZE)

    @skipIfUnsupportedOpsetVersion([10])
    def test_maxpool2d_ceil(self):
        model = nn.MaxPool2d(3, 2, ceil_mode=True)
        x = torch.randn(20, 16, 50, 32, requires_grad=True)
        self.run_model_test(model, train=False, input=x, batch_size=BATCH_SIZE)

    @skipIfUnsupportedOpsetVersion([10])
    def test_maxpool3d_ceil(self):
        model = nn.MaxPool3d(3, 2, ceil_mode=True)
        x = torch.randn(20, 16, 50, 44, 31, requires_grad=True)
        self.run_model_test(model, train=False, input=x, batch_size=BATCH_SIZE)

    @unittest.skip("C2 and PyTorch have small difference in padding implementation")
    def test_avgpool2d(self):
        model = nn.AvgPool2d(5, padding=(2))
        self.run_model_test(model, train=False, batch_size=BATCH_SIZE)

    def test_avgpool2d_with_count_include_pad_set_false(self):
        model = nn.AvgPool2d(7, padding=(2), count_include_pad=False)
        self.run_model_test(model, train=False, batch_size=BATCH_SIZE)

    def test_avgpool2d_with_count_include_pad_set_true(self):
        model = nn.AvgPool2d(7, padding=(2), count_include_pad=True)
        self.run_model_test(model, train=False, batch_size=BATCH_SIZE)

    def test_avgpool2d_no_padding(self):
        model = nn.AvgPool2d(5)
        self.run_model_test(model, train=False, batch_size=BATCH_SIZE)

    @unittest.skip("Disabled due to onnx optimizer deprecation")
    @skipIfUnsupportedOpsetVersion([10])
    def test_avg_pool1D_ceil(self):
        model = torch.nn.AvgPool1d(3, 2, ceil_mode=True)
        x = torch.randn(1, 1, 7, requires_grad=True)
        self.run_model_test(model, train=False, input=x, batch_size=BATCH_SIZE)

    @skipIfUnsupportedOpsetVersion([10])
    def test_avg_pool2D_ceil(self):
        model = torch.nn.AvgPool2d(3, 2, ceil_mode=True)
        x = torch.randn(20, 16, 50, 32, requires_grad=True)
        self.run_model_test(model, train=False, input=x, batch_size=BATCH_SIZE)

    @unittest.skip("Disabled due to onnx optimizer deprecation")
    @skipIfUnsupportedOpsetVersion([10])
    def test_avg_pool3D_ceil(self):
        model = torch.nn.AvgPool3d(3, 2, ceil_mode=True)
        x = torch.randn(20, 16, 50, 44, 31, requires_grad=True)
        self.run_model_test(model, train=False, input=x, batch_size=BATCH_SIZE)

    def test_adaptive_avg_pool1D(self):
        model = torch.nn.AdaptiveAvgPool1d(5)
        x = torch.randn(20, 16, 50, requires_grad=True)
        self.run_model_test(model, train=False, input=x, batch_size=BATCH_SIZE)

    def test_adaptive_avg_pool2D(self):
        model = torch.nn.AdaptiveAvgPool2d((5, 4))
        x = torch.randn(20, 16, 50, 32, requires_grad=True)
        self.run_model_test(model, train=False, input=x, batch_size=BATCH_SIZE)

    def test_adaptive_avg_pool3D(self):
        model = torch.nn.AdaptiveAvgPool3d((5, 4, 3))
        x = torch.randn(20, 16, 50, 44, 30, requires_grad=True)
        self.run_model_test(model, train=False, input=x, batch_size=BATCH_SIZE)

    @skipIfUnsupportedMinOpsetVersion(8)
    def test_adaptive_max_pool1D(self):
        model = torch.nn.AdaptiveMaxPool1d(5)
        x = torch.randn(20, 16, 50, requires_grad=True)
        self.run_model_test(model, train=False, input=x, batch_size=BATCH_SIZE)

    @skipIfUnsupportedMinOpsetVersion(8)
    def test_adaptive_max_pool2D(self):
        model = torch.nn.AdaptiveMaxPool2d((5, 4))
        x = torch.randn(20, 16, 50, 32, requires_grad=True)
        self.run_model_test(model, train=False, input=x, batch_size=BATCH_SIZE)

    @skipIfUnsupportedMinOpsetVersion(8)
    def test_adaptive_max_pool3D(self):
        model = torch.nn.AdaptiveMaxPool3d((5, 4, 3))
        x = torch.randn(20, 16, 50, 44, 30, requires_grad=True)
        self.run_model_test(model, train=False, input=x, batch_size=BATCH_SIZE)

    def test_weight_norm(self):
        model = nn.utils.weight_norm(nn.Conv1d(1, 1, 3))
        input = torch.randn(1, 1, 5, requires_grad=True)
        self.run_model_test(model, train=True, batch_size=0, input=input, use_gpu=False)

    def test_mnist(self):
        model = MNIST()
        input = torch.randn(BATCH_SIZE, 1, 28, 28)
        state_dict = None
        # TODO: test with state_dict
        self.run_model_test(
            model,
            train=False,
            input=input,
            batch_size=BATCH_SIZE,
            state_dict=state_dict,
        )

    def test_mm(self):
        class MyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, m1, m2):
                return torch.mm(m1, m2)

        m1 = torch.randn(3, 4)
        m2 = torch.randn(4, 5)
        self.run_model_test(
            MyModel(), train=False, input=(m1, m2), batch_size=BATCH_SIZE, use_gpu=False
        )

    def test_addmm(self):
        class MyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, ma, m1, m2):
                return torch.addmm(ma, m1, m2)

        ma = torch.randn(5)
        m1 = torch.randn(3, 4)
        m2 = torch.randn(4, 5)
        self.run_model_test(
            MyModel(),
            train=False,
            input=(ma, m1, m2),
            batch_size=BATCH_SIZE,
            use_gpu=False,
        )

    def test_fuse_addmm(self):
        class AddmmModel(torch.nn.Module):
            def forward(self, x):
                return torch.mm(x, x) + x

        x = torch.randn(3, 3)
        self.run_model_test(
            AddmmModel(), train=False, input=x, batch_size=BATCH_SIZE, use_gpu=False
        )

    def test_scalar_type(self):
        class ArithmeticModel(torch.nn.Module):
            def forward(self, x):
                return x.size(0) * 2 * x

        x = torch.ones(2, 3, dtype=torch.float32)
        self.run_model_test(
            ArithmeticModel(), input=x, train=False, batch_size=BATCH_SIZE
        )

        class ReciprocalModel(torch.nn.Module):
            def forward(self, x):
                return torch.reciprocal(x)

        x = torch.tensor([2.0, 4.0], dtype=torch.double)
        self.run_model_test(
            ReciprocalModel(), input=x, train=False, batch_size=BATCH_SIZE
        )

        class ComparisonModel(torch.nn.Module):
            def forward(self, x, y):
                return x.ge(0.5) & y.le(2)

        x = torch.ones(2, 3, dtype=torch.int32)
        y = torch.ones(2, 3, dtype=torch.float32)
        self.run_model_test(
            ComparisonModel(), input=(x, y), train=False, batch_size=BATCH_SIZE
        )

        class MatMulModel(torch.nn.Module):
            def forward(self, x, y):
                return torch.mm(x, y)

        x = torch.ones(3, 4)
        y = torch.ones(4, 5)
        self.run_model_test(
            MatMulModel(), input=(x, y), train=False, batch_size=BATCH_SIZE
        )

        class AddMMModel(torch.nn.Module):
            def forward(self, x):
                return torch.mm(x, x) + x

        x = torch.ones(3, 3)
        self.run_model_test(AddMMModel(), input=x, train=False, batch_size=BATCH_SIZE)

    # test for a pytorch optimization pass, see https://github.com/pytorch/pytorch/pull/7872
    def test_consecutive_transposes(self):
        class MyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x.transpose(1, 2).transpose(2, 3)

        x = torch.randn(5, 6, 7, 8)
        self.run_model_test(
            MyModel(), train=False, input=x, batch_size=BATCH_SIZE, use_gpu=False
        )

    def test_sum(self):
        shape = (3, 4, 5)
        for params in [{}] + [{"dim": i} for i in range(len(shape))]:

            class MyModel(torch.nn.Module):
                def __init__(self):
                    super().__init__()

                def forward(self, x):
                    return torch.sum(x, **params)

            x = torch.randn(*shape)
            self.run_model_test(
                MyModel(), train=False, input=(x), batch_size=BATCH_SIZE, use_gpu=False
            )

    def test_cumsum(self):
        shape = (3, 4, 5)
        for params in [{"dim": i} for i in range(len(shape))]:

            class MyModel(torch.nn.Module):
                def __init__(self):
                    super().__init__()

                def forward(self, x):
                    return torch.cumsum(x, **params)

            x = torch.randn(*shape)
            self.run_model_test(
                MyModel(),
                train=False,
                input=(x),
                batch_size=BATCH_SIZE,
                use_gpu=False,
                operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
            )

    def test_cosine_similarity(self):
        shape = (100, 128)
        x = torch.randn(*shape)
        y = torch.randn(*shape)
        self.run_model_test(
            torch.nn.CosineSimilarity(dim=1, eps=1e-6),
            train=False,
            input=(x, y),
            batch_size=BATCH_SIZE,
            use_gpu=False,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
        )

    @unittest.skip("Disabled due to onnx optimizer deprecation")
    @skipIfUnsupportedOpsetVersion([10])
    def test_lstm_constant_folding(self):
        class LstmNet(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, bidirectional):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size, hidden_size, num_layers, bidirectional=bidirectional
                )

            def forward(self, input, initial_state):
                return self.lstm(input, initial_state)

        def get_LstmNet_model_and_inputs(
            input_size, hidden_size, num_layers, batch_size, seq_len, bidirectional
        ):
            num_directions = 2 if bidirectional else 1
            model = LstmNet(input_size, hidden_size, num_layers, bidirectional)
            input = torch.randn(seq_len, batch_size, input_size)
            h0 = torch.randn(num_layers * num_directions, batch_size, hidden_size)
            c0 = torch.randn(num_layers * num_directions, batch_size, hidden_size)
            return model, (input, (h0, c0))

        batch_size1 = 3
        model1, input1 = get_LstmNet_model_and_inputs(7, 3, 2, batch_size1, 5, True)
        self.run_actual_test(
            model1,
            train=False,
            batch_size=batch_size1,
            input=input1,
            use_gpu=False,
            do_constant_folding=True,
        )

        batch_size2 = 4
        model2, input2 = get_LstmNet_model_and_inputs(5, 4, 3, batch_size2, 7, False)
        self.run_actual_test(
            model2,
            train=False,
            batch_size=batch_size2,
            input=input2,
            use_gpu=False,
            do_constant_folding=True,
        )

    @unittest.skip("Disabled due to onnx optimizer deprecation")
    @skipIfUnsupportedOpsetVersion([10])
    def test_gru_constant_folding(self):
        class GruNet(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, bidirectional):
                super().__init__()
                self.mygru = nn.GRU(
                    input_size, hidden_size, num_layers, bidirectional=bidirectional
                )

            def forward(self, input, initial_state):
                out = self.mygru(input, initial_state)
                return out

        def get_GruNet_model_and_inputs(
            input_size, hidden_size, num_layers, batch_size, seq_len, bidirectional
        ):
            num_directions = 2 if bidirectional else 1
            model = GruNet(input_size, hidden_size, num_layers, bidirectional)
            input = torch.randn(seq_len, batch_size, input_size)
            h0 = torch.randn(num_layers * num_directions, batch_size, hidden_size)
            return model, (input, h0)

        batch_size1 = 3
        model1, input1 = get_GruNet_model_and_inputs(7, 3, 2, batch_size1, 5, True)
        self.run_actual_test(
            model1,
            train=False,
            batch_size=batch_size1,
            input=input1,
            use_gpu=False,
            do_constant_folding=True,
        )

        batch_size2 = 4
        model2, input2 = get_GruNet_model_and_inputs(5, 4, 3, batch_size2, 7, False)
        self.run_actual_test(
            model2,
            train=False,
            batch_size=batch_size2,
            input=input2,
            use_gpu=False,
            do_constant_folding=True,
        )

    def test_repeat(self):
        class MyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x.repeat(1, 2, 3, 4)

        x = torch.randn(4, 3, 2, 1, requires_grad=True)
        self.run_model_test(
            MyModel(), train=False, input=(x), batch_size=BATCH_SIZE, use_gpu=False
        )

    @skipIfUnsupportedOpsetVersion([10])
    def test_upsample(self):
        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        model = nn.Upsample(size=[v * 2 for v in x.size()[2:]], mode="nearest")
        self.run_model_test(
            model, train=False, input=(x), batch_size=BATCH_SIZE, use_gpu=False
        )

    @skipIfUnsupportedOpsetVersion([10])
    def test_interpolate_upsample(self):
        class MyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                size = [v * 2 for v in x.size()[2:]]
                # work around for now: turn the dynamic sizes into constant
                size = [int(i) for i in size]
                return nn.functional.interpolate(x, size=size, mode="nearest")

        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        model = MyModel()
        self.run_model_test(
            model, train=False, input=(x), batch_size=BATCH_SIZE, use_gpu=False
        )

    @skipIfUnsupportedOpsetVersion([7, 8, 10])
    def test_interpolate_upsample_dynamic_sizes(self):
        class MyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                size = [v * 2 for v in x.size()[2:]]
                return nn.functional.interpolate(x, size=size, mode="nearest")

        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        model = MyModel()
        self.run_model_test(
            model, train=False, input=(x), batch_size=BATCH_SIZE, use_gpu=False
        )

    def test_repeat_dim_overflow(self):
        class MyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x.repeat(1, 2, 3, 4)

        x = torch.randn(1, 2, requires_grad=True)
        self.run_model_test(
            MyModel(), train=False, input=(x), batch_size=BATCH_SIZE, use_gpu=False
        )

    def test_repeat_dynamic(self):
        class MyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                return x.repeat(y.size()[0] // 2, y.size()[1] * 2)

        x = torch.randn(1, 2, requires_grad=True)
        y = torch.randn(2, 4, requires_grad=True)
        self.run_model_test(
            MyModel(),
            train=False,
            input=(x, y),
            batch_size=BATCH_SIZE,
            use_gpu=False,
            input_names=["x", "y"],
            dynamic_axes={"x": [0, 1], "y": [0, 1]},
        )
        self.run_model_test(
            MyModel(),
            train=False,
            input=(x, y),
            batch_size=BATCH_SIZE,
            use_gpu=False,
            remained_onnx_input_idx=[0],
        )

    def test_mean(self):
        shape = (3, 4, 5)
        for params in [{}] + [{"dim": i} for i in range(len(shape))]:

            class MyModel(torch.nn.Module):
                def __init__(self):
                    super().__init__()

                def forward(self, x):
                    return torch.mean(x, **params)

            x = torch.randn(*shape)
            self.run_model_test(
                MyModel(), train=False, input=(x), batch_size=BATCH_SIZE, use_gpu=False
            )

    # TODO: Add test cases for prod once Caffe2 has support for ReduceProd
    def test_softmax(self):
        for i in range(2, 8):
            for d in range(0, i - 1):
                model = nn.Softmax(dim=d)
                dims = [2] * (i - 2) + [3, 4]
                input = torch.ones(*dims, requires_grad=True)
                self.run_model_test(
                    model, train=False, batch_size=BATCH_SIZE, input=input
                )

    def test_softmax_dtype(self):
        class SoftmaxModel(torch.nn.Module):
            def forward(self, input):
                return nn.functional.softmax(input, dim=0, dtype=torch.float64)

        x = torch.randn(1, 2, 3, requires_grad=True, dtype=torch.float32)
        self.run_model_test(SoftmaxModel(), train=False, input=x, batch_size=BATCH_SIZE)

    def test_logsoftmax(self):
        for i in range(7)[2:]:
            model = nn.LogSoftmax(dim=i - 1)
            dims = [2] * (i - 2) + [3, 4]
            input = torch.ones(*dims, requires_grad=True)
            self.run_model_test(model, train=False, batch_size=BATCH_SIZE, input=input)

    def test_logsoftmax_dim(self):
        for i in range(-4, 3):
            model = nn.LogSoftmax(dim=i)
            input = torch.randn(3, 4, 5, 6)
            self.run_model_test(model, train=False, batch_size=BATCH_SIZE, input=input)

    def test_randn(self):
        x = torch.randn(1, 2, 3, 4)

        class MyModule(torch.nn.Module):
            def forward(self, x):
                return (torch.randn(1, 2, 3, 4) + x).shape

        self.run_model_test(
            MyModule(),
            train=False,
            input=(x),
            batch_size=BATCH_SIZE,
            use_gpu=False,
            remained_onnx_input_idx=[],
        )

    def test_rand(self):
        x = torch.randn(1, 2, 3, 4)

        class MyModule(torch.nn.Module):
            def forward(self, x):
                return (torch.rand(1, 2, 3, 4) + x).shape

        self.run_model_test(
            MyModule(),
            train=False,
            input=(x),
            batch_size=BATCH_SIZE,
            use_gpu=False,
            remained_onnx_input_idx=[],
        )

    def test_convtranspose(self):
        model = nn.ConvTranspose2d(
            3, 3, 3, stride=3, bias=False, padding=1, output_padding=2
        )
        self.run_model_test(model, train=False, batch_size=BATCH_SIZE, atol=1e-7)

    def test_unsqueeze(self):
        shape = (3, 4, 5)
        # test negative dim as well.
        for dim in range(-len(shape) - 1, len(shape) + 1):

            class MyModel(torch.nn.Module):
                def __init__(self):
                    super().__init__()

                def forward(self, x):
                    return x.unsqueeze(dim)

            x = torch.randn(*shape)
            self.run_model_test(
                MyModel(), train=False, input=(x), batch_size=BATCH_SIZE, atol=1e-7
            )

    def test_squeeze(self):
        shape = (1, 1, 1)
        # test negative dim as well
        for dim in range(-len(shape), len(shape)):

            class MyModel(torch.nn.Module):
                def __init__(self):
                    super().__init__()

                def forward(self, x):
                    return x.squeeze(dim)

            x = torch.randn(*shape)
            self.run_model_test(
                MyModel(), train=False, input=(x), batch_size=BATCH_SIZE, atol=1e-7
            )

    # NB: InstanceNorm model includes unused weights, so skip this in TestCaffe2BackendEmbed
    # TODO: We should have another pass to eliminate the unused initializers in ONNX models.
    @skipIfEmbed
    def test_instance_norm(self):
        underlying = nn.InstanceNorm2d(3)
        self.run_model_test(underlying, train=False, batch_size=BATCH_SIZE)

    @unittest.skip("Disabled due to onnx optimizer deprecation")
    def test_pixel_shuffle(self):
        underlying = nn.PixelShuffle(4)
        shape = (1, 32, 5, 5)
        input = Variable(torch.randn(*shape), requires_grad=True)
        self.run_model_test(
            underlying, train=False, input=(input), batch_size=BATCH_SIZE
        )

    def test_dynamic_sizes(self):
        class MyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                shape = torch.onnx.operators.shape_as_tensor(x)
                new_shape = torch.cat((torch.LongTensor([-1]), shape[0].view(1)))
                return torch.onnx.operators.reshape_from_tensor_shape(x, new_shape)

        x = torch.randn(3, 5, 7)
        self.run_model_test(
            MyModel(), train=False, input=x, batch_size=BATCH_SIZE, use_gpu=False
        )

    def test_advanced_broadcast(self):
        class MyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                return torch.mul(x, y)

        x = torch.randn(1, 5, 10)
        y = torch.randn(1, 5, 1)
        self.run_model_test(
            MyModel(), train=False, input=(x, y), batch_size=BATCH_SIZE, use_gpu=False
        )

    def test_int8_export(self):
        class MyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.param = torch.ByteTensor(3, 4).random_()

            def forward(self, x):
                return x * self.param.float()

        import io

        f = io.BytesIO()
        from torch.onnx import ExportTypes

        torch.onnx._export(
            MyModel(),
            (torch.rand(3, 4),),
            f,
            verbose=True,
            export_type=ExportTypes.ZIP_ARCHIVE,
            keep_initializers_as_inputs=True,
        )

        X = np.random.rand(3, 4).astype(np.float32)

        f.seek(0)
        import caffe2.python.onnx.backend as c2

        model = c2.prepare_zip_archive(f)
        model.run(X)

    @skipIfUnsupportedOpsetVersion([10])
    def test_neg_slice(self):
        class NegSlice(torch.nn.Module):
            def forward(self, x):
                return x[-1, :, :]

        x = torch.randn(3, 4, 5)
        self.run_model_test(
            NegSlice(), train=False, input=(x,), batch_size=BATCH_SIZE, use_gpu=False
        )

    @skipIfUnsupportedOpsetVersion([10])
    def test_neg_slice_large(self):
        class NegSlice(torch.nn.Module):
            def forward(self, x):
                return x[:, :, :, :, -3]

        x = torch.randn(3, 4, 5, 6, 7)
        self.run_model_test(
            NegSlice(), train=False, input=(x,), batch_size=BATCH_SIZE, use_gpu=False
        )

    @unittest.skip("https://github.com/pytorch/pytorch/issues/10984")
    @skipIfUnsupportedOpsetVersion([10])
    def test_neg_slice_large_negone(self):
        class NegSlice(torch.nn.Module):
            def forward(self, x):
                return x[:, :, :, :, -1]

        x = torch.randn(3, 4, 5, 6, 7)
        self.run_model_test(
            NegSlice(), train=False, input=(x,), batch_size=BATCH_SIZE, use_gpu=False
        )

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_dynamic_slice(self):
        class DynamicSliceExportMod(torch.nn.Module):
            def forward(self, x):
                results = []
                for i in range(4):
                    results.append(x[: x.size(0) - i, i : x.size(2), i:3])
                return tuple(results)

        x = torch.rand(5, 5, 5)
        self.run_model_test(
            DynamicSliceExportMod(),
            train=False,
            input=(x,),
            batch_size=BATCH_SIZE,
            use_gpu=False,
        )

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_dynamic_slice_script(self):
        class DynamicSliceModel(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x):
                return x[1 : x.size(0)]

        module = DynamicSliceModel()
        x = torch.rand(1, 2)
        self.run_model_test(
            DynamicSliceModel(),
            train=False,
            input=(x,),
            batch_size=BATCH_SIZE,
            use_gpu=False,
        )

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_dynamic_slice_to_the_end(self):
        class DynamicSliceExportMod(torch.nn.Module):
            def forward(self, x):
                results = []
                for i in range(4):
                    results.append(x[:, i:, x.size(2) - 5])
                return tuple(results)

        x = torch.rand(5, 5, 5)
        self.run_model_test(
            DynamicSliceExportMod(),
            train=False,
            input=(x,),
            batch_size=BATCH_SIZE,
            use_gpu=False,
        )

    def test_unbind(self):
        class UnbindModel(torch.nn.Module):
            def forward(self, input):
                return input.unbind()

        x = torch.randn(3, 4, 5)
        self.run_model_test(
            UnbindModel(), train=False, input=(x,), batch_size=BATCH_SIZE, use_gpu=False
        )

        class UnbindModel2(torch.nn.Module):
            def forward(self, input):
                _, out, _, _ = input.unbind(1)
                return out

        x = torch.randn(3, 4, 5)
        self.run_model_test(
            UnbindModel2(),
            train=False,
            input=(x,),
            batch_size=BATCH_SIZE,
            use_gpu=False,
        )

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_inplace_zero(self):
        class Zero_(torch.nn.Module):
            def forward(self, x):
                return x.zero_()

        x = torch.randn(2, 3, 4)
        self.run_model_test(
            Zero_(),
            train=False,
            input=(x,),
            batch_size=BATCH_SIZE,
            use_gpu=False,
            input_names=["x"],
            dynamic_axes={"x": [0, 1, 2]},
        )
        self.run_model_test(
            Zero_(),
            train=False,
            input=(x,),
            batch_size=BATCH_SIZE,
            use_gpu=False,
            remained_onnx_input_idx=[],
        )

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_inplace_fill(self):
        class Fill_(torch.nn.Module):
            def forward(self, x):
                return x.fill_(3)

        x = torch.randn(2, 3, 4)
        self.run_model_test(
            Fill_(),
            train=False,
            input=(x,),
            batch_size=BATCH_SIZE,
            use_gpu=False,
            input_names=["x"],
            dynamic_axes={"x": [0, 1, 2]},
        )
        self.run_model_test(
            Fill_(),
            train=False,
            input=(x,),
            batch_size=BATCH_SIZE,
            use_gpu=False,
            remained_onnx_input_idx=[],
        )

    # ConstantFill is a deprecated experimental op (used in opsets < 9).
    # Shape inference does not cover this op.
    @skipIfUnsupportedMinOpsetVersion(9)
    def test_inplace_arithmetic(self):
        class Arithmetic(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self):
                x = torch.ones(2, 3, 4)
                y = torch.ones(2, 3, 4) * 2
                x.add_(3)
                y.mul_(x)
                return x, y

        x = torch.ones(2, 3, 4)
        y = torch.ones(2, 3, 4) * 2
        self.run_model_test(
            Arithmetic(), train=False, input=(), batch_size=BATCH_SIZE, use_gpu=False
        )

    def test_tensor_factories(self):
        class TensorFactory(torch.nn.Module):
            def forward(self, x):
                return torch.zeros(x.size()) + torch.ones(x.size())

        x = torch.randn(2, 3, 4)
        self.run_model_test(
            TensorFactory(),
            train=False,
            input=(x,),
            batch_size=BATCH_SIZE,
            use_gpu=False,
            input_names=["x"],
            dynamic_axes={"x": [0, 1, 2]},
        )
        self.run_model_test(
            TensorFactory(),
            train=False,
            input=(x,),
            batch_size=BATCH_SIZE,
            use_gpu=False,
            remained_onnx_input_idx=[],
        )

    def test_tensor_factories_script(self):
        class TensorFactory(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x):
                return torch.zeros(x.shape, dtype=torch.float) + torch.ones(
                    x.shape, dtype=torch.float
                )

        x = torch.randn(2, 3, 4)
        self.run_model_test(
            TensorFactory(),
            train=False,
            input=(x,),
            batch_size=BATCH_SIZE,
            use_gpu=False,
            input_names=["x"],
            dynamic_axes={"x": [0, 1, 2]},
        )
        self.run_model_test(
            TensorFactory(),
            train=False,
            input=(x,),
            batch_size=BATCH_SIZE,
            use_gpu=False,
            remained_onnx_input_idx=[],
        )

    def test_tensor_like_factories_script(self):
        class TensorFactory(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x):
                zeros = torch.zeros_like(
                    x,
                    dtype=torch.float,
                    layout=torch.strided,
                    device=torch.device("cpu"),
                )
                ones = torch.ones_like(
                    x,
                    dtype=torch.float,
                    layout=torch.strided,
                    device=torch.device("cpu"),
                )
                return zeros + ones

        x = torch.randn(2, 3, 4)
        self.run_model_test(
            TensorFactory(),
            train=False,
            input=(x,),
            batch_size=BATCH_SIZE,
            use_gpu=False,
            input_names=["x"],
            dynamic_axes={"x": [0, 1, 2]},
        )
        remained_onnx_input_idx = None if self.opset_version < 9 else []
        self.run_model_test(
            TensorFactory(),
            train=False,
            input=(x,),
            batch_size=BATCH_SIZE,
            use_gpu=False,
            remained_onnx_input_idx=remained_onnx_input_idx,
        )

    def test_full(self):
        class FullModel(torch.nn.Module):
            def forward(self, x):
                return torch.full((3, 4), x, dtype=torch.long)

        x = torch.tensor(12)
        self.run_model_test(
            FullModel(), train=False, input=(x,), batch_size=BATCH_SIZE, use_gpu=False
        )

    def test_full_script(self):
        class FullClass(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x):
                return torch.full((4, 5), x, dtype=torch.long)

        x = torch.tensor(12)
        self.run_model_test(
            FullClass(), train=False, input=(x,), batch_size=BATCH_SIZE, use_gpu=False
        )

    def test_clamp(self):
        class ClampModel(torch.nn.Module):
            def forward(self, x):
                return x.clamp(-0.5, 0.5)

        x = torch.randn(3, 4)
        self.run_model_test(
            ClampModel(), train=False, input=(x,), batch_size=BATCH_SIZE
        )

        class ClampMinModel(torch.nn.Module):
            def forward(self, x):
                return x.clamp(min=-0.5)

        x = torch.randn(3, 4)
        self.run_model_test(
            ClampMinModel(), train=False, input=(x,), batch_size=BATCH_SIZE
        )

        class ClampMaxModel(torch.nn.Module):
            def forward(self, x):
                return x.clamp(max=0.5)

        x = torch.randn(3, 4)
        self.run_model_test(
            ClampMaxModel(), train=False, input=(x,), batch_size=BATCH_SIZE
        )

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_where_functional(self):
        class WhereFunctional(torch.nn.Module):
            def forward(self, x):
                return torch.where(x > 2.0, x, torch.neg(x))

        x = torch.randn(3, 4)
        self.run_model_test(
            WhereFunctional(),
            train=False,
            input=(x,),
            batch_size=BATCH_SIZE,
            use_gpu=False,
        )

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_where_method(self):
        class WhereMethod(torch.nn.Module):
            def forward(self, x):
                return x.where(x > 2.0, torch.neg(x))

        x = torch.randn(3, 4)
        self.run_model_test(
            WhereMethod(), train=False, input=(x,), batch_size=BATCH_SIZE, use_gpu=False
        )

    def test_data_dependent_zeros_factory(self):
        class ZerosFactory(torch.nn.Module):
            def forward(self, input):
                return torch.cat(
                    [input, torch.zeros(input.size(0), 1).type_as(input)], dim=1
                )

        x = torch.zeros(3, 4)
        self.run_model_test(
            ZerosFactory(),
            train=False,
            input=(x,),
            batch_size=BATCH_SIZE,
            use_gpu=False,
        )

    def test_implicit_expand(self):
        class ImplicitExpandExportMod(torch.nn.Module):
            def forward(self, x):
                return x + 1

        x = torch.randn(3, 4)
        self.run_model_test(
            ImplicitExpandExportMod(),
            train=False,
            input=(x,),
            batch_size=BATCH_SIZE,
            use_gpu=False,
        )

    def test_reduce_sum(self):
        class ReduceSumNegativeIndices(torch.nn.Module):
            def forward(self, x):
                return x.sum(-1)

        x = torch.randn(2, 3, 4)
        self.run_model_test(
            ReduceSumNegativeIndices(),
            train=False,
            input=(x,),
            batch_size=BATCH_SIZE,
            use_gpu=False,
        )

    def test_reduce_sum_multi_dim(self):
        class ReduceSumMultipleAxes(torch.nn.Module):
            def forward(self, x):
                return x.sum(dim=(2, 3), keepdim=True)

        x = torch.randn(16, 3, 256, 256)
        self.run_model_test(
            ReduceSumMultipleAxes(),
            train=False,
            input=(x,),
            batch_size=BATCH_SIZE,
            use_gpu=False,
        )

    # InstanceNorm model (used in the subgraph) includes unused weights,
    # so skip this in TestCaffe2BackendEmbed
    @skipIfEmbed
    def test_group_norm(self):
        c = torch.randn(BATCH_SIZE, 6, 224, 224)
        model = nn.GroupNorm(3, 6, eps=0.0002)
        self.run_model_test(model, train=True, input=c, batch_size=BATCH_SIZE)

    # InstanceNorm model (used in the subgraph) includes unused weights,
    # so skip this in TestCaffe2BackendEmbed
    @skipIfEmbed
    def test_group_norm_noaffine(self):
        c = torch.randn(BATCH_SIZE, 6, 224, 224)
        model = nn.GroupNorm(3, 6, eps=0.0002, affine=False)
        self.run_model_test(model, train=True, input=c, batch_size=BATCH_SIZE)

    def test_rsub(self):
        class RsubModel(torch.nn.Module):
            def forward(self, x):
                return 1 - x

        x = torch.randn(1, 2)
        self.run_model_test(
            RsubModel(), train=False, input=(x,), batch_size=BATCH_SIZE, use_gpu=False
        )

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_isnan(self):
        class IsNaNModel(torch.nn.Module):
            def forward(self, input):
                return torch.isnan(input)

        x = torch.tensor([1.0, float("nan"), 2.0])
        self.run_model_test(
            IsNaNModel(), train=False, input=x, batch_size=BATCH_SIZE, use_gpu=False
        )

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_scatter(self):
        class ScatterModel(torch.nn.Module):
            def forward(self, input, indices, values):
                return input.scatter(1, indices, values)

        input = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        indices = torch.tensor([[1, 0], [0, 2], [0, 1]], dtype=torch.int64)
        values = torch.tensor([[1.0, 1.1], [2.0, 2.1], [3.0, 3.1]])
        self.run_model_test(
            ScatterModel(),
            train=False,
            input=(input, indices, values),
            batch_size=BATCH_SIZE,
            use_gpu=False,
        )

        input = torch.zeros(3, 4, 5, 6)
        indices = torch.tensor([[1, 0], [0, 2], [0, 1]], dtype=torch.int64)
        indices = indices.view(3, 2, 1, 1).expand(3, 2, 5, 6)
        values = torch.arange(3 * 2 * 5 * 6, dtype=torch.float32).view(3, 2, 5, 6)
        self.run_model_test(
            ScatterModel(),
            train=False,
            input=(input, indices, values),
            batch_size=BATCH_SIZE,
            use_gpu=False,
        )

        input = torch.zeros(3, 4, 2)
        indices = torch.tensor([[[1, 0], [0, 2]], [[1, 1], [0, 1]], [[2, 1], [2, 2]]])
        values = torch.arange(3 * 2 * 2, dtype=torch.float32).view(3, 2, 2)
        self.run_model_test(
            ScatterModel(),
            train=False,
            input=(input, indices, values),
            batch_size=BATCH_SIZE,
            use_gpu=False,
        )

    @skipIfUnsupportedOpsetVersion([10])
    def test_flatten(self):
        class FlattenModel(torch.nn.Module):
            def forward(self, input):
                return torch.flatten(input)

        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.run_model_test(FlattenModel(), train=False, input=x, batch_size=BATCH_SIZE)

    def test_flatten2D(self):
        class FlattenModel(torch.nn.Module):
            def forward(self, input):
                return torch.flatten(input, 1)

        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.run_model_test(FlattenModel(), train=False, input=x, batch_size=BATCH_SIZE)

    def test_max(self):
        class MaxModel(torch.nn.Module):
            def forward(self, input):
                return torch.max(input, dim=1)

        x = torch.randn(4, 4, requires_grad=True)
        self.run_model_test(MaxModel(), train=False, input=x, batch_size=BATCH_SIZE)

    def test_max_keepdim(self):
        class MaxModel(torch.nn.Module):
            def forward(self, input):
                return torch.max(input, dim=1, keepdim=True)

        x = torch.randn(4, 4, requires_grad=True)
        self.run_model_test(MaxModel(), train=False, input=x, batch_size=BATCH_SIZE)

    def test_max_tensors(self):
        class MaxModel(torch.nn.Module):
            def forward(self, input, other):
                return torch.max(input, other)

        x = torch.randn(4, 4, requires_grad=True)
        y = torch.randn(4, 4, requires_grad=True)
        self.run_model_test(
            MaxModel(), train=False, input=(x, y), batch_size=BATCH_SIZE
        )

    def test_min(self):
        class MinModel(torch.nn.Module):
            def forward(self, input):
                return torch.min(input, dim=1)

        x = torch.randn(4, 4, requires_grad=True)
        self.run_model_test(MinModel(), train=False, input=x, batch_size=BATCH_SIZE)

    def test_argmax(self):
        class ArgmaxModel(torch.nn.Module):
            def forward(self, input):
                return torch.argmax(input, dim=1)

        x = torch.randn(4, 4, requires_grad=True)
        self.run_model_test(ArgmaxModel(), train=False, input=x, batch_size=BATCH_SIZE)

    def test_argmax_none_dim(self):
        class ArgmaxModel(torch.nn.Module):
            def forward(self, input):
                return torch.argmax(input)

        x = torch.randn(4, 4, requires_grad=True)
        self.run_model_test(ArgmaxModel(), train=False, input=x, batch_size=BATCH_SIZE)

    def test_argmin(self):
        class ArgminModel(torch.nn.Module):
            def forward(self, input):
                return torch.argmin(input, dim=1)

        x = torch.randn(4, 4, requires_grad=True)
        self.run_model_test(ArgminModel(), train=False, input=x, batch_size=BATCH_SIZE)

    def test_argmin_none_dim(self):
        class ArgminModel(torch.nn.Module):
            def forward(self, input):
                return torch.argmin(input)

        x = torch.randn(4, 4, requires_grad=True)
        self.run_model_test(ArgminModel(), train=False, input=x, batch_size=BATCH_SIZE)

    def test_reshape(self):
        class ReshapeModel(torch.nn.Module):
            def forward(self, input):
                return input.reshape(1, 1)

        x = torch.randn(1, requires_grad=True)
        self.run_model_test(ReshapeModel(), train=False, input=x, batch_size=BATCH_SIZE)

    def test_reshape_as(self):
        class ReshapeAsModel(torch.nn.Module):
            def forward(self, input):
                y = torch.randn(3, 1, 2, 1, requires_grad=False)
                return input.reshape_as(y)

        x = torch.randn(2, 3, requires_grad=True)
        self.run_model_test(
            ReshapeAsModel(), train=False, input=x, batch_size=BATCH_SIZE
        )

    @skipIfUnsupportedOpsetVersion([10])
    def test_narrow(self):
        class NarrowModel(torch.nn.Module):
            def forward(self, input):
                return torch.narrow(input, 0, 0, 2)

        x = torch.randn(3, 3, requires_grad=True)
        self.run_model_test(NarrowModel(), train=False, input=x, batch_size=BATCH_SIZE)

    def test_randn_like(self):
        class RandNLikeModel(torch.nn.Module):
            def forward(self, input):
                return torch.randn_like(input)

        x = torch.randn(2, 3, 4, requires_grad=False)
        model = RandNLikeModel()
        onnxir, _ = do_export(model, x, keep_initializers_as_inputs=True)
        onnx_model = onnx.ModelProto.FromString(onnxir)
        prepared = c2.prepare(onnx_model)
        caffe2_out = prepared.run(inputs=[x.cpu().numpy()])
        self.assertEqual(caffe2_out[0].shape, x.shape)

    def test_traced_ints(self):
        A = 4
        H = 10
        W = 8
        img_count = 3

        # in this model, the constant propagation in JIT doesn't work
        # so we have ListConstruct in the symbolic
        class MyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(A, 4 * A, 1, stride=1)

            def forward(self, feature, im_info, anchors):
                bbox_deltas = self.conv(feature)
                a, b = torch.ops._caffe2.GenerateProposals(
                    feature,
                    bbox_deltas,
                    im_info,
                    anchors,
                    2.0,
                    6000,
                    300,
                    0.7,
                    16,
                    True,
                    -90,
                    90,
                    1.0,
                    True,
                )
                output = torch.ops._caffe2.RoIAlign(
                    feature,
                    a,
                    order="NCHW",
                    spatial_scale=1.0,
                    pooled_h=3,
                    pooled_w=3,
                    sampling_ratio=0,
                    aligned=False,
                )
                return output

        feature = torch.empty(img_count, A, H, W)
        im_info = torch.ones(img_count, 3, dtype=torch.float32)
        anchors = torch.ones(A, 4, dtype=torch.float32)
        inputs = (feature, im_info, anchors)

        model = MyModel()
        with torch.no_grad():
            self.run_model_test(
                MyModel(), train=False, input=inputs, batch_size=BATCH_SIZE
            )

    def test_c2_roi_align(self):
        class MyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, feature, rois):
                roi_feature = torch.ops._caffe2.RoIAlign(
                    feature,
                    rois,
                    order="NCHW",
                    spatial_scale=1.0,
                    pooled_h=3,
                    pooled_w=3,
                    sampling_ratio=3,
                    aligned=False,
                )
                return roi_feature

        def rand_roi(N, C, H, W):
            return [
                float(int(N * np.random.rand())),
                0.5 * np.random.rand() * W,
                0.5 * np.random.rand() * H,
                (0.5 + 0.5 * np.random.rand()) * W,
                (0.5 + 0.5 * np.random.rand()) * H,
            ]

        N, C, H, W = 1, 4, 10, 8
        feature = torch.randn(N, C, H, W)
        rois = torch.tensor([rand_roi(N, C, H, W) for _ in range(10)])
        inputs = (feature, rois)
        self.run_model_test(MyModel(), train=False, input=inputs, batch_size=3)

    def test_c2_generate_proposals(self):
        class MyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, scores, bbox_deltas, im_info, anchors):
                a, b = torch.ops._caffe2.GenerateProposals(
                    scores,
                    bbox_deltas,
                    im_info,
                    anchors,
                    2.0,
                    6000,
                    300,
                    0.7,
                    16,
                    True,
                    -90,
                    90,
                    1.0,
                    True,
                )
                return a, b

        A = 4
        H = 10
        W = 8
        img_count = 3
        scores = torch.ones(img_count, A, H, W, dtype=torch.float32)
        bbox_deltas = torch.linspace(
            0, 10, steps=img_count * 4 * A * H * W, dtype=torch.float32
        )
        bbox_deltas = bbox_deltas.view(img_count, 4 * A, H, W)
        im_info = torch.ones(img_count, 3, dtype=torch.float32)
        anchors = torch.ones(A, 4, dtype=torch.float32)
        inputs = (scores, bbox_deltas, im_info, anchors)
        self.run_model_test(MyModel(), train=False, input=inputs, batch_size=3)

    def test_c2_bbox_transform(self):
        class MyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, rois, deltas, im_info):
                a, b = torch.ops._caffe2.BBoxTransform(
                    rois,
                    deltas,
                    im_info,
                    weights=[1.0, 1.0, 1.0, 1.0],
                    apply_scale=False,
                    rotated=True,
                    angle_bound_on=True,
                    angle_bound_lo=-90,
                    angle_bound_hi=90,
                    clip_angle_thresh=0.5,
                    legacy_plus_one=True,
                )
                return a, b

        roi_counts = [0, 2, 3, 4, 5]
        batch_size = len(roi_counts)
        total_rois = sum(roi_counts)
        im_dims = np.random.randint(100, 600, batch_size)
        rois = generate_rois_rotated(roi_counts, im_dims)
        box_dim = 5
        num_classes = 7
        deltas = np.random.randn(total_rois, box_dim * num_classes).astype(np.float32)
        im_info = np.zeros((batch_size, 3)).astype(np.float32)
        im_info[:, 0] = im_dims
        im_info[:, 1] = im_dims
        im_info[:, 2] = 1.0
        im_info = torch.zeros((batch_size, 3))
        inputs = (torch.tensor(rois), torch.tensor(deltas), torch.tensor(im_info))
        self.run_model_test(
            MyModel(), train=False, input=inputs, batch_size=3, use_gpu=False
        )

    # BoxWithNMSLimits has requirements for the inputs, so randomly generated inputs
    # in Caffe2BackendTestEmbed doesn't work with this op.
    @skipIfEmbed
    def test_c2_box_with_nms_limits(self):
        roi_counts = [0, 2, 3, 4, 5]
        num_classes = 7
        rotated = False
        angle_bound_on = True
        clip_angle_thresh = 0.5
        rois, deltas, im_info = create_bbox_transform_inputs(
            roi_counts, num_classes, rotated
        )
        pred_bbox, batch_splits = (
            t.detach().numpy()
            for t in torch.ops._caffe2.BBoxTransform(
                torch.tensor(rois),
                torch.tensor(deltas),
                torch.tensor(im_info),
                [1.0, 1.0, 1.0, 1.0],
                False,
                rotated,
                angle_bound_on,
                -90,
                90,
                clip_angle_thresh,
                legacy_plus_one=True,
            )
        )
        class_prob = np.random.randn(sum(roi_counts), num_classes).astype(np.float32)
        score_thresh = 0.5
        nms_thresh = 0.5
        topk_per_image = int(sum(roi_counts) / 2)

        class MyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, class_prob, pred_bbox, batch_splits):
                a, b, c, d, e, f = torch.ops._caffe2.BoxWithNMSLimit(
                    class_prob,
                    pred_bbox,
                    batch_splits,
                    score_thresh=score_thresh,
                    nms=nms_thresh,
                    detections_per_im=topk_per_image,
                    soft_nms_enabled=False,
                    soft_nms_method="linear",
                    soft_nms_sigma=0.5,
                    soft_nms_min_score_thres=0.001,
                    rotated=rotated,
                    cls_agnostic_bbox_reg=False,
                    input_boxes_include_bg_cls=True,
                    output_classes_include_bg_cls=True,
                    legacy_plus_one=True,
                )
                return a, b, c, d, e, f

        inputs = (
            torch.tensor(class_prob),
            torch.tensor(pred_bbox),
            torch.tensor(batch_splits),
        )
        self.run_model_test(
            MyModel(), train=False, input=inputs, batch_size=3, use_gpu=False
        )

    def test_c2_inference_lstm(self):
        num_layers = 4
        seq_lens = 6
        emb_lens = 10
        has_bias = True
        batch_first = True
        is_bidirectional = True

        class MyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, lstm_in):
                a, b, c = torch.ops._caffe2.InferenceLSTM(
                    lstm_in, num_layers, has_bias, batch_first, is_bidirectional
                )
                return a, b, c

        num_directions = 2
        bsz = 5
        hidden_size = 7
        hx = np.zeros((num_layers * num_directions, bsz, hidden_size), dtype=np.float32)
        inputs = np.random.randn(bsz, seq_lens, emb_lens).astype(np.float32)
        torch_lstm = torch.nn.LSTM(
            emb_lens,
            hidden_size,
            batch_first=batch_first,
            bidirectional=is_bidirectional,
            bias=has_bias,
            num_layers=num_layers,
        )
        lstm_in = (
            [
                torch.from_numpy(inputs),
                torch.from_numpy(hx),
                torch.from_numpy(hx),
            ]
            + [param.detach() for param in torch_lstm._flat_weights],
        )

        self.run_model_test(
            MyModel(), train=False, input=lstm_in, batch_size=3, use_gpu=False
        )

    def test_tuple_input_output(self):
        class TupleModel(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(
                self, a: Tuple[torch.Tensor, torch.Tensor]
            ) -> Tuple[torch.Tensor, torch.Tensor]:
                return a

        x = (torch.randn(3, 4), torch.randn(4, 3))
        self.run_model_test(
            TupleModel(), train=False, input=(x,), batch_size=BATCH_SIZE
        )

    def test_nested_tuple_input_output(self):
        class NestedTupleModel(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(
                self,
                a: torch.Tensor,
                b: Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
            ) -> torch.Tensor:
                return a + b[0] + b[1][0] + b[1][1]

        x = torch.randn(4, 5)
        y = (torch.randn(4, 5), (torch.randn(4, 5), torch.randn(4, 5)))
        self.run_model_test(
            NestedTupleModel(), train=False, input=(x, y), batch_size=BATCH_SIZE
        )

    def test_topk(self):
        class TopKModel(torch.nn.Module):
            def forward(self, input):
                return torch.topk(input, 3)

        x = torch.arange(1.0, 6.0)
        self.run_model_test(TopKModel(), train=False, input=x, batch_size=BATCH_SIZE)

    def test_topk_script(self):
        class TopKModel(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, input):
                return torch.topk(input, 3, dim=0)

        x = torch.randn(4, 3, requires_grad=True)
        self.run_model_test(TopKModel(), train=False, input=(x,), batch_size=BATCH_SIZE)

    def test_floor(self):
        class FloorModel(torch.nn.Module):
            def forward(self, input):
                return torch.floor(input)

        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.run_model_test(FloorModel(), train=False, input=x, batch_size=BATCH_SIZE)

    def test_ceil(self):
        class CeilModel(torch.nn.Module):
            def forward(self, input):
                return torch.ceil(input)

        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        self.run_model_test(CeilModel(), train=False, input=x, batch_size=BATCH_SIZE)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test__dim_arange(self):
        class DimArange(torch.nn.Module):
            def forward(self, input):
                return torch._dim_arange(input, 1)

        x = torch.ones(5, 6)
        self.run_model_test(
            DimArange(),
            train=False,
            input=x,
            batch_size=BATCH_SIZE,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
        )

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_arange_end(self):
        class ArangeScript(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, a):
                return torch.arange(a.size(0), dtype=torch.float).view(-1, 1) + a

        x = torch.randn(3, 4, requires_grad=True)
        self.run_model_test(
            ArangeScript(), train=False, input=(x,), batch_size=BATCH_SIZE
        )

        class ArangeModel(torch.nn.Module):
            def forward(self, a):
                return torch.arange(a.size(0), dtype=torch.float).view(-1, 1) + a

        self.run_model_test(
            ArangeModel(), train=False, input=(x,), batch_size=BATCH_SIZE
        )

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_arange_start_end(self):
        class ArangeScript(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, a):
                return torch.arange(2, a.size(0) + 2, dtype=torch.float).view(-1, 1) + a

        x = torch.randn(3, 4, requires_grad=True)
        self.run_model_test(
            ArangeScript(), train=False, input=(x,), batch_size=BATCH_SIZE
        )

        class ArangeModel(torch.nn.Module):
            def forward(self, a):
                return torch.arange(2, a.size(0) + 2, dtype=torch.float).view(-1, 1) + a

        self.run_model_test(
            ArangeModel(), train=False, input=(x,), batch_size=BATCH_SIZE
        )

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_arange_start_end_step(self):
        class ArangeScript(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, a):
                return (
                    torch.arange(
                        2, a.size(0) * a.size(1) + 2, a.size(1), dtype=torch.float
                    ).view(-1, 1)
                    + a
                )

        x = torch.randn(3, 4, requires_grad=True)
        self.run_model_test(
            ArangeScript(), train=False, input=(x,), batch_size=BATCH_SIZE
        )

        class ArangeModel(torch.nn.Module):
            def forward(self, a):
                return (
                    torch.arange(
                        2, a.size(0) * a.size(1) + 2, a.size(1), dtype=torch.float
                    ).view(-1, 1)
                    + a
                )

        self.run_model_test(
            ArangeModel(), train=False, input=(x,), batch_size=BATCH_SIZE
        )

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_size(self):
        class SizeModel(torch.nn.Module):
            def forward(self, input):
                return torch.arange(input.size(0)), torch.arange(input.size(-1))

        x = torch.randn(5, 3, 2)
        self.run_model_test(
            SizeModel(),
            train=False,
            input=(x,),
            batch_size=BATCH_SIZE,
            input_names=["x"],
            dynamic_axes={"x": [0, 1, 2]},
        )
        self.run_model_test(
            SizeModel(),
            train=False,
            input=(x,),
            batch_size=BATCH_SIZE,
            remained_onnx_input_idx=[],
        )

    def test_log2(self):
        class Log2Model(torch.nn.Module):
            def forward(self, input):
                return torch.log2(input)

        x = torch.empty(BATCH_SIZE, 10, 10).uniform_(4, 9)
        self.run_model_test(Log2Model(), train=False, input=x, batch_size=BATCH_SIZE)

    def test__sample_dirichlet(self):
        class DirichletModel(torch.nn.Module):
            def forward(self, input):
                return torch._sample_dirichlet(input)

        x = torch.randn(2, 3, 4, requires_grad=False)
        model = DirichletModel()
        onnxir, _ = do_export(
            model,
            x,
            keep_initializers_as_inputs=True,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
        )
        onnx_model = onnx.ModelProto.FromString(onnxir)
        prepared = c2.prepare(onnx_model)
        caffe2_out = prepared.run(inputs=[x.cpu().numpy()])
        self.assertEqual(caffe2_out[0].shape, x.shape)

    def test__standard_gamma(self):
        class GammaModel(torch.nn.Module):
            def forward(self, input):
                return torch._standard_gamma(input)

        x = torch.randn(2, 3, 4, requires_grad=False)
        model = GammaModel()
        onnxir, _ = do_export(
            model,
            x,
            keep_initializers_as_inputs=True,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
        )
        onnx_model = onnx.ModelProto.FromString(onnxir)
        prepared = c2.prepare(onnx_model)
        caffe2_out = prepared.run(inputs=[x.cpu().numpy()])
        self.assertEqual(caffe2_out[0].shape, x.shape)

    # The order of returned indices from Multinomial is undefined, so randomly generated inputs
    # in Caffe2BackendTestEmbed doesn't work with this op.
    @skipIfEmbed
    def test_multinomial(self):
        class Multinomial(torch.nn.Module):
            def forward(self, weight):
                return torch.multinomial(weight, 3, replacement=True)

        class MultinomialNoReplacement(torch.nn.Module):
            def forward(self, weight):
                return torch.multinomial(weight, 1)

        weight = torch.tensor([[0, 10, 0, 0], [0, 0, 100, 0]], dtype=torch.float)
        self.run_model_test(
            Multinomial(), train=False, input=weight, batch_size=BATCH_SIZE
        )
        self.run_model_test(
            MultinomialNoReplacement(), train=False, input=weight, batch_size=BATCH_SIZE
        )

    def test_prim_shape(self):
        x = torch.randn(4, 5, requires_grad=True)

        @torch.jit.script
        def view_by_prim_shape(x):
            return x.view(x.shape)

        class PrimShapeModel(torch.nn.Module):
            def forward(self, input):
                return view_by_prim_shape(input)

        self.run_model_test(
            PrimShapeModel(), train=False, input=x, batch_size=BATCH_SIZE
        )

    def test_and(self):
        class AndModel(torch.nn.Module):
            def forward(self, x, y):
                return x & y

        x = torch.randint(0, 1, (3, 5), dtype=torch.bool)
        y = torch.randint(0, 1, (3, 5), dtype=torch.bool)
        self.run_model_test(
            AndModel(), train=False, input=(x, y), batch_size=BATCH_SIZE
        )

    def test_or(self):
        class OrModel(torch.nn.Module):
            def forward(self, x, y):
                return x | y

        x = torch.randint(0, 1, (3, 5), dtype=torch.bool)
        y = torch.randint(0, 1, (3, 5), dtype=torch.bool)
        self.run_model_test(OrModel(), train=False, input=(x, y), batch_size=BATCH_SIZE)

    def test_dropout(self):
        class DropoutModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dropout = torch.nn.Dropout(0.5)

            def forward(self, x):
                return self.dropout(x)

        x = torch.randn(1, 2, 3)
        self.run_model_test(DropoutModel(), train=False, input=x, batch_size=BATCH_SIZE)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_while(self):
        class WhileModel(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x):
                a = 0
                while a < 4:
                    a += 1
                return x + a

        model = WhileModel()
        inputs = torch.zeros(1, 2, 3, dtype=torch.long)
        self.run_model_test(
            model,
            train=False,
            input=(inputs,),
            batch_size=BATCH_SIZE,
        )

    def test_while_cond(self):
        class WhileModel(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x, a):
                b = a < 4
                while b:
                    a += b.to(torch.long)
                    b = a < 4
                return x + a

        model = WhileModel()
        x = torch.zeros(1, 2, 3, dtype=torch.long)
        a = torch.tensor([0], dtype=torch.long)
        self.run_model_test(model, train=False, input=(x, a), batch_size=BATCH_SIZE)

    @unittest.skip("Disabled due to onnx optimizer deprecation")
    def test_loop(self):
        class LoopModel(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x):
                for i in range(5):
                    x = x + i
                return x

        model = LoopModel()
        inputs = torch.zeros(1, 2, 3, dtype=torch.long)
        self.run_model_test(model, train=False, input=(inputs,), batch_size=BATCH_SIZE)

    @unittest.skip("Disabled due to onnx optimizer deprecation")
    def test_dynamic_loop(self):
        class LoopModel(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x):
                for i in range(x.size(2)):
                    x = x + i
                return x

        model = LoopModel()
        inputs = torch.zeros(1, 2, 3, dtype=torch.long)
        self.run_model_test(model, train=False, input=(inputs,), batch_size=BATCH_SIZE)

    @unittest.skip("Disabled due to onnx optimizer deprecation")
    @skipIfUnsupportedMinOpsetVersion(9)
    def test_nested_loops(self):
        class NestedLoopsModel(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x):
                for i in range(5):
                    a = 0
                    while a < 4:
                        a += 1
                        for j in range(a):
                            x = x + j
                    x = x + a
                return x

        model = NestedLoopsModel()
        inputs = torch.zeros(1, 2, 3, dtype=torch.long)
        self.run_model_test(
            model,
            train=False,
            input=(inputs,),
            batch_size=BATCH_SIZE,
        )

    def test_select(self):
        class SelectModel(torch.nn.Module):
            def forward(self, x):
                return torch.select(x, 0, 1)

        model = SelectModel()
        inputs = torch.randn(3, 2, 1)
        self.run_model_test(model, train=False, input=(inputs,), batch_size=BATCH_SIZE)

    def test_std(self):
        class StandardDeviation(torch.nn.Module):
            def forward(self, input):
                return torch.std(input, unbiased=False)

        model = StandardDeviation()
        inputs = torch.randn(2, 3, 4)
        self.run_model_test(model, train=False, input=(inputs,), batch_size=BATCH_SIZE)

    def test_std_along_dims(self):
        class StandardDeviationAlongDims(torch.nn.Module):
            def forward(self, input):
                return torch.std(input, dim=(0, 1), unbiased=False, keepdim=False)

        model = StandardDeviationAlongDims()
        inputs = torch.randn(2, 3, 4)
        self.run_model_test(model, train=False, input=(inputs,), batch_size=BATCH_SIZE)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_masked_fill(self):
        class MaskedFillModel(torch.nn.Module):
            def forward(self, x):
                mask = torch.tensor([[0, 0, 1], [1, 1, 0]], dtype=torch.uint8)
                return x.masked_fill(mask, 2)

        x = torch.zeros(4, 2, 3, requires_grad=True)
        self.run_model_test(
            MaskedFillModel(), input=(x,), train=False, batch_size=BATCH_SIZE
        )

        class MaskedFillModel2(torch.nn.Module):
            def forward(self, x):
                return x.masked_fill(x > 3, -1)

        x = torch.arange(16).view(2, 2, 4).to(torch.float32)
        self.run_model_test(
            MaskedFillModel2(), input=(x,), train=False, batch_size=BATCH_SIZE
        )

    @skipIfUnsupportedMinOpsetVersion(8)
    def test_meshgrid(self):
        class MeshgridModel(torch.nn.Module):
            def forward(self, x, y, z):
                return torch.meshgrid(x, y, z)

        x = torch.ones(3, requires_grad=True)
        y = torch.zeros(4, requires_grad=True)
        z = torch.ones(5, requires_grad=True)
        model = MeshgridModel()
        self.run_model_test(model, train=False, input=(x, y, z), batch_size=BATCH_SIZE)

    def test_remainder(self):
        class RemainderModel(torch.nn.Module):
            def forward(self, input, other):
                return torch.remainder(input, other)

        x = torch.randn(4, 2, 3)
        y = torch.randn(1, 2, 1)
        model = RemainderModel()
        self.run_model_test(model, train=False, input=(x, y), batch_size=BATCH_SIZE)

    def test_remainder_scalar(self):
        class RemainderModel(torch.nn.Module):
            def forward(self, input):
                return torch.remainder(input, 2.55)

        inputs = torch.randint(10, (2, 3))
        model = RemainderModel()
        self.run_model_test(
            model,
            train=False,
            input=(inputs,),
            batch_size=BATCH_SIZE,
        )

    def test_baddbmm(self):
        class MyModule(torch.nn.Module):
            def forward(self, input, batch1, batch2):
                return torch.baddbmm(
                    input, batch1, batch2, alpha=torch.tensor(5), beta=3.5
                )

        x = torch.randn(10, 3, 5)
        batch1 = torch.randn(10, 3, 4)
        batch2 = torch.randn(10, 4, 5)
        self.run_model_test(
            MyModule(), input=(x, batch1, batch2), train=False, batch_size=BATCH_SIZE
        )

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_gelu(self):
        class GeluModel(torch.nn.Module):
            def forward(self, x):
                return torch.nn.functional.gelu(x, approximate="none")

        model = GeluModel()
        inputs = torch.randn(2, 4, 5, 6, requires_grad=True)
        self.run_model_test(model, train=False, input=(inputs,), batch_size=BATCH_SIZE)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_tanh_gelu(self):
        class GeluModel(torch.nn.Module):
            def forward(self, x):
                return torch.nn.functional.gelu(x, approximate="tanh")

        model = GeluModel()
        inputs = torch.randn(2, 4, 5, 6, requires_grad=True)
        self.run_model_test(model, train=False, input=(inputs,), batch_size=BATCH_SIZE)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_index_fill(self):
        class IndexFillModel(torch.nn.Module):
            def forward(self, input):
                index = torch.tensor([2, 0])
                return input.index_fill(2, index, -1)

        x = torch.randn(3, 4, 5, requires_grad=True)
        self.run_model_test(
            IndexFillModel(), input=(x,), train=False, batch_size=BATCH_SIZE
        )

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_index_copy(self):
        class IndexCopyModel(torch.nn.Module):
            def forward(self, input):
                index = torch.tensor([2, 0])
                source = torch.ones(3, 2, 5)
                return input.index_copy(1, index, source)

        x = torch.randn(3, 4, 5, requires_grad=True)
        self.run_model_test(
            IndexCopyModel(), input=(x,), train=False, batch_size=BATCH_SIZE
        )


# a bit of metaprogramming to set up all the rnn tests


def make_test(
    name,
    base,
    layer,
    bidirectional,
    initial_state,
    variable_length,
    dropout,
    **extra_kwargs,
):
    test_name = str(
        "_".join(
            [
                "test",
                name,
                layer[1],
                bidirectional[1],
                initial_state[1],
                variable_length[1],
                dropout[1],
            ]
        )
    )

    @unittest.skip("Disabled due to onnx optimizer deprecation")
    @skipIfUnsupportedOpsetVersion([10])
    @skipIfUnsupportedMinOpsetVersion(8)
    def f(self):
        self._dispatch_rnn_test(
            base,
            layers=layer[0],
            bidirectional=bidirectional[0],
            initial_state=initial_state[0],
            packed_sequence=variable_length[0],
            dropout=dropout[0],
            **extra_kwargs,
        )

    f.__name__ = test_name
    setattr(TestCaffe2Backend_opset9, f.__name__, f)


def setup_rnn_tests():
    layers_opts = [(1, "unilayer"), (3, "trilayer")]
    bidirectional_opts = [(False, "forward"), (True, "bidirectional")]
    initial_state_opts = [(True, "with_initial_state"), (False, "no_initial_state")]
    variable_length_opts = [
        (0, "without_sequence_lengths"),
        (1, "with_variable_length_sequences"),
        (2, "with_batch_first_sequence_lengths"),
    ]
    dropout_opts = [(0.2, "with_dropout"), (0.0, "without_dropout")]
    test_count = 0
    for (
        layer,
        bidirectional,
        initial_state,
        variable_length,
        dropout,
    ) in itertools.product(
        layers_opts,
        bidirectional_opts,
        initial_state_opts,
        variable_length_opts,
        dropout_opts,
    ):

        for base, name, extra_kwargs in (
            ("elman", "elman_relu", {"nonlinearity": "relu"}),
            ("elman", "elman_tanh", {"nonlinearity": "tanh"}),
            ("lstm", "lstm", {}),
            ("gru", "gru", {}),
        ):
            make_test(
                name,
                base,
                layer,
                bidirectional,
                initial_state,
                variable_length,
                dropout,
                **extra_kwargs,
            )
            test_count += 1

    # sanity check that a representative example does exist
    TestCaffe2Backend_opset9.test_gru_trilayer_forward_with_initial_state_without_sequence_lengths_with_dropout

    # make sure no one accidentally disables all the tests without
    # noticing
    assert test_count == 192, test_count


setup_rnn_tests()

# add the same test suite as above, but switch embed_params=False
# to embed_params=True
TestCaffe2BackendEmbed_opset9 = type(
    "TestCaffe2BackendEmbed_opset9",
    (common_utils.TestCase,),
    dict(TestCaffe2Backend_opset9.__dict__, embed_params=True),
)

# opset 7 tests
TestCaffe2Backend_opset7 = type(
    "TestCaffe2Backend_opset7",
    (common_utils.TestCase,),
    dict(TestCaffe2Backend_opset9.__dict__, opset_version=7),
)
TestCaffe2BackendEmbed_opset7 = type(
    "TestCaffe2BackendEmbed_opset7",
    (common_utils.TestCase,),
    dict(TestCaffe2Backend_opset9.__dict__, embed_params=True, opset_version=7),
)

# opset 8 tests
TestCaffe2Backend_opset8 = type(
    "TestCaffe2Backend_opset8",
    (common_utils.TestCase,),
    dict(TestCaffe2Backend_opset9.__dict__, opset_version=8),
)
TestCaffe2BackendEmbed_opset8 = type(
    "TestCaffe2BackendEmbed_opset8",
    (common_utils.TestCase,),
    dict(TestCaffe2Backend_opset9.__dict__, embed_params=True, opset_version=8),
)

# opset 10 tests
TestCaffe2Backend_opset10 = type(
    "TestCaffe2Backend_opset10",
    (common_utils.TestCase,),
    dict(TestCaffe2Backend_opset9.__dict__, opset_version=10),
)

TestCaffe2BackendEmbed_opset10 = type(
    "TestCaffe2BackendEmbed_opset10",
    (common_utils.TestCase,),
    dict(TestCaffe2Backend_opset9.__dict__, embed_params=True, opset_version=10),
)

# add the same test suite as above, but switch embed_params=False
# to embed_params=True
TestCaffe2BackendEmbed_opset9_new_jit_API = type(
    "TestCaffe2BackendEmbed_opset9_new_jit_API",
    (common_utils.TestCase,),
    dict(TestCaffe2Backend_opset9.__dict__, embed_params=True),
)

if __name__ == "__main__":
    common_utils.run_tests()
