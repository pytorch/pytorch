from model_defs.alexnet import AlexNet
from model_defs.mnist import MNIST
from model_defs.word_language_model import RNNModel
from model_defs.vgg import *
from model_defs.resnet import Bottleneck, ResNet
from model_defs.inception import Inception3
from model_defs.squeezenet import SqueezeNet
from model_defs.super_resolution import SuperResolutionNet
from model_defs.densenet import DenseNet
from model_defs.srresnet import SRResNet
from model_defs.dcgan import _netD, _netG, weights_init, bsz, imgsz, nz
from model_defs.op_test import DummyNet, ConcatNet, PermuteNet, PReluNet

from common import TestCase, run_tests, skipIfNoLapack, skipIfCI

import torch
import torch.onnx
from torch.autograd import Variable, Function
from torch.nn import Module

import onnx
import onnx.checker
import onnx.helper

import google.protobuf.text_format

import io
import unittest

if torch.cuda.is_available():
    def toC(x):
        return x.cuda()
else:
    def toC(x):
        return x

BATCH_SIZE = 2


def export_to_string(model, inputs, *args, **kwargs):
    f = io.BytesIO()
    with torch.no_grad():
        torch.onnx.export(model, inputs, f, *args, **kwargs)
    return f.getvalue()


class TestModels(TestCase):
    def exportTest(self, model, inputs, subname=None):
        binary_pb = export_to_string(model, inputs, export_params=False)
        model_def = onnx.ModelProto.FromString(binary_pb)
        onnx.checker.check_model(model_def)
        onnx.helper.strip_doc_string(model_def)
        # NB: We prefer to look at printable_model, but it doesn't print
        # all information.  The pbtxt is the *source of truth*.
        self.assertExpected(onnx.helper.printable_graph(model_def.graph), subname)
        if subname is None:
            pbtxt_subname = "pbtxt"
        else:
            pbtxt_subname = "{}-pbtxt".format(subname)
        self.assertExpected(google.protobuf.text_format.MessageToString(model_def, float_format='.15g'), pbtxt_subname)

    def test_ops(self):
        x = Variable(
            torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0)
        )
        self.exportTest(toC(DummyNet()), toC(x))

    def test_prelu(self):
        x = Variable(
            torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0)
        )
        self.exportTest(PReluNet(), x)

    def test_concat(self):
        input_a = Variable(torch.randn(BATCH_SIZE, 3))
        input_b = Variable(torch.randn(BATCH_SIZE, 3))
        inputs = ((toC(input_a), toC(input_b)), )
        self.exportTest(toC(ConcatNet()), inputs)

    def test_permute(self):
        x = Variable(torch.randn(BATCH_SIZE, 3, 10, 12))
        self.exportTest(PermuteNet(), x)

    @unittest.skip("This model takes too much memory")
    def test_srresnet(self):
        x = Variable(torch.randn(1, 3, 224, 224).fill_(1.0))
        self.exportTest(toC(SRResNet(rescale_factor=4, n_filters=64, n_blocks=8)), toC(x))

    @skipIfCI
    @skipIfNoLapack
    def test_super_resolution(self):
        x = Variable(
            torch.randn(BATCH_SIZE, 1, 224, 224).fill_(1.0)
        )
        self.exportTest(toC(SuperResolutionNet(upscale_factor=3)), toC(x))

    def test_alexnet(self):
        x = Variable(
            torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0)
        )
        self.exportTest(toC(AlexNet()), toC(x))

    @unittest.skip("Waiting for https://github.com/pytorch/pytorch/pull/3100")
    def test_mnist(self):
        x = Variable(torch.randn(BATCH_SIZE, 1, 28, 28).fill_(1.0))
        self.exportTest(toC(MNIST()), toC(x))

    @skipIfCI
    def test_vgg(self):
        # VGG 16-layer model (configuration "D")
        x = Variable(torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0))
        vgg16 = make_vgg16()
        self.exportTest(toC(vgg16), toC(x), "16")

        # VGG 16-layer model (configuration "D") with batch normalization
        x = Variable(torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0))
        vgg16_bn = make_vgg16_bn()
        self.exportTest(toC(vgg16_bn), toC(x), "16_bn")

        # VGG 19-layer model (configuration "E")
        x = Variable(torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0))
        vgg19 = make_vgg19()
        self.exportTest(toC(vgg19), toC(x), "19")

        # VGG 19-layer model (configuration 'E') with batch normalization
        x = Variable(torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0))
        vgg19_bn = make_vgg19_bn()
        self.exportTest(toC(vgg19_bn), toC(x), "19_bn")

    def test_resnet(self):
        # ResNet50 model
        x = Variable(torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0))
        resnet50 = ResNet(Bottleneck, [3, 4, 6, 3])
        self.exportTest(toC(resnet50), toC(x), "50")

    def test_inception(self):
        x = Variable(
            torch.randn(BATCH_SIZE, 3, 299, 299).fill_(1.0))
        self.exportTest(toC(Inception3()), toC(x), "3")

    def test_squeezenet(self):
        # SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and
        # <0.5MB model size
        x = Variable(torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0))
        sqnet_v1_0 = SqueezeNet(version=1.1)
        self.exportTest(toC(sqnet_v1_0), toC(x), "1_0")

        # SqueezeNet 1.1 has 2.4x less computation and slightly fewer params
        # than SqueezeNet 1.0, without sacrificing accuracy.
        x = Variable(torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0))
        sqnet_v1_1 = SqueezeNet(version=1.1)
        self.exportTest(toC(sqnet_v1_1), toC(x), "1_1")

    def test_densenet(self):
        # Densenet-121 model
        x = Variable(torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0))
        dense121 = DenseNet(num_init_features=64, growth_rate=32,
                            block_config=(6, 12, 24, 16))
        self.exportTest(toC(dense121), toC(x), "121")

    def test_dcgan(self):
        # note, could have more than 1 gpu
        netG = _netG(1)
        netG.apply(weights_init)
        netD = _netD(1)
        netD.apply(weights_init)

        input = torch.Tensor(bsz, 3, imgsz, imgsz)
        noise = torch.Tensor(bsz, nz, 1, 1)
        fixed_noise = torch.Tensor(bsz, nz, 1, 1).normal_(0, 1)

        fixed_noise = Variable(fixed_noise)

        netD.zero_grad()
        inputv = Variable(input)
        self.exportTest(toC(netD), toC(inputv), "dcgan-netD")

        noise.resize_(bsz, nz, 1, 1).normal_(0, 1)
        noisev = Variable(noise)
        self.exportTest(toC(netG), toC(noisev), "dcgan-netG")

    def run_word_language_model(self, model_name):
        # Args:
        #   model: string, one of RNN_TANH, RNN_RELU, LSTM, GRU
        #   ntokens: int, len(corpus.dictionary)
        #   emsize: int, default 200, size of embedding
        #   nhid: int, default 200, number of hidden units per layer
        #   nlayers: int, default 2
        #   dropout: float, default 0.5
        #   tied: bool, default False
        #   batchsize: int, default 2
        ntokens = 10
        emsize = 5
        nhid = 5
        nlayers = 5
        dropout = 0.2
        tied = False
        batchsize = 5
        model = RNNModel(model_name, ntokens, emsize,
                         nhid, nlayers, dropout,
                         tied, batchsize)
        x = Variable(torch.LongTensor(10, batchsize).fill_(1),
                     volatile=False)
        self.exportTest(model, (x, model.hidden))

    @unittest.expectedFailure
    def test_word_language_model_RNN_TANH(self):
        model_name = 'RNN_TANH'
        self.run_word_language_model(model_name)

    @unittest.expectedFailure
    def test_word_language_model_RNN_RELU(self):
        model_name = 'RNN_RELU'
        self.run_word_language_model(model_name)

    @unittest.expectedFailure
    def test_word_language_model_LSTM(self):
        model_name = 'LSTM'
        self.run_word_language_model(model_name)

    @unittest.expectedFailure
    def test_word_language_model_GRU(self):
        model_name = 'GRU'
        self.run_word_language_model(model_name)


if __name__ == '__main__':
    run_tests()
