from torchvision.models.alexnet import alexnet
from torchvision.models.inception import inception_v3
from torchvision.models.densenet import densenet121
from torchvision.models.resnet import resnet50
from torchvision.models.vgg import vgg16, vgg16_bn, vgg19, vgg19_bn

from model_defs.mnist import MNIST
from model_defs.squeezenet import SqueezeNet
from model_defs.super_resolution import SuperResolutionNet
from model_defs.srresnet import SRResNet
from model_defs.dcgan import _netD, _netG, weights_init, bsz, imgsz, nz
from model_defs.op_test import DummyNet, ConcatNet, PermuteNet, PReluNet

from test_pytorch_common import TestCase, run_tests, skipIfNoLapack

import torch
import torch.onnx
import torch.onnx.utils
from torch.autograd import Variable
from torch.onnx import OperatorExportTypes

import unittest

import caffe2.python.onnx.backend as backend

from verify import verify

if torch.cuda.is_available():
    def toC(x):
        return x.cuda()
else:
    def toC(x):
        return x

BATCH_SIZE = 2


class TestModels(TestCase):
    def exportTest(self, model, inputs, rtol=1e-2, atol=1e-7):
        graph = torch.onnx.utils._trace(model, inputs, OperatorExportTypes.ONNX)
        torch._C._jit_pass_lint(graph)
        verify(model, inputs, backend, rtol=rtol, atol=atol)

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

    @skipIfNoLapack
    @unittest.skip("This model is broken, see https://github.com/pytorch/pytorch/issues/18429")
    def test_super_resolution(self):
        x = Variable(
            torch.randn(BATCH_SIZE, 1, 224, 224).fill_(1.0)
        )
        self.exportTest(toC(SuperResolutionNet(upscale_factor=3)), toC(x), atol=1e-6)

    def test_alexnet(self):
        x = Variable(
            torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0)
        )
        self.exportTest(toC(alexnet()), toC(x))

    @unittest.skip("Waiting for https://github.com/pytorch/pytorch/pull/3100")
    def test_mnist(self):
        x = Variable(torch.randn(BATCH_SIZE, 1, 28, 28).fill_(1.0))
        self.exportTest(toC(MNIST()), toC(x))

    def test_vgg16(self):
        # VGG 16-layer model (configuration "D")
        x = Variable(torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0))
        self.exportTest(toC(vgg16()), toC(x))

    def test_vgg16_bn(self):
        # VGG 16-layer model (configuration "D") with batch normalization
        x = Variable(torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0))
        self.exportTest(toC(vgg16_bn()), toC(x))

    def test_vgg19(self):
        # VGG 19-layer model (configuration "E")
        x = Variable(torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0))
        self.exportTest(toC(vgg19()), toC(x))

    def test_vgg19_bn(self):
        # VGG 19-layer model (configuration 'E') with batch normalization
        x = Variable(torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0))
        self.exportTest(toC(vgg19_bn()), toC(x))

    def test_resnet(self):
        # ResNet50 model
        x = Variable(torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0))
        self.exportTest(toC(resnet50()), toC(x), atol=1e-6)

    def test_inception(self):
        x = Variable(
            torch.randn(BATCH_SIZE, 3, 299, 299) + 1.)
        self.exportTest(toC(inception_v3()), toC(x))

    def test_squeezenet(self):
        # SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and
        # <0.5MB model size
        x = Variable(torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0))
        sqnet_v1_0 = SqueezeNet(version=1.1)
        self.exportTest(toC(sqnet_v1_0), toC(x))

        # SqueezeNet 1.1 has 2.4x less computation and slightly fewer params
        # than SqueezeNet 1.0, without sacrificing accuracy.
        x = Variable(torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0))
        sqnet_v1_1 = SqueezeNet(version=1.1)
        self.exportTest(toC(sqnet_v1_1), toC(x))

    @unittest.skip("Temporary - waiting for https://github.com/onnx/onnx/pull/1773.")
    def test_densenet(self):
        # Densenet-121 model
        x = Variable(torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0))
        self.exportTest(toC(densenet121()), toC(x))

    def test_dcgan_netD(self):
        netD = _netD(1)
        netD.apply(weights_init)
        input = Variable(torch.Tensor(bsz, 3, imgsz, imgsz).normal_(0, 1))
        self.exportTest(toC(netD), toC(input))

    def test_dcgan_netG(self):
        netG = _netG(1)
        netG.apply(weights_init)
        input = Variable(torch.Tensor(bsz, nz, 1, 1).normal_(0, 1))
        self.exportTest(toC(netG), toC(input))

if __name__ == '__main__':
    run_tests()
