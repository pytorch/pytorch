# Owner(s): ["module: onnx"]

import unittest

import torch

from model_defs.dcgan import _netD, _netG, bsz, imgsz, nz, weights_init
from model_defs.emb_seq import EmbeddingNetwork1, EmbeddingNetwork2
from model_defs.mnist import MNIST
from model_defs.op_test import ConcatNet, DummyNet, FakeQuantNet, PermuteNet, PReluNet
from model_defs.squeezenet import SqueezeNet
from model_defs.srresnet import SRResNet
from model_defs.super_resolution import SuperResolutionNet
from pytorch_test_common import skipIfUnsupportedMinOpsetVersion, skipScriptTest
from torch import quantization
from torch.autograd import Variable
from torch.onnx import OperatorExportTypes
from torch.testing._internal import common_utils
from torch.testing._internal.common_utils import skipIfNoLapack
from torchvision.models import shufflenet_v2_x1_0
from torchvision.models.alexnet import alexnet
from torchvision.models.densenet import densenet121
from torchvision.models.googlenet import googlenet
from torchvision.models.inception import inception_v3
from torchvision.models.mnasnet import mnasnet1_0
from torchvision.models.mobilenet import mobilenet_v2
from torchvision.models.resnet import resnet50
from torchvision.models.segmentation import deeplabv3_resnet101, fcn_resnet101
from torchvision.models.vgg import vgg16, vgg16_bn, vgg19, vgg19_bn
from torchvision.models.video import mc3_18, r2plus1d_18, r3d_18
from verify import verify

if torch.cuda.is_available():

    def toC(x):
        return x.cuda()

else:

    def toC(x):
        return x


BATCH_SIZE = 2


class TestModels(common_utils.TestCase):
    opset_version = 9  # Caffe2 doesn't support the default.
    keep_initializers_as_inputs = False

    def exportTest(self, model, inputs, rtol=1e-2, atol=1e-7, **kwargs):
        import caffe2.python.onnx.backend as backend

        with torch.onnx.select_model_mode_for_export(
            model, torch.onnx.TrainingMode.EVAL
        ):
            graph = torch.onnx.utils._trace(model, inputs, OperatorExportTypes.ONNX)
            torch._C._jit_pass_lint(graph)
            verify(
                model,
                inputs,
                backend,
                rtol=rtol,
                atol=atol,
                opset_version=self.opset_version,
            )

    def test_ops(self):
        x = Variable(torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0))
        self.exportTest(toC(DummyNet()), toC(x))

    def test_prelu(self):
        x = Variable(torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0))
        self.exportTest(PReluNet(), x)

    @skipScriptTest()
    def test_concat(self):
        input_a = Variable(torch.randn(BATCH_SIZE, 3))
        input_b = Variable(torch.randn(BATCH_SIZE, 3))
        inputs = ((toC(input_a), toC(input_b)),)
        self.exportTest(toC(ConcatNet()), inputs)

    def test_permute(self):
        x = Variable(torch.randn(BATCH_SIZE, 3, 10, 12))
        self.exportTest(PermuteNet(), x)

    @skipScriptTest()
    def test_embedding_sequential_1(self):
        x = Variable(torch.randint(0, 10, (BATCH_SIZE, 3)))
        self.exportTest(EmbeddingNetwork1(), x)

    @skipScriptTest()
    def test_embedding_sequential_2(self):
        x = Variable(torch.randint(0, 10, (BATCH_SIZE, 3)))
        self.exportTest(EmbeddingNetwork2(), x)

    @unittest.skip("This model takes too much memory")
    def test_srresnet(self):
        x = Variable(torch.randn(1, 3, 224, 224).fill_(1.0))
        self.exportTest(
            toC(SRResNet(rescale_factor=4, n_filters=64, n_blocks=8)), toC(x)
        )

    @skipIfNoLapack
    def test_super_resolution(self):
        x = Variable(torch.randn(BATCH_SIZE, 1, 224, 224).fill_(1.0))
        self.exportTest(toC(SuperResolutionNet(upscale_factor=3)), toC(x), atol=1e-6)

    def test_alexnet(self):
        x = Variable(torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0))
        self.exportTest(toC(alexnet()), toC(x))

    def test_mnist(self):
        x = Variable(torch.randn(BATCH_SIZE, 1, 28, 28).fill_(1.0))
        self.exportTest(toC(MNIST()), toC(x))

    @unittest.skip("This model takes too much memory")
    def test_vgg16(self):
        # VGG 16-layer model (configuration "D")
        x = Variable(torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0))
        self.exportTest(toC(vgg16()), toC(x))

    @unittest.skip("This model takes too much memory")
    def test_vgg16_bn(self):
        # VGG 16-layer model (configuration "D") with batch normalization
        x = Variable(torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0))
        self.exportTest(toC(vgg16_bn()), toC(x))

    @unittest.skip("This model takes too much memory")
    def test_vgg19(self):
        # VGG 19-layer model (configuration "E")
        x = Variable(torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0))
        self.exportTest(toC(vgg19()), toC(x))

    @unittest.skip("This model takes too much memory")
    def test_vgg19_bn(self):
        # VGG 19-layer model (configuration "E") with batch normalization
        x = Variable(torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0))
        self.exportTest(toC(vgg19_bn()), toC(x))

    def test_resnet(self):
        # ResNet50 model
        x = Variable(torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0))
        self.exportTest(toC(resnet50()), toC(x), atol=1e-6)

    # This test is numerically unstable. Sporadic single element mismatch occurs occasionally.
    def test_inception(self):
        x = Variable(torch.randn(BATCH_SIZE, 3, 299, 299))
        self.exportTest(toC(inception_v3()), toC(x), acceptable_error_percentage=0.01)

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

    def test_densenet(self):
        # Densenet-121 model
        x = Variable(torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0))
        self.exportTest(toC(densenet121()), toC(x), rtol=1e-2, atol=1e-5)

    @skipScriptTest()
    def test_dcgan_netD(self):
        netD = _netD(1)
        netD.apply(weights_init)
        input = Variable(torch.empty(bsz, 3, imgsz, imgsz).normal_(0, 1))
        self.exportTest(toC(netD), toC(input))

    @skipScriptTest()
    def test_dcgan_netG(self):
        netG = _netG(1)
        netG.apply(weights_init)
        input = Variable(torch.empty(bsz, nz, 1, 1).normal_(0, 1))
        self.exportTest(toC(netG), toC(input))

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_fake_quant(self):
        x = Variable(torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0))
        self.exportTest(toC(FakeQuantNet()), toC(x))

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_qat_resnet_pertensor(self):
        # Quantize ResNet50 model
        x = Variable(torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0))
        qat_resnet50 = resnet50()

        # Use per tensor for weight. Per channel support will come with opset 13
        qat_resnet50.qconfig = quantization.QConfig(
            activation=quantization.default_fake_quant,
            weight=quantization.default_fake_quant,
        )
        quantization.prepare_qat(qat_resnet50, inplace=True)
        qat_resnet50.apply(torch.ao.quantization.enable_observer)
        qat_resnet50.apply(torch.ao.quantization.enable_fake_quant)

        _ = qat_resnet50(x)
        for module in qat_resnet50.modules():
            if isinstance(module, quantization.FakeQuantize):
                module.calculate_qparams()
        qat_resnet50.apply(torch.ao.quantization.disable_observer)

        self.exportTest(toC(qat_resnet50), toC(x))

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_qat_resnet_per_channel(self):
        # Quantize ResNet50 model
        x = torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0)
        qat_resnet50 = resnet50()

        qat_resnet50.qconfig = quantization.QConfig(
            activation=quantization.default_fake_quant,
            weight=quantization.default_per_channel_weight_fake_quant,
        )
        quantization.prepare_qat(qat_resnet50, inplace=True)
        qat_resnet50.apply(torch.ao.quantization.enable_observer)
        qat_resnet50.apply(torch.ao.quantization.enable_fake_quant)

        _ = qat_resnet50(x)
        for module in qat_resnet50.modules():
            if isinstance(module, quantization.FakeQuantize):
                module.calculate_qparams()
        qat_resnet50.apply(torch.ao.quantization.disable_observer)

        self.exportTest(toC(qat_resnet50), toC(x))

    @skipScriptTest(skip_before_opset_version=15, reason="None type in outputs")
    def test_googlenet(self):
        x = Variable(torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0))
        self.exportTest(toC(googlenet()), toC(x), rtol=1e-3, atol=1e-5)

    def test_mnasnet(self):
        x = Variable(torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0))
        self.exportTest(toC(mnasnet1_0()), toC(x), rtol=1e-3, atol=1e-5)

    def test_mobilenet(self):
        x = Variable(torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0))
        self.exportTest(toC(mobilenet_v2()), toC(x), rtol=1e-3, atol=1e-5)

    @skipScriptTest()  # prim_data
    def test_shufflenet(self):
        x = Variable(torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0))
        self.exportTest(toC(shufflenet_v2_x1_0()), toC(x), rtol=1e-3, atol=1e-5)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_fcn(self):
        x = Variable(torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0))
        self.exportTest(
            toC(fcn_resnet101(pretrained=False, pretrained_backbone=False)),
            toC(x),
            rtol=1e-3,
            atol=1e-5,
        )

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_deeplab(self):
        x = Variable(torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0))
        self.exportTest(
            toC(deeplabv3_resnet101(pretrained=False, pretrained_backbone=False)),
            toC(x),
            rtol=1e-3,
            atol=1e-5,
        )

    def test_r3d_18_video(self):
        x = Variable(torch.randn(1, 3, 4, 112, 112).fill_(1.0))
        self.exportTest(toC(r3d_18()), toC(x), rtol=1e-3, atol=1e-5)

    def test_mc3_18_video(self):
        x = Variable(torch.randn(1, 3, 4, 112, 112).fill_(1.0))
        self.exportTest(toC(mc3_18()), toC(x), rtol=1e-3, atol=1e-5)

    def test_r2plus1d_18_video(self):
        x = Variable(torch.randn(1, 3, 4, 112, 112).fill_(1.0))
        self.exportTest(toC(r2plus1d_18()), toC(x), rtol=1e-3, atol=1e-5)


if __name__ == "__main__":
    common_utils.run_tests()
