# Owner(s): ["oncall: quantization"]

import copy

import torch
import torch.nn as nn
import torch.ao.nn.quantized as nnq
import torch.nn.intrinsic as nni
import torch.nn.intrinsic.quantized as nniq
import torch.nn.intrinsic.qat as nniqat
from torch.ao.quantization import (
    quantize,
    prepare,
    convert,
    prepare_qat,
    quantize_qat,
    fuse_modules,
    fuse_modules_qat,
    QConfig,
    default_qconfig,
    default_qat_qconfig,
)

from torch.testing._internal.common_quantization import (
    QuantizationTestCase,
    ModelForFusion,
    ModelWithSequentialFusion,
    ModelForLinearBNFusion,
    ModelForFusionWithBias,
    ModelForConvTransposeBNFusion,
    test_only_eval_fn,
    test_only_train_fn,
    skipIfNoFBGEMM,
)

from torch.testing._internal.common_quantized import (
    override_quantized_engine,
    supported_qengines,
)


@skipIfNoFBGEMM
class TestFuseEager(QuantizationTestCase):
    def test_fuse_module_train(self):
        model = ModelForFusion(default_qat_qconfig).train()
        # Test step by step fusion
        model = fuse_modules_qat(model, ['conv1', 'bn1', 'relu1'])
        model = fuse_modules_qat(model, ['sub1.conv', 'sub1.bn'])
        self.assertEqual(type(model.conv1), nni.ConvBnReLU2d,
                         msg="Fused Conv + BN + Relu first layer")
        self.assertEqual(type(model.bn1), torch.nn.Identity,
                         msg="Fused Conv + BN + Relu (skipped BN)")
        self.assertEqual(type(model.relu1), torch.nn.Identity,
                         msg="Fused Conv + BN + Relu (skipped Relu)")

        self.assertEqual(type(model.sub1.conv), nni.ConvBn2d,
                         msg="Fused submodule Conv + BN")
        self.assertEqual(type(model.sub1.bn), torch.nn.Identity,
                         msg="Fused submodule Conv + BN (skipped BN)")
        self.assertEqual(type(model.sub2.conv), torch.nn.Conv2d,
                         msg="Non-fused submodule Conv")
        self.assertEqual(type(model.sub2.relu), torch.nn.ReLU,
                         msg="Non-fused submodule ReLU")
        model = prepare_qat(model)
        self.checkObservers(model)

        def checkQAT(model):
            self.assertEqual(type(model.conv1), nniqat.ConvBnReLU2d)
            self.assertEqual(type(model.bn1), nn.Identity)
            self.assertEqual(type(model.relu1), nn.Identity)
            self.assertEqual(type(model.sub1.conv), nniqat.ConvBn2d)
            self.assertEqual(type(model.sub1.bn), nn.Identity)
            self.assertEqual(type(model.sub2.conv), nn.Conv2d)
            self.assertEqual(type(model.sub2.relu), nn.ReLU)

        checkQAT(model)
        test_only_train_fn(model, self.img_data_1d_train)
        model = convert(model)

        def checkQuantized(model):
            self.assertEqual(type(model.conv1), nniq.ConvReLU2d)
            self.assertEqual(type(model.bn1), nn.Identity)
            self.assertEqual(type(model.relu1), nn.Identity)
            self.assertEqual(type(model.sub1.conv), nnq.Conv2d)
            self.assertEqual(type(model.sub1.bn), nn.Identity)
            self.assertEqual(type(model.sub2.conv), nn.Conv2d)
            self.assertEqual(type(model.sub2.relu), nn.ReLU)
            test_only_eval_fn(model, self.img_data_1d)
            self.checkNoQconfig(model)

        with self.assertRaisesRegex(RuntimeError, "Could not run 'aten::native_batch_norm' with arguments from the 'QuantizedCPU'"):
            checkQuantized(model)

        model = ModelForFusion(default_qat_qconfig).train()
        model = fuse_modules_qat(
            model,
            [['conv1', 'bn1', 'relu1'],
             ['sub1.conv', 'sub1.bn']])
        model = quantize_qat(model, test_only_train_fn, [self.img_data_1d_train])
        with self.assertRaisesRegex(RuntimeError, "Could not run 'aten::native_batch_norm' with arguments from the 'QuantizedCPU'"):
            checkQuantized(model)


    def test_fuse_module_eval(self):
        model = ModelForFusion(default_qconfig)
        model.eval()
        model = fuse_modules(
            model,
            [['conv3', 'bn3', 'relu4'],
             ['conv1', 'bn1', 'relu1'],
             ['conv2', 'relu2'],
             ['bn2', 'relu3'],
             ['sub1.conv', 'sub1.bn']])
        self.assertEqual(type(model.conv1), nni.ConvReLU2d,
                         msg="Fused Conv + BN + Relu first layer (BN is folded)")
        self.assertEqual(type(model.conv1[0]), nn.Conv2d,
                         msg="Fused Conv + BN + Relu (Conv + folded BN only)")
        self.assertEqual(type(model.conv1[1]), nn.ReLU,
                         msg="Fused Conv + BN + Relu second layer (Relu only)")
        self.assertEqual(type(model.bn1), nn.Identity,
                         msg="Fused Conv + BN + Relu second layer (Skipped BN)")
        self.assertEqual(type(model.relu1), nn.Identity,
                         msg="Fused Conv + BN + Relu second layer (Skipped Relu)")
        self.assertEqual(type(model.conv2), nni.ConvReLU3d,
                         msg="Fused Conv + BN + Relu first layer (BN is folded)")
        self.assertEqual(type(model.bn2), nni.BNReLU3d,
                         msg="Fused BN + Relu first layer (Relu is folded))")
        self.assertEqual(type(model.relu3), nn.Identity,
                         msg="Fused BN + Relu second layer (Skipped Relu)")
        self.assertEqual(type(model.conv2[0]), nn.Conv3d,
                         msg="Fused Conv + BN + Relu (Conv + folded BN only)")
        self.assertEqual(type(model.conv2[1]), nn.ReLU,
                         msg="Fused Conv + BN + Relu second layer (Relu only)")
        self.assertEqual(type(model.relu2), nn.Identity,
                         msg="Fused Conv + BN + Relu second layer (Skipped Relu)")

        self.assertEqual(type(model.conv3), nni.ConvReLU1d,
                         msg="Fused Conv + Relu for Conv1d (folded BN)")
        self.assertEqual(type(model.conv3[0]), nn.Conv1d,
                         msg="Fused Conv + Relu for Conv1d ")
        self.assertEqual(type(model.conv3[1]), nn.ReLU,
                         msg="Fused Conv + Relu for Conv1d")
        self.assertEqual(type(model.bn3), nn.Identity,
                         msg="Fused Conv + BN + Relu for Conv1d (Skipped BN)")

        self.assertEqual(type(model.sub1.conv), nn.Conv2d,
                         msg="Fused submodule Conv + folded BN")
        self.assertEqual(type(model.sub1.bn), nn.Identity,
                         msg="Fused submodule (skipped BN)")
        self.assertEqual(type(model.sub2.conv), nn.Conv2d,
                         msg="Non-fused submodule Conv")
        self.assertEqual(type(model.sub2.relu), torch.nn.ReLU,
                         msg="Non-fused submodule ReLU")

        model = prepare(model)
        self.checkObservers(model)
        test_only_eval_fn(model, self.img_data_1d)
        model = convert(model)

        def checkQuantized(model):
            self.assertEqual(type(model.conv3), nniq.ConvReLU1d)
            self.assertEqual(type(model.conv1), nniq.ConvReLU2d)
            self.assertEqual(type(model.bn1), nn.Identity)
            self.assertEqual(type(model.relu1), nn.Identity)
            self.assertEqual(type(model.sub1.conv), nnq.Conv2d)
            self.assertEqual(type(model.sub1.bn), nn.Identity)
            self.assertEqual(type(model.sub2.conv), nn.Conv2d)
            self.assertEqual(type(model.sub2.relu), nn.ReLU)
            self.assertEqual(type(model.bn2), nniq.BNReLU3d)
            test_only_eval_fn(model, self.img_data_1d)
            self.checkNoQconfig(model)

        checkQuantized(model)

        model = ModelForFusion(default_qconfig).eval()
        model = fuse_modules(
            model,
            [['conv1', 'bn1', 'relu1'],
             ['conv2', 'relu2'],
             ['bn2', 'relu3'],
             ['sub1.conv', 'sub1.bn'],
             ['conv3', 'bn3', 'relu4']])
        model = quantize(model, test_only_eval_fn, [self.img_data_1d])
        checkQuantized(model)

    def test_fusion_sequential_model_train(self):
        for qengine in supported_qengines:
            with override_quantized_engine(qengine):
                model = ModelWithSequentialFusion().train()
                model.to(torch.float)
                fuse_modules_qat(
                    model, [['conv1', 'relu1'] ,
                            ['features.0.0', 'features.0.1', 'features.0.2'],
                            ['features.1.0', 'features.1.1', 'features.1.2'],
                            ['features.2.0', 'features.2.1', 'features.2.2'],
                            ['classifier.0', 'classifier.1']],
                    inplace=True)
                self.assertEqual(type(model.conv1), nni.ConvReLU2d,
                                 msg="Fused Conv + Relu: nni.ConvReLU2d")
                self.assertEqual(type(model.conv1[0]), nn.Conv2d,
                                 msg="Fused Conv + Relu: Conv2d")
                self.assertEqual(type(model.conv1[1]), nn.ReLU,
                                 msg="Fused Conv + Relu: Relu")
                self.assertEqual(type(model.relu1), nn.Identity,
                                 msg="Fused Conv + Relu: Identity")
                for i in range(3):
                    self.assertEqual(type(model.features[i][0]), nni.ConvBnReLU2d,
                                     msg="Fused submodule Conv + folded BN")
                    self.assertEqual(type(model.features[i][1]), nn.Identity,
                                     msg="Fused submodule (skipped BN)")
                    self.assertEqual(type(model.features[i][2]), nn.Identity,
                                     msg="Non-fused submodule Conv")
                self.assertEqual(type(model.classifier[0]), nni.LinearReLU)
                self.assertEqual(type(model.classifier[1]), nn.Identity)
                model.qconfig = torch.ao.quantization.get_default_qat_qconfig(qengine)
                prepare_qat(model, inplace=True)
                self.checkObservers(model)
                model(self.img_data_2d[0][0])


                def checkQAT(model):
                    self.assertEqual(type(model.conv1), nniqat.ConvReLU2d)
                    self.assertEqual(type(model.relu1), nn.Identity)
                for i in range(3):
                    self.assertEqual(type(model.features[i][0]), nniqat.ConvBnReLU2d,
                                     msg="Fused submodule Conv + folded BN")
                    self.assertEqual(type(model.features[i][1]), nn.Identity,
                                     msg="Fused submodule (skipped BN)")
                    self.assertEqual(type(model.features[i][2]), nn.Identity,
                                     msg="Non-fused submodule Conv")
                self.assertEqual(type(model.classifier[0]), nniqat.LinearReLU)
                self.assertEqual(type(model.classifier[1]), nn.Identity)

                checkQAT(model)
                model(self.img_data_2d[1][0])
                convert(model, inplace=True)
                model(self.img_data_2d[1][0])
                self.checkModelWithSequentialQuantized(model)

    def test_fusion_sequential_model_eval(self):
        for qengine in supported_qengines:
            with override_quantized_engine(qengine):
                model = ModelWithSequentialFusion().eval()
                model.to(torch.float)
                fuse_modules(
                    model,
                    [['conv1', 'relu1'],
                     ['features.0.0', 'features.0.1', 'features.0.2'],
                     ['features.1.0', 'features.1.1', 'features.1.2'],
                     ['features.2.0', 'features.2.1', 'features.2.2'],
                     ['classifier.0', 'classifier.1']],
                    inplace=True)
                self.assertEqual(type(model.conv1), nni.ConvReLU2d,
                                 msg="Fused Conv + Relu: nni.ConvReLU2d")
                self.assertEqual(type(model.conv1[0]), nn.Conv2d,
                                 msg="Fused Conv + Relu: Conv2d")
                self.assertEqual(type(model.conv1[1]), nn.ReLU,
                                 msg="Fused Conv + Relu: Relu")
                self.assertEqual(type(model.relu1), nn.Identity,
                                 msg="Fused Conv + Relu: Identity")
                for i in range(3):
                    self.assertEqual(type(model.features[i][0]), nni.ConvReLU2d,
                                     msg="Fused submodule Conv + folded BN")
                    self.assertEqual(type(model.features[i][1]), nn.Identity,
                                     msg="Fused submodule (skipped BN)")
                    self.assertEqual(type(model.features[i][2]), nn.Identity,
                                     msg="Non-fused submodule Conv")
                self.assertEqual(type(model.classifier[0]), nni.LinearReLU)
                self.assertEqual(type(model.classifier[1]), nn.Identity)
                model.qconfig = torch.ao.quantization.get_default_qconfig(qengine)
                prepare(model, inplace=True)
                self.checkObservers(model)
                model(self.img_data_2d[0][0])
                convert(model, inplace=True)
                model(self.img_data_2d[1][0])
                self.checkModelWithSequentialQuantized(model)

    def checkModelWithSequentialQuantized(self, model):
        self.assertEqual(type(model.conv1), nniq.ConvReLU2d)
        self.assertEqual(type(model.relu1), nn.Identity)
        for i in range(3):
            self.assertEqual(type(model.features[i][0]), nniq.ConvReLU2d)
            self.assertEqual(type(model.features[i][1]), nn.Identity)
            self.assertEqual(type(model.features[i][2]), nn.Identity)
        self.assertEqual(type(model.classifier[0]), nniq.LinearReLU)
        self.assertEqual(type(model.classifier[1]), nn.Identity)

    def test_fusion_conv_with_bias(self):
        for qengine in supported_qengines:
            with override_quantized_engine(qengine):
                model_orig = ModelForFusionWithBias().train()

                # reference model
                model_ref = copy.deepcopy(model_orig)
                # output with no fusion.
                out_ref = model_ref(self.img_data_2d[0][0])

                # fused model
                model_orig.qconfig = QConfig(activation=torch.nn.Identity,
                                             weight=torch.nn.Identity)
                model = fuse_modules_qat(
                    model_orig,
                    [["conv1", "bn1", "relu1"],
                     ["conv2", "bn2"]])
                prep_model = prepare_qat(model, inplace=False)
                # output with fusion but no observers.
                out_fused = prep_model(self.img_data_2d[0][0])

                self.assertEqual(out_ref, out_fused)

                def checkBN(bn_ref, bn):
                    self.assertEqual(bn_ref.weight, bn.weight)
                    self.assertEqual(bn_ref.bias, bn.bias)
                    self.assertEqual(bn_ref.running_mean, bn.running_mean)
                    self.assertEqual(bn_ref.running_var, bn.running_var)

                checkBN(model_ref.bn1, prep_model.conv1.bn)
                checkBN(model_ref.bn2, prep_model.conv2.bn)

                model.qconfig = torch.ao.quantization.get_default_qconfig(qengine)
                prepare_qat(model, inplace=True)

                model(self.img_data_2d[0][0])

                def checkQAT(model):
                    self.assertEqual(type(model.conv1), nniqat.ConvBnReLU2d)
                    self.assertEqual(type(model.bn1), nn.Identity)
                    self.assertEqual(type(model.relu1), nn.Identity)
                    self.assertEqual(type(model.conv2), nniqat.ConvBn2d)
                    self.assertEqual(type(model.bn2), nn.Identity)

                checkQAT(model)


    def test_fusion_linear_bn_eval(self):
        model = ModelForLinearBNFusion().train()
        inp1 = torch.randn(8, 20)
        inp2 = torch.randn(8, 20)

        # Get some interesting values into the running mean and variance.
        model(inp1)
        model.eval()
        golden = model(inp2)

        model = fuse_modules(model, [["fc", "bn"]])
        self.assertEqual(type(model.bn), nn.Identity)
        self.assertEqual(golden, model(inp2))

    def test_fusion_convtranspose_bn_eval(self):
        model = ModelForConvTransposeBNFusion().train()
        inp1 = torch.randn(8, 3, 16)
        inp2 = torch.randn(8, 3, 16)

        # Get some interesting values into the running mean and variance.
        model(inp1)
        model.eval()
        golden = model(inp2)

        model = fuse_modules(model, [["conv1", "bn1"], ["conv2", "bn2"], ["conv3", "bn3"]])
        self.assertEqual(type(model.bn1), nn.Identity)
        self.assertEqual(type(model.bn2), nn.Identity)
        self.assertEqual(type(model.bn3), nn.Identity)

        self.assertEqual(golden, model(inp2))

    def test_forward_hooks_preserved(self):
        r"""Test case that checks whether forward pre hooks of the first module and
        post forward hooks of the last module in modules list passed to fusion function preserved.
        (e.g. before fusion: [nn.Conv2d (with pre forward hooks), nn.BatchNorm2d, nn.ReLU (with post forward hooks)]
        after fusion: [nni.ConvBnReLU2d (with pre and post hooks), nn.Identity, nn.Identity])
        """
        model = ModelForFusion(default_qat_qconfig).train()

        counter = {
            'pre_forwards': 0,
            'forwards': 0,
        }
        fused = False

        def fw_pre_hook(fused_module_class, h_module, input):
            if fused:
                self.assertEqual(type(h_module), fused_module_class,
                                 "After fusion owner of the first module's forward pre hook is not a fused module")
            counter['pre_forwards'] += 1

        def fw_hook(fused_module_class, h_module, input, output):
            if fused:
                self.assertEqual(type(h_module), fused_module_class,
                                 "After fusion owner of the last module's forward hook is not a fused module")
            counter['forwards'] += 1

        # Registering two pre and two post forward hooks, thus expecting counter increment by two each inference
        model.conv1.register_forward_pre_hook(lambda *args: fw_pre_hook(nni.ConvBnReLU2d, *args))
        model.sub1.conv.register_forward_pre_hook(lambda *args: fw_pre_hook(nni.ConvBn2d, *args))
        model.relu1.register_forward_hook(lambda *args: fw_hook(nni.ConvBnReLU2d, *args))
        model.sub1.bn.register_forward_hook(lambda *args: fw_hook(nni.ConvBn2d, *args))

        test_only_eval_fn(model, self.img_data_1d)
        self.assertEqual(counter['pre_forwards'], 2 * len(self.img_data_1d))
        self.assertEqual(counter['forwards'], 2 * len(self.img_data_1d))

        model = fuse_modules_qat(model, ['conv1', 'bn1', 'relu1'])
        model = fuse_modules_qat(model, ['sub1.conv', 'sub1.bn'])

        fused = True
        before_fusion_pre_count = counter['pre_forwards']
        before_fusion_post_count = counter['forwards']
        test_only_eval_fn(model, self.img_data_1d)
        self.assertEqual(counter['pre_forwards'] - before_fusion_pre_count, 2 * len(self.img_data_1d))
        self.assertEqual(counter['forwards'] - before_fusion_post_count, 2 * len(self.img_data_1d))

if __name__ == '__main__':
    raise RuntimeError("This test file is not meant to be run directly, use:\n\n"
                       "\tpython test/test_quantization.py TESTNAME\n\n"
                       "instead.")
