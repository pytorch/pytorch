import torch
import torch.nn as nn
import torch.nn.quantized as nnq
from torch.quantization import (
    prepare,
    convert,
    prepare_qat,
    quantize_qat,
    QuantStub,
    DeQuantStub,
    default_qconfig,
    default_qat_qconfig,
    FixedQParamsFakeQuantize,
)

from torch.testing._internal.common_quantization import (
    QuantizationTestCase,
    QuantStubModel,
    ManualLinearQATModel,
    ManualConvLinearQATModel,
    TwoLayerLinearModel,
    test_only_eval_fn,
    test_only_train_fn,
)

from torch.testing._internal.common_quantized import (
    override_quantized_engine,
    supported_qengines,
    override_qengines,
)

class TestQuantizationAwareTraining(QuantizationTestCase):
    def test_manual(self):
        for qengine in supported_qengines:
            with override_quantized_engine(qengine):
                model = ManualLinearQATModel(qengine)
                model = prepare_qat(model)
                self.checkObservers(model)
                test_only_train_fn(model, self.train_data)
                model = convert(model)

                def checkQuantized(model):
                    self.assertEqual(type(model.fc1), nnq.Linear)
                    self.assertEqual(type(model.fc2), nnq.Linear)
                    test_only_eval_fn(model, self.calib_data)
                    self.checkScriptable(model, self.calib_data)
                    self.checkNoQconfig(model)

                checkQuantized(model)

                model = quantize_qat(ManualLinearQATModel(qengine), test_only_train_fn,
                                     [self.train_data])
                checkQuantized(model)

    def test_eval_only_fake_quant(self):
        r"""Using FakeQuant in evaluation only mode,
        this is useful for estimating accuracy loss when we quantize the
        network
        """
        for qengine in supported_qengines:
            with override_quantized_engine(qengine):
                model = ManualLinearQATModel(qengine)

                model = prepare_qat(model)
                self.checkObservers(model)

                model.eval()
                test_only_eval_fn(model, self.calib_data)

    def test_conv_linear(self):
        for qengine in supported_qengines:
            with override_quantized_engine(qengine):
                model = ManualConvLinearQATModel()

                model = prepare_qat(model)
                self.checkObservers(model)

                test_only_train_fn(model, self.img_data_2d_train)
                model = convert(model)

                def checkQuantized(model):
                    self.assertEqual(type(model.conv), nnq.Conv2d)
                    self.assertEqual(type(model.fc1), nnq.Linear)
                    self.assertEqual(type(model.fc2), nnq.Linear)
                    test_only_eval_fn(model, self.img_data_2d)
                    self.checkScriptable(model, self.img_data_2d)
                    self.checkNoQconfig(model)

                checkQuantized(model)

                model = ManualConvLinearQATModel()
                model = quantize_qat(model, test_only_train_fn, [self.img_data_2d_train])
                checkQuantized(model)

    def test_train_save_load_eval(self):
        r"""Test QAT flow of creating a model, doing QAT and saving the quantized state_dict
        During eval, we first call prepare_qat and conver on the model and then load the state_dict
        and compare results against original model
        """
        for qengine in supported_qengines:
            with override_quantized_engine(qengine):
                model = TwoLayerLinearModel()
                model = torch.quantization.QuantWrapper(model)
                model.qconfig = torch.quantization.get_default_qat_qconfig(qengine)
                model = prepare_qat(model)

                fq_state_dict = model.state_dict()

                test_only_train_fn(model, self.train_data)
                model = convert(model)

                quant_state_dict = model.state_dict()

                x = torch.rand(2, 5, dtype=torch.float)
                ref = model(x)

                # Create model again for eval. Check result using quantized state_dict
                model = TwoLayerLinearModel()
                model = torch.quantization.QuantWrapper(model)
                model.qconfig = torch.quantization.get_default_qat_qconfig(qengine)
                torch.quantization.prepare_qat(model, inplace=True)
                new_state_dict = model.state_dict()

                # Check to make sure the model after prepare_qat has the same state_dict as original.
                self.assertEqual(set(fq_state_dict.keys()), set(new_state_dict.keys()))

                torch.quantization.convert(model, inplace=True)
                model.eval()
                model.load_state_dict(quant_state_dict)
                out = model(x)
                self.assertEqual(ref, out)

                # Check model created using prepare has same state dict as quantized state_dict
                model = TwoLayerLinearModel()
                model.eval()
                model = torch.quantization.QuantWrapper(model)
                model.qconfig = torch.quantization.get_default_qconfig(qengine)
                torch.quantization.prepare(model, inplace=True)
                torch.quantization.convert(model, inplace=True)
                self.assertEqual(set(model.state_dict().keys()), set(quant_state_dict.keys()))
                model.eval()
                model.load_state_dict(quant_state_dict)
                out = model(x)
                self.assertEqual(ref, out)

    @override_qengines
    def test_forward_hooks_preserved(self):
        r"""Test QAT on preserving pre forward and post forward hooks of original model
        """
        qengine = torch.backends.quantized.engine
        model = QuantStubModel()
        counter = {
            'pre_forwards': 0,
            'forwards': 0,
        }

        def fw_pre_hook(h_module, input):
            counter['pre_forwards'] += 1

        def fw_hook(h_module, input, output):
            counter['forwards'] += 1

        model.fc.register_forward_pre_hook(fw_pre_hook)
        model.fc.register_forward_hook(fw_hook)

        model.qconfig = torch.quantization.get_default_qat_qconfig(qengine)
        model = prepare_qat(model)

        def checkHooksIsPresent(model, before_convert=True):
            forward_hooks = 1
            if before_convert:
                self.assertEqual(len(model.quant._forward_hooks.values()), 1,
                                 "Quantization observer hook has disappeared")
                forward_hooks = 2
            self.assertObjectIn(fw_pre_hook, model.fc._forward_pre_hooks.values())
            self.assertObjectIn(fw_hook, model.fc._forward_hooks.values())
            self.assertEqual(len(model.fc._forward_pre_hooks.values()), 1,
                             "Extra pre forward hooks have appeared on a layer")
            self.assertEqual(len(model.fc._forward_hooks.values()), forward_hooks,
                             "Extra post forward hooks have appeared on a layer")

        checkHooksIsPresent(model, True)
        x = torch.rand(2, 5, dtype=torch.float)
        model(x)
        torch.quantization.convert(model, inplace=True)
        checkHooksIsPresent(model, False)

    def test_add_scalar_uses_input_qparams(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.quant = torch.quantization.QuantStub()
                self.ff = torch.nn.quantized.FloatFunctional()

            def forward(self, x):
                x = self.quant(x)
                x = self.ff.add_scalar(x, 1.0)
                return x

        m = M()
        m.qconfig = torch.quantization.default_qconfig
        mp = torch.quantization.prepare_qat(m)
        mp(torch.randn(4, 4))
        mq = torch.quantization.convert(mp)
        res = mq(torch.randn(4, 4))
        eps = 1e-5
        self.assertTrue(torch.abs(mq.quant.scale - res.q_scale()) < eps)

    def test_mul_scalar_uses_input_qparams(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.quant = torch.quantization.QuantStub()
                self.ff = torch.nn.quantized.FloatFunctional()

            def forward(self, x):
                x = self.quant(x)
                x = self.ff.mul_scalar(x, 2.0)
                return x

        m = M()
        m.qconfig = torch.quantization.default_qconfig
        mp = torch.quantization.prepare_qat(m)
        mp(torch.randn(4, 4))
        mq = torch.quantization.convert(mp)
        res = mq(torch.randn(4, 4))
        eps = 1e-5
        self.assertTrue(torch.abs(mq.quant.scale * 2 - res.q_scale()) < eps)


class TestQATActivationOps(QuantizationTestCase):
    def _test_activation_convert_numerics_impl(self, Act, data):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.act = Act()
                self.quant = QuantStub()
                self.dequant = DeQuantStub()

            def forward(self, x):
                x = self.quant(x)
                x = self.act(x)
                x = self.dequant(x)
                return x

        m = M().train()
        m.qconfig = default_qat_qconfig
        m = prepare_qat(m)
        before_convert = m(data)
        m = convert(m)
        after_convert = m(data)
        self.assertEqual(before_convert, after_convert)

    def test_fixed_qparam_ops(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.sigmoid = torch.nn.Sigmoid()
                self.hardsigmoid = torch.nn.Hardsigmoid()
                self.tanh = torch.nn.Tanh()
                self.quant = QuantStub()
                self.dequant = DeQuantStub()

            def forward(self, x):
                x = self.quant(x)
                x = self.sigmoid(x)
                x = self.hardsigmoid(x)
                x = self.tanh(x)
                x = self.dequant(x)
                return x

        m = M().train()
        m.qconfig = default_qat_qconfig
        m = prepare_qat(m)
        for attr in ['sigmoid', 'hardsigmoid', 'tanh']:
            self.assertEqual(type(getattr(m, attr).activation_post_process), FixedQParamsFakeQuantize)
        data = torch.randn(1, 3, 2, 4)
        before_convert = m(data)
        m = convert(m)
        after_convert = m(data)
        self.assertEqual(before_convert, after_convert)
        # make sure activation post process is removed
        for attr in ['sigmoid', 'hardsigmoid', 'tanh']:
            # verify fake quant module is removd
            self.assertFalse(hasattr(getattr(m, attr), 'activation_post_process'))
            # verify that hooks are removed
            self.assertTrue(len(getattr(m, attr)._forward_hooks.items()) == 0)

        # make sure no fake quantize module is inserted for eval mode

        def checkNoFQModule(m):
            for attr in ['sigmoid', 'hardsigmoid', 'tanh']:
                self.assertFalse(hasattr(getattr(m, attr), "activation_post_process"))
                self.assertTrue(len(getattr(m, attr)._forward_hooks.items()) == 0)

        m = M().eval()
        m.qconfig = default_qconfig
        m = prepare(m)
        checkNoFQModule(m)
        m = convert(m)
        checkNoFQModule(m)

    def test_leaky_relu(self):
        data = torch.randn(1, 3, 2, 4)
        self._test_activation_convert_numerics_impl(nn.LeakyReLU, data)

    def test_relu(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.relu(x)
                return x

        m = M().train()
        m.qconfig = default_qconfig
        m = prepare_qat(m)
        # make sure no activation_post_process is inserted for relu
        self.assertFalse(hasattr(m, "activation_post_process"))
        m = convert(m)
        # make sure ReLU module is not changed
        self.assertTrue(type(m.relu), nn.ReLU)

if __name__ == '__main__':
    raise RuntimeError("This test file is not meant to be run directly, use:\n\n"
                       "\tpython test/test_quantization.py TESTNAME\n\n"
                       "instead.")
