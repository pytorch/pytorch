from __future__ import absolute_import, division, print_function, unicode_literals
import torch
import torch.nn.quantized as nnq
import torch.quantization as tq
from torch.quantization import *
from common_utils import TestCase, run_tests

class SingleLayerLinearModel(torch.nn.Module):
    def __init__(self):
        super(SingleLayerLinearModel, self).__init__()
        self.fc1 = torch.nn.Linear(5, 5).to(dtype=torch.float)

    def forward(self, x):
        x = self.fc1(x)
        return x

class TwoLayerLinearModel(torch.nn.Module):
    def __init__(self):
        super(TwoLayerLinearModel, self).__init__()
        self.fc1 = torch.nn.Linear(5, 8).to(dtype=torch.float)
        self.fc2 = torch.nn.Linear(8, 5).to(dtype=torch.float)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class LinearReluModel(torch.nn.Module):
    def __init__(self):
        super(LinearReluModel, self).__init__()
        self.fc = torch.nn.Linear(5, 5).to(dtype=torch.float)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc(x))
        return x

class NestedModel(torch.nn.Module):
    def __init__(self):
        super(NestedModel, self).__init__()
        self.sub1 = LinearReluModel()
        self.sub2 = TwoLayerLinearModel()
        self.fc3 = torch.nn.Linear(5, 5).to(dtype=torch.float)

    def forward(self, x):
        x = self.sub1(x)
        x = self.sub2(x)
        x = self.fc3(x)
        return x

class ManualQuantModel(torch.nn.Module):
    def __init__(self):
        super(ManualQuantModel, self).__init__()
        self.qconfig = default_qconfig
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.fc = torch.nn.Linear(5, 5).to(dtype=torch.float)

    def forward(self, x):
        x = self.quant(x)
        x = self.fc(x)
        return self.dequant(x)

calib_data = [torch.rand(20, 5, dtype=torch.float) for _ in range(20)]

class ModelQuantizeAPITest(TestCase):

    def checkNoPrepModules(self, module):
        r"""Checks the module does not contain child
            modules for quantization prepration, e.g.
            quant, dequant and observer
        """
        self.assertFalse(hasattr(module, 'quant'))
        self.assertFalse(hasattr(module, 'dequant'))

    def checkHasPrepModules(self, module):
        r"""Checks the module contains child
            modules for quantization prepration, e.g.
            quant, dequant and observer
        """
        self.assertTrue(hasattr(module, 'module'))
        self.assertTrue(hasattr(module, 'quant'))
        self.assertTrue(hasattr(module, 'dequant'))

    def checkObservers(self, module):
        if hasattr(module, 'qconfig') and module.qconfig is not None and len(module._modules) == 0:
            self.assertTrue(hasattr(module, 'observer'))
        for child in module.children():
            self.checkObservers(child)

    def checkQuantizedLinear(self, mod):
        self.assertEqual(type(mod.module), nnq.Linear)
        self.assertEqual(type(mod.quant), nnq.Quantize)
        self.assertEqual(type(mod.dequant), nnq.DeQuantize)
        self.assertEqual(mod.module.bias.dtype, torch.qint32)

    def checkLinear(self, mod):
        self.assertEqual(type(mod), torch.nn.Linear)


    def test_quantize_single_layer(self):
        '''
            Quantize SingleLayerLinearModel which has one Linear module, make sure it is swapped
            to nnq.Linear which is the quantized version of the module
        '''
        model = SingleLayerLinearModel()
        qconfig_dict = {
            # Top level qconfig
            '': default_qconfig
        }
        model = tq.quantize(model, qconfig_dict, default_eval_fn, [calib_data])

        self.checkNoPrepModules(model)
        self.checkHasPrepModules(model.fc1)
        self.checkQuantizedLinear(model.fc1)

        default_eval_fn(model, calib_data)

    def test_quantize_two_layers(self):
        '''
            TwoLayerLinearModel has two Linear modules but we only quantize the second one
            `fc2`, and `fc1`is not quantized
        '''
        myModel = TwoLayerLinearModel()
        qconfig_dict = {
            # Only quantize fc2
            'fc2': default_qconfig
        }
        myModel = tq.quantize(myModel, qconfig_dict, default_eval_fn, [calib_data])

        self.checkNoPrepModules(myModel)
        self.checkHasPrepModules(myModel.fc2)
        # check only fc2 is quantized
        self.assertEqual(type(myModel.fc1), torch.nn.Linear)
        self.checkQuantizedLinear(myModel.fc2)

        default_eval_fn(myModel, calib_data)


    def test_steps_single_layer(self):
        myModel = SingleLayerLinearModel()
        qconfig_dict = {
            '': default_qconfig
        }
        myModel = tq.prepare(myModel, qconfig_dict)
        # Check if observers and quant/dequant nodes are inserted
        self.checkNoPrepModules(myModel)
        self.checkHasPrepModules(myModel.fc1)
        self.checkObservers(myModel)

        default_eval_fn(myModel, calib_data)
        tq.convert(myModel)

        self.checkNoPrepModules(myModel)
        self.checkHasPrepModules(myModel.fc1)
        self.checkQuantizedLinear(myModel.fc1)

        default_eval_fn(myModel, calib_data)

    def test_steps_two_layers(self):
        myModel = TwoLayerLinearModel()
        qconfig_dict = {
            'fc2': default_qconfig
        }
        myModel = tq.prepare(myModel, qconfig_dict)
        self.checkNoPrepModules(myModel)
        self.checkObservers(myModel)
        self.checkNoPrepModules(myModel.fc1)
        self.checkHasPrepModules(myModel.fc2)

        default_eval_fn(myModel, calib_data)
        tq.convert(myModel)

        self.checkNoPrepModules(myModel)
        self.checkNoPrepModules(myModel.fc1)
        self.checkHasPrepModules(myModel.fc2)
        self.assertEqual(type(myModel.fc1), torch.nn.Linear)
        self.checkQuantizedLinear(myModel.fc2)

        default_eval_fn(myModel, calib_data)

    def test_nested1(self):
        myModel = NestedModel()
        qconfig_dict = {
            'fc3': default_qconfig,
            'sub2.fc1': default_qconfig
        }

        def checkPrepModules(model, before_calib=False):
            if before_calib:
                self.checkObservers(model)
            self.checkNoPrepModules(model)
            self.checkNoPrepModules(model.sub1)
            self.checkNoPrepModules(model.sub1.fc)
            self.checkNoPrepModules(model.sub1.relu)
            self.checkNoPrepModules(model.sub2)
            self.checkHasPrepModules(model.sub2.fc1)
            self.checkNoPrepModules(model.sub2.fc2)
            self.checkHasPrepModules(model.fc3)

        myModel = tq.prepare(myModel, qconfig_dict)

        checkPrepModules(myModel, True)

        default_eval_fn(myModel, calib_data)
        tq.convert(myModel)

        checkPrepModules(myModel)
        self.checkLinear(myModel.sub1.fc)
        self.checkQuantizedLinear(myModel.fc3)
        self.checkQuantizedLinear(myModel.sub2.fc1)
        self.checkLinear(myModel.sub2.fc2)

        default_eval_fn(myModel, calib_data)

    def test_nested2(self):
        r"""If we add quant dequant for the whole module then we can eliminate
        extra quant dequant between modules. This is what current implementation
        supports.
        However, a more complete support would make test_nested3 work as well,
        but still keep the current behavior. That is to say, when user provides
        configurations for finer grained modules, we operate on that level, e.g.
        if user have a key 'sub2.fc2', then we don't treat 'sub2' as a terminal
        module, instead we'll operate on the same level as 'sub2.fc2'.
        """
        myModel = NestedModel()
        qconfig_dict = {
            'fc3': default_qconfig,
            'sub2': default_qconfig
        }
        myModel = tq.prepare(myModel, qconfig_dict)

        def checkPrepModules(model, before_calib=False):
            if before_calib:
                self.checkObservers(model)
            self.checkNoPrepModules(model)
            self.checkNoPrepModules(model.sub1)
            self.checkNoPrepModules(model.sub1.fc)
            self.checkNoPrepModules(model.sub1.relu)
            self.checkNoPrepModules(model.sub2)
            self.checkHasPrepModules(model.sub2.fc1)
            self.checkHasPrepModules(model.sub2.fc2)
            self.checkHasPrepModules(model.fc3)

        checkPrepModules(myModel, True)

        default_eval_fn(myModel, calib_data)
        tq.convert(myModel)

        checkPrepModules(myModel)
        self.checkLinear(myModel.sub1.fc)
        self.assertEqual(type(myModel.sub1.relu), torch.nn.ReLU)
        self.checkQuantizedLinear(myModel.sub2.fc1)
        self.checkQuantizedLinear(myModel.sub2.fc2)
        self.checkQuantizedLinear(myModel.fc3)

        default_eval_fn(myModel, calib_data)

    def test_nested3(self):
        """
        More complicated nested test case with fallbacks
        this does not work with current implementation
        """
        myModel = NestedModel()
        custum_options = {
            'dtype': torch.quint8,
            'qscheme': torch.per_tensor_affine
        }
        custom_qconfig = QConfig(weight=default_weight_observer(),
                                 activation=default_observer(**custum_options))
        qconfig_dict = {
            'fc3': default_qconfig,
            'sub2': default_qconfig,
            'sub2.fc1': custom_qconfig
        }
        myModel = tq.prepare(myModel, qconfig_dict)

        def checkPrepModules(model, before_calib=False):
            if before_calib:
                self.checkObservers(model)
            self.checkNoPrepModules(model)
            self.checkNoPrepModules(model.sub1)
            self.checkNoPrepModules(model.sub1.fc)
            self.checkNoPrepModules(model.sub1.relu)
            self.checkNoPrepModules(model.sub2)
            self.checkHasPrepModules(model.sub2.fc1)
            self.checkHasPrepModules(model.sub2.fc2)
            self.checkHasPrepModules(model.fc3)

        checkPrepModules(myModel, True)

        default_eval_fn(myModel, calib_data)
        tq.convert(myModel)

        checkPrepModules(myModel)
        self.checkQuantizedLinear(myModel.sub2.fc1)
        self.checkQuantizedLinear(myModel.sub2.fc2)
        self.checkQuantizedLinear(myModel.fc3)

        default_eval_fn(myModel, calib_data)

    def test_manual(self):
        model = ManualQuantModel()
        # propagate the qconfig of parents to children
        tq.propagate_qconfig(model)
        tq.add_observer(model)
        self.checkObservers(model)

        default_eval_fn(model, calib_data)
        tq.convert(model)
        self.assertEqual(type(model.fc), nnq.Linear)
        default_eval_fn(model, calib_data)


if __name__ == '__main__':
    run_tests()
