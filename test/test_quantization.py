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
        self.fc1 = torch.nn.Linear(5, 5).to(dtype=torch.float)
        self.fc2 = torch.nn.Linear(5, 5).to(dtype=torch.float)

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

calib_data = [torch.rand(20, 5, dtype=torch.float) for _ in range(20)]

class ModelQuantizeAPITest(TestCase):

    def checkNoPrepModules(self, module, has_observer=False):
        r"""Checks the module does not contain child
            modules for quantization prepration, e.g.
            quant, dequant and observer
        """
        self.assertFalse(hasattr(module, 'quant'))
        self.assertFalse(hasattr(module, 'dequant'))
        self.assertEqual(hasattr(module, 'observer'), has_observer)

    def checkHasPrepModules(self, module, has_observer=False):
        r"""Checks the module contains child
            modules for quantization prepration, e.g.
            quant, dequant and observer
        """
        self.assertTrue(hasattr(module, 'quant'))
        self.assertTrue(hasattr(module, 'dequant'))
        self.assertEqual(hasattr(module, 'observer'), has_observer)


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
        # self.checkHasPrepModules(model)
        # self.checkNoPrepModules(model.module.fc1)
        # self.assertEqual(type(model.fc1), nnq.Linear)
        # self.assertEqual(type(model.quant), nnq.Quantize)
        # self.assertEqual(type(model.dequant), nnq.DeQuantize)
        # self.assertEqual(model.fc1.bias.dtype, torch.qint32)
        print(model)
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
        # self.checkNoPrepModules(myModel)
        # self.checkHasPrepModules(myModel.fc2)
        # self.assertEqual(type(myModel.fc1), torch.nn.Linear)
        # check only fc2 is quantized
        # self.assertEqual(type(myModel.fc2), nnq.Linear)
        # self.assertEqual(type(myModel.fc2.quant), nnq.Quantize)
        # self.assertEqual(type(myModel.fc2.dequant), nnq.DeQuantize)
        print(myModel)
        default_eval_fn(myModel, calib_data)


    def test_steps_single_layer(self):
        myModel = SingleLayerLinearModel()
        qconfig_dict = {
            '': default_qconfig
        }
        myModel = tq.prepare(myModel, qconfig_dict)
        # Check if observers and quant/dequant nodes are inserted
        # self.checkHasPrepModules(myModel)
        # self.assertTrue(hasattr(myModel.fc1, 'observer'))
        default_eval_fn(myModel, calib_data)

        tq.convert_to_quantized(myModel)
        # self.checkHasPrepModules(myModel)
        # self.checkNoPrepModules(myModel.fc1)
        # # Check if modules are swapped correctly
        # self.assertEqual(type(myModel.fc1), nnq.Linear)
        # self.assertEqual(type(myModel.quant), nnq.Quantize)
        # self.assertEqual(type(myModel.dequant), nnq.DeQuantize)
        # self.assertEqual(myModel.fc1.bias.dtype, torch.qint32)

        default_eval_fn(myModel, calib_data)

    def test_steps_two_layers(self):
        myModel = TwoLayerLinearModel()
        qconfig_dict = {
            'fc2': default_qconfig
        }
        myModel = tq.prepare(myModel, qconfig_dict)
        # self.checkNoPrepModules(myModel)
        # self.checkNoPrepModules(myModel.fc1)
        # self.checkHasPrepModules(myModel.fc2, True)
        default_eval_fn(myModel, calib_data)

        tq.convert_to_quantized(myModel)
        # self.checkNoPrepModules(myModel)
        # self.checkNoPrepModules(myModel.fc1)
        # self.checkHasPrepModules(myModel.fc2, False)
        # self.assertEqual(type(myModel.fc1), torch.nn.Linear)
        # self.assertEqual(type(myModel.fc2), nnq.Linear)
        # self.assertEqual(type(myModel.fc2.quant), nnq.Quantize)
        # self.assertEqual(type(myModel.fc2.dequant), nnq.DeQuantize)
        default_eval_fn(myModel, calib_data)

    def test_nested1(self):
        myModel = NestedModel()
        qconfig_dict = {
            'fc3': default_qconfig,
            'sub2.fc2': default_qconfig
        }
        myModel = tq.prepare(myModel, qconfig_dict)
        # self.checkNoPrepModules(myModel)
        # self.checkNoPrepModules(myModel.sub1)
        # self.checkNoPrepModules(myModel.sub1.fc)
        # self.checkNoPrepModules(myModel.sub1.relu)
        # self.checkNoPrepModules(myModel.sub2)
        # self.checkNoPrepModules(myModel.sub2.fc1)
        # self.checkHasPrepModules(myModel.sub2.fc2, True)
        # self.checkHasPrepModules(myModel.fc3, True)

        default_eval_fn(myModel, calib_data)

        tq.convert_to_quantized(myModel)
        print(myModel)
        # self.checkNoPrepModules(myModel)
        # self.checkNoPrepModules(myModel.sub1)
        # self.checkNoPrepModules(myModel.sub1.fc)
        # self.checkNoPrepModules(myModel.sub1.relu)
        # self.checkNoPrepModules(myModel.sub2)
        # self.checkNoPrepModules(myModel.sub2.fc1)
        # self.checkHasPrepModules(myModel.sub2.fc2)
        # self.checkHasPrepModules(myModel.fc3)

        # self.assertEqual(type(myModel.sub1.fc), torch.nn.Linear)
        # self.assertEqual(type(myModel.sub2.fc1), torch.nn.Linear)
        # self.assertEqual(type(myModel.sub2.fc2), nnq.Linear)
        # self.assertEqual(type(myModel.sub2.fc2.quant), nnq.Quantize)
        # self.assertEqual(type(myModel.sub2.fc2.dequant), nnq.DeQuantize)
        # self.assertEqual(type(myModel.fc3), nnq.Linear)
        # self.assertEqual(type(myModel.fc3.quant), nnq.Quantize)
        # self.assertEqual(type(myModel.fc3.dequant), nnq.DeQuantize)
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
        # self.checkNoPrepModules(myModel)
        # self.checkNoPrepModules(myModel.sub1)
        # self.checkNoPrepModules(myModel.sub1.fc)
        # self.checkNoPrepModules(myModel.sub1.relu)
        # self.checkHasPrepModules(myModel.sub2)
        # self.checkNoPrepModules(myModel.sub2.fc1, True)
        # self.checkNoPrepModules(myModel.sub2.fc2, True)
        # self.checkHasPrepModules(myModel.fc3, True)
        default_eval_fn(myModel, calib_data)

        tq.convert_to_quantized(myModel)
        print('nested:', myModel)
        # self.checkNoPrepModules(myModel)
        # self.checkNoPrepModules(myModel.sub1)
        # self.checkNoPrepModules(myModel.sub1.fc)
        # self.checkNoPrepModules(myModel.sub1.relu)
        # self.checkHasPrepModules(myModel.sub2)
        # self.checkNoPrepModules(myModel.sub2.fc1)
        # self.checkNoPrepModules(myModel.sub2.fc2)
        # self.checkHasPrepModules(myModel.fc3)
        # self.assertEqual(type(myModel.sub1.fc), torch.nn.Linear)
        # self.assertEqual(type(myModel.sub1.relu), torch.nn.ReLU)
        # self.assertEqual(type(myModel.sub2.quant), nnq.Quantize)
        # self.assertEqual(type(myModel.sub2.fc1), nnq.Linear)
        # self.assertEqual(type(myModel.sub2.fc2), nnq.Linear)
        # self.assertEqual(type(myModel.sub2.dequant), nnq.DeQuantize)
        # self.assertEqual(type(myModel.fc3.quant), nnq.Quantize)
        # self.assertEqual(type(myModel.fc3), nnq.Linear)
        # self.assertEqual(type(myModel.fc3.dequant), nnq.DeQuantize)
        default_eval_fn(myModel, calib_data)

    # TODO: Add the support in a separate PR
    # def test_nested3(self):
    #     """
    #     More complicated nested test case with fallbacks
    #     this does not work with current implementation
    #     """
    #     myModel = NestedModel()
    #     custome_options1 = {
    #       'dtype': torch.quint8,
    #       'qscheme': torch.per_tensor_affine
    #     }
    #     custom_qconfig = QConfig(weight=observer(WeightObserver, default_options),
    #                              activation=observer(Observer, custome_options1))
    #     qconfig_dict = {
    #         'fc3': default_qconfig,
    #         'sub2': default_qconfig,
    #         'sub2.fc1': custom_qconfig
    #     }
    #     calib_data = torch.rand(20, 5, dtype=torch.float)
    #     myModel = tq.prepare(myModel, qconfig_dict)
    #
    #     default_eval_fn(myModel, calib_data)
    #     tq.convert_to_quantized(myModel)
    #     myModel(calib_data)



if __name__ == '__main__':
    run_tests()
