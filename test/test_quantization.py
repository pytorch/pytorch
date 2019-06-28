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

# TODO: change assertEqual(..., True) to assertTrue

class ModelQuantizeAPITest(TestCase):

    def test_quantize1(self):
        '''
            Quantize SingleLayerLinearModel which has one Linear module, make sure it is swapped
            to nnq.Linear which is the quantized version of the module
        '''
        model = SingleLayerLinearModel()
        qConfigDict = {
            # Top level qconfig
            '': default_qconfig
        }
        tq.quantize(model, qConfigDict, default_eval_fn, [calib_data])
        print('running model:', model)
        self.assertTrue(hasattr(model, 'quant'))
        self.assertEqual(hasattr(model, 'dequant'), True)
        self.assertEqual(hasattr(model, 'observer'), False)
        # since fc1 is not quantized
        # self.assertEqual(hasattr(model.module, 'module'), False)
        self.assertEqual(type(model.fc1), nnq.Linear)
        self.assertEqual(type(model.quant), nnq.Quantize)
        self.assertEqual(type(model.dequant), nnq.DeQuantize)
        self.assertEqual(model.fc1.bias.dtype, torch.qint32)
        default_eval_fn(model, calib_data)

    def test_quantize2(self):
        '''
            TwoLayerLinearModel has two Linear modules but we only quantize the second one
            `fc2`, and `fc1`is not quantized
        '''
        myModel = TwoLayerLinearModel()
        qConfigDict = {
            # Only quantize fc2
            'fc2': default_qconfig
        }
        tq.quantize(myModel, qConfigDict, default_eval_fn, [calib_data])
        self.assertEqual(hasattr(myModel, 'quant'), False)
        self.assertEqual(hasattr(myModel, 'observer'), False)
        self.assertEqual(hasattr(myModel.fc2, 'quant'), True)
        self.assertEqual(hasattr(myModel.fc2, 'dequant'), True)
        self.assertEqual(hasattr(myModel.fc2, 'observer'), False)
        self.assertEqual(type(myModel.fc1), torch.nn.Linear)
        self.assertEqual(type(myModel.fc2), nnq.Linear)
        default_eval_fn(myModel, calib_data)


    def test_steps1(self):
        myModel = SingleLayerLinearModel()
        qConfigDict = {
            '': default_qconfig
        }
        tq.prepare(myModel, qConfigDict)
        # Check if observers and quant/dequant nodes are inserted
        self.assertEqual(hasattr(myModel, 'quant'), True)
        self.assertEqual(hasattr(myModel, 'dequant'), True)
        self.assertEqual(hasattr(myModel.fc1, 'observer'), True)
        default_eval_fn(myModel, calib_data)

        tq.convert_to_quantized(myModel)
        # Check if modules are swapped correctly
        self.assertEqual(type(myModel.quant), nnq.Quantize)
        self.assertEqual(hasattr(myModel.fc1, 'quant'), False)
        self.assertEqual(hasattr(myModel.fc1, 'observer'), False)
        self.assertEqual(type(myModel.fc1), nnq.Linear)
        self.assertEqual(myModel.fc1.bias.dtype, torch.qint32)
        default_eval_fn(myModel, calib_data)

    def test_steps2(self):
        myModel = TwoLayerLinearModel()
        qConfigDict = {
            'fc2': default_qconfig
        }
        tq.prepare(myModel, qConfigDict)
        self.assertEqual(hasattr(myModel, 'quant'), False)
        self.assertEqual(hasattr(myModel.fc1, 'quant'), False)
        self.assertEqual(hasattr(myModel.fc2, 'quant'), True)
        self.assertEqual(hasattr(myModel.fc2, 'observer'), True)
        default_eval_fn(myModel, calib_data)

        tq.convert_to_quantized(myModel)
        self.assertEqual(hasattr(myModel, 'quant'), False)
        self.assertEqual(hasattr(myModel.fc2, 'quant'), True)
        self.assertEqual(hasattr(myModel.fc2, 'observer'), False)
        self.assertEqual(type(myModel.fc1), torch.nn.Linear)
        self.assertEqual(type(myModel.fc2), nnq.Linear)
        default_eval_fn(myModel, calib_data)

    def test_nested1(self):
        myModel = NestedModel()
        qConfigDict = {
            'fc3': default_qconfig,
            'sub2.fc2': default_qconfig
        }
        tq.prepare(myModel, qConfigDict)
        self.assertEqual(hasattr(myModel, 'quant'), False)
        self.assertEqual(hasattr(myModel.sub1, 'quant'), False)
        self.assertEqual(hasattr(myModel.sub1.fc, 'quant'), False)
        self.assertEqual(hasattr(myModel.sub2, 'quant'), False)
        self.assertEqual(hasattr(myModel.sub2.fc1, 'quant'), False)
        self.assertEqual(hasattr(myModel.sub2.fc2, 'quant'), True)
        self.assertEqual(hasattr(myModel.sub2.fc2, 'observer'), True)
        self.assertEqual(hasattr(myModel.fc3, 'quant'), True)
        self.assertEqual(hasattr(myModel.fc3, 'observer'), True)

        default_eval_fn(myModel, calib_data)

        tq.convert_to_quantized(myModel)
        self.assertEqual(hasattr(myModel, 'quant'), False)
        self.assertEqual(hasattr(myModel.sub1, 'quant'), False)
        self.assertEqual(hasattr(myModel.sub1.fc, 'quant'), False)
        self.assertEqual(hasattr(myModel.sub2, 'quant'), False)
        self.assertEqual(hasattr(myModel.sub2.fc1, 'quant'), False)
        self.assertEqual(hasattr(myModel.sub2.fc2, 'quant'), True)
        self.assertEqual(hasattr(myModel.sub2.fc2, 'observer'), False)
        self.assertEqual(hasattr(myModel.fc3, 'quant'), True)
        self.assertEqual(hasattr(myModel.fc3, 'observer'), False)
        self.assertEqual(type(myModel.sub1.fc), torch.nn.Linear)
        self.assertEqual(type(myModel.sub2.fc1), torch.nn.Linear)
        self.assertEqual(type(myModel.sub2.fc2), nnq.Linear)
        self.assertEqual(type(myModel.fc3), nnq.Linear)
        default_eval_fn(myModel, calib_data)

    def test_nested2(self):
        """
        If we add quant dequant for the whole module then we can eliminate
        extra quant dequant between modules. This is what current implementation
        supports.
        However, a more complete support would make test_nested3 work as well,
        but still keep the current behavior. That is to say, when user provides
        configurations for finer grained modules, we operate on that level, e.g.
        if user have a key 'sub2.fc2', then we don't treat 'sub2' as a terminal
        module, instead we'll operate on the same level as 'sub2.fc2'.
        """
        myModel = NestedModel()
        qConfigDict = {
            'fc3': default_qconfig,
            'sub2': default_qconfig
        }
        tq.prepare(myModel, qConfigDict)
        self.assertEqual(hasattr(myModel, 'quant'), False)
        self.assertEqual(hasattr(myModel.sub1, 'quant'), False)
        self.assertEqual(hasattr(myModel.sub1.fc, 'quant'), False)
        self.assertEqual(hasattr(myModel.sub2, 'quant'), True)
        self.assertEqual(hasattr(myModel.sub2, 'observer'), False)
        self.assertEqual(hasattr(myModel.sub2.fc1, 'observer'), True)
        self.assertEqual(hasattr(myModel.sub2.fc2, 'observer'), True)
        self.assertEqual(hasattr(myModel.sub2.fc1, 'quant'), False)
        self.assertEqual(hasattr(myModel.sub2.fc2, 'quant'), False)
        self.assertEqual(hasattr(myModel.fc3, 'quant'), True)
        self.assertEqual(hasattr(myModel.fc3, 'observer'), True)

        default_eval_fn(myModel, calib_data)

        tq.convert_to_quantized(myModel)
        print('nested:', myModel)
        self.assertEqual(hasattr(myModel, 'quant'), False)
        self.assertEqual(hasattr(myModel.sub1, 'quant'), False)
        self.assertEqual(hasattr(myModel.sub1.fc, 'quant'), False)
        self.assertEqual(hasattr(myModel.sub2, 'quant'), True)
        self.assertEqual(hasattr(myModel.sub2, 'observer'), False)
        self.assertEqual(hasattr(myModel.sub2.fc1, 'observer'), False)
        self.assertEqual(hasattr(myModel.sub2.fc2, 'observer'), False)
        self.assertEqual(hasattr(myModel.sub2.fc1, 'quant'), False)
        self.assertEqual(hasattr(myModel.sub2.fc2, 'quant'), False)
        self.assertEqual(hasattr(myModel.fc3, 'quant'), True)
        self.assertEqual(hasattr(myModel.fc3, 'observer'), False)
        self.assertEqual(type(myModel.sub1.fc), torch.nn.Linear)
        self.assertEqual(type(myModel.sub2.fc1), nnq.Linear)
        self.assertEqual(type(myModel.sub2.fc2), nnq.Linear)
        self.assertEqual(type(myModel.fc3), nnq.Linear)
        default_eval_fn(myModel, calib_data)

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
    #     qConfigDict = {
    #         'fc3': default_qconfig,
    #         'sub2': default_qconfig,
    #         'sub2.fc1': custom_qconfig
    #     }
    #     calib_data = torch.rand(20, 5, dtype=torch.float)
    #     myModel = tq.prepare(myModel, qConfigDict)
    #
    #     default_eval_fn(myModel, calib_data)
    #     tq.convert_to_quantized(myModel)
    #     myModel(calib_data)



if __name__ == '__main__':
    run_tests()
