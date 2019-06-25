from __future__ import absolute_import, division, print_function, unicode_literals
import torch
import torch.nn.quantized as nnq
import torch.quantization as tq
from torch.quantization import \
    observer, Observer, WeightObserver, QConfig, QuantWrapper
from common_utils import TestCase, run_tests

class SingleLinearModel(torch.nn.Module):
    def __init__(self):
        super(SingleLinearModel, self).__init__()
        self.fc1 = torch.nn.Linear(5, 5).to(dtype=torch.float)

    def forward(self, x):
        x = self.fc1(x)
        return x

class Model2(torch.nn.Module):
    def __init__(self):
        super(Model2, self).__init__()
        self.fc1 = torch.nn.Linear(5, 5).to(dtype=torch.float)
        self.fc2 = torch.nn.Linear(5, 5).to(dtype=torch.float)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class Model3(torch.nn.Module):
    def __init__(self):
        super(Model3, self).__init__()
        self.fc = torch.nn.Linear(5, 5).to(dtype=torch.float)

    def forward(self, x):
        x = self.fc(x)
        return x

class NestedModel(torch.nn.Module):
    def __init__(self):
        super(NestedModel, self).__init__()
        self.sub1 = Model3()
        self.sub2 = Model2()
        self.fc3 = torch.nn.Linear(5, 5).to(dtype=torch.float)

    def forward(self, x):
        x = self.sub1(x)
        x = self.sub2(x)
        x = self.fc3(x)
        return x

def eval_fn(model, data):
    model(data)

default_options = {
    'dtype': torch.qint8,
    'qscheme': torch.per_tensor_affine
}

class ModelQuantizeAPITest(TestCase):
    def test_quantize1(self):
        '''
            Quantize SingleLinearModel which has one Linear module, make sure it is swapped
            to nnq.Linear which is the quantized version of the module
        '''
        calib_data = torch.rand(20, 5, dtype=torch.float)
        model = SingleLinearModel()
        qConfigDict = {
            # Top level qconfig
            '': QConfig(weight=observer(WeightObserver, default_options),
                        activation=observer(Observer, default_options))
        }
        eval_args = [calib_data]
        model = tq.quantize(model, qConfigDict, eval_fn, *eval_args)
        self.assertTrue(type(model) == QuantWrapper)
        print(model)
        self.assertTrue(hasattr(model, 'quant'))
        self.assertEqual(hasattr(model, 'dequant'), True)
        self.assertEqual(hasattr(model, 'observer'), False)
        # since fc1 is not quantized
        self.assertEqual(hasattr(model.module, 'module'), False)
        self.assertEqual(type(model.module.fc1), nnq.Linear)
        self.assertEqual(type(model.quant), nnq.Quantize)
        self.assertEqual(type(model.dequant), nnq.DeQuantize)
        self.assertEqual(model.module.fc1.bias.dtype, torch.qint32)
        model(calib_data)

    def test_quantize2(self):
        '''
            Model2 has two Linear modules but we only quantize the second one
            `fc2`, and `fc1`is not quantized
        '''
        calib_data = torch.rand(20, 5, dtype=torch.float)
        myModel = Model2()
        qConfigDict = {
            # Only quantize fc2
            'fc2': QConfig(weight=observer(WeightObserver, default_options),
                           activation=observer(Observer, default_options))
        }
        eval_args = [calib_data]
        myModel = tq.quantize(myModel, qConfigDict, eval_fn, *eval_args)
        # self.assertEqual(hasattr(myModel, 'quant'), False)
        # self.assertEqual(hasattr(myModel, 'observer'), False)
        # self.assertEqual(hasattr(myModel.fc2, 'quant'), True)
        # self.assertEqual(hasattr(myModel.fc2, 'dequant'), True)
        # self.assertEqual(hasattr(myModel.fc2, 'observer'), False)
        # self.assertEqual(type(myModel.fc1), torch.nn.Linear)
        # self.assertEqual(type(myModel.fc2), nnq.Linear)
        myModel(calib_data)

    def test_steps1(self):
        calib_data = torch.rand(20, 5, dtype=torch.float)
        myModel = SingleLinearModel()
        qConfigDict = {
            '': QConfig(weight=observer(WeightObserver, default_options),
                        activation=observer(Observer, default_options))
        }
        eval_args = [calib_data]
        myModel = tq.prepare(myModel, qConfigDict)
        # Check if observers and quant/dequant nodes are inserted
        # self.assertEqual(hasattr(myModel, 'quant'), True)
        # self.assertEqual(hasattr(myModel, 'dequant'), True)
        # self.assertEqual(hasattr(myModel.fc1, 'observer'), True)

        tq.convert_to_quantized(myModel)
        # Check if modules are swapped correctly
        # self.assertEqual(type(myModel.quant), nnq.Quantize)
        # self.assertEqual(hasattr(myModel.fc1, 'quant'), False)
        # self.assertEqual(hasattr(myModel.fc1, 'observer'), False)
        # self.assertEqual(type(myModel.fc1), nnq.Linear)
        # self.assertEqual(myModel.fc1.bias.dtype, torch.qint32)
        myModel(calib_data)

    def test_steps2(self):
        calib_data = torch.rand(20, 5, dtype=torch.float)
        myModel = Model2()
        qConfigDict = {
            'fc2': QConfig(weight=observer(WeightObserver, default_options),
                           activation=observer(Observer, default_options))
        }
        eval_args = [calib_data]
        myModel = tq.prepare(myModel, qConfigDict)
        # self.assertEqual(hasattr(myModel, 'quant'), False)
        # self.assertEqual(hasattr(myModel.fc1, 'quant'), False)
        # self.assertEqual(hasattr(myModel.fc2, 'quant'), True)
        # self.assertEqual(hasattr(myModel.fc2, 'observer'), True)

        tq.convert_to_quantized(myModel)
        # self.assertEqual(hasattr(myModel, 'quant'), False)
        # self.assertEqual(hasattr(myModel.fc2, 'quant'), True)
        # self.assertEqual(hasattr(myModel.fc2, 'observer'), False)
        # self.assertEqual(type(myModel.fc1), torch.nn.Linear)
        # self.assertEqual(type(myModel.fc2), nnq.Linear)
        myModel(calib_data)

    def test_nested(self):
        myModel = NestedModel()
        config = QConfig(weight=observer(WeightObserver, default_options),
                         activation=observer(Observer, default_options))
        qConfigDict = {
            'fc3': config,
            'sub2.fc2': config
        }
        calib_data = torch.rand(20, 5, dtype=torch.float)
        eval_args = [calib_data]
        myModel = tq.prepare(myModel, qConfigDict)
        # self.assertEqual(hasattr(myModel, 'quant'), False)
        # self.assertEqual(hasattr(myModel.sub1, 'quant'), False)
        # self.assertEqual(hasattr(myModel.sub1.fc, 'quant'), False)
        # self.assertEqual(hasattr(myModel.sub2, 'quant'), False)
        # self.assertEqual(hasattr(myModel.sub2.fc1, 'quant'), False)
        # self.assertEqual(hasattr(myModel.sub2.fc2, 'quant'), True)
        # self.assertEqual(hasattr(myModel.sub2.fc2, 'observer'), True)
        # self.assertEqual(hasattr(myModel.fc3, 'quant'), True)
        # self.assertEqual(hasattr(myModel.fc3, 'observer'), True)

        eval_fn(myModel, *eval_args)

        tq.convert_to_quantized(myModel)
        # self.assertEqual(hasattr(myModel, 'quant'), False)
        # self.assertEqual(hasattr(myModel.sub1, 'quant'), False)
        # self.assertEqual(hasattr(myModel.sub1.fc, 'quant'), False)
        # self.assertEqual(hasattr(myModel.sub2, 'quant'), False)
        # self.assertEqual(hasattr(myModel.sub2.fc1, 'quant'), False)
        # self.assertEqual(hasattr(myModel.sub2.fc2, 'quant'), True)
        # self.assertEqual(hasattr(myModel.sub2.fc2, 'observer'), False)
        # self.assertEqual(hasattr(myModel.fc3, 'quant'), True)
        # self.assertEqual(hasattr(myModel.fc3, 'observer'), False)
        # self.assertEqual(type(myModel.sub1.fc), torch.nn.Linear)
        # self.assertEqual(type(myModel.sub2.fc1), torch.nn.Linear)
        # self.assertEqual(type(myModel.sub2.fc2), nnq.Linear)
        # self.assertEqual(type(myModel.fc3), nnq.Linear)
        myModel(calib_data)



if __name__ == '__main__':
    run_tests()
