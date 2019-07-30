r"""Importing this file includes common utility methods and base clases for
checking quantization api and properties of resulting modules.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import torch
import torch.nn.quantized as nnq
from common_utils import TestCase
from torch.quantization import QuantWrapper, QuantStub, DeQuantStub, default_qconfig

def test_only_eval_fn(model, calib_data):
    r"""
    Default evaluation function takes a torch.utils.data.Dataset or a list of
    input Tensors and run the model on the dataset
    """
    total, correct = 0, 0
    for data, target in calib_data:
        output = model(data)
        _, predicted = torch.max(output, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
    return correct / total

_default_loss_fn = torch.nn.CrossEntropyLoss()
def test_only_train_fn(model, train_data, loss_fn=_default_loss_fn):
    r"""
    Default train function takes a torch.utils.data.Dataset and train the model
    on the dataset
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_loss, correct, total = 0, 0, 0
    for i in range(10):
        model.train()
        for data, target in train_data:
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return train_loss, correct, total


# QuantizationTestCase used as a base class for testing quantization on modules
class QuantizationTestCase(TestCase):
    def setUp(self):
        self.calib_data = [(torch.rand(20, 5, dtype=torch.float), torch.randint(0, 1, (20,), dtype=torch.long)) for _ in range(20)]
        self.train_data = [(torch.rand(20, 5, dtype=torch.float), torch.randint(0, 1, (20,), dtype=torch.long)) for _ in range(20)]
        self.img_data = [(torch.rand(20, 3, 10, 10, dtype=torch.float), torch.randint(0, 1, (20,), dtype=torch.long))
                         for _ in range(20)]

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
        r"""Checks the module or module's leaf descendants
            have observers in preperation for quantization
        """
        if hasattr(module, 'qconfig') and module.qconfig is not None and len(module._modules) == 0:
            self.assertTrue(hasattr(module, 'observer'))
        for child in module.children():
            self.checkObservers(child)

    def checkQuantDequant(self, mod):
        r"""Checks that mod has nn.Quantize and
            nn.DeQuantize submodules inserted
        """
        self.assertEqual(type(mod.quant), nnq.Quantize)
        self.assertEqual(type(mod.dequant), nnq.DeQuantize)

    def checkQuantizedLinear(self, mod):
        r"""Checks that mod has been swapped for an nnq.Linear
            module, the bias is qint32, and that the module
            has Quantize and DeQuantize submodules
        """
        self.assertEqual(type(mod.module), nnq.Linear)
        self.assertEqual(mod.module.bias.dtype, torch.qint32)
        self.checkQuantDequant(mod)

    def checkLinear(self, mod):
        self.assertEqual(type(mod), torch.nn.Linear)


# Below are a series of neural net models to use in testing quantization
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

class InnerModule(torch.nn.Module):
    def __init__(self):
        super(InnerModule, self).__init__()
        self.fc1 = torch.nn.Linear(5, 8).to(dtype=torch.float)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(8, 5).to(dtype=torch.float)

    def forward(self, x):
        return self.relu(self.fc2(self.relu(self.fc1(x))))

class WrappedModel(torch.nn.Module):
    def __init__(self):
        super(WrappedModel, self).__init__()
        self.qconfig = default_qconfig
        self.sub = QuantWrapper(InnerModule())
        self.fc = torch.nn.Linear(5, 5).to(dtype=torch.float)
        # don't quantize this fc
        self.fc.qconfig = None

    def forward(self, x):
        return self.fc(self.sub(x))

class ManualQuantModel(torch.nn.Module):
    r"""A Module with manually inserted `QuantStub` and `DeQuantStub`
    """
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

class ManualLinearQATModel(torch.nn.Module):
    r"""A Module with manually inserted `QuantStub` and `DeQuantStub`
    """
    def __init__(self):
        super(ManualLinearQATModel, self).__init__()
        self.qconfig = default_qconfig
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.fc1 = torch.nn.Linear(5, 5).to(dtype=torch.float)
        self.fc2 = torch.nn.Linear(5, 10).to(dtype=torch.float)

    def forward(self, x):
        x = self.quant(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return self.dequant(x)

class ManualConvLinearQATModel(torch.nn.Module):
    r"""A module with manually inserted `QuantStub` and `DeQuantStub`
    and contains both linear and conv modules
    """
    def __init__(self):
        super(ManualConvLinearQATModel, self).__init__()
        self.qconfig = default_qconfig
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.conv = torch.nn.Conv2d(3, 5, kernel_size=3).to(dtype=torch.float)
        self.fc1 = torch.nn.Linear(320, 10).to(dtype=torch.float)
        self.fc2 = torch.nn.Linear(10, 10).to(dtype=torch.float)

    def forward(self, x):
        x = self.quant(x)
        x = self.conv(x)
        # TODO: we can remove these after view is supported
        x = self.dequant(x)
        x = x.view(-1, 320).contiguous()
        x = self.quant(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return self.dequant(x)
