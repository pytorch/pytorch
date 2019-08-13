from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

r"""Importing this file includes common utility methods and base clases for
checking quantization api and properties of resulting modules.
"""

import hypothesis
import io
import torch
import torch.nn.quantized as nnq
from common_utils import TestCase
from torch.quantization import QuantWrapper, QuantStub, DeQuantStub, \
    default_qconfig, QConfig, default_observer, default_weight_observer, \
    default_qat_qconfig

# Disable deadline testing if this version of hypthesis supports it, otherwise
# just return the original function
def no_deadline(fn):
    try:
        return hypothesis.settings(deadline=None)(fn)
    except hypothesis.errors.InvalidArgument:
        return fn

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
        self.calib_data = [(torch.rand(2, 5, dtype=torch.float), torch.randint(0, 1, (2,), dtype=torch.long)) for _ in range(2)]
        self.train_data = [(torch.rand(2, 5, dtype=torch.float), torch.randint(0, 1, (2,), dtype=torch.long)) for _ in range(2)]
        self.img_data = [(torch.rand(2, 3, 10, 10, dtype=torch.float), torch.randint(0, 1, (2,), dtype=torch.long))
                         for _ in range(2)]

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

    def checkWrappedQuantizedLinear(self, mod):
        r"""Checks that mod has been swapped for an nnq.Linear
            module, the bias is qint32, and that the module
            has Quantize and DeQuantize submodules
        """
        self.assertEqual(type(mod.module), nnq.Linear)
        self.assertEqual(mod.module.bias.dtype, torch.qint32)
        self.checkQuantDequant(mod)

    def checkQuantizedLinear(self, mod):
        self.assertEqual(type(mod), nnq.Linear)
        self.assertEqual(mod.bias.dtype, torch.qint32)

    def checkLinear(self, mod):
        self.assertEqual(type(mod), torch.nn.Linear)

    # calib_data follows the same schema as calib_data for
    # test_only_eval_fn, i.e. (input iterable, output iterable)
    def checkScriptable(self, orig_mod, calib_data, check_save_load=False):
        scripted = torch.jit.script(orig_mod)
        self._checkScriptable(orig_mod, scripted, calib_data, check_save_load)

        # Use first calib_data entry as trace input
        #
        # TODO: Trace checking is blocked on this issue:
        # https://github.com/pytorch/pytorch/issues/23986
        #
        # Once that's resolved we can remove `check_trace=False`
        traced = torch.jit.trace(orig_mod, calib_data[0][0], check_trace=False)
        self._checkScriptable(orig_mod, traced, calib_data, check_save_load)

    # Call this twice: once for a scripted module and once for a traced module
    def _checkScriptable(self, orig_mod, script_mod, calib_data, check_save_load):
        self._checkModuleCorrectnessAgainstOrig(orig_mod, script_mod, calib_data)

        # Test save/load
        buffer = io.BytesIO()
        torch.jit.save(script_mod, buffer)

        buffer.seek(0)
        loaded_mod = torch.jit.load(buffer)

        # Pending __get_state_ and __set_state__ support
        # See tracking task https://github.com/pytorch/pytorch/issues/23984
        if check_save_load:
            self._checkModuleCorrectnessAgainstOrig(orig_mod, loaded_mod, calib_data)

    def _checkModuleCorrectnessAgainstOrig(self, orig_mod, test_mod, calib_data):
        for (inp, _) in calib_data:
            ref_output = orig_mod(inp)
            scripted_output = test_mod(inp)
            self.assertEqual(scripted_output, ref_output)

# Below are a series of neural net models to use in testing quantization
class SingleLayerLinearModel(torch.nn.Module):
    def __init__(self):
        super(SingleLayerLinearModel, self).__init__()
        self.qconfig = default_qconfig
        self.fc1 = QuantWrapper(torch.nn.Linear(5, 5).to(dtype=torch.float))

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

class AnnotatedTwoLayerLinearModel(torch.nn.Module):
    def __init__(self):
        super(AnnotatedTwoLayerLinearModel, self).__init__()
        self.fc1 = torch.nn.Linear(5, 8).to(dtype=torch.float)
        self.fc2 = QuantWrapper(torch.nn.Linear(8, 5).to(dtype=torch.float))
        self.fc2.qconfig = default_qconfig

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

class AnnotatedNestedModel(torch.nn.Module):
    def __init__(self):
        super(AnnotatedNestedModel, self).__init__()
        self.sub1 = LinearReluModel()
        self.sub2 = TwoLayerLinearModel()
        self.fc3 = QuantWrapper(torch.nn.Linear(5, 5).to(dtype=torch.float))
        self.fc3.qconfig = default_qconfig
        self.sub2.fc1 = QuantWrapper(self.sub2.fc1)
        self.sub2.fc1.qconfig = default_qconfig

    def forward(self, x):
        x = self.sub1(x)
        x = self.sub2(x)
        x = self.fc3(x)
        return x

class AnnotatedSubNestedModel(torch.nn.Module):
    def __init__(self):
        super(AnnotatedSubNestedModel, self).__init__()
        self.sub1 = LinearReluModel()
        self.sub2 = QuantWrapper(TwoLayerLinearModel())
        self.fc3 = QuantWrapper(torch.nn.Linear(5, 5).to(dtype=torch.float))
        self.fc3.qconfig = default_qconfig
        self.sub2.qconfig = default_qconfig

    def forward(self, x):
        x = self.sub1(x)
        x = self.sub2(x)
        x = self.fc3(x)
        return x

class AnnotatedCustomConfigNestedModel(torch.nn.Module):
    def __init__(self):
        super(AnnotatedCustomConfigNestedModel, self).__init__()
        self.sub1 = LinearReluModel()
        self.sub2 = TwoLayerLinearModel()
        self.fc3 = QuantWrapper(torch.nn.Linear(5, 5).to(dtype=torch.float))
        self.fc3.qconfig = default_qconfig
        self.sub2.qconfig = default_qconfig

        custom_options = {
            'dtype': torch.quint8,
            'qscheme': torch.per_tensor_affine
        }
        custom_qconfig = QConfig(weight=default_weight_observer(),
                                 activation=default_observer(**custom_options))
        self.sub2.fc1.qconfig = custom_qconfig

        self.sub2.fc1 = QuantWrapper(self.sub2.fc1)
        self.sub2.fc2 = QuantWrapper(self.sub2.fc2)

    def forward(self, x):
        x = self.sub1(x)
        x = self.sub2(x)
        x = self.fc3(x)
        return x

class QuantSubModel(torch.nn.Module):
    def __init__(self):
        super(QuantSubModel, self).__init__()
        self.sub1 = LinearReluModel()
        self.sub2 = QuantWrapper(TwoLayerLinearModel())
        self.sub2.qconfig = default_qconfig
        self.fc3 = torch.nn.Linear(5, 5).to(dtype=torch.float)
        self.fc3.qconfig = default_qconfig

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

class SkipQuantModel(torch.nn.Module):
    r"""We can skip quantization by explicitly
    setting qconfig of a submodule to None
    """
    def __init__(self):
        super(SkipQuantModel, self).__init__()
        self.qconfig = default_qconfig
        self.sub = QuantWrapper(InnerModule())
        self.fc = torch.nn.Linear(5, 5).to(dtype=torch.float)
        # don't quantize this fc
        self.fc.qconfig = None

    def forward(self, x):
        return self.fc(self.sub(x))

class QuantStubModel(torch.nn.Module):
    r"""A Module with manually inserted `QuantStub` and `DeQuantStub`
    """
    def __init__(self):
        super(QuantStubModel, self).__init__()
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
        self.qconfig = default_qat_qconfig
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.fc1 = torch.nn.Linear(5, 1).to(dtype=torch.float)
        self.fc2 = torch.nn.Linear(1, 10).to(dtype=torch.float)

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
        self.qconfig = default_qat_qconfig
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.conv = torch.nn.Conv2d(3, 1, kernel_size=3).to(dtype=torch.float)
        self.fc1 = torch.nn.Linear(64, 10).to(dtype=torch.float)
        self.fc2 = torch.nn.Linear(10, 10).to(dtype=torch.float)

    def forward(self, x):
        x = self.quant(x)
        x = self.conv(x)
        x = x.view(-1, 64).contiguous()
        x = self.fc1(x)
        x = self.fc2(x)
        return self.dequant(x)


class SubModForFusion(torch.nn.Module):
    def __init__(self):
        super(SubModForFusion, self).__init__()
        self.conv = torch.nn.Conv2d(20, 20, 1, bias=None)
        self.bn = torch.nn.BatchNorm2d(20)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class ModForFusion(torch.nn.Module):
    def __init__(self):
        super(ModForFusion, self).__init__()
        self.conv1 = torch.nn.Conv2d(10, 20, 5, bias=None)
        self.bn1 = torch.nn.BatchNorm2d(20)
        self.relu1 = torch.nn.ReLU(inplace=False)
        self.sub1 = SubModForFusion()
        self.sub2 = SubModForFusion()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.sub1(x)
        x = self.sub2(x)
        return x


class DummyObserver(torch.nn.Module):
    def calculate_qparams(self):
        return 1.0, 0

    def forward(self, x):
        return x


class ModForWrapping(torch.nn.Module):
    def __init__(self, quantized=False):
        super(ModForWrapping, self).__init__()
        self.qconfig = default_qconfig
        if quantized:
            self.mycat = nnq.QFunctional()
            self.myadd = nnq.QFunctional()
        else:
            self.mycat = nnq.FloatFunctional()
            self.myadd = nnq.FloatFunctional()
            self.mycat.observer = DummyObserver()
            self.myadd.observer = DummyObserver()

    def forward(self, x):
        y = self.mycat.cat([x, x, x])
        z = self.myadd.add(y, y)
        return z

    @classmethod
    def from_float(cls, mod):
        new_mod = cls(quantized=True)
        new_mod.mycat = new_mod.mycat.from_float(mod.mycat)
        new_mod.myadd = new_mod.myadd.from_float(mod.myadd)
        return new_mod
