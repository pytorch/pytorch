from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

r"""Importing this file includes common utility methods and base clases for
checking quantization api and properties of resulting modules.
"""

import io
import functools
import torch
import torch.nn as nn
import torch.nn.quantized as nnq
import torch.nn.quantized.dynamic as nnqd
from torch.testing._internal.common_utils import TestCase
from torch.quantization import QuantWrapper, QuantStub, DeQuantStub, \
    default_qconfig, default_dynamic_qconfig, default_per_channel_qconfig, QConfig, default_observer, default_weight_observer, \
    propagate_qconfig_, convert, get_default_qconfig, quantize_dynamic_script, quantize_script
from torch.quantization.default_mappings import DEFAULT_DYNAMIC_MODULE_MAPPING
import unittest
from torch.testing import FileCheck

def test_only_eval_fn(model, calib_data):
    r"""
    Default evaluation function takes a torch.utils.data.Dataset or a list of
    input Tensors and run the model on the dataset
    """
    total, correct = 0, 0
    for *data, target in calib_data:
        output = model(*data)
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

def convert_dynamic(module):
    convert(module, DEFAULT_DYNAMIC_MODULE_MAPPING, inplace=True)

def prepare_dynamic(model, qconfig_dict=None):
    propagate_qconfig_(model, qconfig_dict)

def _make_conv_test_input(
    batch_size, in_channels_per_group, input_feature_map_size,
    out_channels_per_group, groups, kernel_size, X_scale, X_zero_point, W_scale,
    W_zero_point, use_bias, use_channelwise,
):
    in_channels = in_channels_per_group * groups
    out_channels = out_channels_per_group * groups

    (X_value_min, X_value_max) = (0, 4)
    X_init = torch.randint(
        X_value_min, X_value_max,
        (batch_size, in_channels,) + input_feature_map_size)
    X = X_scale * (X_init - X_zero_point).float()
    X_q = torch.quantize_per_tensor(
        X, scale=X_scale, zero_point=X_zero_point, dtype=torch.quint8)

    W_scale = W_scale * out_channels
    W_zero_point = W_zero_point * out_channels
    # Resize W_scale and W_zero_points arrays equal to out_channels
    W_scale = W_scale[:out_channels]
    W_zero_point = W_zero_point[:out_channels]
    # For testing, we use small values for weights and for activations so that
    # no overflow occurs in vpmaddubsw instruction. If the overflow occurs in
    # qconv implementation and if there is no overflow.
    # In reference we can't exactly match the results with reference.
    # Please see the comment in qconv implementation file
    #   aten/src/ATen/native/quantized/cpu/qconv.cpp for more details.
    (W_value_min, W_value_max) = (-5, 5)
    # The operator expects them in the format
    # (out_channels, in_channels/groups,) + kernel_size
    W_init = torch.randint(
        W_value_min, W_value_max,
        (out_channels, in_channels_per_group,) + kernel_size)
    b_init = torch.randint(0, 10, (out_channels,))

    if use_channelwise:
        W_shape = (-1, 1) + (1,) * len(kernel_size)
        W_scales_tensor = torch.tensor(W_scale, dtype=torch.float)
        W_zero_points_tensor = torch.tensor(W_zero_point, dtype=torch.float)
        W = W_scales_tensor.reshape(*W_shape) * (
            W_init.float() - W_zero_points_tensor.reshape(*W_shape)).float()
        b = X_scale * W_scales_tensor * b_init.float()
        W_q = torch.quantize_per_channel(
            W, W_scales_tensor, W_zero_points_tensor.long(), 0,
            dtype=torch.qint8)
    else:
        W = W_scale[0] * (W_init - W_zero_point[0]).float()
        b = X_scale * W_scale[0] * b_init.float()
        W_q = torch.quantize_per_tensor(
            W, scale=W_scale[0], zero_point=W_zero_point[0], dtype=torch.qint8)

    return (X, X_q, W, W_q, b if use_bias else None)

def skipIfNoFBGEMM(fn):
    reason = 'Quantized operations require FBGEMM. FBGEMM is only optimized for CPUs with instruction set support AVX2 or newer.'
    if isinstance(fn, type):
        if 'fbgemm' not in torch.backends.quantized.supported_engines:
            fn.__unittest_skip__ = True
            fn.__unittest_skip_why__ = reason
        return fn

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if 'fbgemm' not in torch.backends.quantized.supported_engines:
            raise unittest.SkipTest(reason)
        else:
            fn(*args, **kwargs)
    return wrapper

def get_script_module(model, tracing, data):
    return torch.jit.trace(model, data) if tracing else torch.jit.script(model)


# QuantizationTestCase used as a base class for testing quantization on modules
class QuantizationTestCase(TestCase):
    def setUp(self):
        super().setUp()
        self.calib_data = [(torch.rand(2, 5, dtype=torch.float), torch.randint(0, 1, (2,), dtype=torch.long)) for _ in range(2)]
        self.train_data = [(torch.rand(2, 5, dtype=torch.float), torch.randint(0, 1, (2,), dtype=torch.long)) for _ in range(2)]
        # TODO: reame to img_data2d
        self.img_data = [(torch.rand(1, 3, 10, 10, dtype=torch.float),
                          torch.randint(0, 1, (1,), dtype=torch.long))
                         for _ in range(2)]
        self.img_data_1d = [(torch.rand(2, 3, 10, dtype=torch.float),
                             torch.randint(0, 1, (1,), dtype=torch.long))
                            for _ in range(2)]
        self.img_data_3d = [(torch.rand(1, 3, 5, 5, 5, dtype=torch.float),
                             torch.randint(0, 1, (1,), dtype=torch.long))
                            for _ in range(2)]
        self.img_data_dict = {1 : self.img_data_1d,
                              2 : self.img_data,
                              3 : self.img_data_3d}

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
        if hasattr(module, 'qconfig') and module.qconfig is not None and \
           len(module._modules) == 0 and not isinstance(module, torch.nn.Sequential):
            self.assertTrue(hasattr(module, 'activation_post_process'),
                            'module: ' + str(type(module)) + ' do not have observer')
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
        self.checkQuantDequant(mod)

    def checkQuantizedLinear(self, mod):
        self.assertEqual(type(mod), nnq.Linear)

    def checkDynamicQuantizedLinear(self, mod, dtype):
        r"""Checks that mod has been swapped for an nnqd.Linear
            module, the bias is float.
        """
        self.assertEqual(type(mod), nnqd.Linear)
        self.assertEqual(mod._packed_params.dtype, dtype)

    def checkDynamicQuantizedLSTM(self, mod, reference_module_type, dtype):
        r"""Checks that mod has been swapped for an nnqd.LSTM type
            module, the bias is float.
        """
        wt_dtype_map = {torch.qint8: 'quantized_dynamic', torch.float16: 'quantized_fp16'}
        self.assertEqual(type(mod), reference_module_type)
        for packed_params in mod._all_weight_values:
            self.assertEqual(packed_params.param.__getstate__()[0][0], wt_dtype_map[dtype])

    def checkLinear(self, mod):
        self.assertEqual(type(mod), torch.nn.Linear)

    def checkDynamicQuantizedModule(self, mod, reference_module_type, dtype):
        r"""Checks that mod has been swapped for an nnqd.Linear
            module, the bias is float.
        """
        wt_dtype_map = {torch.qint8: 'quantized_dynamic', torch.float16: 'quantized_fp16'}
        self.assertEqual(type(mod), reference_module_type)
        if hasattr(mod, '_all_weight_values'):
            for packed_params in mod._all_weight_values:
                self.assertEqual(packed_params.param.__getstate__()[0][0], wt_dtype_map[dtype])

    def checkScriptable(self, orig_mod, calib_data, check_save_load=False):
        scripted = torch.jit.script(orig_mod)
        self._checkScriptable(orig_mod, scripted, calib_data, check_save_load)

        # Use first calib_data entry as trace input
        traced = torch.jit.trace(orig_mod, calib_data[0][0])
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


    def checkGraphModeOp(self, module, data, quantized_op, tracing=False, debug=False, check=True, eval_mode=True, dynamic=False):
        if debug:
            print('Testing:', str(module))
        qconfig_dict = {'': get_default_qconfig(torch.backends.quantized.engine)}

        if eval_mode:
            module = module.eval()
        if dynamic:
            qconfig_dict = {'': default_dynamic_qconfig}
            inputs = data
        else:
            *inputs, target = data[0]
        model = get_script_module(module, tracing, inputs).eval()
        models = {}
        outputs = {}
        for d in [True, False]:
            # TODO: _test_only_eval_fn --> default_eval_fn
            if dynamic:
                models[d] = quantize_dynamic_script(model, qconfig_dict, debug=d)
                # make sure it runs
                outputs[d] = models[d](inputs)
            else:
                models[d] = quantize_script(
                    model, qconfig_dict, test_only_eval_fn, [data], inplace=False, debug=d)
                # make sure it runs
                outputs[d] = models[d](*inputs)

        if debug:
            print('debug graph:', models[True].graph)
            print('non debug graph:', models[False].graph)

        if check:
            # debug and non-debug option should have the same numerics
            self.assertEqual(outputs[True], outputs[False])

            # non debug graph should produce quantized op
            FileCheck().check(quantized_op) \
                       .run(models[False].graph)

        return models[False]


# Below are a series of neural net models to use in testing quantization
# Single layer models
class SingleLayerLinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(5, 5).to(dtype=torch.float)

    def forward(self, x):
        x = self.fc1(x)
        return x

class AnnotatedSingleLayerLinearModel(torch.nn.Module):
    def __init__(self, qengine='fbgemm'):
        super().__init__()
        self.qconfig = torch.quantization.get_default_qconfig(qengine)
        self.fc1 = QuantWrapper(torch.nn.Linear(5, 5).to(dtype=torch.float))

    def forward(self, x):
        x = self.fc1(x)
        return x

class SingleLayerLinearDynamicModel(torch.nn.Module):
    def __init__(self, qengine='fbgemm'):
        super().__init__()
        self.qconfig = torch.quantization.get_default_qconfig(qengine)
        self.fc1 = torch.nn.Linear(5, 5).to(dtype=torch.float)

    def forward(self, x):
        x = self.fc1(x)
        return x

class RNNDynamicModel(torch.nn.Module):
    def __init__(self, mod_type):
        super().__init__()
        self.qconfig = default_dynamic_qconfig
        if mod_type == 'GRU':
            self.mod = torch.nn.GRU(2, 2).to(dtype=torch.float)
        if mod_type == 'LSTM':
            self.mod = torch.nn.LSTM(2, 2).to(dtype=torch.float)

    def forward(self, x):
        x = self.mod(x)
        return x

class RNNCellDynamicModel(torch.nn.Module):
    def __init__(self, mod_type):
        super().__init__()
        self.qconfig = default_dynamic_qconfig
        if mod_type == 'GRUCell':
            self.mod = torch.nn.GRUCell(2, 2).to(dtype=torch.float)
        if mod_type == 'LSTMCell':
            self.mod = torch.nn.LSTMCell(2, 2).to(dtype=torch.float)
        if mod_type == 'RNNReLU':
            self.mod = torch.nn.RNNCell(2, 2, nonlinearity='relu').to(dtype=torch.float)
        if mod_type == 'RNNTanh':
            self.mod = torch.nn.RNNCell(2, 2, nonlinearity='tanh').to(dtype=torch.float)

    def forward(self, x):
        x = self.mod(x)
        return x

class LSTMwithHiddenDynamicModel(torch.nn.Module):
    def __init__(self, qengine='fbgemm'):
        super().__init__()
        self.qconfig = torch.quantization.get_default_qconfig(qengine)
        self.lstm = torch.nn.LSTM(2, 2).to(dtype=torch.float)

    def forward(self, x, hid):
        x, hid = self.lstm(x, hid)
        return x, hid

class ConvModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 5, 3, bias=False).to(dtype=torch.float)

    def forward(self, x):
        x = self.conv(x)
        return x

class AnnotatedConvModel(torch.nn.Module):
    def __init__(self, qengine):
        super().__init__()
        self.qconfig = torch.quantization.get_default_qconfig(qengine)
        self.conv = torch.nn.Conv2d(3, 5, 3, bias=False).to(dtype=torch.float)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.conv(x)
        x = self.dequant(x)
        return x

class ConvBnModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 5, 3, bias=False).to(dtype=torch.float)
        self.bn = torch.nn.BatchNorm2d(5).to(dtype=torch.float)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class AnnotatedConvBnModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qconfig = default_qconfig
        self.conv = torch.nn.Conv2d(3, 5, 3, bias=False).to(dtype=torch.float)
        self.bn = torch.nn.BatchNorm2d(5).to(dtype=torch.float)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.dequant(x)
        return x

class AnnotatedConvBnReLUModel(torch.nn.Module):
    def __init__(self, qengine='fbgemm'):
        super(AnnotatedConvBnReLUModel, self).__init__()
        self.qconfig = torch.quantization.get_default_qconfig(qengine)
        self.conv = torch.nn.Conv2d(3, 5, 3, bias=False).to(dtype=torch.float)
        self.bn = torch.nn.BatchNorm2d(5).to(dtype=torch.float)
        self.relu = nn.ReLU(inplace=True)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dequant(x)
        return x

    def fuse_model(self):
        torch.quantization.fuse_modules(self, [['conv', 'bn', 'relu']], inplace=True)

class TwoLayerLinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(5, 8).to(dtype=torch.float)
        self.fc2 = torch.nn.Linear(8, 5).to(dtype=torch.float)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class AnnotatedTwoLayerLinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(5, 8).to(dtype=torch.float)
        self.fc2 = QuantWrapper(torch.nn.Linear(8, 5).to(dtype=torch.float))
        self.fc2.qconfig = torch.quantization.get_default_qconfig("fbgemm")

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class ActivationsTestModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qconfig = torch.quantization.get_default_qconfig("fbgemm")
        self.quant = torch.quantization.QuantStub()
        self.hardswish = torch.nn.Hardswish().to(dtype=torch.float)

    def forward(self, x):
        x = self.quant(x)
        x = self.hardswish(x)
        return x

class ActivationsQATTestModel(torch.nn.Module):
    def __init__(self, qengine):
        super().__init__()
        self.qconfig = torch.quantization.get_default_qconfig(qengine)
        self.quant = torch.quantization.QuantStub()
        self.fc1 = torch.nn.Linear(5, 8).to(dtype=torch.float)
        self.hardswish = torch.nn.Hardswish().to(dtype=torch.float)

    def forward(self, x):
        x = self.quant(x)
        x = self.fc1(x)
        x = self.hardswish(x)
        return x

class LinearReluModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(5, 5).to(dtype=torch.float)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc(x))
        return x

class NormalizationTestModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        self.fc1 = torch.nn.Linear(5, 8).to(dtype=torch.float)
        self.layer_norm = torch.nn.LayerNorm((8))
        self.group_norm = torch.nn.GroupNorm(2, 8)
        # TODO: add handling for affine=False (future PR)
        self.instance_norm1d = torch.nn.InstanceNorm1d(8, affine=True)
        self.instance_norm2d = torch.nn.InstanceNorm2d(8, affine=True)
        self.instance_norm3d = torch.nn.InstanceNorm3d(8, affine=True)

    def forward(self, x):
        x = self.quant(x)
        x = self.fc1(x)
        x = self.layer_norm(x)
        x = self.group_norm(x.unsqueeze(-1))
        x = self.instance_norm1d(x)
        x = self.instance_norm2d(x.unsqueeze(-1))
        x = self.instance_norm3d(x.unsqueeze(-1))
        return x

class NormalizationQATTestModel(torch.nn.Module):
    def __init__(self, qengine):
        super().__init__()
        self.qconfig = torch.quantization.get_default_qconfig(qengine)
        self.quant = torch.quantization.QuantStub()
        self.fc1 = torch.nn.Linear(5, 8).to(dtype=torch.float)
        self.layer_norm = torch.nn.LayerNorm((8))
        self.group_norm = torch.nn.GroupNorm(2, 8)
        self.instance_norm1d = torch.nn.InstanceNorm1d(4, affine=True)
        self.instance_norm2d = torch.nn.InstanceNorm2d(4, affine=True)
        self.instance_norm3d = torch.nn.InstanceNorm3d(4, affine=True)
        self.fc2 = torch.nn.Linear(8, 2)

    def forward(self, x):
        x = self.quant(x)
        x = self.fc1(x)
        x = self.layer_norm(x)
        x = self.group_norm(x.unsqueeze(-1))
        x = self.instance_norm1d(x.reshape((2, 4, 2)))
        x = self.instance_norm2d(x.unsqueeze(-1))
        x = self.instance_norm3d(x.unsqueeze(-1))
        x = self.fc2(x.reshape((2, 8)))
        return x

class NestedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sub1 = LinearReluModel()
        self.sub2 = TwoLayerLinearModel()
        self.fc3 = torch.nn.Linear(5, 5).to(dtype=torch.float)

    def forward(self, x):
        x = self.sub1(x)
        x = self.sub2(x)
        x = self.fc3(x)
        return x

class AnnotatedNestedModel(torch.nn.Module):
    def __init__(self, qengine):
        super().__init__()
        self.sub1 = LinearReluModel()
        self.sub2 = TwoLayerLinearModel()
        self.fc3 = QuantWrapper(torch.nn.Linear(5, 5).to(dtype=torch.float))
        self.fc3.qconfig = default_qconfig
        self.sub2.fc1 = QuantWrapper(self.sub2.fc1)
        if qengine == 'fbgemm':
            self.sub2.fc1.qconfig = default_per_channel_qconfig
        else:
            self.sub2.fc1.qconfig = default_qconfig

    def forward(self, x):
        x = self.sub1(x)
        x = self.sub2(x)
        x = self.fc3(x)
        return x

class AnnotatedSubNestedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
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
        super().__init__()
        self.sub1 = LinearReluModel()
        self.sub2 = TwoLayerLinearModel()
        self.fc3 = QuantWrapper(torch.nn.Linear(5, 5).to(dtype=torch.float))
        self.fc3.qconfig = default_qconfig
        self.sub2.qconfig = default_qconfig

        custom_options = {
            'dtype': torch.quint8,
            'qscheme': torch.per_tensor_affine
        }
        custom_qconfig = QConfig(activation=default_observer.with_args(**custom_options),
                                 weight=default_weight_observer)
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
        super().__init__()
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
        super().__init__()
        self.fc1 = torch.nn.Linear(5, 8).to(dtype=torch.float)
        self.relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(8, 5).to(dtype=torch.float)
        self.relu2 = torch.nn.ReLU()

    def forward(self, x):
        return self.relu2(self.fc2(self.relu1(self.fc1(x))))

class SkipQuantModel(torch.nn.Module):
    r"""We can skip quantization by explicitly
    setting qconfig of a submodule to None
    """
    def __init__(self):
        super().__init__()
        self.sub = InnerModule()
        self.fc = torch.nn.Linear(5, 5).to(dtype=torch.float)

    def forward(self, x):
        return self.fc(self.sub(x))

class AnnotatedSkipQuantModel(torch.nn.Module):
    r"""We can skip quantization by explicitly
    setting qconfig of a submodule to None
    """
    def __init__(self, qengine):
        super().__init__()
        self.qconfig = torch.quantization.get_default_qconfig(qengine)
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
        super().__init__()
        self.qconfig = torch.quantization.get_default_qconfig("qnnpack")
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
    def __init__(self, qengine):
        super().__init__()
        self.qconfig = torch.quantization.get_default_qat_qconfig(qengine)
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
        super().__init__()
        self.qconfig = torch.quantization.get_default_qat_qconfig("qnnpack")
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


class SubModelForFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 2, 1, bias=None).to(dtype=torch.float)
        self.bn = nn.BatchNorm2d(2).to(dtype=torch.float)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class SubModelWithoutFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 2, 1, bias=None).to(dtype=torch.float)
        self.relu = nn.ReLU(inplace=False).to(dtype=torch.float)

    def forward(self, x):
        return self.relu(self.conv(x))

class ModelForFusion(nn.Module):
    def __init__(self, qconfig):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 2, 1, bias=None).to(dtype=torch.float)
        self.bn1 = nn.BatchNorm2d(2).to(dtype=torch.float)
        self.relu1 = nn.ReLU(inplace=True).to(dtype=torch.float)
        self.sub1 = SubModelForFusion()
        self.sub2 = SubModelWithoutFusion()
        self.fc = nn.Linear(36, 10).to(dtype=torch.float)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.qconfig = qconfig
        self.conv2 = nn.Conv3d(3, 2, (1, 1, 1), bias=None).to(dtype=torch.float)
        self.relu2 = nn.ReLU(inplace=False).to(dtype=torch.float)
        self.bn2 = nn.BatchNorm3d(2).to(dtype=torch.float)
        self.relu3 = nn.ReLU(inplace=True).to(dtype=torch.float)
        self.conv3 = nn.Conv1d(3, 3, 2).to(dtype=torch.float)
        self.bn3 = nn.BatchNorm1d(3).to(dtype=torch.float)
        self.relu4 = nn.ReLU(inplace=True).to(dtype=torch.float)
        # don't quantize sub2
        self.sub2.qconfig = None
        self.fc.qconfig = None

    def forward(self, x):
        x = x.squeeze(2)
        x = self.quant(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu4(x)
        x = x.unsqueeze(2)
        y = x.unsqueeze(2)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.sub1(x)
        x = self.dequant(x)
        x = self.sub2(x)
        x = x.view(-1, 36).contiguous()
        x = self.fc(x)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.bn2(y)
        y = self.relu3(y)
        y = self.dequant(y)
        return x

class ConvBNReLU(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(3, 3, 1, 1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=False)
        )

class ModelWithSequentialFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 3, 1)
        self.relu1 = nn.ReLU(inplace=False)
        layers = []
        for i in range(3):
            layers.append(ConvBNReLU())
        self.features = nn.Sequential(*layers)
        head = [nn.Linear(300, 10), nn.ReLU(inplace=False)]
        self.classifier = nn.Sequential(*head)
        self.seq = nn.Sequential()
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.features(x)
        x = torch.reshape(x, (-1, 3 * 10 * 10))
        x = self.classifier(x)
        x = self.seq(x)
        x = self.dequant(x)
        return x

class ModelForFusionWithBias(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 2, 5, bias=True).to(dtype=torch.float)
        self.bn1 = nn.BatchNorm2d(2).to(dtype=torch.float)
        self.relu1 = nn.ReLU(inplace=True).to(dtype=torch.float)
        self.conv2 = nn.Conv2d(2, 2, 1, bias=True).to(dtype=torch.float)
        self.bn2 = nn.BatchNorm2d(2).to(dtype=torch.float)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.dequant(x)
        return x

class DummyObserver(torch.nn.Module):
    def calculate_qparams(self):
        return 1.0, 0

    def forward(self, x):
        return x


class ModelWithFunctionals(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mycat = nnq.FloatFunctional()
        self.myadd = nnq.FloatFunctional()
        self.myadd_relu = nnq.FloatFunctional()
        # Tracing doesnt work yet for c10 ops with scalar inputs
        # https://github.com/pytorch/pytorch/issues/27097
        # self.my_scalar_add = nnq.FloatFunctional()
        # self.my_scalar_mul = nnq.FloatFunctional()

    def forward(self, x):
        y = self.mycat.cat([x, x, x])
        z = self.myadd.add(y, y)
        w = self.myadd_relu.add_relu(z, z)
        # Tracing doesnt work yet for c10 ops with scalar inputs
        # https://github.com/pytorch/pytorch/issues/27097
        # w = self.my_scalar_add.add_scalar(w, -0.5)
        # w = self.my_scalar_mul.mul_scalar(w, 0.5)
        return w


class ResNetBase(torch.nn.Module):
    def __init__(self):
        super().__init__()
        norm_layer = nn.BatchNorm2d
        inplanes = 3
        self.conv1 = nn.Conv2d(inplanes, inplanes, (1, 1), bias=False)
        self.bn1 = norm_layer(inplanes)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.downsample = torch.nn.Identity()
        self.myop = nn.quantized.FloatFunctional()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        identity = self.downsample(x)
        out = self.myop.add(out, identity)
        out = self.relu2(out)
        out = self.avgpool(out)
        return out

class ModelMultipleOps(torch.nn.Module):
    def __init__(self):
        super().__init__()
        norm_layer = nn.BatchNorm2d
        inplanes = 3
        self.conv1 = nn.Conv2d(inplanes, inplanes, (1, 1), bias=False)
        self.conv2 = nn.Conv2d(inplanes, inplanes, (1, 1), bias=False)
        self.bn1 = norm_layer(inplanes)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.downsample = torch.nn.Identity()
        self.skip_add = nn.quantized.FloatFunctional()
        self.cat = nn.quantized.FloatFunctional()
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(12, 6)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        identity = self.downsample(x)
        out = self.skip_add.add(out, identity)
        out = self.relu2(out)
        out = self.avgpool(out)
        out = self.conv2(out)
        out = torch.nn.functional.max_pool2d(out, 2, 2)
        out = self.cat.cat([out, out])
        out = out.view(-1, 3 * 2 * 2)
        out = self.fc(out)
        return out

# Model to ensure consistency of fake quant with true quant
# Average pooling and mean operations are not modelled
# accurately with fake-quant so this model does not
# contain those operations
class ModelMultipleOpsNoAvgPool(torch.nn.Module):
    def __init__(self):
        super().__init__()
        norm_layer = nn.BatchNorm2d
        inplanes = 3
        self.conv1 = nn.Conv2d(inplanes, inplanes, (1, 1), bias=False)
        self.conv2 = nn.Conv2d(inplanes, inplanes, (1, 1), bias=False)
        self.bn1 = norm_layer(inplanes)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.skip_add = nn.quantized.FloatFunctional()
        self.cat = nn.quantized.FloatFunctional()
        self.maxpool = nn.MaxPool2d((4, 4))
        self.fc = nn.Linear(12, 6)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        skip = self.conv2(x)
        out = self.skip_add.add(out, skip)
        out = self.relu2(out)
        out = self.maxpool(out)
        out = self.conv2(out)
        out = torch.nn.functional.max_pool2d(out, 2, 2)
        out = self.cat.cat([out, out])
        out = out.view(-1, 3 * 2 * 2)
        out = self.fc(out)
        return out

"""Model to make sure that the observers are not inserted into custom modules.
"""
class ModelWithNoQconfigPropagation(nn.Module):
    class ListOutModule(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            # returns a list of tensors, not supported by observers
            return [x]

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(5, 5).to(dtype=torch.float)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.no_quant_module = self.ListOutModule()

    def forward(self, x):
        x = self.quant(x)
        x = self.fc1(x)
        x = self.dequant(x)
        x = self.no_quant_module(x)
        return x
