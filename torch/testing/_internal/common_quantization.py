r"""Importing this file includes common utility methods and base clases for
checking quantization api and properties of resulting modules.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.ao.nn.intrinsic.quantized.dynamic as nniqd
import torch.ao.nn.quantized as nnq
import torch.ao.nn.quantized.dynamic as nnqd
from torch.ao.nn.intrinsic import _FusedModule
import torch.distributed as dist
from torch.testing._internal.common_utils import TestCase, TEST_WITH_ROCM

from torch.ao.quantization import (
    QuantType,
    default_dynamic_qat_qconfig,
    default_embedding_qat_qconfig,
    default_symmetric_qnnpack_qat_qconfig,
)
from torch.ao.quantization import QuantWrapper, QuantStub, DeQuantStub, \
    default_qconfig, default_dynamic_qconfig, default_per_channel_qconfig, QConfig, default_observer, default_weight_observer, \
    propagate_qconfig_, convert, get_default_qconfig, quantize_dynamic_jit, quantize_jit, float_qparams_weight_only_qconfig, \
    get_default_qat_qconfig, PerChannelMinMaxObserver, default_dynamic_quant_observer, quantize, \
    QConfigMapping, get_default_qconfig_mapping, get_default_qat_qconfig_mapping
from torch.ao.quantization.quantization_mappings import (
    get_default_dynamic_quant_module_mappings,
    get_default_qconfig_propagation_list,
    get_default_qat_module_mappings,
)
from torch.testing._internal.common_quantized import (
    override_quantized_engine,
)
from torch.jit.mobile import _load_for_lite_interpreter

try:
    # graph mode quantization based on fx
    from torch.ao.quantization.quantize_fx import (
        prepare_fx,
        prepare_qat_fx,
        convert_fx,
        convert_to_reference_fx,
    )
    from torch.ao.ns.fx.ns_types import NSSingleResultValuesType, NSSubgraph
    from torch.fx.graph import Node
    from torch.fx import GraphModule
    HAS_FX = True
except ImportError:
    HAS_FX = False

import copy
import io
import functools
import time
import os

import unittest
import numpy as np
from torch.testing import FileCheck
from typing import Callable, Tuple, Dict, Any, Union, Type, Optional
import torch._dynamo as torchdynamo

class NodeSpec:
    ''' Used for checking GraphModule Node
    '''
    def __init__(self, op, target):
        '''
        op: call_function | call_module
        target:
          for call_function, target would be a function
          for call_module, target would be the type of PyTorch module
        '''
        self.op = op
        self.target = target

    @classmethod
    def call_function(cls, target):
        return NodeSpec('call_function', target)

    @classmethod
    def call_method(cls, target):
        return NodeSpec('call_method', target)

    @classmethod
    def call_module(cls, target):
        return NodeSpec('call_module', target)

    def __hash__(self):
        return hash((self.op, self.target))

    def __eq__(self, other):
        if not isinstance(other, NodeSpec):
            return NotImplemented

        return self.op == other.op and self.target == other.target

    def __repr__(self):
        return repr(self.op) + " " + repr(self.target)

def get_supported_device_types():
    return ['cpu', 'cuda'] if torch.cuda.is_available() and not TEST_WITH_ROCM else ['cpu']

def test_only_eval_fn(model, calib_data):
    r"""
    Default evaluation function takes a torch.utils.data.Dataset or a list of
    input Tensors and run the model on the dataset
    """
    for inp in calib_data:
        output = model(*inp)

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

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def train_one_epoch(model, criterion, optimizer, data_loader, device, ntrain_batches):
    model.train()
    cnt = 0
    for image, target in data_loader:
        start_time = time.time()
        print('.', end='')
        cnt += 1
        image, target = image.to(device), target.to(device)
        output = model(image)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        if cnt >= ntrain_batches:
            return
    return

def ddp_setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def ddp_cleanup():
    dist.destroy_process_group()

def run_ddp(rank, world_size, prepared):
    ddp_setup(rank, world_size)
    prepared.cuda()
    prepared = torch.nn.parallel.DistributedDataParallel(prepared, device_ids=[rank])
    prepared.to(rank)
    model_with_ddp = prepared
    optimizer = torch.optim.SGD(model_with_ddp.parameters(), lr=0.0001)
    train_one_epoch(model_with_ddp, criterion, optimizer, dataset, rank, 1)
    ddp_cleanup()


def convert_dynamic(module):
    convert(module, get_default_dynamic_quant_module_mappings(), inplace=True)

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
            W, W_scales_tensor.double(), W_zero_points_tensor.long(), 0,
            dtype=torch.qint8)
    else:
        W = W_scale[0] * (W_init - W_zero_point[0]).float()
        b = X_scale * W_scale[0] * b_init.float()
        W_q = torch.quantize_per_tensor(
            W, scale=W_scale[0], zero_point=W_zero_point[0], dtype=torch.qint8)

    return (X, X_q, W, W_q, b if use_bias else None)

def _make_conv_add_extra_input_tensor(scale, zero_point, sizes):
    (X_value_min, X_value_max) = (0, 4)
    X_init = torch.randint(
        X_value_min,
        X_value_max,
        sizes  # Infer the size of tensor to do the add
    )
    X = scale * (X_init - zero_point).float()
    X_q = torch.quantize_per_tensor(
        X, scale=scale, zero_point=zero_point, dtype=torch.quint8)
    return X, X_q

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

def skipIfNoQNNPACK(fn):
    reason = 'Quantized operations require QNNPACK.'
    if isinstance(fn, type):
        if 'qnnpack' not in torch.backends.quantized.supported_engines:
            fn.__unittest_skip__ = True
            fn.__unittest_skip_why__ = reason
        return fn

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if 'qnnpack' not in torch.backends.quantized.supported_engines:
            raise unittest.SkipTest(reason)
        else:
            fn(*args, **kwargs)
    return wrapper

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if not torch.onnx._CAFFE2_ATEN_FALLBACK:
            raise unittest.SkipTest(reason)
        else:
            fn(*args, **kwargs)
    return wrapper

def withQNNPACKBackend(fn):
    # TODO(future PR): consider combining with skipIfNoQNNPACK,
    # will require testing of existing callsites
    reason = 'Quantized operations require QNNPACK.'
    if isinstance(fn, type):
        if 'qnnpack' not in torch.backends.quantized.supported_engines:
            fn.__unittest_skip__ = True
            fn.__unittest_skip_why__ = reason
        return fn

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if 'qnnpack' not in torch.backends.quantized.supported_engines:
            raise unittest.SkipTest(reason)
        with override_quantized_engine('qnnpack'):
            fn(*args, **kwargs)

    return wrapper

def skipIfNoONEDNN(fn):
    reason = 'Quantized operations require ONEDNN.'
    if isinstance(fn, type):
        if 'onednn' not in torch.backends.quantized.supported_engines:
            fn.__unittest_skip__ = True
            fn.__unittest_skip_why__ = reason
        return fn

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if 'onednn' not in torch.backends.quantized.supported_engines:
            raise unittest.SkipTest(reason)
        else:
            fn(*args, **kwargs)
    return wrapper

def skipIfNoX86(fn):
    reason = 'Quantized operations require X86.'
    if isinstance(fn, type):
        if 'x86' not in torch.backends.quantized.supported_engines:
            fn.__unittest_skip__ = True
            fn.__unittest_skip_why__ = reason
        return fn

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if 'x86' not in torch.backends.quantized.supported_engines:
            raise unittest.SkipTest(reason)
        else:
            fn(*args, **kwargs)
    return wrapper

def skipIfNoDynamoSupport(fn):
    reason = "dynamo doesn't support."
    if isinstance(fn, type):
        if not torchdynamo.is_dynamo_supported():
            fn.__unittest_skip__ = True
            fn.__unittest_skip_why__ = reason
        return fn

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if not torchdynamo.is_dynamo_supported():
            raise unittest.SkipTest(reason)
        else:
            fn(*args, **kwargs)
    return wrapper

try:
    import torchvision  # noqa: F401
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
skip_if_no_torchvision = unittest.skipIf(not HAS_TORCHVISION, "no torchvision")

def get_script_module(model, tracing, data):
    return torch.jit.trace(model, data) if tracing else torch.jit.script(model)

def lengths_to_offsets(t, offset_type=np.int64, use_begin_offset=True):
    """
    Convert lengths to offsets for embedding_bag
    """
    tt = np.zeros((t.shape[0] + 1,), dtype=offset_type)
    tt[1:] = t
    tt = torch.from_numpy(np.cumsum(tt, dtype=offset_type))
    if use_begin_offset:
        return tt[:-1]
    return tt[1:]

# QuantizationTestCase used as a base class for testing quantization on modules
class QuantizationTestCase(TestCase):
    def setUp(self):
        super().setUp()
        self.calib_data = [[torch.rand(2, 5, dtype=torch.float)] for _ in range(2)]
        self.train_data = [[torch.rand(2, 5, dtype=torch.float), torch.randint(0, 1, (2,), dtype=torch.long)] for _ in range(2)]
        self.img_data_1d = [[torch.rand(2, 3, 10, dtype=torch.float)]
                            for _ in range(2)]
        self.img_data_2d = [[torch.rand(1, 3, 10, 10, dtype=torch.float)]
                            for _ in range(2)]
        self.img_data_3d = [[torch.rand(1, 3, 5, 5, 5, dtype=torch.float)]
                            for _ in range(2)]
        self.img_data_1d_train = [[torch.rand(2, 3, 10, dtype=torch.float),
                                   torch.randint(0, 1, (1,), dtype=torch.long)]
                                  for _ in range(2)]
        self.img_data_2d_train = [[torch.rand(1, 3, 10, 10, dtype=torch.float),
                                   torch.randint(0, 1, (1,), dtype=torch.long)]
                                  for _ in range(2)]
        self.img_data_3d_train = [[torch.rand(1, 3, 5, 5, 5, dtype=torch.float),
                                   torch.randint(0, 1, (1,), dtype=torch.long)]
                                  for _ in range(2)]

        self.img_data_dict = {1 : self.img_data_1d,
                              2 : self.img_data_2d,
                              3 : self.img_data_3d}

        # Quant types that produce statically quantized ops
        self.static_quant_types = [QuantType.STATIC, QuantType.QAT]
        # All quant types for (fx based) graph mode quantization
        self.all_quant_types = [QuantType.DYNAMIC, QuantType.STATIC, QuantType.QAT]

    def checkNoPrepModules(self, module):
        r"""Checks the module does not contain child
            modules for quantization preparation, e.g.
            quant, dequant and observer
        """
        self.assertFalse(hasattr(module, 'quant'))
        self.assertFalse(hasattr(module, 'dequant'))

    def checkNoQconfig(self, module):
        r"""Checks the module does not contain qconfig
        """
        self.assertFalse(hasattr(module, 'qconfig'))

        for child in module.children():
            self.checkNoQconfig(child)

    def checkHasPrepModules(self, module):
        r"""Checks the module contains child
            modules for quantization preparation, e.g.
            quant, dequant and observer
        """
        self.assertTrue(hasattr(module, 'module'))
        self.assertTrue(hasattr(module, 'quant'))
        self.assertTrue(hasattr(module, 'dequant'))

    def checkObservers(self, module, propagate_qconfig_list=None, prepare_custom_config_dict=None):
        r"""Checks the module or module's leaf descendants
            have observers in preparation for quantization
        """
        if propagate_qconfig_list is None:
            propagate_qconfig_list = get_default_qconfig_propagation_list()
        if prepare_custom_config_dict is None:
            prepare_custom_config_dict = {}
        float_to_observed_module_class_mapping = prepare_custom_config_dict.get("float_to_observed_custom_module_class", {})

        # check if a module is a leaf module, ignoring activation_post_process attribute
        def is_leaf_module(module):
            submodule_name_count = 0
            for name, _ in module.named_children():
                if name != 'activation_post_process':
                    submodule_name_count += 1
            return submodule_name_count == 0

        if hasattr(module, 'qconfig') and module.qconfig is not None and \
           ((is_leaf_module(module) and not isinstance(module, torch.nn.Sequential)
            and type(module) in propagate_qconfig_list) or
           type(module) in float_to_observed_module_class_mapping.keys()) and \
           not isinstance(module, torch.ao.quantization.DeQuantStub):
            self.assertTrue(hasattr(module, 'activation_post_process'),
                            'module: ' + str(type(module)) + ' do not have observer')
        # we don't need to check observers for child modules of the
        # qat modules
        if type(module) not in get_default_qat_module_mappings().values() and \
           type(module) not in float_to_observed_module_class_mapping.values() and \
           not isinstance(module, _FusedModule):
            for child in module.children():
                if type(child) in [nn.Dropout]:
                    continue
                self.checkObservers(child, propagate_qconfig_list, prepare_custom_config_dict)

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

    def checkDynamicQuantizedLinearRelu(self, mod, dtype):
        r"""Checks that mod has been swapped for an nnqd.Linear
            module, the bias is float.
        """
        self.assertEqual(type(mod), nniqd.LinearReLU)
        self.assertEqual(mod._packed_params.dtype, dtype)

    def check_eager_serialization(self, ref_model, loaded_model, x):
        # Check state dict serialization and torch.save APIs
        model_dict = ref_model.state_dict()
        b = io.BytesIO()
        torch.save(model_dict, b)
        b.seek(0)
        loaded_dict = torch.load(b)
        loaded_model.load_state_dict(loaded_dict)
        ref_out = ref_model(*x)
        load_out = loaded_model(*x)

        def check_outputs(ref_out, load_out):
            self.assertEqual(ref_out[0], load_out[0])
            if isinstance(ref_out[1], tuple):
                self.assertEqual(ref_out[1][0], load_out[1][0])
                self.assertEqual(ref_out[1][1], load_out[1][1])
            else:
                self.assertEqual(ref_out[1], load_out[1])

        check_outputs(ref_out, load_out)
        b = io.BytesIO()
        torch.save(ref_model, b)
        b.seek(0)
        loaded = torch.load(b)
        load_out = loaded(*x)
        check_outputs(ref_out, load_out)

    def check_weight_bias_api(self, ref_model, weight_keys, bias_keys):
        weight = ref_model.get_weight()
        bias = ref_model.get_bias()
        self.assertEqual(weight_keys ^ weight.keys(), set())
        self.assertEqual(bias_keys ^ bias.keys(), set())

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
        traced = torch.jit.trace(orig_mod, calib_data[0])
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
        for inp in calib_data:
            ref_output = orig_mod(*inp)
            scripted_output = test_mod(*inp)
            self.assertEqual(scripted_output, ref_output)


    def checkGraphModeOp(self, module, inputs, quantized_op, tracing=False, debug=False,
                         check=True, eval_mode=True, dynamic=False, qconfig=None):
        if debug:
            print('Testing:', str(module))
        qconfig_dict = {'': get_default_qconfig(torch.backends.quantized.engine)}

        if eval_mode:
            module = module.eval()
        if dynamic:
            qconfig_dict = {'': default_dynamic_qconfig if qconfig is None else qconfig}
        model = get_script_module(module, tracing, inputs[0]).eval()
        if debug:
            print('input graph:', model.graph)
        models = {}
        outputs = {}
        for debug in [True, False]:
            if dynamic:
                models[debug] = quantize_dynamic_jit(model, qconfig_dict, debug=debug)
                # make sure it runs
                outputs[debug] = models[debug](inputs)
            else:
                # module under test can contain in-place ops, and we depend on
                # input data staying constant for comparisons
                inputs_copy = copy.deepcopy(inputs)
                models[debug] = quantize_jit(
                    model, qconfig_dict, test_only_eval_fn, [inputs_copy], inplace=False,
                    debug=debug)
                # make sure it runs
                outputs[debug] = models[debug](*inputs[0])

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

    def checkGraphModuleNodes(
            self, graph_module,
            expected_node=None,
            expected_node_occurrence=None,
            expected_node_list=None):
        """ Check if GraphModule contains the target node
        Args:
            graph_module: the GraphModule instance we want to check
            expected_node, expected_node_occurrence, expected_node_list:
               see docs for checkGraphModeFxOp
        """
        nodes_in_graph = {}
        node_list = []
        modules = dict(graph_module.named_modules(remove_duplicate=False))
        for node in graph_module.graph.nodes:
            n = None
            if node.op == 'call_function' or node.op == 'call_method':
                n = NodeSpec(node.op, node.target)
            elif node.op == 'call_module':
                n = NodeSpec(node.op, type(modules[node.target]))

            if n is not None:
                node_list.append(n)
                if n in nodes_in_graph:
                    nodes_in_graph[n] += 1
                else:
                    nodes_in_graph[n] = 1

        if expected_node is not None:
            self.assertTrue(expected_node in nodes_in_graph, 'node:' + str(expected_node) +
                            ' not found in the graph module')

        if expected_node_occurrence is not None:
            for expected_node, occurrence in expected_node_occurrence.items():
                if occurrence != 0:
                    self.assertTrue(
                        expected_node in nodes_in_graph,
                        'Check failed for node:' + str(expected_node) +
                        ' not found')
                    self.assertTrue(
                        nodes_in_graph[expected_node] == occurrence,
                        'Check failed for node:' + str(expected_node) +
                        ' Expected occurrence:' + str(occurrence) +
                        ' Found occurrence:' + str(nodes_in_graph[expected_node]))
                else:
                    self.assertTrue(
                        expected_node not in nodes_in_graph,
                        'Check failed for node:' + str(expected_node) +
                        ' expected no occurrence but found')

        if expected_node_list is not None:
            cur_index = 0
            for n in node_list:
                if cur_index == len(expected_node_list):
                    return
                if n == expected_node_list[cur_index]:
                    cur_index += 1
            self.assertTrue(
                cur_index == len(expected_node_list),
                "Check failed for graph:" +
                self.printGraphModule(graph_module, print_str=False) +
                "Expected ordered list:" +
                str(expected_node_list))

    def printGraphModule(self, graph_module, print_str=True):
        modules = dict(graph_module.named_modules(remove_duplicate=False))
        node_infos = []
        for n in graph_module.graph.nodes:
            node_info = ' '.join(map(repr, [n.op, n.name, n.target, n.args, n.kwargs]))
            if n.op == 'call_module':
                node_info += ' module type: ' + repr(type(modules[n.target]))
            node_infos.append(node_info)
        str_to_print = '\n'.join(node_infos)
        if print_str:
            print(str_to_print)
        return str_to_print

    if HAS_FX:

        def assert_types_for_matched_subgraph_pairs(
            self,
            matched_subgraph_pairs: Dict[str, Tuple[NSSubgraph, NSSubgraph]],
            expected_types: Dict[str, Tuple[Tuple[Callable, Callable], Tuple[Callable, Callable]]],
            gm_a: GraphModule,
            gm_b: GraphModule,
        ) -> None:
            """
            Verifies that the types specified in expected_types match
            the underlying objects pointed to by the nodes in matched_subgraph_pairs.

            An example successful test case:

              matched_subgraph_pairs = {'x0': (graph_a_conv_0_node, graph_b_conv_0_node)}
              expected_types = {'x0': (nn.Conv2d, nnq.Conv2d)}

            The function tests for key equivalence, and verifies types with
            instance checks.
            """

            def _get_underlying_op_type(
                node: Node, gm: GraphModule
            ) -> Union[Callable, str]:
                if node.op == 'call_module':
                    mod = getattr(gm, node.target)
                    return type(mod)
                else:
                    assert node.op in ('call_function', 'call_method')
                    return node.target

            self.assertTrue(
                len(matched_subgraph_pairs) == len(expected_types),
                f'Expected length of results to match, but got {len(matched_subgraph_pairs)} and {len(expected_types)}'
            )
            for k, v in expected_types.items():
                expected_types_a, expected_types_b = v
                exp_type_start_a, exp_type_end_a = expected_types_a
                exp_type_start_b, exp_type_end_b = expected_types_b
                subgraph_a, subgraph_b = matched_subgraph_pairs[k]

                act_type_start_a = _get_underlying_op_type(subgraph_a.start_node, gm_a)
                act_type_start_b = _get_underlying_op_type(subgraph_b.start_node, gm_b)
                act_type_end_a = _get_underlying_op_type(subgraph_a.end_node, gm_a)
                act_type_end_b = _get_underlying_op_type(subgraph_b.end_node, gm_b)
                types_match = (exp_type_start_a is act_type_start_a) and \
                    (exp_type_end_a is act_type_end_a) and \
                    (exp_type_start_b is act_type_start_b) and \
                    (exp_type_end_b is act_type_end_b)
                self.assertTrue(
                    types_match,
                    'Type mismatch at {}: expected {}, got {}'.format(
                        k,
                        (exp_type_start_a, exp_type_end_a, exp_type_start_b, exp_type_end_b),
                        (act_type_start_a, act_type_end_a, act_type_start_b, act_type_end_b))
                )

        def assert_ns_compare_dict_valid(
            self,
            act_compare_dict: Dict[str, Dict[str, Dict[str, Any]]],
        ) -> None:
            """
            Verifies that the act_compare_dict (output of Numeric Suite APIs) is valid:
            1. for each layer, results are recorded for two models
            2. number of seen tensors match
            3. shapes of each pair of seen tensors match
            """
            for layer_name, result_type_to_data in act_compare_dict.items():
                for result_type, layer_data in result_type_to_data.items():
                    self.assertTrue(
                        len(layer_data) == 2,
                        f"Layer {layer_name} does not have exactly two model results.")
                    model_name_0, model_name_1 = layer_data.keys()
                    for res_idx in range(len(layer_data[model_name_0])):
                        layer_data_0 = layer_data[model_name_0][res_idx]
                        layer_data_1 = layer_data[model_name_1][res_idx]
                        self.assertTrue(
                            layer_data_0['type'] == layer_data_0['type'],
                            f"Layer {layer_name}, {model_name_0} and {model_name_1} do not have the same type.")

                        self.assertTrue(
                            len(layer_data_0['values']) ==
                            len(layer_data_1['values']),
                            f"Layer {layer_name}, {model_name_0} and {model_name_1} do not have the same number of seen Tensors.")

                        # F.conv1d weight has rank 3, and toq.conv1d unpacked weight
                        # has rank 4. For now, skip the length check for conv1d only.
                        is_weight_functional_conv1d = (
                            result_type == NSSingleResultValuesType.WEIGHT.value and
                            (
                                'conv1d' in layer_data_0['prev_node_target_type'] or
                                'conv1d' in layer_data_1['prev_node_target_type']
                            )
                        )
                        if not is_weight_functional_conv1d:
                            for idx in range(len(layer_data_0['values'])):
                                values_0 = layer_data_0['values'][idx]
                                values_1 = layer_data_1['values'][idx]
                                if isinstance(values_0, torch.Tensor):
                                    self.assertTrue(
                                        values_0.shape == values_1.shape,
                                        f"Layer {layer_name}, {model_name_0} and {model_name_1} " +
                                        f"have a shape mismatch at idx {idx}.")
                                elif isinstance(values_0, list):
                                    values_0 = values_0[0]
                                    values_1 = values_1[0]
                                    self.assertTrue(
                                        values_0.shape == values_1.shape,
                                        f"Layer {layer_name}, {model_name_0} and {model_name_1} " +
                                        f"have a shape mismatch at idx {idx}.")
                                else:
                                    assert isinstance(values_0, tuple), \
                                        f"unhandled type {type(values_0)}"
                                    assert len(values_0) == 2
                                    assert len(values_0[1]) == 2
                                    assert values_0[0].shape == values_1[0].shape
                                    assert values_0[1][0].shape == values_1[1][0].shape
                                    assert values_0[1][1].shape == values_1[1][1].shape

                        # verify that ref_node_name is valid
                        ref_node_name_0 = layer_data_0['ref_node_name']
                        ref_node_name_1 = layer_data_1['ref_node_name']
                        prev_node_name_0 = layer_data_0['prev_node_name']
                        prev_node_name_1 = layer_data_1['prev_node_name']
                        if layer_data_0['type'] == NSSingleResultValuesType.NODE_OUTPUT.value:
                            self.assertTrue(ref_node_name_0 == prev_node_name_0)
                            self.assertTrue(ref_node_name_1 == prev_node_name_1)
                        elif layer_data_0['type'] == NSSingleResultValuesType.NODE_INPUT.value:
                            self.assertTrue(ref_node_name_0 != prev_node_name_0)
                            self.assertTrue(ref_node_name_1 != prev_node_name_1)

        def checkGraphModeFxOp(
                self,
                model,
                inputs,
                quant_type,
                expected_node=None,
                expected_node_occurrence=None,
                expected_node_list=None,
                is_reference=False,
                print_debug_info=False,
                custom_qconfig_dict=None,
                prepare_expected_node=None,
                prepare_expected_node_occurrence=None,
                prepare_expected_node_list=None,
                prepare_custom_config=None,
                backend_config=None):
            """ Quantizes model with graph mode quantization on fx and check if the
                quantized model contains the quantized_node

                Args:
                    model: floating point torch.nn.Module
                    inputs: one positional sample input arguments for model
                    expected_node: NodeSpec
                        e.g. NodeSpec.call_function(torch.quantize_per_tensor)
                    expected_node_occurrence: a dict from NodeSpec to
                        expected number of occurrences (int)
                        e.g. {NodeSpec.call_function(torch.quantize_per_tensor) : 1,
                                NodeSpec.call_method('dequantize'): 1}
                    expected_node_list: a list of NodeSpec, used to check the order
                        of the occurrence of Node
                        e.g. [NodeSpec.call_function(torch.quantize_per_tensor),
                                NodeSpec.call_module(nnq.Conv2d),
                                NodeSpec.call_function(F.hardtanh_),
                                NodeSpec.call_method('dequantize')]
                    is_reference: if True, enables reference mode
                    print_debug_info: if True, prints debug info
                    custom_qconfig_dict: overrides default qconfig_dict
                    prepare_expected_node: same as expected_node, but for prepare
                    prepare_expected_node_occurrence: same as
                        expected_node_occurrence, but for prepare
                    prepare_expected_node_list: same as expected_node_list, but
                        for prepare

                Returns:
                    A dictionary with the following structure:
                   {
                       "prepared": ...,  # the prepared model
                       "quantized": ...,  # the quantized non-reference model
                       "quantized_reference": ...,  # the quantized reference model
                       "result": ...,  # the result for either quantized or
                                       # quantized_reference model depending on the
                                       # is_reference argument
                   }
            """
            # TODO: make img_data a single example instead of a list
            if type(inputs) == list:
                inputs = inputs[0]

            if quant_type == QuantType.QAT:
                qconfig_mapping = get_default_qat_qconfig_mapping(torch.backends.quantized.engine)
                model.train()
            elif quant_type == QuantType.STATIC:
                qconfig_mapping = get_default_qconfig_mapping(torch.backends.quantized.engine)
                model.eval()
            else:
                qconfig = default_dynamic_qconfig
                qconfig_mapping = QConfigMapping().set_global(qconfig)
                model.eval()

            if quant_type == QuantType.QAT:
                prepare = prepare_qat_fx
            else:
                prepare = prepare_fx

            # overwrite qconfig_dict with custom_qconfig_dict
            if custom_qconfig_dict is not None:
                assert type(custom_qconfig_dict) in (QConfigMapping, dict), \
                    'custom_qconfig_dict should be a QConfigMapping or a dict'
                if isinstance(custom_qconfig_dict, QConfigMapping):
                    qconfig_mapping = custom_qconfig_dict
                else:
                    qconfig_mapping = QConfigMapping.from_dict(custom_qconfig_dict)
            prepared = prepare(
                model, qconfig_mapping,
                example_inputs=inputs,
                prepare_custom_config=prepare_custom_config,
                backend_config=backend_config)
            if not quant_type == QuantType.DYNAMIC:
                prepared(*inputs)

            if print_debug_info:
                print()
                print('quant type:\n', quant_type)
                print('original model:\n', model)
                print()
                print('prepared model:\n', prepared)

            self.checkGraphModuleNodes(
                prepared, prepare_expected_node,
                prepare_expected_node_occurrence, prepare_expected_node_list)

            prepared_copy = copy.deepcopy(prepared)
            qgraph = convert_fx(copy.deepcopy(prepared))
            qgraph_reference = convert_to_reference_fx(copy.deepcopy(prepared))
            result = qgraph(*inputs)
            result_reference = qgraph_reference(*inputs)
            qgraph_copy = copy.deepcopy(qgraph)
            qgraph_reference_copy = copy.deepcopy(qgraph_reference)

            qgraph_to_check = qgraph_reference if is_reference else qgraph
            if print_debug_info:
                print()
                print('quantized model:\n', qgraph_to_check)
                self.printGraphModule(qgraph_to_check)
                print()
            self.checkGraphModuleNodes(
                qgraph_to_check, expected_node, expected_node_occurrence, expected_node_list)
            return {"prepared": prepared_copy,
                    "quantized": qgraph_copy,
                    "quantized_reference": qgraph_reference_copy,
                    "quantized_output": result,
                    "quantized_reference_output": result_reference}


    def checkEmbeddingSerialization(self, qemb, num_embeddings, embedding_dim, indices, offsets,
                                    set_qconfig, is_emb_bag, dtype=torch.quint8):
        # Test serialization of dynamic EmbeddingBag module using state_dict
        if is_emb_bag:
            inputs = [indices, offsets]
        else:
            inputs = [indices]
        emb_dict = qemb.state_dict()
        b = io.BytesIO()
        torch.save(emb_dict, b)
        b.seek(0)
        loaded_dict = torch.load(b)
        embedding_unpack = torch.ops.quantized.embedding_bag_unpack
        # Check unpacked weight values explicitly
        for key in emb_dict:
            if isinstance(emb_dict[key], torch._C.ScriptObject):
                assert isinstance(loaded_dict[key], torch._C.ScriptObject)
                emb_weight = embedding_unpack(emb_dict[key])
                loaded_weight = embedding_unpack(loaded_dict[key])
                self.assertEqual(emb_weight, loaded_weight)

        # Check state dict serialization and torch.save APIs
        if is_emb_bag:
            loaded_qemb = nnq.EmbeddingBag(num_embeddings=num_embeddings, embedding_dim=embedding_dim,
                                           include_last_offset=True, mode='sum', dtype=dtype)
        else:
            loaded_qemb = nnq.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim, dtype=dtype)
        self.check_eager_serialization(qemb, loaded_qemb, inputs)

        loaded_qemb.load_state_dict(loaded_dict)
        self.assertEqual(embedding_unpack(qemb._packed_params._packed_weight),
                         embedding_unpack(loaded_qemb._packed_params._packed_weight))


        # Test JIT serialization
        self.checkScriptable(qemb, [inputs], check_save_load=True)

        # Test from_float call
        if is_emb_bag:
            float_embedding = torch.nn.EmbeddingBag(num_embeddings=num_embeddings, embedding_dim=embedding_dim,
                                                    include_last_offset=True, scale_grad_by_freq=False, mode='sum')
        else:
            float_embedding = torch.nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)

        if set_qconfig:
            float_qparams_observer = PerChannelMinMaxObserver.with_args(dtype=dtype,
                                                                        qscheme=torch.per_channel_affine_float_qparams,
                                                                        ch_axis=0)
            float_embedding.qconfig = QConfig(activation=default_dynamic_quant_observer,
                                              weight=float_qparams_observer)

        prepare_dynamic(float_embedding)

        float_embedding(*inputs)
        if is_emb_bag:
            q_embeddingbag = nnq.EmbeddingBag.from_float(float_embedding)
            expected_name = "QuantizedEmbeddingBag"
        else:
            q_embeddingbag = nnq.Embedding.from_float(float_embedding)
            expected_name = "QuantizedEmbedding"

        q_embeddingbag(*inputs)

        self.assertTrue(expected_name in str(q_embeddingbag))

class QuantizationLiteTestCase(QuantizationTestCase):
    def _create_quantized_model(self, model_class: Type[torch.nn.Module], **kwargs):
        # Creates quantized model for testing mobile script modules
        qengine = "qnnpack"
        with override_quantized_engine(qengine):
            qconfig = torch.ao.quantization.get_default_qconfig(qengine)
            model = model_class(**kwargs)
            model = quantize(model, test_only_eval_fn, [self.calib_data])

        return model

    def _compare_script_and_mobile(self,
                                   model: torch.nn.Module,
                                   input: torch.Tensor):
        # Compares the numerical outputs for script and lite modules
        qengine = "qnnpack"
        with override_quantized_engine(qengine):
            script_module = torch.jit.script(model)
            script_module_result = script_module(input)

            max_retry = 5
            for retry in range(1, max_retry + 1):
                # retries `max_retry` times; breaks iff succeeds else throws exception
                try:
                    buffer = io.BytesIO(script_module._save_to_buffer_for_lite_interpreter())
                    buffer.seek(0)
                    mobile_module = _load_for_lite_interpreter(buffer)

                    mobile_module_result = mobile_module(input)

                    torch.testing.assert_close(script_module_result, mobile_module_result)
                    mobile_module_forward_result = mobile_module.forward(input)
                    torch.testing.assert_close(script_module_result, mobile_module_forward_result)

                    mobile_module_run_method_result = mobile_module.run_method("forward", input)
                    torch.testing.assert_close(script_module_result, mobile_module_run_method_result)
                except AssertionError as e:
                    if retry == max_retry:
                        raise e
                    else:
                        continue
                break


# Below are a series of toy models to use in testing quantization

class SingleLayerLinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(5, 5).to(dtype=torch.float)

    def forward(self, x):
        x = self.fc1(x)
        return x

    def get_example_inputs(self) -> Tuple[Any, ...]:
        return (torch.rand(1, 5),)

class AnnotatedSingleLayerLinearModel(torch.nn.Module):
    def __init__(self, qengine='fbgemm'):
        super().__init__()
        self.qconfig = torch.ao.quantization.get_default_qconfig(qengine)
        self.fc1 = QuantWrapper(torch.nn.Linear(5, 5).to(dtype=torch.float))

    def forward(self, x):
        x = self.fc1(x)
        return x

    def get_example_inputs(self) -> Tuple[Any, ...]:
        return (torch.rand(1, 5),)

class SingleLayerLinearDynamicModel(torch.nn.Module):
    def __init__(self, qengine='fbgemm'):
        super().__init__()
        self.qconfig = torch.ao.quantization.get_default_qconfig(qengine)
        self.fc1 = torch.nn.Linear(5, 5).to(dtype=torch.float)

    def forward(self, x):
        x = self.fc1(x)
        return x

    def get_example_inputs(self) -> Tuple[Any, ...]:
        return (torch.rand(1, 5),)

class LinearAddModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(5, 8).to(dtype=torch.float)
        self.fc2 = torch.nn.Linear(8, 5).to(dtype=torch.float)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.add(x, 5)
        x = self.fc2(x)
        return x

    def get_example_inputs(self) -> Tuple[Any, ...]:
        return (torch.rand(1, 5),)

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
        self.qconfig = torch.ao.quantization.get_default_qconfig(qengine)
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

    def get_example_inputs(self) -> Tuple[Any, ...]:
        return (torch.rand(1, 3, 5, 5),)

class ConvTransposeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(3, 5, 3, bias=False).to(dtype=torch.float)

    def forward(self, x):
        x = self.conv(x)
        return x

    def get_example_inputs(self) -> Tuple[Any, ...]:
        return (torch.rand(1, 3, 5, 5),)

class AnnotatedConvModel(torch.nn.Module):
    def __init__(self, qengine):
        super().__init__()
        self.qconfig = torch.ao.quantization.get_default_qconfig(qengine)
        self.conv = torch.nn.Conv2d(3, 5, 3, bias=False).to(dtype=torch.float)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.conv(x)
        x = self.dequant(x)
        return x

    def get_example_inputs(self) -> Tuple[Any, ...]:
        return (torch.rand(1, 3, 5, 5),)

class AnnotatedConvTransposeModel(torch.nn.Module):
    def __init__(self, qengine):
        super().__init__()
        self.qconfig = torch.ao.quantization.get_default_qconfig(qengine)
        self.conv = torch.nn.ConvTranspose2d(3, 5, 3, bias=False).to(dtype=torch.float)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.conv(x)
        x = self.dequant(x)
        return x

    def get_example_inputs(self) -> Tuple[Any, ...]:
        return (torch.rand(1, 3, 5, 5),)

class ConvBnModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 5, 3, bias=False).to(dtype=torch.float)
        self.bn = torch.nn.BatchNorm2d(5).to(dtype=torch.float)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

    def get_example_inputs(self) -> Tuple[Any, ...]:
        return (torch.rand(1, 3, 5, 5),)

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

    def get_example_inputs(self) -> Tuple[Any, ...]:
        return (torch.rand(1, 3, 5, 5),)

class ConvBnReLUModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 5, 3, bias=False).to(dtype=torch.float)
        self.bn = torch.nn.BatchNorm2d(5).to(dtype=torch.float)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def get_example_inputs(self) -> Tuple[Any, ...]:
        return (torch.rand(1, 3, 5, 5),)

class AnnotatedConvBnReLUModel(torch.nn.Module):
    def __init__(self, qengine='fbgemm'):
        super().__init__()
        self.qconfig = torch.ao.quantization.get_default_qconfig(qengine)
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
        # TODO: remove this check and define two fuse_modules function on this module
        if self.training:
            torch.ao.quantization.fuse_modules_qat(self, [['conv', 'bn', 'relu']], inplace=True)
        else:
            torch.ao.quantization.fuse_modules(self, [['conv', 'bn', 'relu']], inplace=True)

    def get_example_inputs(self) -> Tuple[Any, ...]:
        return (torch.rand(1, 3, 5, 5),)

class TwoLayerConvModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 5, 3, bias=False).to(dtype=torch.float)
        self.conv2 = torch.nn.Conv2d(5, 5, 1, bias=False).to(dtype=torch.float)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

    def get_example_inputs(self) -> Tuple[Any, ...]:
        return (torch.rand(1, 3, 5, 5),)

class TwoLayerLinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(5, 8).to(dtype=torch.float)
        self.fc2 = torch.nn.Linear(8, 5).to(dtype=torch.float)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def get_example_inputs(self) -> Tuple[Any, ...]:
        return (torch.rand(1, 5),)

class LinearModelWithSubmodule(nn.Module):
    def __init__(self):
        super().__init__()
        self.subm = TwoLayerLinearModel()
        self.fc = nn.Linear(5, 5)

    def forward(self, x):
        x = self.subm(x)
        x = self.fc(x)
        return x

    def get_example_inputs(self) -> Tuple[Any, ...]:
        return self.subm.get_example_inputs()

class AnnotatedTwoLayerLinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(5, 8).to(dtype=torch.float)
        self.fc2 = QuantWrapper(torch.nn.Linear(8, 5).to(dtype=torch.float))
        self.fc2.qconfig = torch.ao.quantization.get_default_qconfig("fbgemm")

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def get_example_inputs(self) -> Tuple[Any, ...]:
        return (torch.rand(1, 5),)

class ActivationsTestModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qconfig = torch.ao.quantization.get_default_qconfig("fbgemm")
        self.quant = torch.ao.quantization.QuantStub()
        self.hardswish = torch.nn.Hardswish().to(dtype=torch.float)
        self.elu = torch.nn.ELU().to(dtype=torch.float)
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.hardswish(x)
        x = self.elu(x)
        x = self.dequant(x)
        return x

class LinearReluModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(5, 5).to(dtype=torch.float)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc(x))
        return x

    def get_example_inputs(self) -> Tuple[Any, ...]:
        return (torch.rand(1, 5),)


class LinearReluLinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(5, 8).to(dtype=torch.float)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(8, 5).to(dtype=torch.float)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def get_example_inputs(self) -> Tuple[Any, ...]:
        return (torch.rand(1, 5),)

class LinearReluAddModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(5, 5).to(dtype=torch.float)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(5, 5).to(dtype=torch.float)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = torch.add(x, 5)
        x = self.fc2(x)
        self.relu = torch.nn.ReLU()
        return x

    def get_example_inputs(self) -> Tuple[Any, ...]:
        return (torch.rand(1, 5),)

class LinearBnLeakyReluModel(torch.nn.Module):
    def __init__(self, with_bn=True):
        super().__init__()
        self.linear = nn.Linear(5, 5)
        self.bn1d = nn.BatchNorm1d(5)
        self.leaky_relu = nn.LeakyReLU(0.01)
        self.with_bn = with_bn

    def forward(self, x):
        x = self.linear(x)
        if self.with_bn:
            x = self.bn1d(x)
        x = self.leaky_relu(x)
        return x

    def get_example_inputs(self) -> Tuple[Any, ...]:
        return (torch.rand(1, 5),)

class LinearTanhModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(5, 5)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.linear(x)
        x = self.tanh(x)
        return x

    def get_example_inputs(self) -> Tuple[Any, ...]:
        return (torch.rand(1, 5),)

class ConvBnAddReluModel(torch.nn.Module):
    def __init__(self,
                 with_bn=True,
                 with_relu=True,
                 left_conv=True,
                 two_conv=True,
                 use_torch_add=True):
        super().__init__()
        self.conv = nn.Conv2d(5, 5, (2, 2))
        self.conv2 = nn.Conv2d(5, 5, (2, 2))
        self.bn = nn.BatchNorm2d(5)
        self.relu = nn.ReLU()
        self.with_bn = with_bn
        self.with_relu = with_relu
        self.two_conv = two_conv
        self.left_conv = left_conv
        self.use_torch_add = use_torch_add

    def forward(self, x1, x2):
        if self.two_conv:
            if self.use_torch_add:
                if self.with_bn:
                    x = torch.add(self.bn(self.conv(x1)), self.conv2(x1))
                else:
                    x = torch.add(self.conv(x1), self.conv2(x1))
            else:
                if self.with_bn:
                    x = self.bn(self.conv(x1)) + self.conv2(x1)
                else:
                    x = self.conv(x1) + self.conv2(x1)
        else:
            if self.use_torch_add:
                if self.left_conv:
                    if self.with_bn:
                        x = torch.add(self.bn(self.conv(x1)), x2)
                    else:
                        x = torch.add(self.conv(x1), x2)
                else:
                    if self.with_bn:
                        x = torch.add(x2, self.bn(self.conv(x1)))
                    else:
                        x = torch.add(x2, self.conv(x1))
            else:
                if self.left_conv:
                    if self.with_bn:
                        x = self.bn(self.conv(x1)) + x2
                    else:
                        x = self.conv(x1) + x2
                else:
                    if self.with_bn:
                        x = x2 + self.bn(self.conv(x1))
                    else:
                        x = x2 + self.conv(x1)
        if self.with_relu:
            x = self.relu(x)
        return x

    def get_example_inputs(self) -> Tuple[Any, ...]:
        return (torch.rand(1, 5, 3, 3), torch.rand(1, 5, 2, 2))

# TODO: self.fc should be self.conv
class ConvReluModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Conv2d(3, 5, 3).to(dtype=torch.float)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc(x))
        return x

    def get_example_inputs(self) -> Tuple[Any, ...]:
        return (torch.rand(1, 3, 5, 5),)

# TODO: self.fc should be self.conv
class ConvReluConvModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Conv2d(3, 5, 3).to(dtype=torch.float)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Conv2d(5, 5, 1).to(dtype=torch.float)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def get_example_inputs(self) -> Tuple[Any, ...]:
        return (torch.rand(1, 3, 5, 5),)

# TODO: self.fc should be self.conv
class ConvReluAddModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Conv2d(3, 5, 3).to(dtype=torch.float)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Conv2d(5, 5, 1).to(dtype=torch.float)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = torch.add(x, 5)
        x = self.fc2(x)
        self.relu = torch.nn.ReLU()
        return x

    def get_example_inputs(self) -> Tuple[Any, ...]:
        return (torch.rand(1, 3, 5, 5),)

class NormalizationTestModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.fc1 = torch.nn.Linear(5, 8).to(dtype=torch.float)
        self.layer_norm = torch.nn.LayerNorm(8)
        self.group_norm = torch.nn.GroupNorm(2, 8)
        self.instance_norm1d = torch.nn.InstanceNorm1d(8)
        self.instance_norm2d = torch.nn.InstanceNorm2d(8)
        self.instance_norm3d = torch.nn.InstanceNorm3d(8)

    def forward(self, x):
        x = self.quant(x)
        x = self.fc1(x)
        x = self.layer_norm(x)
        x = self.group_norm(x.unsqueeze(-1).repeat(1, 1, 3))
        x = self.instance_norm1d(x)
        x = self.instance_norm2d(x.unsqueeze(-1))
        x = self.instance_norm3d(x.unsqueeze(-1))
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

    def fuse_modules(self):
        fusable_layers = []
        named_children = list(self.named_children())
        for idx, (current_name, layer) in enumerate(named_children):
            if isinstance(layer, torch.nn.Linear):
                if idx >= len(named_children) - 1:
                    break
                if isinstance(named_children[idx + 1][1], torch.nn.ReLU):
                    fusable_layers.append([current_name,
                                           named_children[idx + 1][0]])
        # TODO: remove this check and define two fuse_modules function on this module
        if self.training:
            torch.ao.quantization.fuse_modules_qat(self, fusable_layers, inplace=True)
        else:
            torch.ao.quantization.fuse_modules(self, fusable_layers, inplace=True)

class FunctionalLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.rand((5, 5))
        self.bias = torch.zeros(5)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def get_example_inputs(self) -> Tuple[Any, ...]:
        return (torch.rand(1, 5),)

class SingleLayerFunctionalLinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = FunctionalLinear()

    def forward(self, x):
        x = self.linear1(x)
        return x

    def get_example_inputs(self) -> Tuple[Any, ...]:
        return self.linear1.get_example_inputs()

class TwoLayerFunctionalLinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = FunctionalLinear()
        self.linear2 = FunctionalLinear()

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x

    def get_example_inputs(self) -> Tuple[Any, ...]:
        return self.linear1.get_example_inputs()

class FunctionalLinearAddModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = FunctionalLinear()
        self.linear2 = FunctionalLinear()

    def forward(self, x):
        x = self.linear1(x)
        x = torch.add(x, 5)
        x = self.linear2(x)
        return x

    def get_example_inputs(self) -> Tuple[Any, ...]:
        return self.linear1.get_example_inputs()

class FunctionalLinearReluModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = FunctionalLinear()

    def forward(self, x):
        x = self.linear(x)
        x = F.relu(x)
        return x

    def get_example_inputs(self) -> Tuple[Any, ...]:
        return self.linear.get_example_inputs()

class FunctionalLinearReluLinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = FunctionalLinear()
        self.relu = nn.ReLU()
        self.linear2 = FunctionalLinear()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

    def get_example_inputs(self) -> Tuple[Any, ...]:
        return self.linear1.get_example_inputs()

class FunctionalConv2d(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.rand(3, 3, 3, 3)
        self.bias = torch.rand(3)
        self.stride = (1, 1)
        self.padding = (0, 0)
        self.dilation = (1, 1)
        self.groups = 1

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def get_example_inputs(self) -> Tuple[Any, ...]:
        return (torch.rand(1, 3, 5, 5),)

class SingleLayerFunctionalConvModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = FunctionalConv2d()

    def forward(self, x):
        x = self.conv1(x)
        return x

    def get_example_inputs(self) -> Tuple[Any, ...]:
        return self.conv1.get_example_inputs()

class TwoLayerFunctionalConvModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = FunctionalConv2d()
        self.conv2 = FunctionalConv2d()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

    def get_example_inputs(self) -> Tuple[Any, ...]:
        return self.conv1.get_example_inputs()

class FunctionalConvReluModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = FunctionalConv2d()

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        return x

    def get_example_inputs(self) -> Tuple[Any, ...]:
        return self.conv.get_example_inputs()

class FunctionalConvReluConvModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = FunctionalConv2d()
        self.relu = nn.ReLU()
        self.conv2 = FunctionalConv2d()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x

    def get_example_inputs(self) -> Tuple[Any, ...]:
        return self.conv1.get_example_inputs()

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

    def fuse_modules(self):
        self.sub.fuse_modules()

class AnnotatedSkipQuantModel(torch.nn.Module):
    r"""We can skip quantization by explicitly
    setting qconfig of a submodule to None
    """
    def __init__(self, qengine):
        super().__init__()
        self.qconfig = torch.ao.quantization.get_default_qconfig(qengine)
        self.sub = QuantWrapper(InnerModule())
        self.fc = torch.nn.Linear(5, 5).to(dtype=torch.float)
        # don't quantize this fc
        self.fc.qconfig = None

    def forward(self, x):
        return self.fc(self.sub(x))

    def fuse_modules(self):
        self.sub.module.fuse_modules()

class QuantStubModel(torch.nn.Module):
    r"""A Module with manually inserted `QuantStub` and `DeQuantStub`
    """
    def __init__(self):
        super().__init__()
        self.qconfig = torch.ao.quantization.get_default_qconfig("qnnpack")
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
        self.qconfig = torch.ao.quantization.get_default_qat_qconfig(qengine)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.fc1 = torch.nn.Linear(5, 1).to(dtype=torch.float)
        self.fc2 = torch.nn.Linear(1, 10).to(dtype=torch.float)

    def forward(self, x):
        x = self.quant(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return self.dequant(x)

class ManualDropoutQATModel(torch.nn.Module):
    r"""A Module with manually inserted `QuantStub` and `DeQuantStub`
    """
    def __init__(self, qengine):
        super().__init__()
        self.qconfig = torch.ao.quantization.get_default_qat_qconfig(qengine)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.fc1 = torch.nn.Linear(5, 1).to(dtype=torch.float)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x):
        x = self.quant(x)
        x = self.fc1(x)
        x = self.dropout(x)
        return self.dequant(x)

class ManualLinearDynamicQATModel(torch.nn.Module):
    r"""A Module that uses a dynamic QAT by default.
    """
    def __init__(self, qconfig=None):
        super().__init__()
        self.qconfig = qconfig or default_dynamic_qat_qconfig
        self.fc1 = torch.nn.Linear(5, 1).to(dtype=torch.float)
        self.fc2 = torch.nn.Linear(1, 10).to(dtype=torch.float)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class ManualConvLinearQATModel(torch.nn.Module):
    r"""A module with manually inserted `QuantStub` and `DeQuantStub`
    and contains both linear and conv modules
    """
    def __init__(self, qconfig=None):
        super().__init__()
        self.qconfig = qconfig if qconfig else torch.ao.quantization.get_default_qat_qconfig("qnnpack")
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

class ManualConvLinearSymmQATModel(ManualConvLinearQATModel):
    r"""Same as ManualConvLinearQATModule but with Symmetric Quantization.
    Supported only with qnnpack.
    """
    def __init__(self):
        super().__init__(default_symmetric_qnnpack_qat_qconfig)

class ManualEmbeddingBagLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.EmbeddingBag(num_embeddings=10, embedding_dim=12, mode='sum')
        self.emb.qconfig = default_embedding_qat_qconfig
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.linear = nn.Linear(12, 1).to(dtype=torch.float)
        self.qconfig = get_default_qat_qconfig("qnnpack")

    def forward(self, input: torch.Tensor, offsets: Optional[torch.Tensor] = None,
                per_sample_weights: Optional[torch.Tensor] = None):
        x = self.emb(input, offsets, per_sample_weights)
        x = self.quant(x)
        x = self.linear(x)
        return self.dequant(x)

class DeFusedEmbeddingBagLinear(nn.Module):
    r"""A module to simulate QAT embedding bag with a linear layer,
    this module uses a separate embedding and bagging op, similar
    to that which is described in the EmbeddingBag documentation.

    https://pytorch.org/docs/stable/generated/torch.nn.EmbeddingBag.html
    """
    def __init__(self) -> None:
        super().__init__()
        self.emb = nn.Embedding(num_embeddings=10, embedding_dim=12)
        self.emb.qconfig = default_embedding_qat_qconfig
        self.bagging_op = torch.sum
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.linear = nn.Linear(12, 1).to(dtype=torch.float)
        self.qconfig = get_default_qat_qconfig("qnnpack")

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.bagging_op(self.emb(input), dim=1)
        x = self.quant(x)
        x = self.linear(x)
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
        x = x.reshape(-1, 36).contiguous()
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

class ModelForLinearBNFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(20, 10)
        self.bn = nn.BatchNorm1d(10)
        nn.init.uniform_(self.bn.weight)
        nn.init.uniform_(self.bn.bias)

    def forward(self, x):
        return self.bn(self.fc(x))

class DummyObserver(torch.nn.Module):
    def calculate_qparams(self):
        return 1.0, 0

    def forward(self, x):
        return x


class ModelForConvTransposeBNFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.ConvTranspose1d(3, 3, 1)
        self.bn1 = nn.BatchNorm1d(3)
        self.conv2 = nn.ConvTranspose2d(3, 3, 1)
        self.bn2 = nn.BatchNorm2d(3)
        self.conv3 = nn.ConvTranspose3d(3, 3, 1)
        self.bn3 = nn.BatchNorm3d(3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = x.unsqueeze(2)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x.unsqueeze(2)
        x = self.conv3(x)
        x = self.bn3(x)
        return x


class ModelWithFunctionals(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mycat = nnq.FloatFunctional()
        self.myadd = nnq.FloatFunctional()
        self.myadd_relu = nnq.FloatFunctional()
        self.mymatmul = nnq.FloatFunctional()
        # Tracing doesnt work yet for c10 ops with scalar inputs
        # https://github.com/pytorch/pytorch/issues/27097
        # self.my_scalar_add = nnq.FloatFunctional()
        # self.my_scalar_mul = nnq.FloatFunctional()

    def forward(self, x):
        y = self.mycat.cat([x, x, x])
        z = self.myadd.add(y, y)
        w = self.myadd_relu.add_relu(z, z)
        u = self.mymatmul.matmul(w, w.T)
        # Tracing doesnt work yet for c10 ops with scalar inputs
        # https://github.com/pytorch/pytorch/issues/27097
        # w = self.my_scalar_add.add_scalar(w, -0.5)
        # w = self.my_scalar_mul.mul_scalar(w, 0.5)
        return u


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
        self.fc = torch.nn.Linear(inplanes, 1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        identity = self.downsample(x)
        out = self.myop.add(out, identity)
        out = self.relu2(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

    def fuse_model(self):
        # TODO: remove this check and define two fuse_model function on this module
        if self.training:
            torch.ao.quantization.fuse_modules_qat(self, [['conv1', 'bn1', 'relu1']], inplace=True)
        else:
            torch.ao.quantization.fuse_modules(self, [['conv1', 'bn1', 'relu1']], inplace=True)

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
        out = out.reshape(-1, 3 * 2 * 2)
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
        out = out.reshape(-1, 3 * 2 * 2)
        out = self.fc(out)
        return out

class EmbeddingBagModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.EmbeddingBag(num_embeddings=10, embedding_dim=12,
                                         include_last_offset=True, scale_grad_by_freq=False, mode='sum')

    def forward(self, indices, offsets, per_sample_weights):
        return self.emb(indices, offsets, per_sample_weights)

class EmbeddingModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(num_embeddings=10, embedding_dim=12)

    def forward(self, indices):
        return self.emb(indices)

class EmbeddingWithStaticLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.EmbeddingBag(num_embeddings=10, embedding_dim=12)
        self.fc = torch.nn.Linear(4, 2)
        self.emb.qconfig = float_qparams_weight_only_qconfig
        self.qconfig = default_qconfig
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, indices, offsets, linear_in):
        emb = self.emb(indices, offsets)
        q_x = self.quant(linear_in)
        fc = self.fc(q_x)
        fc = self.dequant(fc)
        features = torch.cat([fc] + [emb], dim=1)
        return features

class DenseTopMLP(nn.Module):

    def __init__(self, dense_dim, dense_out, embedding_dim, top_out_in, top_out_out) -> None:
        super().__init__()

        self.dense_mlp = nn.Sequential(
            nn.Linear(dense_dim, dense_out),
        )
        self.top_mlp = nn.Sequential(
            nn.Linear(dense_out + embedding_dim, top_out_in),
            nn.Linear(top_out_in, top_out_out),
        )

    def forward(
        self,
        sparse_feature: torch.Tensor,
        dense: torch.Tensor,
    ) -> torch.Tensor:
        dense_feature = self.dense_mlp(dense)
        features = torch.cat([dense_feature] + [sparse_feature], dim=1)

        out = self.top_mlp(features)
        return out

# thin wrapper around embedding bag, because tracing inside nn.Embedding
# bag is not supported at the moment and this is top level
class EmbBagWrapper(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.emb_bag = nn.EmbeddingBag(num_embeddings, embedding_dim, mode='sum')

    def forward(self, indices, offsets):
        return self.emb_bag(indices, offsets)

class SparseNNModel(nn.Module):
    _NUM_EMBEDDINGS = 10
    _EMBEDDING_DIM = 5
    _DENSE_DIM = 4
    _DENSE_OUTPUT = 2
    _TOP_OUT_IN = 2
    _TOP_OUT_OUT = 2
    _TOP_MLP_DIM = 1

    def __init__(self) -> None:
        super().__init__()

        self.model_sparse = EmbBagWrapper(self._NUM_EMBEDDINGS, self._EMBEDDING_DIM)
        self.dense_top = DenseTopMLP(
            self._DENSE_DIM, self._DENSE_OUTPUT, self._EMBEDDING_DIM, self._TOP_OUT_IN,
            self._TOP_OUT_OUT)

    def forward(
        self,
        sparse_indices: torch.Tensor,
        sparse_offsets: torch.Tensor,
        dense: torch.Tensor,
    ) -> torch.Tensor:

        sparse_feature = self.model_sparse(sparse_indices, sparse_offsets)
        out = self.dense_top(sparse_feature, dense)

        return out

class TestHelperModules:
    class Conv2dPropAnnotaton(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 3, 3)
            self.linear = torch.nn.Linear(3, 3)

        def forward(self, x):
            x = self.conv(x)
            x = x.view(-1, 3)
            x = torch.nn.functional.hardtanh(x, -0.5, 0.5)
            x = self.linear(x)
            return x

    class Conv2dWithObsSharingOps(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 3, 3)
            self.hardtanh = torch.nn.Hardtanh()
            self.adaptive_avg_pool2d = torch.nn.AdaptiveAvgPool2d((1, 1))

        def forward(self, x):
            x = self.conv(x)
            x = self.adaptive_avg_pool2d(x)
            x = self.hardtanh(x)
            x = torch.mean(x)
            return x

    class Conv2dWithTwoLinearPermute(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 16, 3)
            self.linear1 = torch.nn.Linear(16, 8, bias=False)
            self.linear2 = torch.nn.Linear(8, 8)

        def forward(self, x):
            conv_out = self.conv(x)
            permute_out = torch.permute(conv_out, (0, 2, 3, 1))
            return self.linear2(self.linear1(permute_out))

    class Conv2dWithTwoLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 16, 3)
            self.linear1 = torch.nn.Linear(64, 8, bias=False)
            self.linear2 = torch.nn.Linear(8, 8)

        def forward(self, x):
            conv_out = self.conv(x)
            reshape_out = torch.reshape(conv_out, (2, 64))
            return self.linear2(self.linear1(reshape_out))

    class ConvLinearWPermute(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 8, 3)
            self.linear1 = torch.nn.Linear(8, 8)

        def forward(self, x):
            conv_out = self.conv(x)
            permute_out = torch.permute(conv_out, (0, 2, 3, 1))
            return self.linear1(permute_out)

    class TwoLinearModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = torch.nn.Linear(8, 16, bias=False)
            self.linear2 = torch.nn.Linear(16, 8)

        def forward(self, x):
            return self.linear2(self.linear1(x))

    class ConvMaxPool2d(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(2, 2, 1)
            self.pool = torch.nn.MaxPool2d(1, 1)

        def forward(self, x):
            x = self.conv(x)
            x = self.pool(x)
            return x

    class ConvWithAdaptiveAvgPool2d(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 3, 3)
            self.adaptive_avg_pool2d = torch.nn.AdaptiveAvgPool2d((1, 1))

        def forward(self, x):
            x = self.conv(x)
            x = self.adaptive_avg_pool2d(x)
            return x

    class ConvWithBNRelu(torch.nn.Module):
        def __init__(self, relu, bn=True, bias=True):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 3, 3, bias=bias)
            if bn:
                self.bn = torch.nn.BatchNorm2d(3)
            else:
                self.bn = torch.nn.Identity()
            if relu:
                self.relu = torch.nn.ReLU()
            else:
                self.relu = torch.nn.Identity()

        def forward(self, x):
            x = self.conv(x)
            x = self.bn(x)
            return self.relu(x)

    class Conv2dWithCat(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(3, 3, 3)
            self.conv2 = torch.nn.Conv2d(3, 3, 3)

        def forward(self, x, y):
            x = self.conv1(x)
            y = self.conv2(y)
            z = torch.cat([x, y], dim=1)
            return z

    class EmbeddingModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = torch.nn.Embedding(num_embeddings=10, embedding_dim=12)

        def forward(self, indices):
            return self.emb(indices)

    class EmbeddingConvLinearModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = torch.nn.Embedding(num_embeddings=10, embedding_dim=8)
            self.conv = torch.nn.Conv2d(8, 16, (1, 3))
            self.linear = torch.nn.Linear(16, 8)

        def forward(self, indices):
            embeddings = self.emb(indices)
            embeddings = torch.unsqueeze(embeddings, dim=0)
            embeddings = torch.permute(embeddings, (0, 3, 1, 2))
            conv_out = self.conv(embeddings)
            conv_out = torch.permute(conv_out, (0, 2, 3, 1))
            conv_out = torch.squeeze(conv_out, dim=0)
            return self.linear(conv_out)

    class AddInplaceAdd(torch.nn.Module):
        def forward(self, x, y):
            x = x + y
            x += y
            return x

    class MulInplaceMul(torch.nn.Module):
        def forward(self, x, y):
            x = x * y
            x *= y
            return x

    class ConvBnReLU2dAndLinearReLU(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv_bn_relu = TestHelperModules.ConvWithBNRelu(relu=True)
            self.linear = torch.nn.Linear(3, 8, bias=False)
            self.relu = torch.nn.ReLU()

        def forward(self, x):
            x = self.conv_bn_relu(x)
            permute_out = torch.permute(x, (0, 2, 3, 1))
            linear_out = self.linear(permute_out)
            return linear_out
