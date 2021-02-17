r"""Importing this file includes common utility methods and base clases for
checking quantization api and properties of resulting modules.
"""

import torch
import torch.nn as nn
import torch.nn.quantized as nnq
import torch.nn.quantized.dynamic as nnqd
from torch.nn.intrinsic import _FusedModule
import torch.distributed as dist

from torch.testing._internal.common_utils import TestCase
from torch.quantization import QuantWrapper, QuantStub, DeQuantStub, \
    default_qconfig, default_dynamic_qconfig, default_per_channel_qconfig, QConfig, default_observer, default_weight_observer, \
    propagate_qconfig_, convert, get_default_qconfig, quantize_dynamic_jit, quantize_jit, float_qparams_weight_only_qconfig, \
    get_default_qat_qconfig, PerChannelMinMaxObserver, default_dynamic_quant_observer, QConfigDynamic, QuantType
from torch.quantization.quantization_mappings import (
    get_default_dynamic_quant_module_mappings,
    get_default_qconfig_propagation_list,
    get_default_qat_module_mappings,
)

try:
    # graph mode quantization based on fx
    from torch.quantization.quantize_fx import (
        prepare_fx,
        prepare_qat_fx,
        convert_fx,
    )
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
from typing import Callable, Tuple, Dict

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

class AverageMeter(object):
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
            modules for quantization prepration, e.g.
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
            modules for quantization prepration, e.g.
            quant, dequant and observer
        """
        self.assertTrue(hasattr(module, 'module'))
        self.assertTrue(hasattr(module, 'quant'))
        self.assertTrue(hasattr(module, 'dequant'))

    def checkObservers(self, module, propagate_qconfig_list=None, prepare_custom_config_dict=None):
        r"""Checks the module or module's leaf descendants
            have observers in preperation for quantization
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
           not isinstance(module, torch.quantization.DeQuantStub):
            self.assertTrue(hasattr(module, 'activation_post_process'),
                            'module: ' + str(type(module)) + ' do not have observer')
        # we don't need to check observers for child modules of the
        # qat modules
        if type(module) not in get_default_qat_module_mappings().values() and \
           type(module) not in float_to_observed_module_class_mapping.values() and \
           not isinstance(module, _FusedModule):
            for child in module.children():
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
        for d in [True, False]:
            if dynamic:
                models[d] = quantize_dynamic_jit(model, qconfig_dict, debug=d)
                # make sure it runs
                outputs[d] = models[d](inputs)
            else:
                # module under test can contain in-place ops, and we depend on
                # input data staying constant for comparisons
                inputs_copy = copy.deepcopy(inputs)
                models[d] = quantize_jit(
                    model, qconfig_dict, test_only_eval_fn, [inputs_copy], inplace=False,
                    debug=d)
                # make sure it runs
                outputs[d] = models[d](*inputs[0])

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
        nodes_in_graph = dict()
        node_list = []
        modules = dict(graph_module.named_modules())
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
        modules = dict(graph_module.named_modules())
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

        def assert_types_for_matched_node_pairs(
            self,
            matched_node_pairs: Dict[str, Tuple[Node, Node]],
            expected_types: Dict[str, Tuple[Callable, Callable]],
            gm_a: GraphModule,
            gm_b: GraphModule,
        ) -> None:
            """
            Verifies that the types specified in expected_types match
            the underlying objects pointed to by the nodes in matched_node_pairs.

            An example successful test case:

              matched_node_pairs = {'x0': (graph_a_conv_0_node, graph_b_conv_0_node)}
              expected_types = {'x0': (nn.Conv2d, nnq.Conv2d)}

            The function tests for key equivalence, and verifies types with
            instance checks.
            """

            def _get_underlying_op_type(node: Node, gm: GraphModule) -> Callable:
                if node.op == 'call_module':
                    mod = getattr(gm, node.target)
                    return type(mod)
                else:
                    assert node.op == 'call_function'
                    return node.target

            self.assertTrue(
                len(matched_node_pairs) == len(expected_types),
                'Expected length of results to match, but got %d and %d' %
                (len(matched_node_pairs), len(expected_types))
            )
            for k, v in expected_types.items():
                expected_type_a, expected_type_b = v
                node_a, node_b = matched_node_pairs[k]
                actual_type_a = _get_underlying_op_type(node_a, gm_a)
                actual_type_b = _get_underlying_op_type(node_b, gm_b)
                types_match = (expected_type_a is actual_type_a) and \
                    (expected_type_b is actual_type_b)
                self.assertTrue(
                    types_match,
                    'Type mismatch at %s: expected %s, got %s' %
                    (k, (expected_type_a, expected_type_b), (actual_type_a, actual_type_b))
                )

        def checkGraphModeFxOp(self, model, inputs, quant_type,
                               expected_node=None,
                               expected_node_occurrence=None,
                               expected_node_list=None,
                               debug=False,
                               print_debug_info=False,
                               custom_qconfig=None,
                               prepare_expected_node=None,
                               prepare_expected_node_occurrence=None,
                               prepare_expected_node_list=None,
                               prepare_custom_config_dict=None):
            """ Quantizes model with graph mode quantization on fx and check if the
                quantized model contains the quantized_node

                Args:
                    model: floating point torch.nn.Module
                    inputs: one positional sample input arguments for model
                    expected_node: NodeSpec
                        e.g. NodeSpec.call_function(torch.quantize_per_tensor)
                    expected_node_occurrence: a dict from NodeSpec to
                        expected number of occurences (int)
                        e.g. {NodeSpec.call_function(torch.quantize_per_tensor) : 1,
                                NodeSpec.call_method('dequantize'): 1}
                    expected_node_list: a list of NodeSpec, used to check the order
                        of the occurrence of Node
                        e.g. [NodeSpec.call_function(torch.quantize_per_tensor),
                                NodeSpec.call_module(nnq.Conv2d),
                                NodeSpec.call_function(F.hardtanh_),
                                NodeSpec.call_method('dequantize')]
                    debug: if True, enables debug mode
                    print_debug_info: if True, prints debug info
                    custom_qconfig: overrides default qconfig
                    prepare_expected_node: same as expected_node, but for prepare
                    prepare_expected_node_occurrence: same as
                        expected_node_occurrence, but for prepare
                    prepare_expected_node_list: same as expected_node_list, but
                        for prepare
            """
            # TODO: make img_data a single example instead of a list
            if type(inputs) == list:
                inputs = inputs[0]

            if quant_type == QuantType.QAT:
                qconfig = get_default_qat_qconfig(torch.backends.quantized.engine)
                model.train()
            elif quant_type == QuantType.STATIC:
                qconfig = get_default_qconfig(torch.backends.quantized.engine)
                model.eval()
            else:
                qconfig = default_dynamic_qconfig
                model.eval()

            # overwrite qconfig with custom_qconfig
            if custom_qconfig is not None:
                qconfig = custom_qconfig

            if quant_type == QuantType.QAT:
                prepare = prepare_qat_fx
            else:
                prepare = prepare_fx

            qconfig_dict = {'': qconfig}
            prepared = prepare(
                model, qconfig_dict,
                prepare_custom_config_dict=prepare_custom_config_dict)
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
            qgraph = convert_fx(prepared)
            qgraph_debug = convert_fx(prepared_copy, debug=True)
            result = qgraph(*inputs)
            result_debug = qgraph_debug(*inputs)

            qgraph_to_check = qgraph_debug if debug else qgraph
            if print_debug_info:
                print()
                print('quantized model:\n', qgraph_to_check)
                self.printGraphModule(qgraph_to_check)
                print()
            self.checkGraphModuleNodes(
                qgraph_to_check, expected_node, expected_node_occurrence, expected_node_list)
            return result


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
            float_embedding.qconfig = QConfigDynamic(activation=default_dynamic_quant_observer,
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


# Below are a series of toy models to use in testing quantization

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

class ConvTransposeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(3, 5, 3, bias=False).to(dtype=torch.float)

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

class AnnotatedConvTransposeModel(torch.nn.Module):
    def __init__(self, qengine):
        super().__init__()
        self.qconfig = torch.quantization.get_default_qconfig(qengine)
        self.conv = torch.nn.ConvTranspose2d(3, 5, 3, bias=False).to(dtype=torch.float)
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

class LinearModelWithSubmodule(nn.Module):
    def __init__(self):
        super(LinearModelWithSubmodule, self).__init__()
        self.subm = TwoLayerLinearModel()
        self.fc = nn.Linear(5, 5)

    def forward(self, x):
        x = self.subm(x)
        x = self.fc(x)
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
        self.elu = torch.nn.ELU().to(dtype=torch.float)
        self.dequant = torch.quantization.DeQuantStub()

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

class NormalizationTestModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        self.fc1 = torch.nn.Linear(5, 8).to(dtype=torch.float)
        self.layer_norm = torch.nn.LayerNorm((8))
        self.group_norm = torch.nn.GroupNorm(2, 8)
        self.instance_norm1d = torch.nn.InstanceNorm1d(8)
        self.instance_norm2d = torch.nn.InstanceNorm2d(8)
        self.instance_norm3d = torch.nn.InstanceNorm3d(8)

    def forward(self, x):
        x = self.quant(x)
        x = self.fc1(x)
        x = self.layer_norm(x)
        x = self.group_norm(x.unsqueeze(-1))
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
        torch.quantization.fuse_modules(self, fusable_layers, inplace=True)

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
        self.qconfig = torch.quantization.get_default_qconfig(qengine)
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
        torch.quantization.fuse_modules(self, [['conv1', 'bn1', 'relu1']], inplace=True)

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

class EmbeddingWithLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(num_embeddings=10, embedding_dim=12)
        self.fc = torch.nn.Linear(5, 5)
        self.emb.qconfig = float_qparams_weight_only_qconfig
        self.qconfig = default_qconfig

    def forward(self, indices, linear_in):
        return self.emb(indices), self.fc(linear_in)
