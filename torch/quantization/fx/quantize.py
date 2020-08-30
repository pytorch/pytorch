import torch
from torch.quantization import (
    propagate_qconfig_,
    convert,
)

from torch.quantization.default_mappings import (
    DEFAULT_QAT_MODULE_MAPPING,
    DEFAULT_MODULE_MAPPING,
    DEFAULT_OPERATOR_MAPPING,
)

from torch.fx import (
    GraphModule,
    Proxy,
)

from torch.fx.graph import (
    Graph,
    Node,
    map_arg,
)

from .pattern_utils import (
    matches,
    register_quant_pattern,
    get_quant_patterns,
    register_dynamic_pattern,
    get_dynamic_quant_patterns,
)

from .utils import _parent_name

from abc import ABC, abstractmethod
import copy
import operator

# ------------------------
# Helper Functions
# ------------------------
def get_qparams(activation_post_process):
    scale, zero_point = activation_post_process.calculate_qparams()
    scale = float(scale)
    zero_point = int(zero_point)
    dtype = activation_post_process.dtype
    return scale, zero_point, dtype

def quantize_node(node, activation_post_process):
    scale, zero_point, dtype = get_qparams(activation_post_process)
    return torch.quantize_per_tensor(node, scale, zero_point, dtype)

def quantize(quantizer, node):
    quantize_node(node, quantizer.activation_post_process_map[node.name])

# Returns a function that can get a new attribute name for module with given prefix
# for example,
# >> get_new_observer_name = get_new_attr_name_with_prefix('_observer')
# >> new_name = get_new_observer_name(module)
# new_name will be an unused attribute name on module, e.g. `_observer_1`
def get_new_attr_name_with_prefix(prefix):
    def get_new_attr_name(module):
        def get_attr_name(i):
            return prefix + str(i)
        i = 0
        attr_name = get_attr_name(i)
        while hasattr(module, attr_name):
            i += 1
            attr_name = get_attr_name(i)
        return attr_name
    return get_new_attr_name

# A dictionary for querying the weight index for a given op
WEIGHT_INDEX_DICT = {
    torch.nn.functional.conv2d : [1],
    torch.nn.functional.linear : [1],
}

# Pattern Registrations

# 1. Post Training Static Quantization and Quantization Aware Training Patterns

# Base Pattern Handler
class QuantizeHandler(ABC):
    """ Base handler class for the quantizer patterns
    """
    def __init__(self, quantizer, node):
        """ Records pattern information in __init__, which will be used
        in convert
        """
        # this is an indicator of whether all the inputs are Node or not
        # since some op might be quantized differently depending on whether
        # all inputs are tensors or not, e.g. add/mul
        self.all_nodes = True

    @abstractmethod
    def convert(self, quantizer, node, load_arg, debug=False):
        """ Convert the given node to a quantized node and insert
        it to the quantized graph
        """
        return NotImplemented

@register_quant_pattern(operator.add)
@register_quant_pattern((torch.nn.ReLU, operator.add))
@register_quant_pattern((torch.nn.functional.relu, operator.add))
class Add(QuantizeHandler):
    def __init__(self, quantizer, node):
        super().__init__(quantizer, node)
        self.relu_node = None
        if (node.op == 'call_function' and node.target is torch.nn.functional.relu) or \
           (node.op == 'call_module' and isinstance(quantizer.modules[node.target], torch.nn.ReLU)):
            self.relu_node = node
            node = node.args[0]
        assert node.op == 'call_function' and node.target == operator.add
        self.add_node = node
        self.all_nodes = all([isinstance(a, Node) for a in self.add_node.args[:2]])

    def convert(self, quantizer, node, load_arg, debug=False):
        if not self.all_nodes:
            # add scalar
            if self.relu_node is not None:
                op = torch.ops.quantized.add_relu
            else:
                op = torch.ops.quantized.add
            return quantizer.quantized_graph.create_node(
                'call_function', op,
                load_arg(quantized=[0])(self.add_node.args), self.add_node.kwargs)
        else:
            activation_post_process = quantizer.activation_post_process_map[node.name]
            scale, zero_point = activation_post_process.calculate_qparams()
            scale = float(scale)
            zero_point = int(zero_point)
            if self.relu_node is not None:
                op = torch.ops.quantized.add_relu
            else:
                op = torch.ops.quantized.add
            kwargs = self.add_node.kwargs
            kwargs.update({'scale': scale, 'zero_point': zero_point})
            return quantizer.quantized_graph.create_node(
                'call_function', op, load_arg(quantized=True)(self.add_node.args), kwargs)

@register_quant_pattern(operator.mul)
@register_quant_pattern((torch.nn.ReLU, operator.mul))
@register_quant_pattern((torch.nn.functional.relu, operator.mul))
class Mul(QuantizeHandler):
    def __init__(self, quantizer, node):
        super().__init__(quantizer, node)
        self.relu_node = None
        if (node.op == 'call_function' and node.target is torch.nn.functional.relu) or \
           (node.op == 'call_module' and isinstance(quantizer.modules[node.target], torch.nn.ReLU)):
            self.relu_node = node
            node = node.args[0]
        assert node.op == 'call_function' and node.target == operator.mul
        self.mul_node = node
        self.all_nodes = all([isinstance(a, Node) for a in self.mul_node.args[:2]])

    def convert(self, quantizer, node, load_arg, debug=False):
        if not self.all_nodes:
            # mul scalar
            if self.relu_node is not None:
                op = torch.ops.quantized.mul_relu
            else:
                op = torch.ops.quantized.mul
            return quantizer.quantized_graph.create_node(
                'call_function', op, load_arg(quantized=[0])(self.mul_node.args), self.mul_node.kwargs)
        else:
            activation_post_process = quantizer.activation_post_process_map[node.name]
            scale, zero_point = activation_post_process.calculate_qparams()
            scale = float(scale)
            zero_point = int(zero_point)
            if self.relu_node is not None:
                op = torch.ops.quantized.mul_relu
            else:
                op = torch.ops.quantized.mul
            kwargs = self.mul_node.kwargs
            kwargs.update({'scale': scale, 'zero_point': zero_point})
            return quantizer.quantized_graph.create_node('call_function', op, load_arg(quantized=True)(self.mul_node.args), kwargs)

@register_quant_pattern(torch.cat)
class Cat(QuantizeHandler):
    def convert(self, quantizer, node, load_arg, debug=False):
        if not self.all_nodes:
            return NotImplemented
        activation_post_process = quantizer.activation_post_process_map[node.name]
        scale, zero_point = activation_post_process.calculate_qparams()
        scale = float(scale)
        zero_point = int(zero_point)
        kwargs = load_arg(quantized=False)(node.kwargs)
        kwargs.update({'scale': scale, 'zero_point': zero_point})
        return quantizer.quantized_graph.create_node(
            'call_function', torch.ops.quantized.cat, load_arg(quantized=[0])(node.args), kwargs)

# handle conv, maybe followed by relu
# NB: matching order is reversed, that is we match from the bottom of this list to the beginning
@register_quant_pattern(torch.nn.Conv1d)
@register_quant_pattern(torch.nn.Conv2d)
@register_quant_pattern(torch.nn.Conv3d)
@register_quant_pattern(torch.nn.functional.conv2d)
@register_quant_pattern(torch.nn.qat.Conv2d)
@register_quant_pattern(torch.nn.intrinsic.ConvReLU1d)
@register_quant_pattern(torch.nn.intrinsic.ConvReLU2d)
@register_quant_pattern(torch.nn.intrinsic.ConvReLU3d)
@register_quant_pattern(torch.nn.intrinsic.qat.ConvBn2d)
@register_quant_pattern(torch.nn.intrinsic.qat.ConvBnReLU2d)
@register_quant_pattern(torch.nn.intrinsic.qat.ConvReLU2d)
@register_quant_pattern((torch.nn.functional.relu, torch.nn.functional.conv2d))
@register_quant_pattern((torch.nn.ReLU, torch.nn.functional.conv2d))
# just for error checks
@register_quant_pattern((torch.nn.ReLU, torch.nn.Conv2d))
@register_quant_pattern((torch.nn.functional.relu, torch.nn.Conv2d))
class ConvRelu(QuantizeHandler):
    def __init__(self, quantizer, node):
        super().__init__(quantizer, node)
        self.relu_node = None
        if (node.op == 'call_function' and node.target is torch.nn.functional.relu) or \
           (node.op == 'call_module' and isinstance(quantizer.modules[node.target], torch.nn.ReLU)):
            self.relu_node = node
            node = node.args[0]
        self.conv_node = node
        if node.op == 'call_module':
            self.conv = quantizer.modules[self.conv_node.target]

    def convert(self, quantizer, node, load_arg, debug=False):
        # TODO: debug option for conv module
        if self.conv_node.op == 'call_module':
            # note that relu should already be fused into conv module in the fusion step
            assert self.relu_node is None, 'conv module and relu fusion is not executed, ' \
                'please make sure to run fusion before prepare'
            # 1. attach activation post process to module
            if type(self.conv) in [
                    torch.nn.intrinsic.ConvReLU1d,
                    torch.nn.intrinsic.ConvReLU2d,
                    torch.nn.intrinsic.ConvReLU3d
            ]:
                self.conv[1].activation_post_process = quantizer.activation_post_process_map[node.name]
            else:
                self.conv.activation_post_process = quantizer.activation_post_process_map[node.name]
            # 2. select quantized class
            # TODO: make the mapping configurable?
            assert type(self.conv) in DEFAULT_MODULE_MAPPING, \
                'unhandled conv type:{}'.format(type(self.conv))
            qconv_cls = DEFAULT_MODULE_MAPPING[type(self.conv)]
            quantized = qconv_cls.from_float(self.conv)
            parent_name, name = _parent_name(self.conv_node.target)
            setattr(quantizer.modules[parent_name], name, quantized)
            return quantizer.quantized_graph.create_node(
                'call_module',
                self.conv_node.target,
                (load_arg(quantized=True)(self.conv_node.args[0]),),
                {})
        elif self.conv_node.op == 'call_function':
            if self.relu_node is not None:
                raise Exception("functional conv + relu is not supported yet")
            if debug:
                args = load_arg(quantized=[0, 1])(self.conv_node.args)
                args = load_arg(quantized=False)(self.conv_node.args)
                kwargs = load_arg(quantized=False)(self.conv_node.kwargs)
                conv_out = quantizer.quantized_graph.create_node(
                    'call_function', torch.nn.functional.conv2d, args, kwargs)
                return quantize_node(
                    conv_out, quantizer.activation_post_process_map[self.conv_node.name])
            else:
                assert len(self.conv_node.args) == 7, \
                    'only conv2d calls with all arguments specified is support right now in debug=False option'
                args = load_arg(quantized=[0, 1])(self.conv_node.args)
                # pack weight
                weight = load_arg(quantized=True)(self.conv_node.args[1])
                other_args = load_arg(quantized=False)(self.conv_node.args[2:])
                prepack_args = tuple([weight] + list(other_args))
                packed_weight = quantizer.quantized_graph.create_node(
                    'call_function', torch.ops.quantized.conv2d_prepack, prepack_args, {})
                # construct conv input
                conv_input = load_arg(quantized=True)(self.conv_node.args[0])
                activation_post_process = quantizer.activation_post_process_map[self.conv_node.name]
                scale, zero_point, _ = get_qparams(activation_post_process)
                qconv_args = (conv_input, packed_weight, scale, zero_point)
                kwargs = load_arg(quantized=False)(self.conv_node.kwargs)
                return quantizer.quantized_graph.create_node(
                    'call_function', torch.ops.quantized.conv2d, qconv_args, kwargs)

# handle linear, maybe followed by relu
@register_quant_pattern(torch.nn.Linear)
@register_quant_pattern(torch.nn.functional.linear)
@register_quant_pattern(torch.nn.qat.Linear)
@register_quant_pattern(torch.nn.intrinsic.LinearReLU)
@register_quant_pattern(torch.nn.intrinsic.qat.LinearReLU)
@register_quant_pattern((torch.nn.functional.relu, torch.nn.functional.linear))
@register_quant_pattern((torch.nn.ReLU, torch.nn.functional.linear))
# for error checks
@register_quant_pattern((torch.nn.ReLU, torch.nn.Linear))
@register_quant_pattern((torch.nn.functional.relu, torch.nn.Linear))
class LinearReLU(QuantizeHandler):
    def __init__(self, quantizer, node):
        super().__init__(quantizer, node)
        self.relu_node = None
        if (node.op == 'call_function' and node.target is torch.nn.functional.relu) or \
           (node.op == 'call_module' and isinstance(quantizer.modules[node.target], torch.nn.ReLU)):
            self.relu_node = node
            node = node.args[0]
        self.linear_node = node
        if node.op == 'call_module':
            self.linear = quantizer.modules[self.linear_node.target]

    def convert(self, quantizer, node, load_arg, debug=False):
        # TODO: debug option for linear module
        if self.linear_node.op == 'call_module':
            # note that relu should already be fused into conv module in the fusion step
            assert self.relu_node is None, 'linear module and relu fusion is not executed, ' \
                'please make sure to run fusion before prepare'
            # 1. attach activation post process to module
            if type(self.linear) == torch.nn.intrinsic.LinearReLU:
                self.linear[1].activation_post_process = quantizer.activation_post_process_map[node.name]
            else:
                self.linear.activation_post_process = quantizer.activation_post_process_map[node.name]
            # 2. select quantized class
            if type(self.linear) in [torch.nn.Linear, torch.nn.qat.Linear]:
                qlinear = torch.nn.quantized.Linear
            elif type(self.linear) in [torch.nn.intrinsic.LinearReLU, torch.nn.intrinsic.qat.LinearReLU]:
                qlinear = torch.nn.intrinsic.quantized.LinearReLU
            else:
                raise Exception("unhandled linear type:", type(self.linear))
            quantized = qlinear.from_float(self.linear)
            parent_name, name = _parent_name(self.linear_node.target)
            setattr(quantizer.modules[parent_name], name, quantized)
            return quantizer.quantized_graph.create_node(
                'call_module',
                self.linear_node.target, (load_arg(quantized=True)(self.linear_node.args[0]),), {})
        elif self.linear_node.op == 'call_function':
            if debug:
                args = load_arg(quantized=[0, 1])(self.linear_node.args)
                args = load_arg(quantized=False)(self.linear_node.args)
                kwargs = load_arg(quantized=False)(self.linear_node.kwargs)
                linear_out = quantizer.quantized_graph.create_node(
                    'call_function', torch.nn.functional.linear, args, kwargs)
                return quantize_node(
                    linear_out,
                    quantizer.activation_post_process_map[self.linear_node.name])
            else:
                # TODO: this code can be merged with dynamic linear code
                # linear args
                # (x, weight, bias, ...)
                args = load_arg(quantized=[0, 1])(self.linear_node.args)
                kwargs = load_arg(quantized=False)(self.linear_node.kwargs)
                # pack weight
                weight = load_arg(quantized=True)(self.linear_node.args[1])
                bias = None
                # all args after bias, including bias
                other_args = load_arg(quantized=False)(self.linear_node.args[2:])
                if len(self.linear_node.args) > 2:
                    bias = load_arg(quantized=False)(self.linear_node.args[2])
                    other_args = other_args[1:]  # remove the bias argument
                else:
                    assert 'bias' in kwargs, \
                        'expect bias provided as a keyword argument when it is not a positional argument'
                    bias = kwargs['bias']
                    kwargs.pop('bias')
                prepack_args = (weight, bias)
                packed_weight = quantizer.quantized_graph.create_node(
                    'call_function', torch.ops.quantized.linear_prepack, prepack_args, {})
                # construct linear input
                linear_input = load_arg(quantized=True)(self.linear_node.args[0])
                activation_post_process = \
                    quantizer.activation_post_process_map[self.linear_node.name]
                scale, zero_point, _ = get_qparams(activation_post_process)
                qlinear_args = (linear_input, packed_weight, scale, zero_point)
                return quantizer.quantized_graph.create_node(
                    'call_function', torch.ops.quantized.linear, qlinear_args, kwargs)

@register_quant_pattern(torch.nn.BatchNorm2d)
@register_quant_pattern(torch.nn.BatchNorm3d)
@register_quant_pattern(torch.nn.intrinsic.BNReLU2d)
@register_quant_pattern(torch.nn.intrinsic.BNReLU3d)
class BatchNorm(QuantizeHandler):
    def __init__(self, quantizer, node):
        super().__init__(quantizer, node)
        assert node.op == 'call_module'
        self.bn_node = node
        self.bn = quantizer.modules[self.bn_node.target]

    def convert(self, quantizer, node, load_arg, debug=False):
        # 1. attach activation post process to module
        activation_post_process = quantizer.activation_post_process_map[node.name]
        if type(self.bn) in \
            [torch.nn.intrinsic.BNReLU2d,
             torch.nn.intrinsic.BNReLU3d]:
            self.bn[1].activation_post_process = activation_post_process
        else:
            self.bn.activation_post_process = activation_post_process
        qbn_cls = DEFAULT_MODULE_MAPPING[type(self.bn)]
        quantized = qbn_cls.from_float(self.bn)
        parent_name, name = _parent_name(self.bn_node.target)
        setattr(quantizer.modules[parent_name], name, quantized)
        return quantizer.quantized_graph.create_node(
            'call_module',
            self.bn_node.target,
            load_arg(quantized=[0])(self.bn_node.args),
            load_arg(quantized=False)(self.bn_node.kwargs))

ARGS_TO_SKIP = {
    torch._ops.ops.quantized.hardswish: ['inplace'],
    torch._ops.ops.quantized.instance_norm:
    ['running_mean', 'running_var', 'use_input_stats', 'momentum'],
}
@register_quant_pattern(torch.nn.ELU)
@register_quant_pattern(torch.nn.Hardswish)
@register_quant_pattern(torch.nn.InstanceNorm1d)
@register_quant_pattern(torch.nn.InstanceNorm2d)
@register_quant_pattern(torch.nn.InstanceNorm3d)
@register_quant_pattern(torch.nn.LayerNorm)
@register_quant_pattern(torch.nn.functional.hardswish)
@register_quant_pattern(torch.nn.functional.instance_norm)
@register_quant_pattern(torch.nn.functional.layer_norm)
class DefaultNode(QuantizeHandler):
    ''' Common quantized op, first input and first output will be quantized
    '''
    def convert(self, quantizer, node, load_arg, debug=False):
        if not self.all_nodes:
            return NotImplemented
        assert node.op in ['call_module', 'call_function'], 'Only call_module and ' + \
            'call_function are handled in DefaultNode'
        activation_post_process = quantizer.activation_post_process_map[node.name]
        if node.op == 'call_module':
            module = quantizer.modules[node.target]
            module.activation_post_process = activation_post_process
            quantized_module = DEFAULT_MODULE_MAPPING[type(module)].from_float(module)
            parent_name, name = _parent_name(node.target)
            setattr(quantizer.modules[parent_name], name, quantized_module)
            return quantizer.quantized_graph.create_node(
                'call_module',
                node.target,
                load_arg(quantized=[0])(node.args),
                load_arg(quantized=False)(node.kwargs))
        else:
            # call_function
            scale, zero_point = activation_post_process.calculate_qparams()
            scale = float(scale)
            zero_point = int(zero_point)

            quantized_op = DEFAULT_OPERATOR_MAPPING[node.target]
            args = load_arg(quantized=[0])(node.args)
            kwargs = load_arg(quantized=False)(node.kwargs)
            kwargs.update({'output_scale': scale, 'output_zero_point': zero_point})
            if quantized_op in ARGS_TO_SKIP:
                args_to_skip = ARGS_TO_SKIP[quantized_op]
                for arg in args_to_skip:
                    if arg in kwargs:
                        kwargs.pop(arg)
            return quantizer.quantized_graph.create_node(
                'call_function', quantized_op, args, kwargs)

# TODO: elu is using scale/zero_point instead of output_scale, output_zero_point
@register_quant_pattern(torch.nn.functional.elu)
class ELU(QuantizeHandler):
    def convert(self, quantizer, node, load_arg, debug=False):
        activation_post_process = quantizer.activation_post_process_map[node.name]
        scale, zero_point = activation_post_process.calculate_qparams()
        scale = float(scale)
        zero_point = int(zero_point)
        quantized_op = DEFAULT_OPERATOR_MAPPING[node.target]
        args = load_arg(quantized=[0])(node.args)
        kwargs = load_arg(quantized=False)(node.kwargs)
        kwargs.update({'output_scale': scale, 'output_zero_point': zero_point})
        kwargs.pop('inplace')
        return quantizer.quantized_graph.create_node(
            'call_function', quantized_op, args, kwargs)

# these ops have quantized equivalents that do not need any extra information
@register_quant_pattern(torch.nn.AdaptiveAvgPool1d)
@register_quant_pattern(torch.nn.AdaptiveAvgPool2d)
@register_quant_pattern(torch.nn.AdaptiveAvgPool3d)
@register_quant_pattern(torch.nn.AvgPool1d)
@register_quant_pattern(torch.nn.AvgPool2d)
@register_quant_pattern(torch.nn.AvgPool3d)
@register_quant_pattern(torch.nn.Dropout)
@register_quant_pattern(torch.nn.Hardsigmoid)
@register_quant_pattern(torch.nn.Hardtanh)
@register_quant_pattern(torch.nn.LeakyReLU)
@register_quant_pattern(torch.nn.MaxPool1d)
@register_quant_pattern(torch.nn.MaxPool2d)
@register_quant_pattern(torch.nn.MaxPool3d)
@register_quant_pattern(torch.nn.ReLU)
@register_quant_pattern(torch.nn.ReLU6)
@register_quant_pattern(torch.nn.Sigmoid)
@register_quant_pattern(torch.nn.Tanh)
@register_quant_pattern(torch.adaptive_avg_pool1d)
@register_quant_pattern(torch.nn.functional.adaptive_avg_pool2d)
@register_quant_pattern(torch.nn.functional.adaptive_avg_pool3d)
@register_quant_pattern(torch.nn.functional.dropout)
@register_quant_pattern(torch.nn.functional.hardsigmoid)
@register_quant_pattern(torch.nn.functional.hardtanh)
@register_quant_pattern(torch.nn.functional.hardtanh_)
@register_quant_pattern(torch.nn.functional.interpolate)
@register_quant_pattern(torch.nn.functional.leaky_relu)
@register_quant_pattern(torch.nn.functional.max_pool1d)
@register_quant_pattern(torch.nn.functional.max_pool2d)
@register_quant_pattern(torch.nn.functional.max_pool3d)
@register_quant_pattern(torch.nn.functional.relu)
@register_quant_pattern(torch.nn.functional.relu6)
@register_quant_pattern(torch.avg_pool1d)
@register_quant_pattern(torch._C._nn.avg_pool2d)
@register_quant_pattern(torch._C._nn.avg_pool3d)
@register_quant_pattern(torch.chunk)
@register_quant_pattern(torch.clamp)
@register_quant_pattern(torch.flatten)
@register_quant_pattern(torch.transpose)
@register_quant_pattern(torch.max)
@register_quant_pattern(torch.mean)
@register_quant_pattern(torch.min)
@register_quant_pattern(torch.repeat_interleave)
@register_quant_pattern(torch.sigmoid)
@register_quant_pattern(torch.sort)
@register_quant_pattern(torch.squeeze)
@register_quant_pattern(torch.stack)
@register_quant_pattern(torch.tanh)
@register_quant_pattern(torch.unsqueeze)
@register_quant_pattern(operator.getitem)
@register_quant_pattern(operator.floordiv)
@register_quant_pattern('chunk')
@register_quant_pattern('clamp')
@register_quant_pattern('contiguous')
@register_quant_pattern('detach')
@register_quant_pattern('detach_')
@register_quant_pattern('hardsigmoid')
@register_quant_pattern('hardsigmoid_')
@register_quant_pattern('leaky_relu')
@register_quant_pattern('leaky_relu_')
@register_quant_pattern('mean')
@register_quant_pattern('numel')
@register_quant_pattern('permute')
@register_quant_pattern('relu')
@register_quant_pattern('relu_')
@register_quant_pattern('repeat')
@register_quant_pattern('repeat_interleave')
@register_quant_pattern('reshape')
@register_quant_pattern('resize_')
@register_quant_pattern('shape')
@register_quant_pattern('sigmoid')
@register_quant_pattern('sigmoid_')
@register_quant_pattern('size')
@register_quant_pattern('squeeze')
@register_quant_pattern('squeeze_')
@register_quant_pattern('tanh')
@register_quant_pattern('tanh_')
@register_quant_pattern('transpose')
@register_quant_pattern('unsqueeze')
@register_quant_pattern('unsqueeze_')
@register_quant_pattern('view')
class CopyNode(QuantizeHandler):
    def convert(self, quantizer, node, load_arg, debug=False):
        return quantizer.quantized_graph.node_copy(node, load_arg(quantized=None))

class DefaultQuant(QuantizeHandler):
    def convert(self, quantizer, node):
        assert self.all_nodes
        return quantize(quantizer, node)

# 2. Post Training Dynamic Quantizatoin Patterns
@register_dynamic_pattern(torch.nn.Linear)
@register_dynamic_pattern(torch.nn.functional.linear)
class DynamicLinear(QuantizeHandler):
    def __init__(self, quantizer, node):
        super().__init__(quantizer, node)
        self.linear_node = node
        if node.op == 'call_module':
            assert isinstance(quantizer.modules[node.target], torch.nn.Linear)
            self.linear = quantizer.modules[self.linear_node.target]

    def convert(self, quantizer, node, load_arg, debug=False):
        if self.linear_node.op == 'call_module':
            quantized = torch.nn.quantized.dynamic.Linear.from_float(self.linear)
            parent_name, name = _parent_name(self.linear_node.target)
            setattr(quantizer.modules[parent_name], name, quantized)
            return quantizer.quantized_graph.create_node(
                'call_module',
                self.linear_node.target,
                (load_arg(quantized=False)(self.linear_node.args[0]),),
                {})
        elif self.linear_node.op == 'call_function':
            if debug:
                # quantize and dequantize weight
                args = load_arg(quantized=[1])(self.linear_node.args)
                args = load_arg(quantized=False)(self.linear_node.args)
                kwargs = load_arg(quantized=False)(self.linear_node.kwargs)
                return quantizer.quantized_graph.create_node(
                    'call_function', torch.nn.functional.linear, args, kwargs)
            else:
                # linear args:
                # (x, weight, bias)
                # quantize weight
                quantized_weight = load_arg(quantized=True)(self.linear_node.args[1])
                bias = None
                # all args after bias, including bias
                other_args = load_arg(quantized=False)(self.linear_node.args[2:])
                kwargs = load_arg(quantized=False)(self.linear_node.kwargs)
                if len(self.linear_node.args) > 2:
                    bias = load_arg(quantized=False)(self.linear_node.args[2])
                    other_args = other_args[1:]  # remove the bias argument
                else:
                    assert 'bias' in kwargs, \
                        'expect bias provided as a keyword argument when it is not a positional argument'
                    bias = kwargs['bias']
                    kwargs.pop('bias')
                prepack_args = (quantized_weight, bias)
                # pack weight
                packed_weight = quantizer.quantized_graph.create_node(
                    'call_function', torch.ops.quantized.linear_prepack, prepack_args, {})
                # construct dynamic linear input
                non_quantized_input = load_arg(quantized=False)(self.linear_node.args[0])
                qdynamic_linear_args = (non_quantized_input, packed_weight)
                return quantizer.quantized_graph.create_node(
                    'call_function', torch.ops.quantized.linear_dynamic, qdynamic_linear_args, kwargs)


# weight prepacking ops
WEIGHT_PREPACK_OPS = {
    torch._ops.ops.quantized.linear_prepack,
    torch._ops.ops.quantized.conv2d_prepack,
}

class Quantizer:
    def __init__(self):
        # mapping from matched node to activation_post_process
        # must be filled before convert
        self.activation_post_process_map = None

    def _qat_swap_modules(self, root):
        convert(root, mapping=DEFAULT_QAT_MODULE_MAPPING, inplace=True, remove_qconfig=False)

    def _generate_qconfig_map(self, root, input_graph):
        def get_qconfig(module):
            return module.qconfig if hasattr(module, 'qconfig') else None

        self.qconfig_map = dict()
        for node in input_graph.nodes:
            if node.op == 'get_param':
                parent, _ = _parent_name(node.target)
                self.qconfig_map[node.name] = get_qconfig(self.modules[parent])
            elif node.op == 'call_function':
                self.qconfig_map[node.name] = get_qconfig(root)
            elif node.op == 'call_method':
                self_obj = node.args[0]
                # qconfig for call_method should be the same as the `self` object for the call
                self.qconfig_map[node.name] = self.qconfig_map[self_obj.name]
            elif node.op == 'call_module':
                self.qconfig_map[node.name] = get_qconfig(self.modules[node.target])

    def _prepare(self, model, qconfig_dict, inplace, is_dynamic_quant):
        assert not inplace, 'inplace prepare is not supported yet'
        input_root = model.root
        if not inplace:
            input_root = copy.deepcopy(input_root)

        input_graph = model.graph
        self.is_dynamic_quant = is_dynamic_quant
        # TODO: allow user specified patterns
        if self.is_dynamic_quant:
            self.patterns = get_dynamic_quant_patterns()
        else:
            self.patterns = get_quant_patterns()

        propagate_qconfig_(input_root, qconfig_dict)
        if input_root.training:
            self._qat_swap_modules(input_root)

        self.modules = dict(input_root.named_modules())

        # map from node name to qconfig, used in _find_matches
        self._generate_qconfig_map(input_root, input_graph)

        # match the patterns that will get quantized
        matches = self._find_matches(input_graph, self.modules, self.patterns)

        # find _inputs_ to matched nodes that are not quantized, these
        # have to be quantized, which requires measuring stats,
        # initialize an DefaultQuant object for each
        quants = self._find_quants(input_graph, matches)

        self.activation_post_process_map = dict()

        env = {}
        observed_graph = Graph()
        observed = set()

        def load_arg(a):
            return map_arg(a, lambda node: env[node.name])

        for node in input_graph.nodes:
            if node.name in observed:
                continue

            get_new_observer_name = get_new_attr_name_with_prefix('activation_post_process_')
            root_node, _, obj, qconfig = matches.get(node.name, (None, None, None, None))
            if root_node is None:
                env[node.name] = observed_graph.node_copy(node, load_arg)
            elif root_node is node:
                env[node.name] = observed_graph.node_copy(node, load_arg)

                def insert_observer(node, observer):
                    observer_name = get_new_observer_name(input_root)
                    setattr(input_root, observer_name, observer)
                    self.activation_post_process_map[node.name] = observer
                    env[node.name] = observed_graph.create_node('call_module', observer_name, [load_arg(node)], {})
                    observed.add(node.name)

                # don't need to insert observer for output in dynamic quantization
                if self.is_dynamic_quant:
                    continue

                if isinstance(obj, CopyNode):
                    assert node.op in [
                        'call_module',
                        'call_function',
                        'call_method'], \
                        'CopyNode of type ' + node.op + ' is not handled'

                    def is_observed(input_arg):
                        if isinstance(input_arg, Node):
                            return input_arg.name in observed
                        elif isinstance(input_arg, list):
                            return all(map(is_observed, input_arg))
                    # propagate observed property from input
                    if is_observed(node.args[0]):
                        observed.add(node.name)
                elif (isinstance(obj, Add) or isinstance(obj, Mul)) and not obj.all_nodes:
                    if node.args[0].name in observed:
                        observed.add(node.name)
                elif qconfig is not None and obj.all_nodes:
                    # observer for outputs
                    insert_observer(node, qconfig.activation())
            else:
                env[node.name] = observed_graph.node_copy(node, load_arg)

            if node.name not in observed and node.name in quants:
                observer_name = get_new_observer_name(input_root)
                _, qconfig, is_weight = quants[node.name]
                if qconfig is not None:
                    self.activation_post_process_map[node.name] = qconfig.weight() if is_weight else qconfig.activation()
                    setattr(input_root, observer_name, self.activation_post_process_map[node.name])
                    env[node.name] = observed_graph.create_node('call_module', observer_name, [load_arg(node)], {})
                    observed.add(node.name)
        observed_graph.output(load_arg(input_graph.result))

        observed = GraphModule(input_root, observed_graph)
        self.save_state(observed)
        return observed

    def save_state(self, observed):
        observed._activation_post_process_map = self.activation_post_process_map
        observed._patterns = self.patterns
        observed._qconfig_map = self.qconfig_map

    def restore_state(self, observed):
        err_msg = 'please make sure the model is produced by prepare'
        assert hasattr(observed, '_activation_post_process_map'), 'did not found ' + \
            '_activation_post_process attribute ' + err_msg
        assert hasattr(observed, '_patterns'), 'did not found ' + \
            '_patterns attribute ' + err_msg
        assert hasattr(observed, '_qconfig_map'), 'did not found ' + \
            '_qconfig_map attribute ' + err_msg
        self.activation_post_process_map = observed._activation_post_process_map
        self.patterns = observed._patterns
        self.qconfig_map = observed._qconfig_map

    def prepare(self, model, qconfig_dict, inplace=False):
        return self._prepare(model, qconfig_dict, inplace, is_dynamic_quant=False)

    def prepare_dynamic(self, model, qconfig_dict, inplace=False):
        return self._prepare(model, qconfig_dict, inplace, is_dynamic_quant=True)

    def _convert(self, observed, inplace=False, debug=False, is_dynamic_quant=False):
        assert not inplace, 'inplace convert is not supported yet'
        self.restore_state(observed)
        self.is_dynamic_quant = is_dynamic_quant
        # move to cpu since we only have quantized cpu kernels
        observed.eval().cpu()
        observed_root = observed.root
        observed_graph = observed.graph
        if not inplace:
            observed_root = copy.deepcopy(observed_root)

        self.modules = dict(observed_root.named_modules())

        matches = self._find_matches(observed.graph, self.modules, self.patterns)
        quants = self._find_quants(observed.graph, matches)
        self.quantized_graph = Graph()
        env = {}
        quant_env = {}

        def load_non_quantized(n):
            if n.name not in env:
                assert n.name in quant_env, \
                    'trying to load float node but did not find node:' + n.name + \
                    ' in quantized environment:' + str(quant_env)
                env[n.name] = Proxy(quant_env[n.name]).dequantize().node
            return env[n.name]

        def load_quantized(n):
            if n.name not in quant_env:
                assert n.name in env, \
                    'trying to load quantized node but did not find node:' + n.name + \
                    ' in float environment:' + str(env)
                assert n.name in quants, 'did not find quant object for node:' + n.name
                quant = quants[n.name][0]
                quant_env[n.name] = quant.convert(self, env[n.name])
            return quant_env[n.name]

        def load_x(n):
            assert n.name in env or n.name in quant_env, \
                'node ' + n.name + ' does not exist in either of the environment'
            if n.name in quant_env:
                return quant_env[n.name]
            else:
                return env[n.name]

        def load_arg(quantized):
            """
            if quantized is a list, then arg should be a list and the args with corresponding
            indexes will be quantized
            if quantized is a boolean, then all args will be quantized/not quantized
            if quantized is None, then we'll load the node as long as it exists
            """
            assert quantized is None or isinstance(quantized, (tuple, list, bool)), type(quantized)

            def load_arg_impl(arg):
                if quantized is None:
                    return map_arg(arg, load_x)
                if isinstance(quantized, bool):
                    return map_arg(arg, load_quantized if quantized else load_non_quantized)
                elif isinstance(quantized, (tuple, list)):
                    assert isinstance(arg, (tuple, list)), arg
                    loaded_arg = []
                    # for now, we only support quantizing positional arguments
                    for i, a in enumerate(arg):
                        if i in quantized:
                            loaded_arg.append(map_arg(a, load_quantized))
                        else:
                            loaded_arg.append(map_arg(a, load_non_quantized))
                    return type(arg)(loaded_arg)
            return load_arg_impl

        def is_quantized(node):
            if isinstance(node, Node):
                assert node.name in env or node.name in quant_env, 'Expecting node to be in the environment'
                # there might be nodes appearing in both environemnts, but quant_env will take
                # precedence
                if node.name in quant_env:
                    return True
                elif node.name in env:
                    return False
            elif isinstance(node, list):
                quantized = map(is_quantized, node)
                if all(quantized):
                    return True
                elif not any(quantized):
                    return False
                else:
                    raise Exception("partially quantized inputs in list not handled yet")

        for node in observed_graph.nodes:
            root_node, matched, obj, qconfig = matches.get(node.name, (None, None, None, None))
            if root_node is node:
                result = obj.convert(self, node, load_arg)
                quantized = True
                # Need to get correct quantized/non-quantized state for the output of CopyNode
                if isinstance(obj, CopyNode):
                    assert node.op in [
                        'call_module',
                        'call_function',
                        'call_method'], \
                        'CopyNode of type ' + node.op + ' is not handled'
                    quantized = is_quantized(node.args[0])

                # output of dynamic quantization is not quantized
                if self.is_dynamic_quant:
                    quantized = False

                if quantized:
                    quant_env[node.name] = result
                else:
                    env[node.name] = result
                continue
            elif root_node is not None:
                continue

            # handle activation post process calls
            if node.op == 'call_module':
                if node.target.split('.')[-1].startswith('activation_post_process_'):
                    observer_module = self.modules[node.target]
                    prev_node = node.args[0]
                    if prev_node.name in quant_env:
                        # if previous node is already quantized, we'll just remove the activation_post_process
                        quant_env[node.name] = quant_env[prev_node.name]
                        continue
                    # replace activation post process with quantization ops
                    parent_name = ''

                    scale, zero_point = observer_module.calculate_qparams()
                    dtype = observer_module.dtype

                    def is_per_channel(qscheme):
                        return qscheme == torch.per_channel_affine or \
                            qscheme == torch.per_channel_symmetric

                    if is_per_channel(observer_module.qscheme):
                        ch_axis = int(observer_module.ch_axis)
                        qparams = {'_scale_': scale, '_zero_point_': zero_point, '_axis': ch_axis, '_dtype_': dtype}
                        quantize_op = torch.quantize_per_channel
                    else:
                        scale = float(scale)
                        zero_point = int(zero_point)
                        qparams = {'_scale_': scale, '_zero_point_': zero_point, '_dtype_': dtype}
                        quantize_op = torch.quantize_per_tensor
                    i = 0

                    def noattr(module, qparams, i):
                        for name in qparams.keys():
                            if hasattr(module, name + str(i)):
                                return False
                        return True

                    def get_next_i(module, qparams):
                        i = 0
                        while not noattr(module, qparams, i):
                            i += 1
                        return i

                    parent_module = self.modules[parent_name]
                    i = get_next_i(parent_module, qparams)
                    inputs = [load_non_quantized(node.args[0])]
                    for key, value in qparams.items():
                        setattr(parent_module, key + str(i), value)
                        qparam_full_path = key + str(i)
                        if parent_name:
                            qparam_full_path = parent_name + '.' + qparam_full_path
                        inputs.append(self.quantized_graph.create_node('get_param', qparam_full_path))
                    quant_env[node.name] = self.quantized_graph.create_node('call_function', quantize_op, tuple(inputs), {})
                    continue
            # dequantize inputs for the node that are not quantized
            env[node.name] = self.quantized_graph.node_copy(node, load_non_quantized)

        self.quantized_graph.output(load_non_quantized(observed_graph.result))

        to_be_removed = []
        for name, _ in observed_root.named_modules():
            if name.split('.')[-1].startswith('activation_post_process_'):
                to_be_removed.append(name)
        for n in to_be_removed:
            delattr(observed_root, n)
        return GraphModule(observed_root, self.quantized_graph)

    # Trace back from the weight node util we hit getattr, reconstruct the graph module
    # with the traced nodes and run the graph module to pack the weight. then replace
    # the original chain of ops with the packed weight.
    def _fold_weight(self, quantized):
        def collect_nodes_to_fold(node):
            nodes = [node]
            frontier = [node]
            while frontier:
                node = frontier.pop()
                all_args = list(node.args) + list(node.kwargs.values())
                for arg in all_args:
                    if not isinstance(arg, Node):
                        continue
                    if arg.op == 'placeholder':
                        # hit input, can't fold in this case
                        return None
                    nodes.append(arg)
                    if not (arg.op == 'call_function' and arg.target == getattr):
                        frontier.append(arg)
            return nodes

        packed_weights = dict()
        # map from folded node name to the prepacked weight name
        folded_nodes = dict()
        # get packed weights
        for node in quantized.graph.nodes:
            if node.op == 'call_function' and node.target in WEIGHT_PREPACK_OPS:
                nodes_to_fold = collect_nodes_to_fold(node)
                if nodes_to_fold is not None:
                    # since we traced back from weight node to getattrr
                    nodes_to_fold.reverse()
                    prepacking_graph = Graph()
                    env = {}

                    def load_arg(a):
                        return map_arg(a, lambda node: env[node.name])
                    for node_to_fold in nodes_to_fold:
                        env[node_to_fold.name] = prepacking_graph.node_copy(node_to_fold, load_arg)
                        folded_nodes[node_to_fold.name] = node
                    prepacking_graph.output(load_arg(node.name))
                    prepacking_module = GraphModule(quantized.root, prepacking_graph)
                    packed_weight = prepacking_module()
                    packed_weights[node.name] = packed_weight

        # remove folded nodes and replace the prepacking node with getattr
        folded_graph = Graph()
        env = {}

        def load_arg(a):
            return map_arg(a, lambda node: env[node.name])
        get_new_packed_weight_name = get_new_attr_name_with_prefix('_fx_pass_packed_weight_')
        quantized_root = quantized.root
        quantized_graph = quantized.graph
        for node in quantized_graph.nodes:
            prepack_node = folded_nodes.get(node.name, None)
            if prepack_node is node:
                packed_weight = packed_weights[node.name]
                # add a prepacked attribute to root
                packed_weight_name = get_new_packed_weight_name(quantized_root)
                setattr(quantized_root, packed_weight_name, packed_weight)
                # replace prepack node with a getattr node
                env[node.name] = folded_graph.create_node(
                    'get_param', packed_weight_name, (), {})
            elif prepack_node is not None:
                # remove the foled node
                continue
            else:
                # copy other nodes
                env[node.name] = folded_graph.node_copy(node, load_arg)
        folded_graph.output(load_arg(quantized_graph.result))
        return GraphModule(quantized_root, folded_graph)

    def convert(self, observed, inplace=False, debug=False, is_dynamic=False):
        quantized = self._convert(observed, inplace, debug, is_dynamic)
        if not debug:
            quantized = self._fold_weight(quantized)
        return quantized

    def _find_matches(self, graph, modules, patterns):
        match_map = {}  # node name -> (root_node, match_value?)
        all_matched = set()

        def record_match(pattern, node, matched):
            if isinstance(pattern, tuple):
                s, *args = pattern
                record_match(s, node, matched)
                if pattern[0] is not getattr:
                    for subpattern, arg in zip(args, node.args):
                        record_match(subpattern, arg, matched)
            else:
                matched.append(node)

        for node in reversed(graph.nodes):
            if node.name not in match_map and node.name not in all_matched:
                for pattern, value in patterns.items():
                    if matches(modules, node, pattern):
                        matched = []
                        record_match(pattern, node, matched)
                        for n in matched:
                            match_map[n.name] = (node, matched, value(self, node), self.qconfig_map[n.name])
                            all_matched.add(n.name)
                        # break after finding the first match
                        break
        return match_map

    def _find_quants(self, graph, matches):
        quants = {}

        def visit(node, qconfig):
            def visit_arg(arg):
                # note: we have to measure quantization information
                # even for nodes where we might not use it because it is already
                # quantized. This is because each match has the option to
                # say NotImplemented (if for instance, it is an __add__ and the data type is not appropriate)
                is_weight = False
                if isinstance(node, Node) and node.op == 'call_function' and node.target in WEIGHT_INDEX_DICT:
                    for i, node_arg in enumerate(node.args):
                        if arg is node_arg and i in WEIGHT_INDEX_DICT[node.target]:
                            is_weight = True
                if (not self.is_dynamic_quant) or is_weight:
                    # overwrite previous quant config
                    quants[arg.name] = (DefaultQuant(self, arg), qconfig, is_weight)
            return visit_arg

        for node in graph.nodes:
            if node.name in matches:
                root_node, matched, obj, qconfig = matches[node.name]
                # don't attach observer/fake_quant for CopyNode
                if isinstance(obj, CopyNode):
                    qconfig = None
                if root_node is node:
                    # matched[-1] is the first op in the sequence and
                    # matched[0] is the last op in the sequence
                    # inputs
                    map_arg(matched[-1].args, visit(matched[-1], qconfig))
                    map_arg(matched[-1].kwargs, visit(matched[-1], qconfig))
                    # output
                    map_arg(matched[0], visit(None, qconfig))
        return quants
