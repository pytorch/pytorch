from __future__ import print_function, division, absolute_import
import torch
import torch.nn as nn
import torch.nn.quantized as nnq

import torch.quantization
import torch.quantization._numeric_suite as ns
from torch.quantization import QuantStub, DeQuantStub
from torch.quantization import (
    default_eval_fn,
    default_qconfig,
    quantize,
)

import copy

_supported_modules = {nn.Linear, nn.Conv2d}
_supported_modules_q = {nnq.Linear, nnq.Conv2d}

def get_module(model, name):
    ''' Given name of submodule, this function grabs the submodule from given model
    '''
    curr = model
    name = name.split('.')
    for subname in name:
        # print(curr)
        # print(subname)
        if subname == '':
            return curr
        curr = curr._modules[subname]
    return curr

def get_parent_module(model, name):
    ''' Given name of submodule, this function grabs the parent of the submodule, from given model
    '''
    parent_name = name.rsplit('.', 1)[0]
    if parent_name == name:
        parent_name = ''
    return get_module(model, parent_name)

def get_param(module, attr):
    ''' Sometimes the weights/bias attribute gives you the raw tensor, but sometimes
    gives a function that will give you the raw tensor, this function takes care of that logic
    '''
    param = getattr(module, attr, None)
    if isinstance(param, nn.Parameter) or isinstance(param, torch.Tensor):
        return param
    elif callable(param):
        return param()
    return None

class MeanShadowLogger(ns.Logger):
    r"""Class used in Shadow module to record the outputs of the original and
    shadow modules.
    """

    def __init__(self):
        super(MeanShadowLogger, self).__init__()
        self.stats["float"] = None
        self.stats["quantized"] = None
        self.count = 0
        self.float_sum = None
        self.quant_sum = None

    def forward(self, x, y):
        if len(x) > 1:
            x = x[0]
        if len(y) > 1:
            y = y[0]
        if x.is_quantized:
            x = x.dequantize()

        self.count += 1
        if self.stats["quantized"] is None:
            self.stats["quantized"] = x
            # self.count = 1
            self.quant_sum = x
        else:
            # self.stats["quantized"] = torch.cat((self.stats["quantized"], x.detach()))
            self.quant_sum += x
            self.stats["quantized"] = self.quant_sum / self.count

        if self.stats["float"] is None:
            self.stats["float"] = y
            # self.count = 1
            self.float_sum = y
        else:
            # self.stats["float"] = torch.cat((self.stats["float"], y.detach()))
            self.float_sum += y
            self.stats["float"] = self.float_sum / self.count

        # print("recorded difference: ", x-y)
        # print("rolling mean difference: ", )

def bias_correction(float_model, quantized_model, img_data, neval_batches=30, white_list=_supported_modules):
    # marking what modules to do bias correction on
    biased_modules = {}
    for name, submodule in quantized_model.named_modules():
        if type(submodule) in _supported_modules_q:
            print("hi :), ", name, type(submodule))
            biased_modules[name] = submodule
    # TODO: figure out way to ensure the order of biased_modules is the same
    # as their order of execution, and/or a way to reorder if not

    for biased_module in biased_modules:
        float_submodule = get_module(float_model, biased_module )
        quantized_submodule = get_module(quantized_model, biased_module)
        bias = get_param(quantized_submodule, 'bias')
        # print(type(float_submodule))
        # print(type(quantized_submodule))

        if bias is not None:
            # Collecting data
            ns.prepare_model_with_stubs(float_model, quantized_model, white_list, MeanShadowLogger)
            # print(quantized_model)
            # print(float_model)
            count = 0
            for data in img_data:
                quantized_model(data[0])
                count += 1
                if count == neval_batches:
                    break
            ob_dict = ns.get_logger_dict(quantized_model)
            # print("break keys from ob_dict: ", ob_dict.keys())
            float_data = ob_dict[biased_module + '.stats']['float']
            quant_data = ob_dict[biased_module + '.stats']['quantized']
            # print(ob_dict)



            # calculuating bias deviation
            epsilon_x =  quant_data - float_data
            # print("difference: ", epsilon_x)
            dims = list(range(epsilon_x.dim()))
            dims.remove(0)
            expected_epsilon_x = torch.mean(epsilon_x, dims)
            # print(expected_epsilon_x)

            # calculating new value for bias parameter

            updated_bias = bias.data - expected_epsilon_x
            updated_bias = updated_bias.reshape(bias.data.size())
            print("update: ", updated_bias)

            # setting new bias
            if isinstance(quantized_submodule.bias, torch.nn.parameter.Parameter):
                quantized_submodule.bias.data = updated_bias
            else:
                quantized_submodule.bias().data = updated_bias
            print("change-oo happened")
            print("bias: ", quantized_submodule.bias())

            # need to remove shadows down here
            for name, submodule in quantized_model.named_modules():
                if isinstance(submodule, ns.Shadow):
                    parent = get_parent_module(quantized_model, name)
                    child_name = name.rsplit('.', 1)[-1]
                    parent._modules[child_name] = submodule.orig_module

        # print('finished ' + biased_module)
    # print(quantized_model)
