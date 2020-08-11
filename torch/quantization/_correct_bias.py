from __future__ import print_function, division, absolute_import
import torch
import torch.nn as nn
import torch.nn.quantized as nnq

import torch.quantization
import torch.quantization._numeric_suite as ns

_supported_modules = {nn.Linear, nn.Conv2d}
_supported_modules_quantized = {nnq.Linear, nnq.Conv2d}

def get_module(model, name):
    ''' Given name of submodule, this function grabs the submodule from given model
    '''
    return {name: submodule for name, submodule in model.named_modules()}[name]

def parent_child_names(name):
    '''Splits full name of submodule into parent submodule's full name and submodule's name
    '''
    split_name = name.rsplit('.', 1)
    if len(split_name) == 1:
        return '', split_name[0]
    else:
        return split_name[0], split_name[1]

def get_param(module, attr):
    ''' Sometimes the weights/bias attribute gives you the raw tensor, but sometimes
    gives a function that will give you the raw tensor, this function takes care of that logic
    '''
    param = getattr(module, attr, None)
    if callable(param):
        return param()
    else:
        return param
    return None

class MeanShadowLogger(ns.Logger):
    r"""A logger for a Shadow module whose purpose is to record the rolling mean
    of the data passed to the floating point and quantized models
    """
    def __init__(self):
        super(MeanShadowLogger, self).__init__()
        self.stats["float"] = None
        self.stats["quantized"] = None
        self.count = 0
        self.float_sum = None
        self.quant_sum = None

    def forward(self, x, y):
        ''' The inputs x,y are output data from the quantized and floating-point modules.
        x is for the quantized module, y is for the floating point module
        '''
        if x.is_quantized:
            x = x.dequantize()

        self.count += x.size(0)
        if self.stats["quantized"] is None:
            self.stats["quantized"] = torch.sum(x, 0)
            self.quant_sum = torch.sum(x, 0)
        else:
            self.quant_sum += torch.sum(x, 0)
            self.stats["quantized"] = self.quant_sum / self.count

        if self.stats["float"] is None:
            self.stats["float"] = torch.sum(y, 0)
            self.float_sum = torch.sum(y, 0)
        else:
            self.float_sum += torch.sum(y, 0)
            self.stats["float"] = self.float_sum / self.count

def bias_correction(float_model, quantized_model, img_data, neval_batches=30):
    ''' Using numeric suite shadow module, the expected output of the floating point and quantized modules
    is recorded. Using that data the bias of supported modules is shifted to compensate for the drift caused
    by quantization
    Paper reference: https://arxiv.org/pdf/1906.04721.pdf (Section 4.2)
    '''
    uncorrected_modules = {}
    for name, submodule in quantized_model.named_modules():
        if type(submodule) in _supported_modules_quantized:
            uncorrected_modules[name] = submodule

    for uncorrected_module in uncorrected_modules:
        quantized_submodule = get_module(quantized_model, uncorrected_module)
        bias = get_param(quantized_submodule, 'bias')
        if bias is not None:
            ns.prepare_model_with_stubs(float_model, quantized_model, _supported_modules, MeanShadowLogger)
            count = 0
            for data in img_data:
                quantized_model(data[0])
                count += 1
                if count == neval_batches:
                    break
            ob_dict = ns.get_logger_dict(quantized_model)

            float_data = ob_dict[uncorrected_module + '.stats']['float']
            quant_data = ob_dict[uncorrected_module + '.stats']['quantized']

            # math for expected_error
            quantization_error = quant_data - float_data
            dims = list(range(1, quantization_error.dim()))
            expected_error = torch.mean(quantization_error, dims)

            updated_bias = bias.data - expected_error

            bias.data = updated_bias

            # Removing shadows from model, needed to prevent nesting of shadow modules
            for name, submodule in quantized_model.named_modules():
                if isinstance(submodule, ns.Shadow):
                    parent_name, child_name = parent_child_names(name)
                    parent = get_module(quantized_model, parent_name)
                    parent._modules[child_name] = submodule.orig_module
