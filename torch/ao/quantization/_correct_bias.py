import torch
import torch.nn as nn
import torch.ao.nn.quantized as nnq

import torch.ao.quantization
import torch.ao.ns._numeric_suite as ns

__all__ = [
    "get_module",
    "parent_child_names",
    "get_param",
    "MeanShadowLogger",
    "bias_correction",
]

_supported_modules = {nn.Linear, nn.Conv2d}
_supported_modules_quantized = {nnq.Linear, nnq.Conv2d}

def get_module(model, name):
    ''' Given name of submodule, this function grabs the submodule from given model
    '''
    return dict(model.named_modules())[name]

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

class MeanShadowLogger(ns.Logger):
    r"""A logger for a Shadow module whose purpose is to record the rolling mean
    of the data passed to the floating point and quantized models
    """
    def __init__(self):
        super().__init__()
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

        self.count += 1
        if self.stats["quantized"] is None:
            self.stats["quantized"] = x
            self.quant_sum = x
        else:
            self.quant_sum += x
            self.stats["quantized"] = self.quant_sum / self.count

        if self.stats["float"] is None:
            self.stats["float"] = y
            self.float_sum = y
        else:
            self.float_sum += y
            self.stats["float"] = self.float_sum / self.count

    def clear(self):
        self.stats["float"] = None
        self.stats["quantized"] = None
        self.count = 0
        self.float_sum = None
        self.quant_sum = None

def bias_correction(float_model, quantized_model, img_data, target_modules=_supported_modules_quantized, neval_batches=None):
    ''' Using numeric suite shadow module, the expected output of the floating point and quantized modules
    is recorded. Using that data the bias of supported modules is shifted to compensate for the drift caused
    by quantization
    Paper reference: https://arxiv.org/pdf/1906.04721.pdf (Section 4.2)

    Args:
        float_model: a trained model that serves as a reference to what bias correction should aim for
        quantized_model: quantized form of float_model that bias correction is to applied to
        img_data: calibration data to estimate the expected output (used to find quantization error)
        target_modules: specifies what submodules in quantized_model need bias correction (can be extended to
                unquantized submodules)
        neval_batches: a cap to the number of batches you want to be used for estimating the expected output
    '''
    ns.prepare_model_with_stubs(float_model, quantized_model, _supported_modules, MeanShadowLogger)

    uncorrected_modules = {}
    for name, submodule in quantized_model.named_modules():
        if type(submodule) in target_modules:
            uncorrected_modules[name] = submodule

    for uncorrected_module in uncorrected_modules:
        quantized_submodule = get_module(quantized_model, uncorrected_module)
        bias = get_param(quantized_submodule, 'bias')
        if bias is not None:

            count = 0
            for data in img_data:
                quantized_model(data[0])
                count += 1
                if count == neval_batches:
                    break
            ob_dict = ns.get_logger_dict(quantized_model)
            parent_name, _ = parent_child_names(uncorrected_module)

            float_data = ob_dict[parent_name + '.stats']['float']
            quant_data = ob_dict[parent_name + '.stats']['quantized']

            # math for expected_error
            quantization_error = quant_data - float_data
            dims = list(range(quantization_error.dim()))
            # Note: we don't want to take the mean over the output channel dimension
            dims.remove(1)
            expected_error = torch.mean(quantization_error, dims)

            updated_bias = bias.data - expected_error

            bias.data = updated_bias

            # Resets the data contained in the loggers
            for name, submodule in quantized_model.named_modules():
                if isinstance(submodule, MeanShadowLogger):
                    submodule.clear()
