import torch
import torch.nn as nn
# import torch.nn.init as init
import math
# from torch.quantization.fake_quantize import FakeQuantize
# from torch.quantization.observer import HistogramObserver
# from torch.quantization.qconfig import *
# from torch.quantization.fake_quantize import *
import torch.quantization._numeric_suite as ns
# import torch.nn.qat.modules as nnqat
# from torch.quantization.default_mappings import DEFAULT_QAT_MODULE_MAPPING
# from torch.quantization import QuantStub, DeQuantStub
import copy
_supported_modules = {nn.Conv2d, nn.Linear}

def computeSqnr(x, y):
    Ps = torch.norm(x)
    Pn = torch.norm(x - y)
    return 20 * torch.log10(Ps / Pn)

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




class OutputWrapper(nn.Module):
    def __init__(self, model):
        super(OutputWrapper, self).__init__()
        self.wrapped_module = model

        self.float_output = None
        self.quantized_output = None

        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        self.wrapped_module.activation_post_process.disable_fake_quant()
        self.wrapped_module.weight_fake_quant.disable_fake_quant()
        self.float_output = self.wrapped_module(x).detach()

        self.wrapped_module.activation_post_process.enable_fake_quant()
        self.wrapped_module.weight_fake_quant.enable_fake_quant()
        self.quantized_output = self.wrapped_module(x)

        print("norm in forward outputwrapper: ", torch.norm(self.float_output - self.quantized_output))
        return self.dequant(self.quantized_output)

class LastOutputLogger(ns.Logger):
    r"""A logger for a Shadow module whose purpose is to record the last data point
    passed to the floating point and quantized models
    """
    def __init__(self):
        super(MeanShadowLogger, self).__init__()
        self.stats["float"] = None
        self.stats["quantized"] = None

    def forward(self, x, y):
        ''' The inputs x,y are output data from the quantized and floating-point modules.
        x is for the quantized module, y is for the floating point module
        '''
        if x.is_quantized:
            x = x.dequantize()

        self.stats["quantized"] = x
        self.stats["float"] = y




def loss_function(model, count, target_layers):
    ''' Given model, the loss function of all of its whitelisted leaf modules will
    be added up and returned.
    '''
    result = torch.Tensor([0])
    for name, submodule in model.named_modules():
        if name in target_layers:
            result = result + loss_function_leaf(submodule, count)
            result = result * .90
    return result

def optimize_V(leaf_module, target_layers):
    '''Takes in a leaf module with an adaround attached to its
    weight_fake_quant attribute'''
    def dummy_generator():
        yield leaf_module.wrapped_module.weight_fake_quant.continous_V
    optimizer = torch.optim.Adam(dummy_generator(), lr=learning_rate)

    count = 0
    for data in tuning_dataset:
        output = float_model(data[0])
        loss = loss_function(float_model, count, target_layers)
        # loss = loss_function_leaf(leaf_module, count)

        print("loss: ", loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        count += 1
        print("running count during optimazation: ", count)
        if count == number_of_epochs:
            return

def learn_adaround(float_model, tuning_dataset, target_layers=None, norm_output=True,
                    number_of_epochs=10, number_of_calibration_batches=30, learning_rate=.1):
    ''' Implements the learning procedure for tuning the rounding scheme of the layers specified
    for the given model
    '''
    quantized_model = ????
    ns.prepare_model_with_stubs(float_model, quantized_model, _supported_modules, MeanShadowLogger)


    if target_layers is None:
        target_layers = []
        for name, submodule in float_model.named_modules():
            if type(submodule) in _supported_modules:
                target_layers.append(name)

    print(target_layers)
    for layer_name in target_layers:
        layer = get_module(float_model, layer_name)
        print(layer)
        print(layer_name)
        print("quantized submodule")
        optimize_V(layer, target_layers)
        print("finished optimizing adaround instance")


    return float_model

if __name__ == "__main__":
    # main()
    learn_adaround(*load_conv())
