import torch
import torch.nn as nn
import torch.nn.qat as nnqat
# import torch.nn.init as init

import torch.quantization._numeric_suite as ns
import copy
from torch.quantization.boiler_code import load_conv
from torch.quantization.adaround_fake_quantize import *

_supported_modules = {nn.Conv2d, nn.Linear}
_supported_modules_qat = {nnqat.Conv2d, nnqat.Linear}

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

def get_param(module, attr):
    ''' Sometimes the weights/bias attribute gives you the raw tensor, but sometimes
    gives a function that will give you the raw tensor, this function takes care of that logic
    '''
    param = getattr(module, attr, None)
    if callable(param):
        return param()
    else:
        return param

class LastOutputLogger(ns.Logger):
    r"""A logger for a Shadow module whose purpose is to record the last data point
    passed to the floating point and quantized models and is used to compute the norm
    of the difference between the two models. To be passed to AdaRoundFakeQuantize's
    layer_loss_function
    """
    def __init__(self):
        super(LastOutputLogger, self).__init__()
        self.stats["tensor_val"] = None
        # self.stats["quantized"] = None

    def forward(self, x):
        ''' The inputs x,y are output data from the quantized and floating-point modules.
        x is for the quantized module, y is for the floating point module
        '''
        if x.is_quantized:
            x = x.dequantize()

        self.stats["tensor_val"] = x
        # self.stats["float"] = y

def learn_adaround(float_model, quantized_model, tuning_dataset, target_layers=None, norm_output=False,
                    number_of_epochs=15, learning_rate=.5):
    ''' Implements the learning procedure for tuning the rounding scheme of the layers specified
    for the given model

    Args:
        float_model:
        quantized_model:
        tuning_dataset:
        target_layers:
        norm_output:
        number_of_epochs:
        learning_rate:
    '''
    def optimize_V(leaf_module):
        '''Takes in a leaf module with an adaround attached to its
        weight_fake_quant attribute and runs an adam optimizer on the continous_V
        attribute on the adaround module
        '''
        print(leaf_module)
        leaf_module.weight_fake_quant.tuning = True
        def dummy_generator():
            yield leaf_module.weight_fake_quant.continous_V
        optimizer = torch.optim.Adam(dummy_generator(), lr=learning_rate)

        count = 0
        for data in tuning_dataset:
            # output = float_model(data[0])
            output = quantized_model(data[0])
            # act_compare_dict = ns.get_matching_activations(float_model, quantized_model)
            # wt_compare_dict = ns.compare_weights(float_model.state_dict(), quantized_model.state_dict())
            # print(quantized_model)
            # print(wt_compare_dict.keys())
            # ob_dict = ns.get_logger_dict(quantized_model)
            # ob_dict2 = ns.get_logger_dict(leaf_module)
            # print("dict 1: ", ob_dict)
            # print("dict 2: ", ob_dict2)
            # parent_name, _ = parent_child_names(uncorrected_module)

            # float_data = ob_dict[parent_name + '.stats']['float']
            # quant_data = ob_dict[parent_name + '.stats']['quantized']
            # need to handle more norm_output stuff
            loss = leaf_module.weight_fake_quant.layer_loss_function(count / number_of_epochs, leaf_module.weight)

            print("loss: ", loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            count += 1
            print("running count during optimazation: ", count)
            if count == number_of_epochs:
                return
    # this might be wrong setup, idt shadow module is needed?
    # if norm_output:
    #     # ns.prepare_model_with_stubs(float_model, quantized_model, _supported_modules, LastOutputLogger)
    #     ns.prepare_model_outputs(float_model, quantized_model, LastOutputLogger)

    if target_layers is None:
        target_layers = []
        for name, submodule in float_model.named_modules():
            if type(submodule) in _supported_modules:
                target_layers.append(name)

    for layer_name in target_layers:
        layer = get_module(quantized_model, layer_name)
        print("quantized submodule")
        optimize_V(layer)
        print("finished optimizing adaround instance")


    return float_model

if __name__ == "__main__":
    # main()
    x,y = load_conv()
    z = copy.deepcopy(x)
    z.qconfig = adaround_qconfig
    torch.quantization.prepare_qat(z, inplace=True)
    print(z)
    z(y[0][0])
    learn_adaround(x, z, y, norm_output=False)
