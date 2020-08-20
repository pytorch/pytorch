import torch
import torch.nn as nn
# import torch.nn.init as init
import math
import torch.quantization._numeric_suite as ns
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

def loss_function(model, count, target_layers, norm_output):
    ''' Given model, the loss function of all of its whitelisted leaf modules will
    be added up and returned.
    '''
    result = torch.Tensor([0])
    for name, submodule in model.named_modules():
        if name in target_layers:
            if norm_output:
                # TODO: collect output from logger
                pass
            else:
                result = result + layer_loss_function(submodule, count, get_param(submodule, 'weight'))
            result = result * .90
    return result

def optimize_V(leaf_module, target_layers, number_of_epochs, norm_output):
    '''Takes in a leaf module with an adaround attached to its
    weight_fake_quant attribute

    Args:
        leaf_module:
        target_layes:
        number_of_epochs:
        norm_output:
    '''
    def dummy_generator():
        yield leaf_module.weight_fake_quant.continous_V
    optimizer = torch.optim.Adam(dummy_generator(), lr=learning_rate)

    count = 0
    for data in tuning_dataset:
        output = float_model(data[0])
        # ob_dict = ns.get_logger_dict(quantized_model)
        #     parent_name, _ = parent_child_names(uncorrected_module)

        #     float_data = ob_dict[parent_name + '.stats']['float']
        #     quant_data = ob_dict[parent_name + '.stats']['quantized']
        # loss = loss_function(float_model, count, target_layers, norm_output)

        # only work if one layer's loss for each training
        loss = loss_function_leaf(leaf_module, count, norm_output)

        print("loss: ", loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        count += 1
        print("running count during optimazation: ", count)
        if count == number_of_epochs:
            return

def learn_adaround(float_model, quantized_model, tuning_dataset, target_layers=None, norm_output=False,
                    number_of_epochs=10, number_of_calibration_batches=30, learning_rate=.1):
    ''' Implements the learning procedure for tuning the rounding scheme of the layers specified
    for the given model

    Args:
        float_model:
        quantized_model:
        tuning_dataset:
        target_layers:
        norm_output:
        number_of_epochs:
        number_of_calibration_batches:
        learning_rate:
    '''
    # this might be wrong setup, idt shadow module is needed?
    if norm_output:
        ns.prepare_model_with_stubs(float_model, quantized_model, _supported_modules, LastOutputLogger)


    if target_layers is None:
        target_layers = []
        for name, submodule in float_model.named_modules():
            if type(submodule) in _supported_modules:
                target_layers.append(name)

    for layer_name in target_layers:
        layer = get_module(float_model, layer_name)
        print("quantized submodule")
        optimize_V(layer, target_layers, number_of_epochs, norm_output)
        print("finished optimizing adaround instance")


    return float_model

if __name__ == "__main__":
    # main()
    learn_adaround(*load_conv())
