import torch
import torch.nn as nn
import torch.nn.init as init
import math
from torch.quantization.fake_quantize import FakeQuantize
from torch.quantization.observer import HistogramObserver
from torch.quantization.qconfig import *
from torch.quantization.fake_quantize import *
import torch.nn.qat.modules as nnqat
from torch.quantization.default_mappings import DEFAULT_QAT_MODULE_MAPPING
from torch.quantization import QuantStub, DeQuantStub
import copy
_supported_modules = {nn.Conv2d, nn.Linear}

# Hyper parameters for loss function
beta_high = 8
beta_low = 2
norm_scaling = 100
regularization_scaling = .01

#
number_of_epochs = 100
number_of_calibration_batches = 10
learning_rate = .1

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

class adaround(FakeQuantize):
    def __init__(self, *args, **keywords):
        super(adaround, self).__init__(*args, **keywords)
        self.continous_V = None

    def forward(self, X):
        if self.continous_V is None:
            # small initial values are intended so no bias is introduced on initialization or calibration
            self.continous_V = torch.nn.Parameter(torch.ones(X.size()) / 10000)

        if self.observer_enabled[0] == 1:
            self.activation_post_process(X.detach())
            _scale, _zero_point = self.calculate_qparams()
            _scale, _zero_point = _scale.to(self.scale.device), _zero_point.to(self.zero_point.device)
            self.scale = _scale
            self.zero_point = _zero_point

        if self.fake_quant_enabled[0] == 1:
            X = adaround_rounding(self, X)
            if self.qscheme == torch.per_channel_symmetric or self.qscheme == torch.per_channel_affine:
                X = torch.fake_quantize_per_channel_affine(X, self.scale, self.zero_point,
                                                           self.ch_axis, self.quant_min, self.quant_max)
            else:
                X = torch.fake_quantize_per_tensor_affine(X, float(self.scale),
                                                          int(self.zero_point), self.quant_min,
                                                          self.quant_max)
        return X

    def randomize(self):
        # uniform distribution of vals
        init.kaiming_uniform_(self.continous_V, a=math.sqrt(5))


class OuputWrapper(nn.Module):
    def __init__(self, model):
        super(OuputWrapper, self).__init__()
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


araround_fake_quant = adaround.with_args(observer=HistogramObserver, quant_min=-128, quant_max=127,
                                         dtype=torch.qint8, qscheme=torch.per_tensor_symmetric, reduce_range=False)

adaround_qconfig = QConfig(activation=default_fake_quant,
                           weight=araround_fake_quant)

def clipped_sigmoid(continous_V):
    ''' Function to create a non-vanishing gradient for V

    Paper Reference: https://arxiv.org/pdf/2004.10568.pdf Eq. 23
    '''
    sigmoid_of_V = torch.sigmoid(continous_V)
    scale_and_add = (sigmoid_of_V * 1.2) - 0.1
    return torch.clamp(scale_and_add, 0, 1)

def adaround_rounding(model, x):
    ''' Using the scale and continous_V parameters of the model, the given tensor x is
    rounded using adaround

    Paper Reference: https://arxiv.org/pdf/2004.10568.pdf Eq. 22
    '''
    weight = x
    continous_V = model.continous_V
    scale = model.scale

    weights_divided_by_scale = torch.div(weight, scale)
    weights_divided_by_scale = torch.floor(weights_divided_by_scale)
    weights_clipped = weights_divided_by_scale + clipped_sigmoid(continous_V)

    weights_w_adaround_rounding = scale * torch.clamp(weights_clipped, model.quant_min, model.quant_max)
    return weights_w_adaround_rounding

def loss_function_leaf(model, count):
    ''' Calculates the loss function for a submodule

    Paper Reference: https://arxiv.org/pdf/2004.10568.pdf Eq. 25
    '''
    beta = count / number_of_epochs * (beta_high - beta_low) + beta_low


    # Calculates the difference between floating point and quantized models
    adaround_instance = model.wrapped_module.weight_fake_quant
    float_weight = model.wrapped_module.weight
    clipped_weight = adaround_rounding(adaround_instance, float_weight)
    quantized_weight = torch.fake_quantize_per_tensor_affine(clipped_weight, float(adaround_instance.scale),
                                                             int(adaround_instance.zero_point), adaround_instance.quant_min,
                                                             adaround_instance.quant_max)
    # Frobenius_norm = torch.norm(float_weight - quantized_weight)  # norm(W - ~W)
    Frobenius_norm = torch.norm(model.float_output - model.quantized_output)  # norm(Wx - ~Wx)

    # calculating regularization factor -> forces values of continous_V to be 0 or 1
    scale = adaround_instance.scale
    continous_V = adaround_instance.continous_V
    clip_V = clipped_sigmoid(continous_V)
    spreading_range = torch.abs((2 * clip_V) - 1)
    one_minus_beta = 1 - (spreading_range ** beta)  # torch.exp
    regulization = torch.sum(one_minus_beta)

    print("loss function break down: ", Frobenius_norm * norm_scaling, regularization_scaling * regulization)
    # print("sqnr of float and quantized: ", computeSqnr(float_weight, quantized_weight))
    return Frobenius_norm * norm_scaling + regularization_scaling * regulization

def loss_function(model, count):
    ''' Given model, the loss function of all of its whitelisted leaf modules will
    be added up and returned.
    '''
    result = torch.Tensor([0])
    for name, submodule in model.named_modules():
        if isinstance(submodule, OuputWrapper):
            result = result + loss_function_leaf(submodule, count)
            result = result * .95
    return result

def add_wrapper_class(model, white_list=DEFAULT_QAT_MODULE_MAPPING.keys()):
    ''' Throws on a wrapper class to collect the last output passed through it.
    This information is used in computing the loss function.
    '''
    for name, submodule in model.named_modules():
        print("type of submodule in add_wrapper_class: ", type(submodule))
        if type(submodule) in white_list:
            parent_name, child_name = parent_child_names(name)
            parent_module = get_module(model, parent_name)
            parent_module._modules[child_name] = OuputWrapper(submodule)
            submodule.qconfig = adaround_qconfig



def learn_adaround_sequential(float_model, data_loader_test, target_layers=None, with_adaround=True):
    ''' Convert number of batches to a dict? that tells you names of layer you want adaround applied to
    '''
    if target_layers is None:
        target_layers = []
        for name, submodule in float_model.named_modules():
            if type(submodule) in _supported_modules:
                target_layers.append(name)

    def optimize_V(leaf_module):
        '''Takes in a leaf module with an adaround attached to its
        weight_fake_quant attribute'''
        def dummy_generator():
            yield leaf_module.wrapped_module.weight_fake_quant.continous_V
        optimizer = torch.optim.Adam(dummy_generator(), lr=learning_rate)

        count = 0
        for data in data_loader_test:
            output = float_model(data[0])
            loss = loss_function(float_model, count)
            # loss = loss_function_leaf(leaf_module, count)

            print("loss: ", loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            count += 1
            print("running count during optimazation: ", count)
            if count == number_of_epochs:
                return

    # initiallizing all the wrappers
    V_s = add_wrapper_class(float_model, _supported_modules)
    float_model.qconfig = torch.quantization.default_qat_qconfig
    torch.quantization.prepare_qat(float_model, inplace=True)

    count = 0
    for data in data_loader_test:
        print(float_model)
        float_model(data[0])
        count += 1
        if count == number_of_calibration_batches:
            break

    for layer_name in target_layers:
        print(target_layers)
        layer = get_module(float_model, layer_name)
        print("quantized submodule")
        optimize_V(layer)
        print("finished optimizing adaround instance")


    return float_model

def learn_adaround_parallel(float_model, data_loader_test):
    # initializing V's and adding wrapper modules
    V_s = add_wrapper_class(float_model, _supported_modules)

    for name, submodule in float_model.named_modules():
        if isinstance(submodule, OuputWrapper):
            submodule.wrapped_module.qconfig = adaround_qconfig
            torch.quantization.prepare_qat(submodule, inplace=True)

    # calibrating the scale and offset parameters
    count = 0
    for data in data_loader_test:
        float_model(data[0])
        count += 1
        if count == number_of_calibration_batches:
            break

    def dummy_generator():
        for name, submodule in float_model.named_modules():
            if isinstance(submodule, OuputWrapper):
                yield submodule.wrapped_module.weight_fake_quant.continous_V
    optimizer = torch.optim.Adam(dummy_generator(), lr=.1)

    # training all the continous V variables
    count = 0
    for data in data_loader_test:
        output = float_model(data[0])
        loss = loss_function(float_model, count)
        # loss = loss_function_leaf(leaf_module, count)

        print("loss: ", loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        count += 1
        print("running count during optimazation: ", count)
        if count == number_of_epochs:
            return



if __name__ == "__main__":
    # main()
    learn_adaround_sequential(*load_conv())
