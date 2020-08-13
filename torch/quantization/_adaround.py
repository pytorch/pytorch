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

def computeSqnr(x, y):
    Ps = torch.norm(x)
    Pn = torch.norm(x - y)
    return 20 * torch.log10(Ps / Pn)

def get_module(model, name):
    ''' Given name of submodule, this function grabs the submodule from given model
    '''
    curr = model
    name = name.split('.')
    for subname in name:
        if subname == '':
            return curr
        curr = curr._modules[subname]
    return curr

def get_parent_module(model, name):
    ''' Given name of submodule, this function grabs the parent of the submodule, from given model
    '''
    curr = model
    name = name.split('.')[:-1]
    for subname in name:
        if subname == '':
            return curr
        curr = curr._modules[subname]
    return curr

class adaround(FakeQuantize):
    def __init__(self, *args, **keywords):
        super(adaround, self).__init__(*args, **keywords)
        self.continous_V = None
        self.tuning = False

    def forward(self, X):
        if self.observer_enabled[0] == 1:
            self.activation_post_process(X.detach())
            _scale, _zero_point = self.calculate_qparams()
            _scale, _zero_point = _scale.to(self.scale.device), _zero_point.to(self.zero_point.device)
            self.scale = _scale
            self.zero_point = _zero_point

        if self.tuning:
            X = modified_quantized(self, X)

        if self.fake_quant_enabled[0] == 1:
            if self.qscheme == torch.per_channel_symmetric or self.qscheme == torch.per_channel_affine:
                X = torch.fake_quantize_per_channel_affine(X, self.scale, self.zero_point,
                                                           self.ch_axis, self.quant_min, self.quant_max)
            else:
                X = torch.fake_quantize_per_tensor_affine(X, float(self.scale),
                                                          int(self.zero_point), self.quant_min,
                                                          self.quant_max)
        return X

class ConvChain(nn.Module):
    def __init__(self):
        super(ConvChain, self).__init__()
        self.conv2d1 = nn.Conv2d(3, 4, 5, 5)
        self.conv2d2 = nn.Conv2d(4, 5, 5, 5)
        self.conv2d3 = nn.Conv2d(5, 6, 5, 5)

    def forward(self, x):
        x1 = self.conv2d1(x)
        x2 = self.conv2d2(x1)
        x3 = self.conv2d3(x2)
        return x3

class OuputWrapper(nn.Module):
    def __init__(self, model):
        super(OuputWrapper, self).__init__()
        self.wrapped_module = model
        self.float_output = None
        self.quantized_output = None
        self.on = False
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        rtn = None
        if self.on:
            self.wrapped_module.activation_post_process.disable_fake_quant()
            self.wrapped_module.weight_fake_quant.disable_fake_quant()
            self.float_output = self.wrapped_module(x).detach()

            self.wrapped_module.activation_post_process.enable_fake_quant()
            self.wrapped_module.weight_fake_quant.enable_fake_quant()
            self.quantized_output = self.wrapped_module(x)

            print("norm in forward outputwrapper: ", torch.norm(self.float_output - self.quantized_output))

            rtn = self.quantized_output
        else:
            rtn = self.wrapped_module(x)

        return self.dequant(rtn)


araround_fake_quant = adaround.with_args(observer=HistogramObserver, quant_min=-128, quant_max=127,
                                         dtype=torch.qint8, qscheme=torch.per_tensor_symmetric, reduce_range=False)

adaround_qconfig = QConfig(activation=default_fake_quant,
                           weight=araround_fake_quant)

def clipped_sigmoid(continous_V):
    ''' Function to create a non-vanishing gradient for V

    Paper Reference: https://arxiv.org/pdf/2004.10568.pdf Eq. 23
    '''
    sigmoid_applied = torch.sigmoid(continous_V)
    scale_n_add = (sigmoid_applied * 1.2) - 0.1
    clip = torch.clamp(scale_n_add, 0, 1)
    return clip

def modified_quantized(model, x):
    ''' Given a tensor x and variable V and scale from a model, the adaround logic for clipping
    is applied

    Paper Reference: https://arxiv.org/pdf/2004.10568.pdf Eq. 22
    '''
    weight = x
    continous_V = model.continous_V
    scale = model.scale

    W_over_S = torch.div(weight, scale)
    W_over_S = torch.floor(W_over_S)
    W_plus_H = W_over_S + clipped_sigmoid(continous_V)

    soft_quantized_weights = scale * torch.clamp(W_plus_H, model.quant_min, model.quant_max)
    return soft_quantized_weights

def loss_function_leaf(model, count):
    ''' Calculates the loss function for a submodule

    Paper Reference: https://arxiv.org/pdf/2004.10568.pdf Eq. 25
    '''
    high = 8
    low = 2
    beta = count / 100 * (high - low) + low
    _lambda = .01

    # Calculates the difference between floating point and quantized models
    adaround_instance = model.wrapped_module.weight_fake_quant
    float_weight = model.wrapped_module.weight
    clipped_weight = modified_quantized(adaround_instance, float_weight)
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

    print("loss function break down: ", Frobenius_norm * 100, _lambda * regulization)
    # print("sqnr of float and quantized: ", computeSqnr(float_weight, quantized_weight))
    return Frobenius_norm * 100 + _lambda * regulization

def loss_function(model, count, white_list=(nnqat.Conv2d,)):
    ''' Given model, the loss function of all of its whitelisted leaf modules will
    be added up and returned.
    '''
    result = torch.Tensor([0])
    for name, submodule in model.named_modules():
        if isinstance(submodule, OuputWrapper):
            result = result + loss_function_leaf(submodule, count)
    return result

def add_wrapper_class(model, white_list=DEFAULT_QAT_MODULE_MAPPING.keys()):
    ''' Throws on a wrapper class to collect the last output passed through it.
    This information is used in computing the loss function.
    '''
    for name, submodule in model.named_modules():
        print("type of submodule in add_wrapper_class: ", type(submodule))
        if type(submodule) in white_list:
            parent = get_parent_module(model, name)
            submodule_name = name.split('.')[-1]
            parent._modules[submodule_name] = OuputWrapper(submodule)

def learn_adaround_sequential(float_model, data_loader_test, number_of_batches=1, with_adaround=True):
    def optimize_V(leaf_module):
        '''Takes in a leaf module with an adaround attached to its
        weight_fake_quant attribute'''
        def dummy_generator():
            yield leaf_module.wrapped_module.weight_fake_quant.continous_V
        optimizer = torch.optim.Adam(dummy_generator(), lr=.1)

        count = 0
        for data in data_loader_test:
            output = float_model(data[0])
            # loss = loss_function(qat_model, count)
            loss = loss_function_leaf(leaf_module, count)

            print("loss: ", loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            count += 1
            print("running count during optimazation: ", count)
            if count == 100:
                return
            # try:  # try and accept here is because the classifier at the end of the mobilenet has different dims
            #     pass
            #     # print(leaf_module.wrapped_module.weight_fake_quant.continous_V[0][0][:][:])
            # except IndexError:
            #     print("ruh roh")

    # initiallizing all the wrappers on the first layer updating V
    if number_of_batches == 1:
        V_s = add_wrapper_class(float_model, _supported_modules)

    batch = 0
    for name, submodule in float_model.named_modules():
        if isinstance(submodule, OuputWrapper):
            batch += 1
            if batch == number_of_batches:
                # setting up submodule for standard qat training
                if with_adaround:
                    submodule.wrapped_module.qconfig = adaround_qconfig
                else:
                    submodule.wrapped_module.qconfig = torch.quantization.default_qat_qconfig

                # Calibrate the scale and offset parameters
                torch.quantization.prepare_qat(submodule, inplace=True)
                count = 0
                for data in data_loader_test:
                    float_model(data[0])
                    count += 1
                    if count == 10:
                        break

                if with_adaround:
                    submodule.wrapped_module.weight_fake_quant.continous_V = \
                        torch.nn.Parameter(torch.ones(submodule.wrapped_module.weight.size()) / 10)  # dummy input
                    init.kaiming_uniform_(submodule.wrapped_module.weight_fake_quant.continous_V, a=math.sqrt(5))  # uniform distribution of vals
                    submodule.on = True  # wrapper starts to collect outputs
                    submodule.wrapped_module.tuning = True  # continous V is now being applied in the forward of the fakequant
                    submodule.wrapped_module.weight_fake_quant.disable_observer()

                    print("quantized submodule")
                    optimize_V(submodule)
                    print("finished optimizing adaround instance")
                    submodule.on = False
            if batch == number_of_batches:
                return float_model

    return float_model

def learn_adaround_parallel(float_model, data_loader_test):
    # initializing V's and adding wrapper modules
    V_s = add_wrapper_class(float_model, _supported_modules)

    for name, submodule in float_model.named_modules():
        if isinstance(submodule, OuputWrapper):
            submodule.wrapped_module.qconfig = adaround_qconfig
            torch.quantization.prepare_qat(submodule, inplace=True)

            submodule.wrapped_module.weight_fake_quant.continous_V = \
                torch.nn.Parameter(torch.ones(submodule.wrapped_module.weight.size()) / 10)  # dummy input
            init.kaiming_uniform_(submodule.wrapped_module.weight_fake_quant.continous_V, a=math.sqrt(5))  # uniform distribution of vals

    # calibrating the scale and offset parameters
    count = 0
    for data in data_loader_test:
        float_model(data[0])
        count += 1
        if count == 100:
            break

    for name, submodule in float_model.named_modules():
        if isinstance(submodule, OuputWrapper):
            submodule.wrapped_module.tuning = True  # continous V is now being applied in the forward of the fakequant
            submodule.on = True  # wrapper starts to collect outputs


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
        if count == 100:
            return



def load_conv():
    model = ConvChain()
    copy_of_model = copy.deepcopy(model)
    model.train()
    img_data = [(torch.rand(10, 3, 125, 125, dtype=torch.float, requires_grad=True), torch.randint(0, 1, (2,), dtype=torch.long))
                for _ in range(500)]
    return model, img_data

if __name__ == "__main__":
    # main()
    learn_adaround_sequential(*load_conv())
