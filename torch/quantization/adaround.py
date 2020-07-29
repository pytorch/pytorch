import torch
import torch.nn as nn
from torch.quantization.fake_quantize import FakeQuantize
from torch.quantization.observer import MovingAverageMinMaxObserver, HistogramObserver, MovingAveragePerChannelMinMaxObserver, _with_args
from torch.quantization.qconfig import *
from torch.quantization.fake_quantize import *
import torch.nn.qat.modules as nnqat
import torch.quantization._numeric_suite as ns

import copy


def clipped_sigmoid(continous_V):
    sigmoid_applied = torch.sigmoid(continous_V)

    #sigmoid_applied vs continues_V
    scale_n_add = (sigmoid_applied * 1.2) - 0.1  # broadcast should work?
    # TODO: dtypes
    # if continous_V.dtype == torch.int8:
    #     clip = torch.clamp(scale_n_add, -128, 127)
    # else:  # add other dtypes
    #     clip = torch.clamp(scale_n_add, -128, 127)
    clip = torch.clamp(scale_n_add, 0, 1)
    return clip

def modified_quantized(model, x):
    weight = x
    continous_V = model.continous_V
    scale = model.scale

    # W_over_s = torch.floor_divide(weight, scale)
    W_over_S = torch.div(weight, scale)
    W_over_S = torch.floor(W_over_S)
    W_plus_H = W_over_S + clipped_sigmoid(continous_V)


    # # TODO: dtypes
    # if weight.dtype == torch.int8:
    #     soft_quantized_weights = scale * torch.clamp(W_plus_H, model.quant_min, model.quant_max)
    # else:  # add dtype conditional for clambing range
    #     soft_quantized_weights = scale * torch.clamp(W_plus_H, model.quant_min, model.quant_max)
    # return soft_quantized_weights
    return scale * torch.clamp(W_plus_H, model.quant_min, model.quant_max)

def loss_function_leaf(model):
    beta = 2
    _lambda = .01

    float_output = model.float_output
    quantized_output = model.quantized_output

    scale = model.wrapped_module.weight_fake_quant.scale
    continous_V = model.wrapped_module.weight_fake_quant.continous_V

    spreading_range = 2 * continous_V - 1
    one_minus_beta = 1 - (spreading_range ** beta)  # torch.exp
    regulization = torch.sum(one_minus_beta)

    Frobenius_norm = torch.norm(float_output - quantized_output)

    # out_of_bounds penalty
    clipped = spreading_range  - torch.clamp(spreading_range, -1, 1)
    clipped = torch.abs(clipped * 100) ** 5
    clipped = torch.sum(clipped)

    # return _lambda * regulization
    print(Frobenius_norm, _lambda * regulization, clipped)
    return Frobenius_norm + _lambda * regulization + clipped

def loss_function(model, input, white_list={nnqat.Conv2d}):
    result = torch.Tensor([0])
    print()
    for name, submodule in model.named_modules():
        if isinstance(submodule, OuputWrapper):
            # print("results running: ", result)
            # return loss_function_leaf(submodule)
            print(name)
            result = result + loss_function_leaf(submodule)
            print("results running: ", result)
    return result

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
    def __init__(self):
        super(adaround, self).__init__()
        self.continous_V = None

    def forward(self, X):
        if self.observer_enabled[0] == 1:
            self.activation_post_process(X.detach())
            _scale, _zero_point = self.calculate_qparams()
            _scale, _zero_point = _scale.to(self.scale.device), _zero_point.to(self.zero_point.device)
            self.scale = _scale
            self.zero_point = _zero_point

        if self.fake_quant_enabled[0] == 1:
            X = modified_quantized(self, X)

        if self.fake_quant_enabled[0] == 1:
            if self.qscheme == torch.per_channel_symmetric or self.qscheme == torch.per_channel_affine:
                X = torch.fake_quantize_per_channel_affine(X, self.scale, self.zero_point,
                                                           self.ch_axis, self.quant_min, self.quant_max)
            else:
                X = torch.fake_quantize_per_tensor_affine(X, float(self.scale),
                                                          int(self.zero_point), self.quant_min,
                                                          self.quant_max)
        # if self.fake_quant_enabled[0] == 1:
        #     X = modified_quantized(self, X)
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

    def forward(self, x):
        self.wrapped_module.activation_post_process.disable_fake_quant()
        self.wrapped_module.weight_fake_quant.disable_fake_quant()
        self.float_output = self.wrapped_module(x)

        self.wrapped_module.activation_post_process.enable_fake_quant()
        self.wrapped_module.weight_fake_quant.enable_fake_quant()
        self.quantized_output = self.wrapped_module(x)

        print(computeSqnr(self.float_output, self.quantized_output))

        return self.quantized_output

def add_wrapper_class(model, white_list):
    V_s = []
    for name, submodule in model.named_modules():
        if type(submodule) in white_list:
            print(type(submodule))
            parent = get_parent_module(model, name)
            submodule_name = name.split('.')[-1]
            parent._modules[submodule_name] = OuputWrapper(submodule)

            submodule.weight_fake_quant.continous_V = copy.deepcopy(submodule.weight)
            V_s.append(parent._modules[submodule_name])
    return V_s


araround_fake_quant = adaround.with_args()
default_fake_quant = FakeQuantize.with_args(observer=MovingAverageMinMaxObserver, quant_min=0, quant_max=255,
                                            dtype=torch.quint8, qscheme=torch.per_tensor_affine, reduce_range=True)

araround_qconfig = QConfig(activation=default_fake_quant,
                          weight=araround_fake_quant)
default_qat_qconfig = QConfig(activation=default_fake_quant,
                              weight=default_weight_fake_quant)

def quick_function(float_model, quantized_model, data_loader_test):
    quantized_model.qconfig = araround_qconfig
    print(quantized_model)
    quantized_model = torch.quantization.prepare_qat(quantized_model, inplace=False)
    V_s = add_wrapper_class(quantized_model, {nnqat.Conv2d})


    for V in V_s:
        def dummy_generator():
            yield V.wrapped_module.weight_fake_quant.continous_V
        optimizer = torch.optim.Adam(dummy_generator(), lr=0.01)
        count = 0
        for image, target in data_loader_test:
            with torch.autograd.set_detect_anomaly(True):
                optimizer.zero_grad()
                output = quantized_model(image)
                # loss = loss_function(quantized_model, image)
                loss = loss_function_leaf(V)
                print("loss: ", loss)
                loss.backward()
                optimizer.step()
                count += 1
                if count > 50:
                    break

    # torch.quantization.convert(quantized_model, inplace=True)

    for image, target in data_loader_test:
        float_output = float_model(image)
        quantized_output = quantized_model(image)
        print(computeSqnr(float_output, quantized_output))

def load_conv():
    model = ConvChain()
    copy_of_model = copy.deepcopy(model)
    model.train()
    img_data = [(torch.rand(10, 3, 125, 125, dtype=torch.float, requires_grad=True), torch.randint(0, 1, (2,), dtype=torch.long))
                for _ in range(500)]
    return copy_of_model, model, img_data


if __name__ == "__main__":
    # main()
    quick_function(*load_conv())
