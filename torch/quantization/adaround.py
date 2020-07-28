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
    scale_n_add = (continous_V * 1.2) - 0.1  # broadcast should work?
    # TODO: dtypes
    if continous_V.dtype == torch.int8:
        clip = torch.clamp(scale_n_add, -128, 127)
    else:  # add other dtypes
        clip = torch.clamp(scale_n_add, -128, 127)
    return clip

def modified_quantized(model, x):
    weight = x
    continous_V = model.continous_V
    # scale = model.observer_ref.scale
    scale = model.scale
    W_over_s = torch.floor_divide(weight, scale)
    W_plus_H = W_over_s + clipped_sigmoid(continous_V)


    # TODO: dtypes
    if weight.dtype == torch.int8:
        soft_quantized_weights = scale * torch.clamp(W_plus_H, model.quant_min, model.quant_max)
    else:  # add dtype conditional for clambing range
        soft_quantized_weights = scale * torch.clamp(W_plus_H, model.quant_min, model.quant_max)
    return soft_quantized_weights

def loss_function_leaf(model):
    beta = 2
    _lambda = .25

    float_output = model.float_output
    quantized_output = model.quantized_output



    scale = model.wrapped_module.weight_fake_quant.scale
    continous_V = model.wrapped_module.weight_fake_quant.continous_V

    spreading_range = 2 * continous_V - 1
    one_minus_beta = 1 - (spreading_range ** beta)  # torch.exp
    regulization = torch.sum(one_minus_beta)

    Frobenius_norm = torch.norm(float_output - quantized_output)

    # return _lambda * regulization
    return Frobenius_norm + _lambda * regulization

def loss_function(model, input, white_list={nnqat.Conv2d}):
    # model.disable_fake_quant()
    # normal_output = model(input)
    # model.enable_fake_quant()
    # fake_ouput = model(input)


    result = torch.Tensor([0])
    print()
    for name, submodule in model.named_modules():
        # if type(submodule) in white_list and hasattr(submodule, 'weight_fake_quant'):
        if isinstance(submodule, OuputWrapper):
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

    def forward(self, x):
        if self.fake_quant_enabled[0] == 1:
            y = modified_quantized(self, x)
            return super(adaround, self).forward(y)
        return super(adaround, self).forward(x)


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

        # self.output = torch.Tensor.detach(x)
        return self.quantized_output

def add_wrapper_class(model, white_list):
    for name, submodule in model.named_modules():
        if type(submodule) in white_list:
            print(type(submodule))
            parent = get_parent_module(model, name)
            submodule_name = name.split('.')[-1]
            parent._modules[submodule_name] = OuputWrapper(submodule)

            submodule.weight_fake_quant.continous_V = torch.rand(submodule.weight.size(), requires_grad=True)

def init_V(model):
    for name, submodule in model.named_modules():
        if isinstance(submodule, adaround):
            parent = get_parent_module(model, name)
            submodule.continous_V = torch.rand(parent.weight.size())

araround_fake_quant = adaround.with_args()
# get rid of activation param
araround_qconfig = QConfig(activation=FakeQuantize.with_args(observer=MovingAverageMinMaxObserver,
                                                            quant_min=0,
                                                            quant_max=255,
                                                            dtype=torch.quint8, qscheme=torch.per_tensor_affine,
                                                            reduce_range=True),
                          weight=araround_fake_quant)
default_fake_quant = FakeQuantize.with_args(observer=MovingAverageMinMaxObserver, quant_min=0, quant_max=255,
                                            dtype=torch.quint8, qscheme=torch.per_tensor_affine, reduce_range=True)

default_qat_qconfig = QConfig(activation=default_fake_quant,
                              weight=default_weight_fake_quant)

def quick_function():
    prepared_model = ConvChain()
    copy_of_model = copy.deepcopy(prepared_model)
    prepared_model.train()
    img_data = [(torch.rand(10, 3, 125, 125, dtype=torch.float, requires_grad=True), torch.randint(0, 1, (2,), dtype=torch.long))
                for _ in range(500)]
    prepared_model.qconfig = default_qat_qconfig
    prepared_model = torch.quantization.prepare_qat(prepared_model, inplace=False)
    prepared_model.conv2d2.weight_fake_quant = adaround()
    def dummy_generator():
        yield prepared_model.conv2d2.wrapped_module.weight_fake_quant.continous_V

    # init_V(prepared_model)
    add_wrapper_class(prepared_model, {nnqat.Conv2d})
    V_s = [prepared_model.conv2d1.wrapped_module.weight_fake_quant.continous_V,
            prepared_model.conv2d2.wrapped_module.weight_fake_quant.continous_V,
            prepared_model.conv2d3.wrapped_module.weight_fake_quant.continous_V]

    for V in V_s:
        def dummy_generator():
            yield V
        optimizer = torch.optim.Adam(dummy_generator(), lr=0.001)
        for image, target in img_data:
            with torch.autograd.set_detect_anomaly(True):
                output = prepared_model(image)
                loss = loss_function(prepared_model, image)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    # torch.quantization.convert(prepared_model, inplace=True)

    for image, target in img_data:
        float_output = copy_of_model(image)
        quantized_output = prepared_model(image)
        print(computeSqnr(float_output, quantized_output))

def main():
    ## Want to do this sequentially, so we going to want to do the stack thing again, after that
    # the first lead node is gonna need its og weight and the fake stuff, then pump out the modified output
    prepared_model = ConvChain()
    prepared_model.train()
    img_data = [(torch.rand(10, 3, 125, 125, dtype=torch.float), torch.randint(0, 1, (2,), dtype=torch.long))
                for _ in range(5)]
    prepared_model.qconfig = araround_qconfig
    prepared_model = torch.quantization.prepare_qat(prepared_model, inplace=False)

    add_wrapper_class(prepared_model, {nn.Conv2d, nnqat.Conv2d})

    for name, module in prepared_model.named_parameters():
        print(name, module.data.__dict__)
    print(prepared_model)
    # targets = set()
    # for name, module in prepared_model.named_modules():
    #     if hasattr(module, 'weight_fake_quant') and isinstance(module.weight_fake_quant, adaround):
    #         targets.add(module)
    # for target in targets:
    #     target.weight_fake_quant.observer_ref = target.activation_post_process


    optimizer = torch.optim.Adam(prepared_model.parameters(), lr=0.001)

    for image, target in img_data:
        output = prepared_model(image)
        loss = loss_function(prepared_model, image)
        print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(prepared_model)

    # convert prepared_model down here


if __name__ == "__main__":
    # main()
    quick_function()
