import torch
import torch.nn as nn
from torch.quantization.fake_quantize import FakeQuantize
from torch.quantization.observer import MovingAverageMinMaxObserver, HistogramObserver, MovingAveragePerChannelMinMaxObserver, _with_args
from torch.quantization.qconfig import *
from torch.quantization.fake_quantize import *
import torch.nn.qat.modules as nnqat
import torch.quantization._numeric_suite as ns
from torch.quantization.default_mappings import DEFAULT_QAT_MODULE_MAPPING
import copy
_supported_modules = {nn.Conv2d, nn.Linear}




def clipped_sigmoid(continous_V):
    sigmoid_applied = torch.sigmoid(continous_V)
    scale_n_add = (sigmoid_applied * 1.2) - 0.1
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

    rtn = scale * torch.clamp(W_plus_H, model.quant_min, model.quant_max)
    return rtn

def loss_function_leaf(model, count):
    # print("model: ", model)
    high = 4
    low = 1
    beta = count/10 * (high - low) + low
    _lambda = .01

    adaround_instance = model.wrapped_module.weight_fake_quant
    float_weight = model.wrapped_module.weight
    clipped_weight = modified_quantized(adaround_instance, float_weight)
    # FW_over_scale = torch.div(float_weight, adaround_instance.scale)
    # _w_Floor = torch.floor(FW_over_scale)
    # W_plus_sig = _w_Floor + clipped_sigmoid(adaround_instance.continous_V)
    # clipp = torch.clamp(W_plus_sig, adaround_instance.quant_min, adaround_instance.quant_max)
    # scaled_final_answer = adaround_instance.scale * clipp
    # clipped_weight = scaled_final_answer
    ## next is clippy


    quantized_weight = torch.fake_quantize_per_tensor_affine(clipped_weight, float(adaround_instance.scale),
                                                int(adaround_instance.zero_point), adaround_instance.quant_min,
                                                adaround_instance.quant_max)

    scale = model.wrapped_module.weight_fake_quant.scale
    continous_V = model.wrapped_module.weight_fake_quant.continous_V

    clip_V = clipped_sigmoid(continous_V)
    spreading_range = torch.abs((2 * clip_V) - 1)
    one_minus_beta = 1 - (spreading_range ** beta)  # torch.exp
    print(one_minus_beta.size())
    regulization = torch.sum(one_minus_beta)

    # Frobenius_norm = torch.norm(continous_V- float_weight)
    # up to clip is ok, scaled_final _answer isn't ok
    Frobenius_norm = torch.norm(float_weight - quantized_weight)
    # Frobenius_norm = torch.norm(model.float_output - model.quantized_output)
    print("float size: ", float_weight.size())

    print("loss function break down: ", Frobenius_norm*100, _lambda * regulization)
    print("sqnr of float and quantized: ", computeSqnr(float_weight, quantized_weight))
    return Frobenius_norm*100 + _lambda * regulization #+ clipped

def loss_function(model, count, white_list={nnqat.Conv2d}):
    result = torch.Tensor([0])
    for name, submodule in model.named_modules():
        if isinstance(submodule, OuputWrapper):
            result = result + loss_function_leaf(submodule, count)
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
            assert X is not None
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
        # self.hacky_input = None

    def forward(self, x):
        # self.hacky_input = x.detach()
        self.wrapped_module.activation_post_process.disable_fake_quant()
        self.wrapped_module.weight_fake_quant.disable_fake_quant()
        self.float_output = self.wrapped_module(x).detach()

        self.wrapped_module.activation_post_process.enable_fake_quant()
        self.wrapped_module.weight_fake_quant.enable_fake_quant()
        self.quantized_output = self.wrapped_module(x)

        return self.quantized_output

default_weight_fake_quant = FakeQuantize.with_args(observer=MovingAverageMinMaxObserver, quant_min=-128, quant_max=127,
                                                   dtype=torch.qint8, qscheme=torch.per_tensor_symmetric, reduce_range=False)

araround_fake_quant = adaround.with_args(observer=MovingAverageMinMaxObserver, quant_min=-128, quant_max=127,
                                                   dtype=torch.qint8, qscheme=torch.per_tensor_symmetric, reduce_range=False)

adaround_qconfig = QConfig(activation=default_fake_quant,
                          weight=araround_fake_quant)
default_qat_qconfig22 = QConfig(activation=default_fake_quant,
                              weight=default_weight_fake_quant)

def add_wrapper_class(model, white_list=DEFAULT_QAT_MODULE_MAPPING.values()):
    V_s = []
    for name, submodule in model.named_modules():
        # print(type(submodule), white_list)
        if type(submodule) in white_list:
            print("adding wrapper")
            parent = get_parent_module(model, name)
            submodule_name = name.split('.')[-1]
            parent._modules[submodule_name] = OuputWrapper(submodule)

            # submodule.weight_fake_quant.continous_V = copy.deepcopy(submodule.weight)
            submodule.weight_fake_quant.continous_V = torch.nn.Parameter(torch.ones(submodule.weight.size()) / 10)
            assert submodule.weight_fake_quant.continous_V is not None
            V_s.append(parent._modules[submodule_name])
    return V_s


def load_conv():
    model = ConvChain()
    copy_of_model = copy.deepcopy(model)
    model.train()
    img_data = [(torch.rand(10, 3, 125, 125, dtype=torch.float, requires_grad=True), torch.randint(0, 1, (2,), dtype=torch.long))
                for _ in range(500)]
    return copy_of_model, model, img_data

def quick_function(qat_model, dummy, data_loader_test):
    # turning off observer and turning on tuning
    for name, submodule in qat_model.named_modules():
        if type(submodule) in _supported_modules:
            # submodule.weight_fake_quant.disable_observer()
            # submodule.weight_fake_quant.enable_fake_quant()
            submodule.weight_fake_quant.tuning = True


    V_s = add_wrapper_class(qat_model)
    def uniform_images():
        for image, target in data_loader_test:
            yield image
    generator = uniform_images()

    batch = 0
    for name, submodule in qat_model.named_modules():
        if isinstance(submodule, OuputWrapper):
            # submodule.wrapped_module.weight_fake_quant.enable_observer()
            if batch < 3:
                def dummy_generator():
                    yield submodule.wrapped_module.weight_fake_quant.continous_V
                optimizer = torch.optim.Adam(dummy_generator(), lr=.1)

                for count in range(10):
                    output = qat_model(next(generator))
                    # loss = loss_function(qat_model, count)
                    loss = loss_function_leaf(submodule, count)

                    print("loss: ", loss)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    try:
                        print(submodule.wrapped_module.weight_fake_quant.continous_V[0][0][:][:])
                    except IndexError:
                        print("ruh roh")

                    print("running count during optimazation: ", count)
            if batch == 3:
                return qat_model
            batch +=1
            # submodule.wrapped_module.weight_fake_quant.disable_observer()


    # qat_model.eval()
    # torch.quantization.convert(qat_model, inplace=True)
    return qat_model


if __name__ == "__main__":
    # main()
    quick_function(*load_conv())
