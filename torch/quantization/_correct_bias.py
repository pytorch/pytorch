from __future__ import print_function, division, absolute_import
import numpy as np
import torch
import torch.nn as nn
import torch.nn.quantized as nnq
import torch.nn.quantized.dynamic as nnqd
from  torch.nn.quantized.modules import Quantize, DeQuantize
from torch._ops import ops

from torch.quantization import QuantStub, DeQuantStub


import torchvision
import torchvision.transforms as transforms
import os
import torch.quantization
import torch.quantization._numeric_suite as ns
from torchvision.models.quantization.mobilenet import mobilenet_v2
from torch.quantization import (
    default_eval_fn,
    default_qconfig,
    quantize,
)
import copy

from  mobilenet_classes import (
    ConvBNReLU,
    InvertedResidual,
    MobileNetV2,
    ChainModule,
    ChainModule2
)


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

def get_param(module, attr):
    ''' Sometime the weights/bias attribute gives you the raw tensor, but sometimes
    is a function that will give you the raw tensor, this function takes care of that logic
    '''
    if isinstance(getattr(module, attr, None), nn.Parameter):
        return getattr(module, attr, None)
    else:
        return getattr(module, attr, None)()

module_swap_list = [torch.nn.intrinsic.quantized.modules.conv_relu.ConvReLU2d,
                    torch.nn.Conv2d,
                    ConvBNReLU,
                    torchvision.models.mobilenet.ConvBNReLU,
                    torch.quantization.observer.MinMaxObserver,  # might be a problem
                    torchvision.models.quantization.mobilenet.QuantizableInvertedResidual,
                    torch.nn.modules.batchnorm.BatchNorm2d,
                    torch.nn.modules.linear.Linear,
                    torch.nn.modules.conv.Conv2d,
                    ]

class MeanLogger(ns.Logger):
    id =0
    def __init__(self):
        super(MeanLogger, self).__init__()
        self.stats["tensor_val"] = None
        self.count = 0
        self.sum = None
        self.id = MeanLogger.id
        MeanLogger.id += 1

    def forward(self, x):
        y = copy.deepcopy(x)
        if x.is_quantized:
            x = x.dequantize()

        if self.stats["tensor_val"] is None:
            self.stats["tensor_val"] = x
            self.sum = x
            self.count = 1
        else:
            torch.Tensor.add_(self.sum, x)
            self.count += 1
            self.stats["tensor_val"] = self.sum / self.count
        return y

class ShadowModule(nn.Module):
    def __init__(self, float_module=None, quantized_module=None):
        super(ShadowModule, self).__init__()
        self.float_module = float_module
        self.quantized_module = quantized_module

    def forward(self, x):
        a = self.float_module(x.dequantize())
        self.quantized_module(x)
        # hooks should already been added, running the data here is the calibration

        output_logger, input_logger = get_matching_activations(self.float_module, self.quantized_module)
        correct_quantized_bias_V2(output_logger, input_logger, self.float_module, self.quantized_module)
        return a



def sequential_bias_correction(float_model, quantized_model, img_data, white_list = [nn.Linear, nnq.Linear, nn.Conv2d, nnq.Conv2d]):
    # adding hooks
    # setting up qconfigs to add post hook
    qconfig_debug = torch.quantization.QConfig(activation=MeanLogger, weight=None)
    float_model.qconfig = qconfig_debug
    quantized_model.qconfig = qconfig_debug
    # calling prepare with prehook param, and throwing the whitelist to make sure supported modules
    # get the qconfig and thus post hook
    torch.quantization.prepare(float_model, inplace=True, white_list=white_list, prehook=MeanLogger)
    torch.quantization.prepare(quantized_model, inplace=True, white_list=white_list, observer_non_leaf_module_list=[nnq.Linear], prehook=MeanLogger)

    add_shadow(float_model, quantized_model, white_list=white_list)

    # format the data
    # the shadow takes care of everything :O
    # for data in img_data:
    #     with torch.no_grad():
    #         print(data[0].dtype)
    #         quantized_model(data[0])
    #         # break
    stack = None
    for data in img_data:
        with torch.no_grad():
            if stack is None:
                stack = data[0]
            else:
                stack = torch.cat((stack, data[0]))
            break
    # stack = tuple(img_data)
    print(stack.size())
    quantized_model(stack)

    strip_hooks_and_qconfigs(quantized_model)
    strip_shadow(quantized_model)
    return quantized_model

def add_shadow(float_model, quantized_model, white_list = [nn.Linear, nnq.Linear, nn.Conv2d, nnq.Conv2d]):
    float_module_children = {}
    for name, mod in float_model.named_children():
        float_module_children[name] = mod

    reassign = {}
    for name, mod in quantized_model.named_children():
        if name not in float_module_children:
            continue

        float_mod = float_module_children[name]

        if type(float_mod) not in white_list:
            add_shadow(float_mod, mod, white_list)

        if type(float_mod) in white_list:
            reassign[name] = ShadowModule(float_mod, mod)

    for key, value in reassign.items():
        quantized_model._modules[key] = value

def strip_hooks_and_qconfigs(model):
    pass

def strip_shadow(model):
    pass




def correct_quantized_bias(output_dict, float_model, qmodel):
    ''' Inplace function to modify the weights of the qmodel based off the
    recorded differences in weights provided by output_dict

    '''
    for key in output_dict:
        q_mod = get_module(qmodel, key[:-6])  #.orig_module
        if hasattr(q_mod, 'bias') and type(q_mod) in [nn.Linear, nn.Conv2d, nnq.Linear, nnq.Conv2d]: # get rid of linear later
            print(type(q_mod))
            if (isinstance(q_mod.bias, torch.nn.parameter.Parameter) and (q_mod.bias is not None)) or \
                (q_mod.bias() is not None):
                bias = None
                if isinstance(q_mod.bias, torch.nn.parameter.Parameter):
                    bias = q_mod.bias
                else:
                    bias = q_mod.bias()
                if output_dict[key]['quantized'].is_quantized:
                    output_dict[key]['quantized'] = output_dict[key]['quantized'].dequantize()

                difference = output_dict[key]['float'] - output_dict[key]['quantized']
                dims_to_mean_over = [i for i in range(difference.dim())]
                dims_to_mean_over.remove(1)
                difference = torch.mean(difference, dims_to_mean_over)

                updated_bias = bias.data - difference

                if isinstance(q_mod.bias, torch.nn.parameter.Parameter):
                    q_mod.bias.data = updated_bias
                else:
                    q_mod.bias().data = updated_bias

def correct_quantized_bias_V2(expected_output, expected_input, float_model, qmodel):
    ''' Inplace functino to modify the weights of the qmodel based off the
    recorded differences in weights provided by output_dict

    '''
    # print(expected_output)
    for key in expected_output:
        q_mod = get_module(qmodel, key[:-1]) #removing decimal
        if hasattr(q_mod, 'bias') and type(q_mod) in [nn.Linear, nn.Conv2d, nnq.Linear, nnq.Conv2d]:
            if (isinstance(q_mod.bias, torch.nn.parameter.Parameter) and (q_mod.bias is not None)) or \
                (q_mod.bias() is not None):
                bias = get_param(q_mod, 'bias')

                float_submodule = get_module(float_model, key[:-1])
                float_weight = get_param(float_submodule, 'weight')

                quantized_submodule = get_module(qmodel, key[:-1])
                quantized_weight = get_param(quantized_submodule, 'weight')

                if quantized_weight.is_quantized:
                    quantized_weight = quantized_weight.dequantize()

                error_matrix = float_weight - quantized_weight

                if type(q_mod) in [nn.Conv2d, nnq.Conv2d]:
                    sum_kernels = torch.sum(error_matrix, (2,3)) #c_o, c_i
                    expected_c_i = torch.mean(expected_input[key]['float'], (0,2,3)) #c_i
                    expected_c_i = expected_c_i.reshape(expected_c_i.size()[0], 1) #flipping into column vector
                    expected_c_o = torch.matmul(sum_kernels, expected_c_i) #c_o
                    expected_c_o = expected_c_o.squeeze(1)
                elif type(q_mod) in [nn.Linear, nnq.Linear]:
                    expected_c_o = torch.mean(error_matrix, 1)

                updated_bias = bias.data - expected_c_o

                updated_bias = updated_bias.reshape(bias.data.size())

                if isinstance(q_mod.bias, torch.nn.parameter.Parameter):
                    q_mod.bias.data = updated_bias
                else:
                    q_mod.bias().data = updated_bias

def correct_quantized_bias_V3(output_dict, float_model, qmodel):
    # approach without having to do the fuse modules
    for key in output_dict:
        q_mod = get_module(qmodel, key[:-6])
        if hasattr(q_mod, 'bias') and not isinstance(q_mod, torch.nn.intrinsic.quantized.ConvReLU2d):
            if (isinstance(q_mod.bias, torch.nn.parameter.Parameter) and (q_mod.bias is not None)) or \
                (q_mod.bias() is not None):
                bias = None
                if isinstance(q_mod.bias, torch.nn.parameter.Parameter):
                    bias = q_mod.bias
                else:
                    bias = q_mod.bias()
                if output_dict[key]['quantized'].is_quantized:
                    output_dict[key]['quantized'] = output_dict[key]['quantized'].dequantize()

                float_values = output_dict[key]['float']
                quantized_values = output_dict[key]['quantized']

                difference = float_values - quantized_values
                # element wise x>0
                positive_pre_activations_float = torch.gt(float_values, torch.zeros(float_values.size()))
                positive_pre_activations_quantized = torch.gt(quantized_values, torch.zeros(quantized_values.size()))
                mask = torch.mul(positive_pre_activations_float, positive_pre_activations_quantized)
                # converting bool values to 0's and 1's
                mask = mask.to(torch.float)
                masked_difference = torch.mul(difference, mask)

                counts = torch.sum(mask, (2,3))
                sum = torch.sum(masked_difference, (2,3))
                avg = sum/counts
                avg_over_batches = 0
                # need to edit

                updated_bias = bias.data - difference

                if isinstance(q_mod.bias, torch.nn.parameter.Parameter):
                    q_mod.bias.data = updated_bias
                else:
                    q_mod.bias().data = updated_bias

def get_local_modilenet():
    saved_model_dir = '/home/edmundjr/local/pytorch/torch/quantization/data/'
    float_model_file = 'mobilenet_pretrained_float.pth'
    # float_model = torchvision.models.quantization.mobilenet_v2(pretrained=True, quantize=False)  # mobilenet_v2, resnet18
    float_model = load_model(saved_model_dir + float_model_file)
    float_model.to('cpu')
    float_model.eval()
    float_model.qconfig = torch.quantization.default_qconfig

    img_data = [(torch.rand(2, 3, 10, 10, dtype=torch.float), torch.randint(0, 1, (2,), dtype=torch.long))
                            for _ in range(30)]
    # lost code on how to get local data loaders

def get_online_mobilenet():
    float_model = torchvision.models.quantization.mobilenet_v2(pretrained=True, quantize=False)  # mobilenet_v2, resnet18
    float_model.to('cpu')
    float_model.eval()
    float_model.fuse_model()
    float_model.qconfig = torch.quantization.default_qconfig

    img_data = [(torch.rand(10, 3, 224, 224, dtype=torch.float), torch.randint(0, 1, (2,), dtype=torch.long))
                            for _ in range(30)]

    qmodel = copy.deepcopy(float_model)
    quantize(qmodel, default_eval_fn, img_data, inplace=True)

    return float_model, qmodel, img_data

def get_local_linear_chain():
    float_model = ChainModule(True)
    img_data = [(torch.rand(1000, 3, dtype=torch.float), torch.randint(0, 1, (2,), dtype=torch.long))
                            for _ in range(30)]
    float_model.qconfig = torch.quantization.default_qconfig
    qmodel = quantize(float_model, default_eval_fn, img_data, inplace=False)

    return float_model, qmodel, img_data

def get_local_conv_chain():
    float_model = ChainModule2(True)
    img_data = [(torch.rand(100, 3, 125, 125, dtype=torch.float), torch.randint(0, 1, (2,), dtype=torch.long))
                            for _ in range(30)]
    float_model.qconfig = torch.quantization.default_qconfig
    qmodel = quantize(float_model, default_eval_fn, img_data, inplace=False)

    return float_model, qmodel, img_data

def setup_foward_hook(logger=ns.OutputLogger, dataset=get_local_linear_chain):
    float_model, qmodel, img_data = dataset()

    ns.prepare_model_outputs(float_model, qmodel, logger)
    for data in img_data:
        with torch.no_grad():
            float_model(data[0])
            qmodel(data[0])

    act_compare_dict = ns.get_matching_activations(float_model, qmodel)
    return act_compare_dict, float_model, qmodel, img_data

def setup_forward_pre_hook(logger=ns.OutputLogger, dataset=get_local_linear_chain):
    # debugged/hacked around a bunch, don't particularly remember the reason for
    # the second round of deleting qconfigs
    float_model, qmodel, img_data = dataset()

    def _remove_qconfig(model_prime):
        for child in model_prime.children():
            _remove_qconfig(child)
        if hasattr(model_prime, "qconfig"):
            del model_prime.qconfig

    _remove_qconfig(qmodel)
    _remove_qconfig(float_model)

    torch.quantization.prepare(float_model, inplace=True, prehook=MeanLogger)
    torch.quantization.prepare(qmodel, inplace=True, prehook=MeanLogger)

    _remove_qconfig(qmodel)
    _remove_qconfig(float_model)

    for data in img_data:
        with torch.no_grad():
            print("what if")
            float_model(data[0])
            qmodel(data[0])
    compare_dict = ns.get_matching_activations(float_model, qmodel)

    return compare_dict, float_model, qmodel

def setup_double_hook(logger=ns.OutputLogger, dataset=get_local_linear_chain):
    float_model, qmodel, img_data = dataset()

    # setting up forward hook and forward  prehook
    qconfig_debug = torch.quantization.QConfig(activation=MeanLogger, weight=None)
    float_model.qconfig = qconfig_debug
    qmodel.qconfig = qconfig_debug
    white_list = [nn.Linear, nnq.Linear, nn.Conv2d, nnq.Conv2d]

    torch.quantization.prepare(float_model, inplace=True, white_list=white_list, prehook=MeanLogger)
    torch.quantization.prepare(qmodel, inplace=True, white_list=white_list, observer_non_leaf_module_list=[nnq.Linear], prehook=MeanLogger)

    # logging and calibration under the hood here
    for data in img_data:
        with torch.no_grad():
            float_model(data[0])
            qmodel(data[0])

    output_logger, input_logger = get_matching_activations(float_model, qmodel)
    # print("output_logger: ", output_logger)
    return output_logger, input_logger, float_model, qmodel, img_data

    # correct_quantized_bias_V2(output_logger, input_logger, float_model, qmodel)

def compute_error(x,y):
    Ps = torch.norm(x)
    Pn = torch.norm(x-y)
    return 20*torch.log10(Ps/Pn)

def sanity_checks_sqnr():
    x = setup_double_hook(MeanLogger, get_online_mobilenet)

    # correct_quantized_bias(*x[:-1])
    correct_quantized_bias_V2(*x[:-1])
    output_logger, input_logger, float_model, qmodel, img_data = x

    correct_quantized_bias_V2(output_logger, input_logger, float_model, qmodel)


    for key in output_logger:
        float_submodule = get_module(float_model, key[:-1])
        quantized_submodule = get_module(qmodel, key[:-1])
        if hasattr(quantized_submodule, 'bias'):
            print(key)
            print(type(quantized_submodule))
            float_weight = get_param(float_submodule, 'weight')
            quantized_weight = get_param(quantized_submodule, 'weight')
            if quantized_weight.is_quantized:
                quantized_weight = quantized_weight.dequantize()
            print(compute_error(float_weight, quantized_weight))

def sanity_checks_accuracy():
    x = setup_foward_hook(MeanLogger, get_local_conv_chain)
    correct_quantized_bias(*x[:-1])
    act_compare_dict, float_model, qmodel, img_data = x
    # need to verify that qmodel did get changed by correct_quantized_bias

    results = []
    total = 0
    for data in img_data:
        with torch.no_grad():
            results.append(float_model(data[0]))
            total += data[0].size()[0]

    index = 0
    for data in img_data:
        with torch.no_grad():
            output = qmodel(data[0])
            #hm i want to loop over different imgs in single batch
            output = torch.split(output,1)
            expected = torch.split(results[index], 1)
            index += 1

            for indx in range(len(output)):
                print(compute_error(output[indx],expected[indx]))

def get_logger_entries(mod, target_dict = {}, prefix=""):
    r"""This is the helper function for get_logger_dict

    Args:
        mod: module we want to save all logger stats
        prefix: prefix for the current module
        target_dict: the dictionary used to save all logger stats
    """

    def get_prefix(prefix):
        return prefix if prefix == "" else prefix + "."

    for name, child in mod.named_children():
        if isinstance(child, ns.Logger):
            if get_prefix(prefix) not in target_dict:
                target_dict[get_prefix(prefix)] = {str(name): child.stats["tensor_val"]}
            else:
                target_dict[get_prefix(prefix)][str(name)] = child.stats["tensor_val"]
    for name, child in mod.named_children():
        module_prefix = get_prefix(prefix) + name if prefix else name
        get_logger_entries(child, target_dict, module_prefix)

def get_matching_activations(float_module, q_module):
    r"""Find the matching activation between float and quantized modules.

    Args:
        float_module: float module used to generate the q_module
        q_module: module quantized from float_module

    Return:
        act_dict: dict with key corresponding to quantized module names and each
        entry being a dictionary with two keys 'float' and 'quantized', containing
        the matching float and quantized activations
    """
    float_dict = {}
    get_logger_entries(float_module, float_dict)

    quantized_dict = {}
    get_logger_entries(q_module, quantized_dict)

    output_logger = {}
    input_logger = {}

    for key in quantized_dict:
        input_logger[key] = {}
        input_logger[key]['float'] = float_dict[key]['activation_pre_process']
        input_logger[key]['quantized'] = quantized_dict[key]['activation_pre_process']
        output_logger[key] = {}
        output_logger[key]['float'] = float_dict[key]['activation_post_process']
        output_logger[key]['quantized'] = quantized_dict[key]['activation_post_process']
    return output_logger, input_logger

if __name__ == "__main__":
    # act_compare_dict, float_model, qmodel, img_data = setup_foward_hook(MeanLogger, get_online_mobilenet)
    # print(qmodel)

    sequential_bias_correction(*get_online_mobilenet())
    # sanity_checks_sqnr()
    # sanity_checks_accuracy()
    # correct_quantized_bias(*setup_foward_hook(MeanLogger))
    # main()
    # double_hook_dictionaries()
    # loading_running_mobilenet_img()
    # loading_running_chainmodule_img()
