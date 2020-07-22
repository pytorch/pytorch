from __future__ import print_function, division, absolute_import
import numpy as np
import torch
import torch.nn as nn
import torch.nn.quantized as nnq
import torch.nn.quantized.dynamic as nnqd
# import torch.nn.quantized.QFunctional as QFunctional
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
    curr = model
    name = name.split('.')
    for subname in name:
        curr = curr._modules[subname]
    return curr

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

def correct_quantized_bias(output_dict, float_model, qmodel):
    ''' Inplace functino to modify the weights of the qmodel based off the
    recorded differences in weights provided by output_dict

    '''
    for key in output_dict:
        q_mod = get_module(qmodel, key[:-6])  #.orig_module
        # print(key[:-6], type(q_mod))
    ###################
    # todo
        if hasattr(q_mod, 'bias') and not isinstance(q_mod, torch.nn.intrinsic.quantized.ConvReLU2d): # get rid of linear later
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
    for key in expected_output:
        q_mod = get_module(qmodel, key)
        print(key)

        if hasattr(q_mod, 'bias'):
            if (isinstance(q_mod.bias, torch.nn.parameter.Parameter) and (q_mod.bias is not None)) or \
                (q_mod.bias() is not None):
                bias = None
                if isinstance(q_mod.bias, torch.nn.parameter.Parameter):
                    bias = q_mod.bias
                else:
                    bias = q_mod.bias()
                if expected_output[key]['quantized'].is_quantized:
                    expected_output[key]['quantized'] = expected_output[key]['quantized'].dequantize()

                float_submodule = get_module(float_model, key)
                quantized_submodule = get_module(qmodel, key)

                error_matrix = float_submodule.weight() - quantized_submodule.weight()
                expected_input_to_submodule = expected_input[key]['float']

                try: # should auto distinguish between conv2d and others?
                    error_matrix = torch.sum(error_matrix, (2,3))

                except:
                    pass

                biased_output = error_matrix * expected_input_to_submodule
                # biased_output is the biased expected output, so its already meaned
                difference = expected_output[key]['float'] - biased_output

                updated_bias = bias.data - difference

                if isinstance(q_mod.bias, torch.nn.parameter.Parameter):
                    q_mod.bias.data = updated_bias
                else:
                    q_mod.bias().data = updated_bias

def bias_absorption(float_model, qmodel, paired_modules_list, output_dict):
    #max(0, mean - 3*std)
    pass

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
    img_data = [(torch.rand(10, 3, dtype=torch.float), torch.randint(0, 1, (2,), dtype=torch.long))
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

    # print(get_module(float_model, 'features.0.0')._modules['0'].activation_pre_process.stats["tensor_val"].size())
    # print(get_module(float_model, 'features.0.0')._modules['1'].activation_pre_process.stats["tensor_val"].size())
    # print(get_module(qmodel, 'features.0.0').activation_pre_process.stats["tensor_val"].size())
    # print("welp")
    # print(get_module(float_model, 'features.0.0'))
    # print(get_module(qmodel, 'features.0.0'))

    return compare_dict, float_model, qmodel

def compute_error(x,y):
    Ps = torch.norm(x)
    Pn = torch.norm(x-y)
    return 20*torch.log10(Ps/Pn)

def sanity_checks_sqnr():
    x = setup_foward_hook(MeanLogger, get_online_mobilenet)


    correct_quantized_bias(*x[:-1])
    act_compare_dict, float_model, qmodel, img_data = x

    print(act_compare_dict.keys())
    for key in act_compare_dict:
        # print(key, type(get_module(float_model, key)))
        print(key, type(act_compare_dict[key]['quantized']))
    print("BREAK    ")
    for key in act_compare_dict:
        q_mod = get_module(qmodel, key[:-6])  #.orig_module

        if hasattr(q_mod, 'bias'):
            print(key[:-6])
            print(type(q_mod))
            print(compute_error(act_compare_dict[key]['float'], act_compare_dict[key]['quantized']))

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



def main():
    float_model, qmodel, img_data = get_local_linear_chain()

    # running data and logging entries
    logger_data = double_hook_dictionaries(float_model, qmodel, img_data)

    print(logger_data.keys())

    for key in logger_data:
        print(key)
        print(logger_data[key]['float']['stats'].size())
        print(logger_data[key]['float']['id'])
        print(logger_data[key]['quantized']['stats'].size())
        print(logger_data[key]['quantized']['id'])
        print()
        # print(compute_error(logger_data[key]['float'], logger_data[key]['quantized']))

    output_dict, input_dict = split_dictionaries(logger_data)
    print("log keys: ", logger_data.keys())
    print("output keys: ", output_dict.keys())
    print("input keys: ", input_dict.keys())
    print("bruh")

    correct_quantized_bias_V2(output_dict, input_dict, float_model, qmodel)

def split_dictionaries(logger_data):
    input_dict = {}
    output_dict = {}
    for key in logger_data:
        if 'activation_pre_process' in key:
            input_dict[key] = logger_data[key]
        if 'activation_post_process' in key:
            output_dict[key] = logger_data[key]

    return output_dict, input_dict



def double_hook_dictionaries(float_model, qmodel, input_img_data):
    # setting up forward hook and forward  prehook
    qconfig_debug = torch.quantization.QConfig(activation=MeanLogger, weight=None)
    float_model.qconfig = qconfig_debug
    qmodel.qconfig = qconfig_debug
    white_list = {nn.Linear: nnq.Linear, nnq.Linear: nnq.Linear}

    torch.quantization.prepare(float_model, inplace=True, white_list=white_list, prehook=MeanLogger)
    for name, module in float_model.named_modules():
        if isinstance(module, ns.Logger):
            print("grabbing id's: ", module.id)
    torch.quantization.prepare(qmodel, inplace=True, white_list=white_list, prehook=MeanLogger)
    # fix up white_list to have quantized modules in the non-lead-mod-  whitelist

    # logging and calibration under the hood here
    for data in input_img_data:
        with torch.no_grad():
            float_model(data[0])
            qmodel(data[0])
    for name, module in float_model.named_modules():
        if isinstance(module, ns.Logger):
            print("grabbing id's: ", module.id)

    return get_matching_activations(float_model, qmodel)

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
            target_dict[get_prefix(prefix) + str(name)] = {'stats': child.stats["tensor_val"], 'id':child.id}

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
    for key in float_dict:
        pass
    quantized_dict = {}
    get_logger_entries(q_module, quantized_dict)
    act_dict = {}
    for key in quantized_dict:
        match_key = ns._find_match(sorted(float_dict, reverse=False), key, "activation_post_process")
        if match_key is not None:
            act_dict[key] = {}
            act_dict[key]["float"] = float_dict[match_key] #['stats']["tensor_val"]
            act_dict[key]["quantized"] = quantized_dict[key] #['stats']["tensor_val"]
        match_key = ns._find_match(sorted(float_dict, reverse=False), key, "activation_pre_process")
        if match_key is not None:
            act_dict[key] = {}
            act_dict[key]["float"] = float_dict[match_key] #['stats']["tensor_val"]
            act_dict[key]["quantized"] = quantized_dict[key] #['stats']["tensor_val"]
    return act_dict

if __name__ == "__main__":
    # act_compare_dict, float_model, qmodel, img_data = setup_foward_hook(MeanLogger, get_online_mobilenet)
    # print(qmodel)

    # sanity_checks_sqnr()
    # sanity_checks_accuracy()
    # correct_quantized_bias(*setup_foward_hook(MeanLogger))
    main()
    # double_hook_dictionaries()
    # loading_running_mobilenet_img()
    # loading_running_chainmodule_img()
