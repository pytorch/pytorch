from __future__ import print_function, division, absolute_import
import torch
import torch.nn as nn
import torch.nn.quantized as nnq

import torch.quantization
import torch.quantization._numeric_suite as ns

import copy

_supported_modules = {nn.Linear, nnq.Linear, nn.Conv2d, nnq.Conv2d}

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

def get_param(module, attr):
    ''' Sometimes the weights/bias attribute gives you the raw tensor, but sometimes
    gives a function that will give you the raw tensor, this function takes care of that logic
    '''
    param = getattr(module, attr, None)
    if isinstance(param, nn.Parameter):
        return param
    elif param is not None:
        return param()


class MeanLogger(ns.Logger):
    def __init__(self):
        super(MeanLogger, self).__init__()
        self.stats["tensor_val"] = None
        self.count = 0
        self.sum = None

    def forward(self, x):
        ''' takes the rolling mean of the data being passed through
        '''
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
        ''' Reads in a batch of data, using the computed means of the input and output of the
        float and quantized modules, bias correction will be applied on the quantized module
        '''
        self.float_module(x.dequantize())
        a = self.quantized_module(x)

        output_logger, input_logger = get_inputs_n_outputs(self.float_module, self.quantized_module)
        correct_quantized_bias(output_logger, input_logger, self.float_module, self.quantized_module)
        return a

def add_shadow(float_model, quantized_model, white_list=_supported_modules):
    ''' Goal: while data is being passed from submodule to submodule when executing we want to compare
    the inputs and outputs of the whitelisted submodules of the floating and quantized models.

    To do this, evey whitelisted submodule in the quantized model will be wrapped with the
    shadowModule class (along with its floating_point counterpart).
    '''
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

def remove_hooks(model, white_list=_supported_modules):
    ''' Removes all the hooks from the whitelisted submodules
    '''
    for name, module in model.named_modules():
        if type(module) in white_list:
            if hasattr(module, 'activation_post_process'):
                for key in module._forward_hooks:
                    del module._forward_hooks[key]
                del module.activation_post_process
            if hasattr(module, 'activation_pre_process'):
                for key in module._forward_pre_hooks:
                    del module._forward_pre_hooks[key]
                del module.activation_pre_process

def remove_shadow(model):
    ''' Goes through the submodules of a quantized model and replaces every
    shadowModule with just the quantized submodule it contains
    '''
    reassign = {}
    # logging occurences of shadowModules
    for name, module in model.named_modules():
        if isinstance(module, ShadowModule):
            reassign[name] = module.quantized_module
    # removing occurences of shadowModules
    for name, module in reassign.items():
        name_split = name.split('.')
        get_parent_module(model, name)._modules[name_split[-1]] = reassign[name]

def sequential_bias_correction(float_model, quantized_model, img_data, white_list=_supported_modules):
    ''' Applies bias correction on the whitelisted submodules of the quantized model
    After adding shadow modules and MeanLoggers to the quantized model the img_data will get reshaped
    into one large batch and then this batch will be ran through the quantized model

    at every whitelisted submodule during this execution, bias correction will be applied
    on the quantized submodule within the shadowModule, before moving to the next submodule
    '''
    # adding hooks
    # setting up qconfigs to add post hook
    qconfig_debug = torch.quantization.QConfig(activation=MeanLogger, weight=None)
    float_model.qconfig = qconfig_debug
    quantized_model.qconfig = qconfig_debug
    # calling prepare with prehook param, and throwing the whitelist to make sure supported modules
    # get the qconfig and thus post hook
    torch.quantization.prepare(float_model, inplace=True, white_list=white_list, prehook=MeanLogger)
    torch.quantization.prepare(quantized_model, inplace=True, white_list=white_list,
                            observer_non_leaf_module_list=[nnq.Linear], prehook=MeanLogger)

    add_shadow(float_model, quantized_model, white_list=white_list)

    # reshaping img_data into one large batch
    stack = None
    for data in img_data:
        if stack is None:
            stack = data[0]
        else:
            stack = torch.cat((stack, data[0]))

    # bias correction is happening under the hood here
    with torch.no_grad():
        quantized_model(stack)

    remove_hooks(quantized_model)
    remove_shadow(quantized_model)

def parallel_bias_correction(float_model, quantized_model, img_data, white_list=_supported_modules):
    ''' Applies bias correction on the whitelisted submodules of the quantized model
    MeanLoggers are added to float and quantized models to record the running means of
    their inputs and outputs, after running all the data provided by img_data, the results
    from the loggers are fed into the bias correction function
    '''

    qconfig_debug = torch.quantization.QConfig(activation=MeanLogger, weight=None)
    float_model.qconfig = qconfig_debug
    quantized_model.qconfig = qconfig_debug

    torch.quantization.prepare(float_model, inplace=True, white_list=white_list, prehook=MeanLogger)
    torch.quantization.prepare(quantized_model, inplace=True, white_list=white_list,
                            observer_non_leaf_module_list=[nnq.Linear], prehook=MeanLogger)
    batch_size = None
    # batch size is used here to avoid an adding error in the MeanLogger due
    # to the last batch of the dataset possibly being a different size than
    # the previous batches
    for data in img_data:
        with torch.no_grad():
            if batch_size is None:
                batch_size = data[0].size(0)  # getting batch size
            if data[0].size(0) == batch_size:
                float_model(data[0])
                quantized_model(data[0])


    output_logger, input_logger = get_inputs_n_outputs(float_model, quantized_model)
    correct_quantized_bias(output_logger, input_logger, float_model, quantized_model)

def correct_quantized_bias(expected_output, expected_input, float_model, quantized_model):
    ''' Inplace function to modify the weights of the quantized_model based off the
    recorded differences in weights provided by output_dict

    '''
    for key in expected_output:
        # submodules associated with key
        float_submodule = get_module(float_model, key[:-1])
        quantized_submodule = get_module(quantized_model, key[:-1])

        # checking for existence of bias attribute
        if hasattr(quantized_submodule, 'bias') and type(quantized_submodule) in _supported_modules:
            if (isinstance(quantized_submodule.bias, torch.nn.parameter.Parameter) and (quantized_submodule.bias is not None)) or \
                            (quantized_submodule.bias() is not None):

                bias = get_param(quantized_submodule, 'bias')

                # grabbing the weights
                float_weight = get_param(float_submodule, 'weight')
                quantized_weight = get_param(quantized_submodule, 'weight')
                if quantized_weight.is_quantized:
                    quantized_weight = quantized_weight.dequantize()

                error_matrix = float_weight - quantized_weight

                # bias correction logic
                if type(quantized_submodule) in [nn.Conv2d, nnq.Conv2d]:
                    sum_kernels = torch.sum(error_matrix, (2, 3))  # c_o, c_i
                    expected_c_i = torch.mean(expected_input[key]['float'], (0, 2, 3))  # c_i
                    expected_c_i = expected_c_i.reshape(expected_c_i.size()[0], 1)  # flipping into column vector
                    expected_c_o = torch.matmul(sum_kernels, expected_c_i)  # c_o
                    expected_c_o = expected_c_o.squeeze(1)
                elif type(quantized_submodule) in [nn.Linear, nnq.Linear]:
                    expected_c_o = torch.mean(error_matrix, 1)

                # new value for bias attribute
                updated_bias = bias.data - expected_c_o
                updated_bias = updated_bias.reshape(bias.data.size())

                # setting new bias
                if isinstance(quantized_submodule.bias, torch.nn.parameter.Parameter):
                    quantized_submodule.bias.data = updated_bias
                else:
                    quantized_submodule.bias().data = updated_bias


def get_logger_entries(mod, target_dict, prefix=""):
    """ Reads in the data from a Logger object and formats it into a dictionary,
    where the key to the dict is the name of a submodule and the assoicated value
    are the results of the logger (stored in logger.stats["tensor_val"])
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

def get_inputs_n_outputs(float_module, q_module):
    r"""Given two models (a floating point model and is quantized counterpart)
    two dictionaries representing the statistics the forward_pre_hook and forward_hook
    Loggers collected in the two given models
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
