import torch
import copy
from typing import Dict, Any

_supported_types = {torch.nn.Conv2d, torch.nn.Linear}
_supported_intrinsic_types = {torch.nn.intrinsic.ConvReLU2d, torch.nn.intrinsic.LinearReLU}
_all_supported_types = _supported_types.union(_supported_intrinsic_types)

def set_module_weight(module, weight) -> None:
    if type(module) in _supported_types:
        module.weight = torch.nn.Parameter(weight)
    else:
        module[0].weight = torch.nn.Parameter(weight)

def set_module_bias(module, bias) -> None:
    if type(module) in _supported_types:
        module.bias = torch.nn.Parameter(bias)
    else:
        module[0].bias = torch.nn.Parameter(bias)

def get_module_weight(module):
    if type(module) in _supported_types:
        return module.weight
    else:
        return module[0].weight

def get_module_bias(module):
    if type(module) in _supported_types:
        return module.bias
    else:
        return module[0].bias

def max_over_ndim(input, axis_list, keepdim=False):
    ''' Applies 'torch.max' over the given axises
    '''
    axis_list.sort(reverse=True)
    for axis in axis_list:
        input, _ = input.max(axis, keepdim)
    return input

def min_over_ndim(input, axis_list, keepdim=False):
    ''' Applies 'torch.min' over the given axises
    '''
    axis_list.sort(reverse=True)
    for axis in axis_list:
        input, _ = input.min(axis, keepdim)
    return input

def channel_range(input, axis=0):
    ''' finds the range of weights associated with a specific channel
    '''
    size_of_tensor_dim = input.ndim
    axis_list = list(range(size_of_tensor_dim))
    axis_list.remove(axis)

    mins = min_over_ndim(input, axis_list)
    maxs = max_over_ndim(input, axis_list)

    assert mins.size(0) == input.size(axis), "Dimensions of resultant channel range does not match size of requested axis"
    return maxs - mins

def cross_layer_equalization(module1, module2, output_axis=0, input_axis=1):
    ''' Given two adjacent tensors', the weights are scaled such that
    the ranges of the first tensors' output channel are equal to the
    ranges of the second tensors' input channel
    '''
    if type(module1) not in _all_supported_types or type(module2) not in _all_supported_types:
        raise ValueError("module type not supported:", type(module1), " ", type(module2))

    weight1 = get_module_weight(module1)
    weight2 = get_module_weight(module2)

    if weight1.size(output_axis) != weight2.size(input_axis):
        raise TypeError("Number of output channels of first arg do not match \
        number input channels of second arg")

    bias = get_module_bias(module1)

    weight1_range = channel_range(weight1, output_axis)
    weight2_range = channel_range(weight2, input_axis)

    # producing scaling factors to applied
    weight2_range += 1e-9
    scaling_factors = torch.sqrt(weight1_range / weight2_range)
    inverse_scaling_factors = torch.reciprocal(scaling_factors)

    bias = bias * inverse_scaling_factors

    # formatting the scaling (1D) tensors to be applied on the given argument tensors
    # pads axis to (1D) tensors to then be broadcasted
    size1 = [1] * weight1.ndim
    size1[output_axis] = weight1.size(output_axis)
    size2 = [1] * weight2.ndim
    size2[input_axis] = weight2.size(input_axis)

    scaling_factors = torch.reshape(scaling_factors, size2)
    inverse_scaling_factors = torch.reshape(inverse_scaling_factors, size1)

    weight1 = weight1 * inverse_scaling_factors
    weight2 = weight2 * scaling_factors

    set_module_weight(module1, weight1)
    set_module_bias(module1, bias)
    set_module_weight(module2, weight2)

def equalize(model, paired_modules_list, threshold=1e-4, inplace=True):
    ''' Given a list of adjacent modules within a model, equalization will
    be applied between each pair, this will repeated until convergence is achieved

    Keeps a copy of the changing modules from the previous iteration, if the copies
    are not that different than the current modules (determined by converged_test),
    then the modules have converged enough that further equalizing is not necessary

    Implementation of this referced section 4.1 of this paper https://arxiv.org/pdf/1906.04721.pdf

    Args:
        model: a model (nn.module) that equalization is to be applied on
        paired_modules_list: a list of lists where each sublist is a pair of two
            submodules found in the model, for each pair the two submodules generally
            have to be adjacent in the model to get expected/reasonable results
        threshold: a number used by the converged function to determine what degree
            similarity between models is necessary for them to be called equivalent
        inplace: determines if function is inplace or not
    '''
    if not inplace:
        model = copy.deepcopy(model)

    name_to_module : Dict[str, torch.nn.Module] = {}
    previous_name_to_module: Dict[str, Any] = {}
    name_set = {name for pair in paired_modules_list for name in pair}

    for name, module in model.named_modules():
        if name in name_set:
            name_to_module[name] = module
            previous_name_to_module[name] = None
    while not converged(name_to_module, previous_name_to_module, threshold):
        for pair in paired_modules_list:
            previous_name_to_module[pair[0]] = copy.deepcopy(name_to_module[pair[0]])
            previous_name_to_module[pair[1]] = copy.deepcopy(name_to_module[pair[1]])

            cross_layer_equalization(name_to_module[pair[0]], name_to_module[pair[1]])

    return model

def converged(curr_modules, prev_modules, threshold=1e-4):
    ''' Tests for the summed norm of the differences between each set of modules
    being less than the given threshold

    Takes two dictionaries mapping names to modules, the set of names for each dictionary
    should be the same, looping over the set of names, for each name take the differnce
    between the associated modules in each dictionary

    '''
    if curr_modules.keys() != prev_modules.keys():
        raise ValueError("The keys to the given mappings must have the same set of names of modules")

    summed_norms = torch.tensor(0.)
    if None in prev_modules.values():
        return False
    for name in curr_modules.keys():
        curr_weight = get_module_weight(curr_modules[name])
        prev_weight = get_module_weight(prev_modules[name])

        difference = curr_weight.sub(prev_weight)
        summed_norms += torch.norm(difference)
    return bool(summed_norms < threshold)
