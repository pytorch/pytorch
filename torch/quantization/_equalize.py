import torch
import torch.nn as nn
import copy

def channel_range(input, axis=0):
    ''' finds the range of weights associated with a specific channel

    '''
    size_of_tensor_dim = len(input.size())
    mins = input
    maxs = input

    # reshape input to make specified axis the last axis in input
    temp = list(range(size_of_tensor_dim))
    temp[size_of_tensor_dim - 1] = axis
    temp[axis] = size_of_tensor_dim - 1

    mins = mins.permute(temp)
    maxs = maxs.permute(temp)

    # minimizing over all axises except the last axis
    for i in range(size_of_tensor_dim - 1):
        mins = torch.min(mins, 0)[0]
        maxs = torch.max(maxs, 0)[0]

    assert(mins.size()[0] == input.size()[axis])
    return maxs - mins

def cross_layer_equalization(module1, module2, output_axis=0, input_axis=1):
    ''' Given two adjacent tensors', the weights are scaled such that
    the ranges of the first tensors' output channel are equal to the
    ranges of the second tensors' input channel

    '''
    if module1.weight.size()[output_axis] != module2.weight.size()[input_axis]:
        raise TypeError("Incompatible tensors")

    weight1 = module1.weight
    weight2 = module2.weight
    bias = module1.bias

    weight1_range = channel_range(weight1, output_axis)
    weight2_range = channel_range(weight2, input_axis)

    # producing scaling factors to applied
    weight2_range += 1e-9
    scaling_factors = torch.sqrt(weight1_range / weight2_range)
    r_scaling_factors = torch.reciprocal(scaling_factors)

    bias = bias * r_scaling_factors

    # formatting the scaling (1D) tensors to be applied on the given argument tensors
    # pads axis to (1D) tensors to then be broadcasted
    size1 = [1] * len(weight1.size())
    size1[output_axis] = weight1.size()[output_axis]
    size2 = [1] * len(weight2.size())
    size2[input_axis] = weight2.size()[input_axis]

    scaling_factors = torch.reshape(scaling_factors, size2)
    r_scaling_factors = torch.reshape(r_scaling_factors, size1)

    weight1 = weight1 * r_scaling_factors
    weight2 = weight2 * scaling_factors

    module1.weight.data = weight1
    module1.bias.data = bias
    module2.weight.data = weight2

def equalize(model, paired_modules_list, threshold = 1e-4):
    ''' Given a list of adjacent modules within a model, equalization will
    be applied between each pair, this will repeated until convergence is achieved

    Keeps a copy of the changing modules from the previous iteration, if the copies
    are not that different than the current modules (determined by converged_test),
    then the modules have converged enough that further equalizing is not necessary

    '''
    name_to_module = {}
    previous_name_to_module = {}
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

def converged(curr_modules, prev_modules, threshold):
    ''' Tests for the summed norm of the differences between each set of modules
    being less than the given threshold

    Takes two dictionaries mapping names to modules, the set of names for each dictionary
    should be the same, looping over the set of names, for each name take the differnce
    between the associated modules in each dictionary

    '''
    if len(curr_modules) != len(prev_modules):
        raise TypeError("in compatiables modules in convergence condition")

    summed_norms = 0
    for name in curr_modules.keys():
        if prev_modules[name] is None:
            return False
        difference = curr_modules[name].weight.sub(prev_modules[name].weight)
        summed_norms += torch.norm(difference)
    return summed_norms < threshold
