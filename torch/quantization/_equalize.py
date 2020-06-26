import torch
import torch.nn as nn
import copy

def channel_range(input, axis=0):
    ''' finds the range of weights associated with a specific channel

    '''
    size_of_tensor_dim = len(input.size())
    mins = input
    maxs = input
    for i in range(size_of_tensor_dim - 1, axis, -1):
        mins = torch.min(mins, i)[0]
        maxs = torch.max(maxs, i)[0]
    for i in range(axis):
        mins = torch.min(mins, 0)[0]
        maxs = torch.max(maxs, 0)[0]
    return maxs.sub(mins)

def scaling_channels(tensor1, tensor2, output_axis=0, input_axis=1):
    ''' Given two adjacent tensors', the weights are scaled such that
    the ranges of the first tensors' output channel are equal to the
    ranges of the second tensors' input channel

    Note: assumes the input/output channels are the first two axises of the given tensors
    '''
    if tensor1.size()[output_axis] != tensor2.size()[input_axis]:
        raise TypeError("Incompatible tensors")

    output_channel_tensor1 = channel_range(tensor1, output_axis)
    input_channel_tensor2 = channel_range(tensor2, input_axis)

    # producing scaling factors to applied
    # zero_input = torch.zeros(output_channel_tensor1.size())
    # scaling_factors = torch.sqrt(torch.addcdiv(zero_input, output_channel_tensor1, input_channel_tensor2))
    input_channel_tensor2 += 1e-9
    scaling_factors = torch.sqrt(output_channel_tensor1 / input_channel_tensor2)
    r_scaling_factors = torch.reciprocal(scaling_factors)

    # formatting the scaling (1D) tensors to be applied on the given argument tensors
    scaling_factors = torch.unsqueeze(scaling_factors, output_axis)
    r_scaling_factors = torch.unsqueeze(r_scaling_factors, input_axis)

    for i in range(len(tensor1.size()) - 2):
        r_scaling_factors = torch.unsqueeze(r_scaling_factors, -1)
    for i in range(len(tensor2.size()) - 2):
        scaling_factors = torch.unsqueeze(scaling_factors, -1)

    # mod_tensor1 = torch.mul(tensor1, r_scaling_factors)
    # mod_tensor2 = torch.mul(tensor2, scaling_factors)
    mod_tensor1 = tensor1 * r_scaling_factors
    mod_tensor2 = tensor2 * scaling_factors

    return mod_tensor1, mod_tensor2


def cross_layer_equalization(module1, module2):
    ''' Given two adjacent modules, the weights are scaled such that
    the ranges of the first modules' output channel are equal to the
    ranges of the second modules' input channel

    Scaling work is done in scaling_channels()
    '''
    module1_output_channel_axis = -1
    module2_input_channel_axis = -1

    # additional modules can be added, given the axises of the input/output channels are known
    if isinstance(module1, nn.Linear) or isinstance(module1, nn.Conv2d):
        module1_output_channel_axis = 0
    else:
        print(module1)
        raise TypeError("Only Linear and Conv2d modules are supported at this time")
    if isinstance(module2, nn.Linear) or isinstance(module2, nn.Conv2d):
        module2_input_channel_axis = 1
    else:
        print(module2)
        raise TypeError("Only Linear and Conv2d modules are supported at this time")

    tensor1, tensor2 = scaling_channels(module1.weight, module2.weight, module1_output_channel_axis, module2_input_channel_axis)

    module1.weight = nn.parameter.Parameter(tensor1)
    module2.weight = nn.parameter.Parameter(tensor2)

def equalize(model, paired_modules_list, threshold):
    ''' Given a list of adjacent modules within a model, equalization will
    be applied between each pair, this will repeated until convergence is achieved

    '''
    name_to_module = {}
    copies = {}
    name_set = {name for pair in paired_modules_list for name in pair}

    for name, module in model.named_modules():
        if name in name_set:
            name_to_module[name] = module
            copies[name] = None

    while not converged_test(name_to_module, copies, threshold):
        for pair in paired_modules_list:
            copies[pair[0]] = copy.deepcopy(name_to_module[pair[0]])
            copies[pair[1]] = copy.deepcopy(name_to_module[pair[1]])

            cross_layer_equalization(name_to_module[pair[0]], name_to_module[pair[1]])

def converged_test(curr_modules, prev_modules, threshold):
    if len(curr_modules) != len(prev_modules):
        raise TypeError("in compatiables modules in convergence condition")

    summed_norms = 0
    for name in curr_modules.keys():
        if prev_modules[name] is None:
            return False
        difference = curr_modules[name].weight.sub(prev_modules[name].weight)
        summed_norms += torch.norm(difference)
    return summed_norms < threshold
