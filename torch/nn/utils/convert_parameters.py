import torch
from torch.autograd import Variable


def parameters_to_vector(parameters):
    """Convert parameters to one vector

    Arguments:
        parameters (Iterable[Variable]): an iterator of Variables that are the
            parameters of a model.

    Returns:
        The parameters represented by a single vector
    """
    vec = []
    for param in parameters:
        vec.append(param.view(-1))
    return torch.cat(vec)


def vector_to_parameters(vec, parameters):
    """Convert one vector to the parameters

    Arguments:
        vec (Variable): a single vector represents the parameters of a model.
        parameters (Iterable[Variable]): an iterator of Variables that are the
            parameters of a model.
    """
    # Ensure vec of type Variable
    if not isinstance(vec, Variable):
        raise TypeError('expected torch.autograd.Variable, but got: {}'
                        .format(torch.typename(vec)))
    # Flag for the device where the parameter is located
    param_device = None

    # Pointer for slicing the vector for each parameter
    pointer = 0
    for param in parameters:
        # Ensure the parameters are located in the same device
        if param_device is None:
            param_device = param.get_device() if param.is_cuda else -1
        else:
            warn = False
            if param.is_cuda:
                warn = (param.get_device() != param_device)
            else:
                warn = (param_device != -1)
            if warn:
                raise TypeError('Found two parameters on different devices, '
                                'this is currently not supported.')

        # The length of the parameter
        num_param = torch.prod(torch.LongTensor(list(param.size())))
        # Slice the vector, reshape it, and replace the old data of the parameter
        param.data = vec[pointer:pointer + num_param].view(param.size()).data

        # Increment the pointer
        pointer += num_param
