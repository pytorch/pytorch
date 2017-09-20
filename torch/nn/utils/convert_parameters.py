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

    pointer = 0
    for param in parameters:
        # The length of the parameter
        num_param = torch.prod(torch.LongTensor(list(param.size())))
        # Slice the vector, reshape it, and replace the old data of the parameter
        param.data = vec[pointer:pointer + num_param].view(param.size()).data

        # Increment the pointer
        pointer += num_param
