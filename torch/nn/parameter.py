from torch.autograd import Variable


class Parameter(Variable):

    def __init__(self, data, requires_grad=True):
        super(Parameter, self).__init__(data, requires_grad=requires_grad)

    def __deepcopy__(self, memo):
        result = type(self)(self.data.clone(), self.requires_grad)
        memo[id(self)] = result
        return result

    def __repr__(self):
        return 'Parameter containing:' + self.data.__repr__()
