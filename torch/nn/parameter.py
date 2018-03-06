from torch.autograd import Variable


class Parameter(Variable):
    r"""A kind of Variable that is to be considered a module parameter.

    Parameters are :class:`~torch.autograd.Variable` subclasses, that have a
    very special property when used with :class:`Module` s - when they're
    assigned as Module attributes they are automatically added to the list of
    its parameters, and will appear e.g. in :meth:`~Module.parameters` iterator.
    Assigning a Variable doesn't have such effect. This is because one might
    want to cache some temporary state, like last hidden state of the RNN, in
    the model. If there was no such class as :class:`Parameter`, these
    temporaries would get registered too.

    Arguments:
        data (Tensor): parameter tensor.
        requires_grad (bool, optional): if the parameter requires gradient. See
            :ref:`excluding-subgraphs` for more details.
    """
    def __new__(cls, data=None, requires_grad=True):
        return super(Parameter, cls).__new__(cls, data, requires_grad=requires_grad)

    def __repr__(self):
        return super(Parameter, self).__repr__().replace('Variable', 'Parameter')
