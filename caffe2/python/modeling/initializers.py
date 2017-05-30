from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.python.modeling.parameter_info import ParameterInfo


class Initializer(object):
    '''
    This class abstracts out parameter creation. One can  come up with a new
    Initializer in order to implement more complex parameter initializaion logic
    '''
    def __init__(self, operator_name=None, **kwargs):
        self.operator_name = operator_name
        self.operator_kwargs = kwargs

    def update(self, operator_name, kwargs):
        if self.operator_name is not None:
            raise Exception("Operator name overwrites are not allowed")
        self.operator_name = operator_name
        self.operator_kwargs = kwargs

    def create_param(self, param_name, init_net, shape):
        param = init_net.__getattr__(self.operator_name)(
            [], param_name, shape=shape, **self.operator_kwargs)
        return ParameterInfo(
            param_id=None,
            param=param,
            shape=shape,
        )

def update_initializer(initializer_class,
                       operator_name_and_kwargs,
                       default_operator_name_and_kwargs):
    '''
    A helper function to convert from operator_name_and_kwargs to new
    object of type initializer_class. This function serves two purposes:

    1. Support for custom initialization operators being passed in
    2. Allow user to specify a custom Initializer without overwriting
       default operators used for initialization

    If initializer_class is None, creates a default initializer using
    the Initializer class and operator_name_and_kwargs provided

    If operator_name_and_kwargs is None, uses default_operator_name_and_kwargs

    returns an instantiated Initializer object
    '''
    def get_initializer_args():
        return (
            operator_name_and_kwargs or
            default_operator_name_and_kwargs
        )

    if initializer_class is not None:
        init = initializer_class(get_initializer_args()[0],
                                 **get_initializer_args()[1])
    else:
        init = Initializer(
            get_initializer_args()[0],
            **get_initializer_args()[1]
        )
    return init
