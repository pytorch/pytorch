




from caffe2.python.core import DataType, BlobReference, ScopedBlobReference
from caffe2.python.modeling.parameter_info import ParameterInfo

import six


class Initializer(object):
    '''
    This class abstracts out parameter creation. One can come up with a new
    Initializer in order to implement more complex parameter initialization logic
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


class ExternalInitializer(object):
    '''
    This class is used in cases when the parameter should not be initialized by
    the initializer, but rather provided in the workspace when param_init_net is
    executed.

    Current version is not doing any real sanity checks to the parameter.
    '''

    def create_param(self, param_name, init_net, shape):
        if isinstance(param_name, BlobReference):
            param = BlobReference(str(param_name), init_net)
        elif isinstance(param_name, six.string_types):
            param = ScopedBlobReference(param_name, init_net)
        else:
            raise TypeError("Unsupported type for param_name")
        # TODO(amalevich): Add operator that will check param in the workspace
        return ParameterInfo(
            param_id=None,
            param=param,
            shape=shape,
        )


class PseudoFP16Initializer(Initializer):
    '''
    Used in cases when the parameter should be used at half (16-bit) precision
    for compute purposes (i.e. on the forward and backward pass) but
    needs to be stored and optimized at single (32-bit) precision so tiny
    gradients with small learning rates don't underflow FP16 precision.
    A 32-bit copy of the 16-bit blob is stored in the ParameterInfo.
    This is helpful for mixed-precision training, see
    https://arxiv.org/abs/1710.03740 for details.
    '''
    def update(self, operator_name, kwargs):
        if self.operator_name is not None:
            raise Exception("Operator name overwrites are not allowed")
        self.operator_name = operator_name
        self.operator_kwargs = kwargs

    def create_param(self, param_name, init_net, shape):
        # create master fp32 copy
        param_fp32 = init_net.__getattr__(self.operator_name)(
            [], param_name + "_fp32", shape=shape,
            **self.operator_kwargs)
        # cast to fp16 copy
        param = init_net.FloatToHalf(
            param_fp32, param_name)

        return ParameterInfo(
            param_id=None,
            param=param,
            shape=shape,
            blob_copy={DataType.FLOAT: param_fp32}
        )


class ReversePseudoFP16Initializer(Initializer):
    '''
    Like PseudoFP16Initializer above, except the primary blob is taken to
    be the 32-bit precision parameter, and the 16-bit version of the blob
    is stored in blob_copy instead.
    '''
    def update(self, operator_name, kwargs):
        if self.operator_name is not None:
            raise Exception("Operator name overwrites are not allowed")
        self.operator_name = operator_name
        self.operator_kwargs = kwargs

    def create_param(self, param_name, init_net, shape):
        # create master fp32 copy
        param_fp32 = init_net.__getattr__(self.operator_name)(
            [], param_name, shape=shape,
            **self.operator_kwargs)
        # cast to fp16 copy
        param_fp16 = init_net.FloatToHalf(
            param_fp32, param_name + "_fp16")

        return ParameterInfo(
            param_id=None,
            param=param_fp32,
            shape=shape,
            blob_copy={DataType.FLOAT16: param_fp16}
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
