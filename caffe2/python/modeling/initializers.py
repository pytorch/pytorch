# Copyright (c) 2016-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python.core import DataType, BlobReference, ScopedBlobReference
from caffe2.python.modeling.parameter_info import ParameterInfo

import six


class Initializer(object):
    '''
    This class abstracts out parameter creation. One cancome up with a new
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
            raise "Unsupported type for param_name"
        # TODO(amalevich): Add operator that will check param in the workspace
        return ParameterInfo(
            param_id=None,
            param=param,
            shape=shape,
        )


class pFP16Initializer(Initializer):

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
