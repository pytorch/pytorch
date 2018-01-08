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

# @package optimizer
# Module caffe2.python.optimizer
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


from caffe2.python import core


class Regularizer(object):
    def __init__(self):
        self.apply_after_optimizer = False

    '''
    Adds regularization to train_net for given parameter. Its factor ahead of
    regularization is given when initialization.
    The param should be a BlobReference.
    '''

    def __call__(self, net, param_init_net, param, grad=None):
        assert isinstance(param, core.BlobReference)
        return self._run(net, param_init_net, param)

    def _run(self, net, param_init_net, param):
        raise Exception("Not Impelemented")


class L1Norm(Regularizer):
    def __init__(self, reg_lambda):
        super(L1Norm, self).__init__()
        assert reg_lambda >= 0,\
            'factor ahead of regularization should be 0 or positive'

        self.reg_lambda = reg_lambda

    def _run(self, net, param_init_net, param):
        output_blob = net.NextScopedBlob(param + '_l1_regularization')
        net.LpNorm([param], [output_blob], p=1)
        net.Scale([output_blob], [output_blob], scale=self.reg_lambda)
        return output_blob


class L2Norm(Regularizer):
    def __init__(self, reg_lambda):
        super(L2Norm, self).__init__()
        assert reg_lambda >= 0,\
            'factor ahead of regularization should be 0 or positive'

        self.reg_lambda = reg_lambda

    def _run(self, net, param_init_net, param):
        output_blob = net.NextScopedBlob(param + '_l2_regularization')
        net.LpNorm([param], [output_blob], p=2)
        net.Scale([output_blob], [output_blob], scale=self.reg_lambda)
        return output_blob
