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

## @package sampling_trainable_mixin
# Module caffe2.python.layers.sampling_trainable_mixin
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import abc
import six


class SamplingTrainableMixin(six.with_metaclass(abc.ABCMeta, object)):

    def __init__(self, *args, **kwargs):
        super(SamplingTrainableMixin, self).__init__(*args, **kwargs)
        self._train_param_blobs = None
        self._train_param_blobs_frozen = False

    @property
    @abc.abstractmethod
    def param_blobs(self):
        """
        List of parameter blobs for prediction net
        """
        pass

    @property
    def train_param_blobs(self):
        """
        If train_param_blobs is not set before used, default to param_blobs
        """
        if self._train_param_blobs is None:
            self.train_param_blobs = self.param_blobs
        return self._train_param_blobs

    @train_param_blobs.setter
    def train_param_blobs(self, blobs):
        assert not self._train_param_blobs_frozen
        assert blobs is not None
        self._train_param_blobs_frozen = True
        self._train_param_blobs = blobs

    @abc.abstractmethod
    def _add_ops(self, net, param_blobs):
        """
        Add ops to the given net, using the given param_blobs
        """
        pass

    def add_ops(self, net):
        self._add_ops(net, self.param_blobs)

    def add_train_ops(self, net):
        self._add_ops(net, self.train_param_blobs)
