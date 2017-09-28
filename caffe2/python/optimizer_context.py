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

## @package optimizer_context
# Module caffe2.python.optimizer_context
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import context
from caffe2.python.modifier_context import (
    ModifierContext, UseModifierBase)


DEFAULT_OPTIM = 'DEFAULT'


@context.define_context(allow_default=True)
class OptimizerContext(ModifierContext):
    """
    provide context to allow param_info to have different optimizers
    """

    def has_optimizer(self, name):
        return self._has_modifier(name)

    def get_optimizer(self, name):
        assert self.has_optimizer(name), (
            "{} optimizer is not provided!".format(name))
        return self._get_modifier(name)


class UseOptimizer(UseModifierBase):
    '''
    context class to allow setting the current context.
    Example usage with brew:
        - with UseOptimizer(optim):
            brew.func
        - with UseOptimizer({'WEIGHT': weight_optim}):
            brew.func
        - with UseOptimizer({'DEFAULT': optim, 'BIAS': bias_optim,
                                'WEIGHT': weight_optim}):
            brew.func
        - with UseOptimizer(optim1):
            brew.func
            with UseOptimizer(optim2):
                brew.func

    Example useage with layer:
        optimizers = {'optim1': optim1, 'optim2': optim2}
        with Optimizers(optimizers):
            optim = OptimizerContext.current().get_optimizer('optim1')
            layer(optim=optim)
    '''
    def _context_class(self):
        return OptimizerContext
