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

# @package regularizer_context
# Module caffe2.python.regularizer_context
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import context
from caffe2.python.modifier_context import (
    ModifierContext, UseModifierBase)


@context.define_context(allow_default=True)
class RegularizerContext(ModifierContext):
    """
    provide context to allow param_info to have different regularizers
    """

    def has_regularizer(self, name):
        return self._has_modifier(name)

    def get_regularizer(self, name):
        assert self.has_regularizer(name), (
            "{} regularizer is not provided!".format(name))
        return self._get_modifier(name)


class UseRegularizer(UseModifierBase):
    '''
    context class to allow setting the current context.
    Example useage with layer:
        regularizers = {'reg1': reg1, 'reg2': reg2}
        with Regularizers(regularizers):
            reg = RegularizerContext.current().get_regularizer('reg1')
            layer(reg=reg)
    '''
    def _context_class(self):
        return RegularizerContext
