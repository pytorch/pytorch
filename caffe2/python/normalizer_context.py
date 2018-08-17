# @package regularizer_context
# Module caffe2.python.normalizer_context
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import context
from caffe2.python.modifier_context import (
    ModifierContext, UseModifierBase)


@context.define_context(allow_default=True)
class NormalizerContext(ModifierContext):
    """
    provide context to allow param_info to have different normalizers
    """

    def has_normalizer(self, name):
        return self._has_modifier(name)

    def get_normalizer(self, name):
        assert self.has_normalizer(name), (
            "{} normalizer is not provided!".format(name))
        return self._get_modifier(name)


class UseNormalizer(UseModifierBase):
    '''
    context class to allow setting the current context.
    Example useage with layer:
        normalizers = {'norm1': norm1, 'norm2': norm2}
        with UseNormalizer(normalizers):
            norm = NormalizerContext.current().get_normalizer('norm1')
            layer(norm=norm)
    '''
    def _context_class(self):
        return NormalizerContext
