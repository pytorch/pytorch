# @package regularizer_context
# Module caffe2.python.regularizer_context





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
    Example usage with layer:
        regularizers = {'reg1': reg1, 'reg2': reg2}
        with UseRegularizer(regularizers):
            reg = RegularizerContext.current().get_regularizer('reg1')
            layer(reg=reg)
    '''
    def _context_class(self):
        return RegularizerContext
