# @package modifier_context
# Module caffe2.python.modifier_context
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


DEFAULT_MODIFIER = 'DEFAULT'


class ModifierContext(object):
    """
    provide context to allow param_info to have different modifiers
    """

    def __init__(self):
        self._modifiers = {}
        self._modifiers_list = []

    def _rebuild_modifiers(self):
        self._modifiers = {}
        for m in self._modifiers_list:
            self._modifiers.update(m)

    def _has_modifier(self, name):
        return name in self._modifiers

    def _get_modifier(self, name):
        return self._modifiers.get(name)

    def push_modifiers(self, modifiers):
        # modifier override is allowed
        self._modifiers_list.append(modifiers)
        self._modifiers.update(modifiers)

    def pop_modifiers(self):
        assert len(self._modifiers_list) > 0
        self._modifiers_list.pop()
        self._rebuild_modifiers()


class UseModifierBase(object):
    '''
    context class to allow setting the current context.
    Example useage with layer:
        modifiers = {'modifier1': modifier1, 'modifier2': modifier2}
        with Modifiers(modifiers):
            modifier = ModifierContext.current().get_modifier('modifier1')
            layer(modifier=modifier)
    '''

    def __init__(self, modifier_or_dict):
        if isinstance(modifier_or_dict, dict):
            self._modifiers = modifier_or_dict
        else:
            self._modifiers = {DEFAULT_MODIFIER: modifier_or_dict}

    def _context_class(self):
        raise NotImplementedError

    def __enter__(self):
        self._context_class().current().push_modifiers(self._modifiers)
        return self

    def __exit__(self, type, value, traceback):
        self._context_class().current().pop_modifiers()
