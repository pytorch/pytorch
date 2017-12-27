import os
from tools.shared.module_loader import import_module
from .nested_dict import nested_dict


__all__ = ['CodeTemplate', 'nested_dict', 'write']


CodeTemplate = import_module('code_template', 'aten/src/ATen/code_template.py').CodeTemplate

try:
    # use faster C loader if available
    from yaml import CLoader as YamlLoader
except ImportError:
    from yaml import YamlLoader


GENERATED_COMMENT = CodeTemplate("""\
generated from tools/autograd/templates/${filename}""")


def write(dirname, name, template, env):
    env['generated_comment'] = GENERATED_COMMENT.substitute(filename=name)
    path = os.path.join(dirname, name)
    with open(path, 'w') as f:
        f.write(template.substitute(env))
