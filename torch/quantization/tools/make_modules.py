import os

operations = (
    # (non-quantized op, quantized op)
    ('torch.add', 'torch.ops.quantized.add'),
)

_disclaimer = r"""
# @generated This file is produced by `torch/quantization/tools/make_module`.
"""

_imports = r"""
import torch
from torch.nn.modules import Module
"""

_module_body = r'''
r"""{module_name} wraps the {op_string} function."""
class {module_name}(Module):
    def __init__(self):
        super({module_name}, self).__init__()

    def forward(self, *args):
        return {op_string}(*args)
'''

_qmodule_body = r'''
r"""{module_name} wraps the {qop_string} function."""
class {module_name}(Module):
    __FLOAT_MODULE = torch.nn.modules.{float_op_name}

    def __init__(self):
        super({module_name}, self).__init__()
        self.register_buffer('scale', torch.tensor([1.0], dtype=torch.double))
        self.register_buffer('zero_point', torch.tensor([0], dtype=torch.long))

    def forward(self, *args):
        return {qop_string}(
            *args, scale=self.scale, zero_point=self.zero_point)

    @classmethod
    def from_float(cls, mod):
        assert (hasattr(mod, 'observer')),\
            "Input float module must have observer attached"
        assert (type(mod) == cls.__FLOAT_MODULE),\
            "nnq." + cls.__name__ + ".from_float only works for " \
            + cls.__FLOAT_MODULE.__name__
        scale, zero_point = mod.observer.calculate_qparams()[:2]
        mod = cls()
        mod.scale = torch.tensor(scale, dtype=torch.double)
        mod.zero_point = torch.tensor(zero_point, dtype=torch.long)
        return mod
'''

"""Factory method to generate the ops files.

Args:
    op: The operation to wrap the module around
"""
def _make_module(op_string=None, qop_string=None):
    if op_string is None or qop_string is None:
        raise ValueError("Both op_string and qop_string must be set.")
    name = op_string.split('.')[-1].capitalize()

    params = {
        'module_name': name,
        'op_string': op_string,
    }

    generated_module = _module_body.format(**params)
    qparams = {
        'module_name': name,
        'qop_string': qop_string,
        'float_op_name': name
    }

    generated_qmodule = _qmodule_body.format(**qparams)

    return generated_module, generated_qmodule

def make_modules(basepath='../..'):
    module_path = os.path.join(os.path.abspath(basepath), 'nn', 'modules')
    qmodule_path = os.path.join(os.path.abspath(basepath), 'nn', 'quantized',
                                'modules')
    filename = '_generated.py'
    f_name = os.path.join(module_path, filename)
    qf_name = os.path.join(qmodule_path, filename)
    file_body = ''
    qfile_body = ''
    for op, qop in operations:
        module_body, qmodule_body = _make_module(op, qop)
        file_body += module_body
        qfile_body += qmodule_body
        file_body += '\n'
        qfile_body += '\n'
    file_body = file_body[:-1]
    qfile_body = qfile_body[:-1]

    # Write the files
    message = "Generated '{}'. Please, amend {}/__init__.py"
    with open(f_name, 'w') as f:
        f.write(_disclaimer)
        f.write(_imports)
        f.write(file_body)
    print(message.format(f_name, module_path))
    with open(qf_name, 'w') as qf:
        qf.write(_disclaimer)
        qf.write(_imports)
        qf.write(qfile_body)
    print(message.format(qf_name, qmodule_path))

if __name__ == '__main__':
    make_modules()
