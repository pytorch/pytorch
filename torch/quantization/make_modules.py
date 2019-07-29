import os

_disclaimer = r"""
# This file is generated using `torch.nn.quantization.make_module`
# and saved as `{filename}`.

"""

_module_head = r'''
r"""{module_name} wraps the {op_string} function."""
class {module_name}(torch.nn.Module):
'''

_module_init = r'''    def __init__(self, **kwargs):
        super({module_name}, self).__init__()
'''

_module_init_extras = r'''
        self.scale = kwargs.get('scale', 1.0)
        self.zero_point = swargs.get('zero_point', 0)
'''

_module_forward = r'''
    def forward(self, *args):
        return self.{op_string}(*args)
'''

_module_from_float = r'''
    def from_float(self, mod):
        return {module_name}()
'''

"""Factory method to generate the ops files.

Args:
    op: The operation to wrap the module around
"""
def _make_module(op_string=None, qop_string=None):
    if op_string is None or qop_string is None:
        raise ValueError("Both op_string and qop_string must be set.")
    name = op_string.split('.')[-1].capitalize()
    name = name + "Gen"

    filename = 'Q' + name + '.py'
    params = {
        'module_name': name,
        'op_string': op_string,
        'filename': filename
    }

    generated_module = _disclaimer
    generated_module += _module_head
    generated_module += _module_init
    generated_module += _module_forward
    generated_module = generated_module.format(**params)

    qname = "Q" + name
    qparams = {
        'module_name': qname,
        'op_string': qop_string,
    }

    generated_qmodule = '\n\n' + _module_head
    generated_qmodule += _module_init
    generated_qmodule += _module_init_extras
    generated_qmodule += _module_forward
    generated_qmodule += _module_from_float
    generated_qmodule = generated_qmodule.format(**qparams)

    return generated_module + generated_qmodule, filename

def make_modules(location='./_generated'):
    operations = (
        ('torch.add', 'torch.ops.quantized.add'),
    )

    basepath = os.path.abspath(location)
    for op, qop in operations:
        py_body, filename = _make_module(op, qop)
        filename = os.path.join(basepath, filename)
        with open(filename, 'w') as genf:
            genf.write(py_body)

if __name__ == '__main__':
    make_modules()
