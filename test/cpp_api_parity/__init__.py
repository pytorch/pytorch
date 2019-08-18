from collections import namedtuple

CppArgDeclaration = namedtuple('CppArgDeclaration', ['arg_type', 'arg_name'])

TorchNNTestParams = namedtuple(
    'TorchNNTestParams',
    [
        'module_name',
        'module_variant_name',
        'python_constructor_args',
        'cpp_constructor_args',
        'input_size',
        'input_fn',
        'has_parity',
        'cpp_forward_arg_declarations',
        'python_module_class',
        'cpp_source',
        'device',
    ]
)

class set_has_parity(object):
    has_parity = False

    def __init__(self, parity):
        self.prev_has_parity = set_has_parity.has_parity
        set_has_parity.has_parity = parity

    def __enter__(self):
        pass

    def __exit__(self, *args):
        set_has_parity.has_parity = self.prev_has_parity
        return False


def has_parity():
    return set_has_parity.has_parity
