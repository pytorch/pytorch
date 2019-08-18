from collections import namedtuple

TorchNNTestParams = namedtuple(
    'TorchNNTestParams',
    [
        'module_name',
        'module_variant_name',
        'python_constructor_args',
        'cpp_constructor_args',
        'example_inputs',
        'has_parity',
        'python_module_class',
        'cpp_sources',
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
