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
        'num_attrs_recursive',
        'device',
    ]
)

CppArg = namedtuple('CppArg', ['type', 'value'])
