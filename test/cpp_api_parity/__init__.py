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
        'expect_parity_error',
        'cpp_forward_arg_declarations',
        'python_module_class',
        'cpp_source',
        'device',
    ]
)
