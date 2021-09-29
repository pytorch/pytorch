# The purpose of this test is to check that we have implementation parity between
# a Python `torch.nn` module and its corresponding C++ `torch::nn` module. Concretely,
# this test does the following:
#
# 1. Get a test params dict from common_nn.py, run forward and backward on the
# Python module created using the test params.
#
# 2. Serialize the Python module's parameters / buffers and its forward input
# arguments, deserialize them in C++ and load them into the C++ module.
#
# 3. Run the same forward and backward passes on the C++ module, and serialize
# the C++ module's forward output and backward gradients.
#
# 4. Compare Python/C++ module's forward output and backward gradients. If they
# are the same, then we have implementation parity between Python/C++ module.

import tempfile
from collections import defaultdict

import torch
import torch.testing._internal.common_modules as common_modules
from cpp_api_parity.utils import TorchNNModuleInfoTestParams, TORCH_NN_COMMON_TEST_HARNESS, \
    compile_cpp_code_inline, add_test, \
    decorate_test_fn, try_remove_folder, CppArg, convert_to_list
from cpp_api_parity.sample_module import SAMPLE_MODULE_CPP_SOURCE
from cpp_api_parity.module_utils import test_forward_backward, TORCH_NN_MODULE_TEST_FORWARD_BACKWARD, \
    generate_test_cpp_sources

def process_test_params_for_module(module_info, sample, device):
    module_name = module_info.name.replace('nn_', '')
    module_variant_name = module_name + sample.desc + (('_' + device) if device != 'cpu' else '')

    def compute_arg_dict(sample):
        arg_dict = defaultdict(list)

        def put_args_into_arg_dict(arg_type, arg_type_prefix, args):
            for i, arg in enumerate(args):
                arg_dict[arg_type].append(CppArg(name=arg_type_prefix + str(i), value=arg))

        err_msg = ("Sample's forward input can't have kwarg if cpp_parity is True"
                   " as C++ does not have kwarg")
        assert len(sample.forward_input.kwargs) == 0, err_msg
        inputs = convert_to_list(sample.forward_input.args)
        put_args_into_arg_dict('input', 'i', inputs)
        return arg_dict

    arg_dict = compute_arg_dict(sample)

    return TorchNNModuleInfoTestParams(
        module_name=module_name,
        module_variant_name=module_variant_name,
        module_info=module_info,
        sample=sample,
        cpp_constructor_args=getattr(sample, 'cpp_constructor_args', ''),
        arg_dict=arg_dict,
        has_parity=True,
        device=device,
        cpp_tmp_folder=tempfile.mkdtemp(),
    )


def write_test_to_test_class(
        unit_test_class, module_info, devices):
    assert isinstance(module_info, common_modules.ModuleInfo)
    module_name = module_info.name.replace('nn_', '')
    assert hasattr(torch.nn, module_name), (
        "`torch.nn` doesn't have module `{}` for module_info"
    ).format(module_name)

    for device in devices:
        for sample in module_info.module_inputs_func(module_info, device='cpu',
                                                     dtype=torch.get_default_dtype(), requires_grad=False):
            if sample.test_cpp_parity:
                test_params = process_test_params_for_module(
                    module_info=module_info,
                    sample=sample,
                    device=device,
                )
                try_remove_folder(test_params.cpp_tmp_folder)
                unit_test_name = 'test_torch_module_info_nn_{}'.format(test_params.module_variant_name)
                unit_test_class.module_info_test_params_map[unit_test_name] = test_params

                def test_fn(self):
                    test_forward_backward(
                        unit_test_class=self, test_params=unit_test_class.module_info_test_params_map[self._testMethodName],
                        cpp_module=unit_test_class.module_info_impl_check_cpp_module)

                test_fn = decorate_test_fn(
                    test_fn=test_fn,
                    test_cuda=True,
                    has_impl_parity=True,
                    device=device)

                add_test(unit_test_class, unit_test_name, test_fn)

# Build all C++ tests together, instead of once per test.
def build_cpp_tests(unit_test_class, print_cpp_source=False):
    assert len(unit_test_class.module_info_test_params_map) > 0
    cpp_sources = TORCH_NN_COMMON_TEST_HARNESS + SAMPLE_MODULE_CPP_SOURCE
    functions = []
    for _, test_params in unit_test_class.module_info_test_params_map.items():
        cpp_sources += generate_test_cpp_sources(
            test_params=test_params, template=TORCH_NN_MODULE_TEST_FORWARD_BACKWARD)
        functions.append('{}_test_forward_backward'.format(test_params.module_variant_name))
    if print_cpp_source:
        print(cpp_sources)

    cpp_module = compile_cpp_code_inline(
        name='module_info_impl_check',
        cpp_sources=cpp_sources,
        functions=functions)
    unit_test_class.module_info_impl_check_cpp_module = cpp_module
