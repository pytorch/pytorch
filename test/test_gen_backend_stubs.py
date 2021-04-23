import os
import tempfile

from torch.testing._internal.common_utils import TestCase, run_tests
import tools.codegen.gen_backend_stubs

path = os.path.dirname(os.path.realpath(__file__))
gen_backend_stubs_path = os.path.join(path, '../tools/codegen/gen_backend_stubs.py')

# gen_backend_stubs.py is an integration point that is called directly by external backends.
# The tests here are to confirm that badly formed inputs result in reasonable error messages.
class TestGenBackendStubs(TestCase):

    def get_errors_from_gen_backend_stubs(self, yaml_str: str) -> str:
        with tempfile.NamedTemporaryFile(mode='w') as fp:
            fp.write(yaml_str)
            fp.flush()
            try:
                tools.codegen.gen_backend_stubs.run(fp.name, '', True)
            except AssertionError as e:
                # Scrub out the temp file name from any error messages to simplify assertions.
                return str(e).replace(fp.name, '')
            self.fail('Expected gen_backend_stubs to raise an AssertionError, but it did not.')

    def test_missing_backend(self):
        yaml_str = '''\
cpp_namespace: torch_xla
supported:
- abs'''
        output_error = self.get_errors_from_gen_backend_stubs(yaml_str)
        self.assertExpectedInline(output_error, '''You must provide a value for "backend"''')

    def test_empty_backend(self):
        yaml_str = '''\
backend:
cpp_namespace: torch_xla
supported:
- abs'''
        output_error = self.get_errors_from_gen_backend_stubs(yaml_str)
        self.assertExpectedInline(output_error, '''You must provide a value for "backend"''')

    def test_backend_invalid_dispatch_key(self):
        yaml_str = '''\
backend: NOT_XLA
cpp_namespace: torch_xla
supported:
- abs'''
        output_error = self.get_errors_from_gen_backend_stubs(yaml_str)
        self.assertExpectedInline(output_error, '''The provided value for "backend" must be a valid DispatchKey, but got NOT_XLA. The set of valid dispatch keys is: Undefined, CatchAll, CPU, CUDA, HIP, FPGA, MSNPU, XLA, Vulkan, Metal, XPU, MKLDNN, OpenGL, OpenCL, IDEEP, QuantizedCPU, QuantizedCUDA, QuantizedXPU, CustomRNGKeyId, MkldnnCPU, SparseCPU, SparseCUDA, SparseCsrCPU, SparseCsrCUDA, SparseHIP, SparseXPU, NestedTensor, PrivateUse1, PrivateUse2, PrivateUse3, EndOfBackendKeys, Meta, BackendSelect, Named, AutogradOther, AutogradCPU, AutogradCUDA, AutogradXLA, AutogradNestedTensor, AutogradXPU, AutogradPrivateUse1, AutogradPrivateUse2, AutogradPrivateUse3, Tracer, Autocast, Batched, VmapMode, TESTING_ONLY_GenericWrapper, TESTING_ONLY_GenericMode, NumDispatchKeys, Autograd, CompositeImplicitAutograd, CompositeExplicitAutograd, EndOfAliasKeys, CPUTensorId, CUDATensorId, PrivateUse1_PreAutograd, PrivateUse2_PreAutograd, PrivateUse3_PreAutograd''')  # noqa: B950

    def test_missing_cpp_namespace(self):
        yaml_str = '''\
backend: XLA
supported:
- abs'''
        output_error = self.get_errors_from_gen_backend_stubs(yaml_str)
        self.assertExpectedInline(output_error, '''You must provide a value for "cpp_namespace"''')

    def test_whitespace_cpp_namespace(self):
        yaml_str = '''\
backend: XLA
cpp_namespace:\t
supported:
- abs'''
        output_error = self.get_errors_from_gen_backend_stubs(yaml_str)
        self.assertExpectedInline(output_error, '''You must provide a value for "cpp_namespace"''')

    # supported is a single item (it should be a list)
    def test_nonlist_supported(self):
        yaml_str = '''\
backend: XLA
cpp_namespace: torch_xla
supported: abs'''
        output_error = self.get_errors_from_gen_backend_stubs(yaml_str)
        self.assertExpectedInline(output_error, '''expected "supported" to be a list, but got: abs (of type <class 'str'>)''')

    # supported contains an op that isn't in native_functions.yaml
    def test_supported_invalid_op(self):
        yaml_str = '''\
backend: XLA
cpp_namespace: torch_xla
supported:
- abs_BAD'''
        output_error = self.get_errors_from_gen_backend_stubs(yaml_str)
        self.assertExpectedInline(output_error, '''Found an invalid operator name: abs_BAD''')

    # The backend is valid, but doesn't have a valid autograd key. They can't override autograd kernels in that case.
    # Only using MSNPU here because it has a valid backend key but not an autograd key- if this changes we can update the test.
    def test_backend_has_no_autograd_key_but_provides_entries(self):
        yaml_str = '''\
backend: MSNPU
cpp_namespace: torch_msnpu
supported:
- add
autograd:
- sub'''
        output_error = self.get_errors_from_gen_backend_stubs(yaml_str)
        self.assertExpectedInline(output_error, '''The "autograd" key was specified, which indicates that you would like to override the behavior of autograd for some operators on your backend. However "AutogradMSNPU" is not a valid DispatchKey. The set of valid dispatch keys is: Undefined, CatchAll, CPU, CUDA, HIP, FPGA, MSNPU, XLA, Vulkan, Metal, XPU, MKLDNN, OpenGL, OpenCL, IDEEP, QuantizedCPU, QuantizedCUDA, QuantizedXPU, CustomRNGKeyId, MkldnnCPU, SparseCPU, SparseCUDA, SparseCsrCPU, SparseCsrCUDA, SparseHIP, SparseXPU, NestedTensor, PrivateUse1, PrivateUse2, PrivateUse3, EndOfBackendKeys, Meta, BackendSelect, Named, AutogradOther, AutogradCPU, AutogradCUDA, AutogradXLA, AutogradNestedTensor, AutogradXPU, AutogradPrivateUse1, AutogradPrivateUse2, AutogradPrivateUse3, Tracer, Autocast, Batched, VmapMode, TESTING_ONLY_GenericWrapper, TESTING_ONLY_GenericMode, NumDispatchKeys, Autograd, CompositeImplicitAutograd, CompositeExplicitAutograd, EndOfAliasKeys, CPUTensorId, CUDATensorId, PrivateUse1_PreAutograd, PrivateUse2_PreAutograd, PrivateUse3_PreAutograd''')  # noqa: B950

    # in an operator group, currently all operators must either be registered to the backend or autograd kernel.
    # Here, functional and out mismatch
    def test_backend_autograd_kernel_mismatch_out_functional(self):
        yaml_str = '''\
backend: XLA
cpp_namespace: torch_msnpu
supported:
- add.Tensor
autograd:
- add.out'''
        output_error = self.get_errors_from_gen_backend_stubs(yaml_str)
        self.assertExpectedInline(output_error, '''Currently, all variants of an op must either be registered to a backend key, or to a backend's autograd key. They can not be mix and matched. If this is something you need, feel free to create an issue! add.out is listed under autograd, but add.Tensor is listed under supported''')  # noqa: B950

    # in an operator group, currently all operators must either be registered to the backend or autograd kernel.
    # Here, functional and inplace mismatch
    def test_backend_autograd_kernel_mismatch_functional_inplace(self):
        yaml_str = '''\
backend: XLA
cpp_namespace: torch_msnpu
supported:
- add.Tensor
autograd:
- add_.Tensor'''
        output_error = self.get_errors_from_gen_backend_stubs(yaml_str)
        self.assertExpectedInline(output_error, '''Currently, all variants of an op must either be registered to a backend key, or to a backend's autograd key. They can not be mix and matched. If this is something you need, feel free to create an issue! add_.Tensor is listed under autograd, but add.Tensor is listed under supported''')  # noqa: B950

    # unrecognized extra yaml key
    def test_unrecognized_key(self):
        yaml_str = '''\
backend: XLA
cpp_namespace: torch_xla
supported:
- abs
invalid_key: invalid_val'''
        output_error = self.get_errors_from_gen_backend_stubs(yaml_str)
        self.assertExpectedInline(output_error, ''' contains unexpected keys: invalid_key. Only the following keys are supported: backend, cpp_namespace, supported, autograd''')  # noqa: B950


if __name__ == '__main__':
    run_tests()
