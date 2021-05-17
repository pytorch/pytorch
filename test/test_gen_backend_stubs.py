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
