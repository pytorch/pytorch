import os
import tempfile
import subprocess

from torch.testing._internal.common_utils import TestCase, run_tests

path = os.path.dirname(os.path.realpath(__file__))
gen_backend_stubs_path = os.path.join(path, '../tools/codegen/gen_backend_stubs.py')

yaml_with_errors = {
    # Missing key (backend)
    '''\
cpp_namespace: torch_xla
supported:
- abs
    ''': 'AssertionError: You must provide a value for "backend"',
    # 'backend' key provided but not specified
    '''\
backend:
cpp_namespace: torch_xla
supported:
- abs
    ''': 'AssertionError: You must provide a value for "backend"',

    # Missing key (cpp_namespace)
    '''\
backend: XLA
supported:
- abs
    ''': 'AssertionError: You must provide a value for "cpp_namespace"',


    # 'cpp_namespace' key provided but not specified (including whitespace)
    '''\
backend: XLA
cpp_namespace:\t
supported:
- abs
    ''': 'AssertionError: You must provide a value for "cpp_namespace"',

    # supported is empty (it should be a list)
    '''\
backend: XLA
cpp_namespace: torch_xla
supported:
    ''': 'AssertionError: expected "supported" to be a list, but got: None',

    # supported is a single item (it should be a list)
    '''\
backend: XLA
cpp_namespace: torch_xla
supported: abs
    ''': "AssertionError: expected \"supported\" to be a list, but got: abs (of type <class 'str'>)",

    # supported contains an op that isn't in native_functions.yaml
    '''\
backend: XLA
cpp_namespace: torch_xla
supported:
- abs_BAD
    ''': 'AssertionError: Found an invalid operator name: abs_BAD',

    # unrecognized extra yaml key
    '''\
backend: XLA
cpp_namespace: torch_xla
supported:
- abs
invalid_key: invalid_val
    ''': 'contains unexpected keys: invalid_key. \
Only the following keys are supported: backend, cpp_namespace, supported, autograd',
}

# gen_backend_stubs.py is an integration point that is called directly by external backends.
# The tests here are to confirm that badly formed inputs result in reasonable error messages.
class TestGenBackendStubs(TestCase):

    def assert_yaml_causes_error(self, yaml_str: str, error_msg: str) -> None:
        with tempfile.NamedTemporaryFile(mode='w') as fp:
            fp.write(yaml_str)
            fp.flush()
            command = ['python',
                       gen_backend_stubs_path,
                       '--output_dir=""',
                       '--dry_run=True',
                       f'--source_yaml={fp.name}']

            subprocess_output = subprocess.run(command, encoding='utf-8', stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            std_err = subprocess_output.stderr
            self.assertTrue(
                error_msg in std_err,
                f'Expected a string containing \n{error_msg}\n, but got \n{std_err}\n')

    def test_error_message(self):
        for yaml_str, error_msg in yaml_with_errors.items():
            self.assert_yaml_causes_error(yaml_str, error_msg)

if __name__ == '__main__':
    run_tests()
