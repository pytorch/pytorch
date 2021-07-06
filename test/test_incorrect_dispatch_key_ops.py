import os
import re
from typing import List

from tools.codegen.gen import parse_native_yaml, get_grouped_native_functions
from tools.codegen.model import DispatchKey, NativeFunction, NativeFunctionsGroup
from torch.testing._internal.common_utils import TestCase, run_tests

# The test below is a quick-and-dirty check to ensure that people mark operators in native_functions.yaml
# with valid dispatch keys.
# In particular, it's INCORRECT to implement an operator using DispatchStub, but mark the operator
# with the CompositExplicitAutograd key in native_functions.yaml.
# This is bad for two reasons:
# - CompositeExplicitAutograd implies that the op is valid on all backends, but it isn't;
#   it will break for external backends
# - The cuda kernel registered to DispatchStub expects a device guard to be set; but the codegen
#   doesn't create one for ComositeExplicitAutograd entries, because it expects Composite ops
#   to call into sub-ops that themselves are registered to the CUDA key, which will have
#   their own device guards.
# See an example bug from this at https://github.com/pytorch/pytorch/issues/60892

# The test written below is pretty fragile; it parses the aten/ folder to look for all
# dispatch stub kernels, and maps them to native_functions.yaml ops through naming conventions.
# I had to add a list of ignored ops here to avoid false positives.
# For example, `at::native::cumprod` is a (valid) CompositeExplicitAutograd kernel, which is implemented
# by calling directly into `at::native::_cumprod` - which has a CPU/CUDA kernel that calls `cumprod_stub()`,
# which is registered to DispatchStub.
# This is a kind of weird pattern, but it's valid, and hard to avoid catching in this test.
#
# The hope is that this test saves more overall time by catching bugs such as #60892,
# than it does generating new false positives.
#
# If you create a future op that's implemented similarly and you're sure it's a false positive,
# add it to this list.
IGNORED_OPS = [
    "cumprod",
    "cumsum",
    "logcumsumexp",
]

class TestIncorrectDispatchKeyOps(TestCase):

    def find_dispatch_stub_kernels(self) -> List[str]:
        curr_path = os.path.dirname(os.path.realpath(__file__))
        root_dir = os.path.join(curr_path, '../aten/src/ATen/')
        regex = re.compile("DEFINE_DISPATCH\((.*)_stub\)")

        matches = []
        for root, dirs, files in os.walk(root_dir):
            # Skip hidden files and directories
            files = [f for f in files if not f[0] == '.']
            dirs[:] = [d for d in dirs if not d[0] == '.']
            for f_name in files:
                if not f_name.endswith(".cpp"):
                    continue
                f_path = os.path.join(root, f_name)
                with open(f_path, "r") as f:
                    f_str = f.read()
                    matches += regex.findall(f_str)

        return matches

    def test_find_invalid_CompositeExplicitAutograd_ops(self):

        def op_name(f: NativeFunction) -> str:
            op_name = f.func.name
            op_name_str = str(op_name.name)
            if op_name.overload_name:
                op_name_str += f'_{op_name.overload_name}'
            return op_name_str

        def invalid(op: str) -> bool:
            return op in dispatch_stub_kernels and op not in IGNORED_OPS

        curr_path = os.path.dirname(os.path.realpath(__file__))
        yaml_path = os.path.join(curr_path, '../aten/src/ATen/native/native_functions.yaml')

        parsed_yaml = parse_native_yaml(yaml_path)
        native_functions, backend_indices = parsed_yaml.native_functions, parsed_yaml.backend_indices
        grouped_native_functions = get_grouped_native_functions(native_functions)

        dispatch_stub_kernels = self.find_dispatch_stub_kernels()

        invalid_op_entries: List[str] = []
        for g in grouped_native_functions:
            if isinstance(g, NativeFunction):
                f = g
                if not backend_indices[DispatchKey.CompositeExplicitAutograd].has_kernel(f):
                    continue
                op_name_str = op_name(f)
                if invalid(op_name_str):
                    invalid_op_entries.append(op_name_str)

            elif isinstance(g, NativeFunctionsGroup):
                if not all([backend_indices[DispatchKey.CompositeExplicitAutograd].has_kernel(f) for f in g.functions()]):
                    continue
                op_name_str = op_name(g.functional)
                if invalid(op_name_str):
                    invalid_op_entries.append(op_name_str)
            else:
                raise Exception()
        invalid_op_entries_str = '\n'.join(invalid_op_entries)
        self.assertEqual([], invalid_op_entries, f"""
Detected an operator that is marked as CompositeExplicitAutograd in native_functions.yaml, but appears to
be implemented using DispatchStub.

detected operators:
{invalid_op_entries_str}

If you think that this is a false positive add the operator name to IGNORED_OPS in the file {__file__}.
A common false positive is if you wrote your operator `at::native::{{op}}` as a wrapper that calls into
`at::native::{{_op}}` to do the real, backend-specific work. In this case, marking {{op}} with
CompositeExplicitAutograd and {{_op}} with CPU,CUDA is completely valid.""")


if __name__ == '__main__':
    run_tests()
