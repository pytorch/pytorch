import os
import sys

import torch
from torch.testing import FileCheck

# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.jit_utils import JitTestCase, op_alias_mappings
from torch.testing._internal.jit_metaprogramming_utils import create_script_fn, create_traced_fn
from torch.testing._internal.common_methods_invocations import method_tests, create_input

if __name__ == '__main__':
    raise RuntimeError("This test file is not meant to be run directly, use:\n\n"
                       "\tpython test/test_jit.py TESTNAME\n\n"
                       "instead.")

def get_defaults(
        name,
        self_size,
        args,
        variant_name='',
        check_ad=(),
        dim_args_idx=(),
        skipTestIf=(),
        output_process_fn=lambda x: x,
        kwargs=None):
    args = tuple(enumerate(args))
    kwargs = kwargs if kwargs else {}
    return name, self_size, args, kwargs, output_process_fn

class TestOpNormalization(JitTestCase):

    @torch.no_grad()
    def test_aliases(self):
        # tests that op aliases are correctly being normalized
        # does not check for other properties such as correctness because
        # the common method registry gets tested for those in test_jit.py

        op_registry = {}
        for op in method_tests():
            op_registry[op[0]] = op

        for alias, mapping in op_alias_mappings.items():
            assert alias in op_registry, "Test not found for {} alias".format(alias)

            name, self_size, args, kwargs, output_process_fn = get_defaults(*op_registry[alias])

            def fn(*inputs, **kwargs):
                attr = getattr(inputs[0], name)
                output = attr(*inputs[1:], **kwargs)
                return output_process_fn(output)

            self_variable = create_input((self_size,))[0][0]
            args_variable, kwargs_variable = create_input(args, requires_grad=False, call_kwargs=kwargs)

            traced_fn = create_traced_fn(self, fn)
            inputs = (self_variable,) + args_variable
            traced_fn(*inputs, **kwargs)
            last_graph = traced_fn.last_graph
            FileCheck().check(mapping).check_not(alias).run(last_graph)

            script_fn = create_script_fn(self, name, 'method', output_process_fn)
            script_fn(*inputs, **kwargs)
            last_graph = script_fn.last_graph
            FileCheck().check(mapping).check_not(alias).run(last_graph)
