# Owner(s): ["oncall: jit"]

import contextlib
import os
import sys
import unittest

import torch

# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.common_utils import (
    IS_FBCODE,
    run_tests,
    set_default_dtype,
    suppress_warnings,
)
from torch.testing._internal.jit_metaprogramming_utils import (
    get_all_nn_module_tests,
    get_nn_functional_compiled_fn_and_inputs,
    get_nn_mod_test_name,
    nn_functional_tests,
    try_get_nn_module_compiled_mod_and_inputs,
)
from torch.testing._internal.jit_utils import enable_profiling_mode, JitTestCase


def num_ifs_loops(graph):
    graph_str = str(graph)
    # only look at body of graph
    graph_body = graph_str[0 : graph_str.find("return")]
    return graph_body.count("prim::Loop") + graph_body.count("prim::If")


def num_non_tensor_nodes(block):
    num_non_tensor = 0
    for node in block.nodes():
        kind = node.kind()
        # GetAttr don't provide useful signal here, since they are non-optimizable except with freezing
        # Constant is not executed, bailouts should be a separate tests, don't provide useful signal here
        if kind == "prim::Constant" or "prim::Bailout" in kind or "GetAttr" in kind:
            continue
        for b in node.blocks():
            num_non_tensor += num_non_tensor_nodes(b)
        tensor_out = False
        for out in node.outputs():
            if "Tensor" in str(out.type()):
                tensor_out = True
                break
        num_non_tensor += int(not tensor_out)
    return num_non_tensor


class TestComplexity(JitTestCase):
    def setUp(self):
        super().setUp()
        self.grad_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(False)
        self._stack = contextlib.ExitStack()
        self._stack.enter_context(set_default_dtype(torch.double))

    def tearDown(self):
        self._stack.close()
        torch.set_grad_enabled(self.grad_enabled)
        super().tearDown()

    @suppress_warnings
    def test_generated_functional_tests(self):
        with enable_profiling_mode():
            stats = [("Name", "Ifs/Loops", "non-tensor ops")]
            for test in nn_functional_tests:
                test_name = test[0]

                fn, inputs = get_nn_functional_compiled_fn_and_inputs(*test)
                for _ in range(6):
                    fn(*inputs)

                g = torch.jit.last_executed_optimized_graph()
                stats.append((test_name, num_ifs_loops(g), num_non_tensor_nodes(g)))
        for line in stats:
            print(line)

    @suppress_warnings
    @unittest.skipIf(IS_FBCODE, "Causes a RecursionError in fbcode")
    def test_nn_module_tests(self):
        with enable_profiling_mode():
            stats = [("Name", "Ifs/Loops", "non-tensor ops")]
            for test in get_all_nn_module_tests():
                out = try_get_nn_module_compiled_mod_and_inputs(**test)
                if not out:
                    continue

                mod, inputs = out
                test_name = get_nn_mod_test_name(**test)
                for _ in range(6):
                    mod(*inputs)

                g = torch.jit.last_executed_optimized_graph()
                stats.append((test_name, num_ifs_loops(g), num_non_tensor_nodes(g)))

            for line in stats:
                print(line)


if __name__ == "__main__":
    run_tests()
