import os
import sys

import torch

# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.jit_utils import JitTestCase, enable_profiling_mode
from torch.testing._internal.common_utils import run_tests
import torch.testing._internal.jit_utils as jit_utils

def num_ifs_loops(graph):
    graph_str = str(graph)
    # only look at body of graph
    graph_body = graph_str[0:graph_str.find("return")]
    return graph_body.count("prim::Loop") + graph_body.count("prim::If")

def num_non_tensor_nodes(block):
    num_non_tensor = 0
    for node in block.nodes():
        kind = node.kind()
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

    def test_generated_functional_tests(self):
        with enable_profiling_mode():
            stats = [("Name", "Ifs/Loops", "non-tensor ops")]
            for test in jit_utils.nn_functional_tests:
                test_name = test[0]
                if test_name in jit_utils.EXCLUDE_SCRIPT or test_name != "avg_pool2d":
                    continue

                print(test)
                fn, inputs = jit_utils.get_nn_functional_compiled_fn_and_inputs(*test)
                for _ in range(6):
                    fn(*inputs)

                g = torch.jit.last_executed_optimized_graph()
                stats.append((test_name, num_ifs_loops(g), num_non_tensor_nodes(g)))
        print(stats)


if __name__ == '__main__':
    run_tests()
