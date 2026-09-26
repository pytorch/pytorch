# Owner(s): ["module: fx"]

import copy
from collections import defaultdict

import torch
import torch.fx as fx
from torch._dynamo.source import LocalSource
from torch.fx.experimental.shape_inference.infer_shape import infer_shape, mksym
from torch.fx.experimental.shape_inference.infer_symbol_values import (
    infer_symbol_values,
)
from torch.fx.experimental.symbolic_shapes import DimDynamic, ShapeEnv
from torch.testing._internal.common_utils import HardwareClassification, TestCase


class TestShapeInference(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_infer_symbol_values(self):
        shape_env = ShapeEnv()
        N = 8
        sample = {f"s{i}": 2 for i in range(N)}
        init_symints = [
            mksym(shape_env, v, LocalSource(k), DimDynamic.DYNAMIC)
            for k, v in sample.items()
        ]
        symints = copy.deepcopy(init_symints)
        name = [str(s) for s in init_symints]  # non-sequential names like s48, s49, ...
        symbol_to_idx_dict = {str(init_symints[i]): i for i in range(N)}
        padding_constraints = defaultdict(list)

        # prepare constraints strings
        constraints = []
        constraints.append(
            f"The size of tensor a ({name[1]}) must match the size of tensor b (1773) at non-singleton dimension 1)"
        )
        constraints.append(
            f"Expected size for first two dimensions of batch2 tensor to be: [{name[0]}, ({name[2]}//2) + 12] but got: [{name[0]}, 120]."
        )
        constraints.append(
            f"shape '[{name[0]}, -1, 32]' is invalid for input of size {name[0]}*{name[3]}"
        )
        constraints.append(
            f"a and b must have same reduction dim, but got [32*{name[0]}, {name[3]}] X [20, 15]."
        )
        constraints.append(
            f"a and b must have same reduction dim, but got [{name[0]}, {name[4]} + 1568] X [5728, 1024]."
        )
        constraints.append(
            f"Expected size for first two dimensions of batch2 tensor to be: [{name[0]}, 40] but got: [{name[0]}, {name[5]}]."
        )
        constraints.append(
            f"shape '[{name[0]}, -1, 32]' is invalid for input of size {name[0]}*{name[6]} + 1344*{name[0]}"
        )
        constraints.append(
            f"shape '[-1, 47]' is invalid for input of size 32*{name[0]}*{name[6]} + 1344*{name[0]}"
        )
        constraints.append(
            f"Expected size for first two dimensions of batch2 tensor to be: [{name[0]}, 47*{name[6]}] but got: [{name[0]}*{name[6]}, 47]."
        )
        constraints.append(
            f"Split sizes add up to 4258 but got the tensor's size of {name[7]}"
        )

        for constraint in constraints:
            infer_symbol_values(
                symints,
                init_symints,
                symbol_to_idx_dict,
                padding_constraints,
                constraint,
            )

        self.assertEqual(symints[1], 1773)
        self.assertEqual(symints[2], 216)
        self.assertEqual(symints[3], 640)
        self.assertEqual(symints[4], 4160)
        self.assertEqual(symints[5], 40)
        self.assertEqual(symints[6], 160)
        self.assertEqual(symints[7], 4258)

    def test_infer_shape(self):
        class TestModule(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.w_1 = torch.empty([256, 328])
                self.b_1 = torch.empty([256])
                self.w_2 = torch.empty([328, 256])
                self.b_2 = torch.empty([328])

            def forward(self, x):
                l_1 = torch.nn.functional.linear(x, self.w_1, bias=self.b_1)
                s_1 = torch.sigmoid(l_1)
                l_2 = torch.nn.functional.linear(s_1, self.w_2, bias=self.b_2)
                t_1 = torch.tanh(l_2)
                return t_1

        def generate_graph_module(model):
            gm = fx.symbolic_trace(model)
            return gm

        m = TestModule()
        gm = generate_graph_module(m)
        input_tensors = [torch.randn(1, 1)]
        infer_shape(gm, input_tensors)


if __name__ == "__main__":
    raise RuntimeError(
        "This test is not currently used and should be "
        "enabled in discover_tests.py if required."
    )
