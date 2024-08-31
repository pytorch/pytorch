# Owner(s): ["module: fx"]

import unittest

import torch
import torch.fx
from torch.testing._internal.common_utils import TestCase


class MyModuleBase(torch.nn.Module):
    def forward(self, x):
        matrx = self.get_mul_matrix()
        if self.no_relu():
            return torch.mm(x, matrx)
        else:
            return torch.relu(torch.mm(x, matrx))

    def get_mul_matrix(self):
        return self.param

    def no_relu(self):
        raise Exception("not implemented")  # noqa: TRY002


class MyModuleParamShape(MyModuleBase):
    def __init__(self, in_channels):
        super().__init__()
        self.param = torch.nn.Parameter(torch.randn(in_channels, 3))

    def no_relu(self):
        return self.param.shape[0] < 10


class MyModuleParamSize(MyModuleBase):
    def __init__(self, in_channels):
        super().__init__()
        self.param = torch.nn.Parameter(torch.randn(in_channels, 3))

    def no_relu(self):
        return self.param.size()[0] < 10


class MyModuleParamDim(MyModuleBase):
    def __init__(self, param):
        super().__init__()
        self.param = param

    def get_mul_matrix(self):
        return self.param[0] if (self.param.dim() == 3) else self.param

    def no_relu(self):
        return self.param.dim() == 3


class MyModuleParamNDim(MyModuleBase):
    def __init__(self, param):
        super().__init__()
        self.param = param

    def get_mul_matrix(self):
        return self.param[0] if (self.param.ndim == 3) else self.param

    def no_relu(self):
        return self.param.ndim == 3


class MyModuleParamNumEl(MyModuleBase):
    def __init__(self, in_channels):
        super().__init__()
        self.param = torch.nn.Parameter(torch.randn(in_channels, 3))

    def no_relu(self):
        return self.param.numel() < 10 * 3


class MyModuleParamNElement(MyModuleBase):
    def __init__(self, in_channels):
        super().__init__()
        self.param = torch.nn.Parameter(torch.randn(in_channels, 3))

    def no_relu(self):
        return self.param.nelement() < 10 * 3


class TestConstParamShapeInControlFlow(TestCase):
    def verify_mm_relu_mods(self, mm_only_mod, relu_mod):
        """
        Verify one module only does a mm op while the other
        performs both mm and relu ops in cascade
        """
        x = torch.randn(10, 5)
        torch.testing.assert_close(
            mm_only_mod(x), torch.mm(x, mm_only_mod.get_mul_matrix())
        )
        tracer = torch.fx.Tracer(param_shapes_constant=True)
        traced_graph = tracer.trace(mm_only_mod)

        # verify the graph module calculates the same result
        graph_mod_mm = torch.fx.GraphModule(mm_only_mod, traced_graph)
        torch.testing.assert_close(
            graph_mod_mm(x), torch.mm(x, mm_only_mod.get_mul_matrix())
        )

        # Make a new module with different parameter shape to go down the different
        # code path
        x = torch.randn(10, 15)
        torch.testing.assert_close(
            relu_mod(x), torch.relu(torch.mm(x, relu_mod.get_mul_matrix()))
        )

        tracer2 = torch.fx.Tracer(param_shapes_constant=True)
        traced_graph2 = tracer2.trace(relu_mod)

        # verify the graph module calculates the same result
        graph_mod_relu = torch.fx.GraphModule(relu_mod, traced_graph2)
        torch.testing.assert_close(
            graph_mod_relu(x), torch.relu(torch.mm(x, relu_mod.get_mul_matrix()))
        )

        graph1_node_targets = [n.target for n in traced_graph.nodes]
        graph2_node_targets = [n.target for n in traced_graph2.nodes]

        # the second graph has an exta relu function call node
        assert torch.mm in graph1_node_targets and torch.mm in graph2_node_targets
        assert (
            torch.relu not in graph1_node_targets and torch.relu in graph2_node_targets
        )

    def test_param_shape_const(self):
        mymod = MyModuleParamShape(in_channels=5)
        mymod2 = MyModuleParamShape(in_channels=15)
        self.verify_mm_relu_mods(mymod, mymod2)

    def test_param_size_const(self):
        mymod = MyModuleParamSize(in_channels=5)
        mymod2 = MyModuleParamSize(in_channels=15)
        self.verify_mm_relu_mods(mymod, mymod2)

    def test_param_dim_const(self):
        mymod = MyModuleParamDim(torch.nn.Parameter(torch.randn(2, 5, 3)))
        mymod2 = MyModuleParamDim(torch.nn.Parameter(torch.randn(15, 3)))
        self.verify_mm_relu_mods(mymod, mymod2)

    def test_param_ndim_const(self):
        mymod = MyModuleParamNDim(torch.nn.Parameter(torch.randn(2, 5, 3)))
        mymod2 = MyModuleParamNDim(torch.nn.Parameter(torch.randn(15, 3)))
        self.verify_mm_relu_mods(mymod, mymod2)

    def test_param_numel_const(self):
        mymod = MyModuleParamNumEl(in_channels=5)
        mymod2 = MyModuleParamNumEl(in_channels=15)
        self.verify_mm_relu_mods(mymod, mymod2)

    def test_param_nelement_const(self):
        mymod = MyModuleParamNElement(in_channels=5)
        mymod2 = MyModuleParamNElement(in_channels=15)
        self.verify_mm_relu_mods(mymod, mymod2)


if __name__ == "__main__":
    unittest.main()
