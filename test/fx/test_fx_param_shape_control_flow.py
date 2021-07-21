import unittest
import torch
import torch.fx


class MyModule(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.param = torch.nn.Parameter(torch.randn(in_channels, 3))

    def forward(self, x):
        if self.param.shape[0] < 10:
            return torch.mm(x, self.param)
        else:
            return torch.relu(torch.mm(x, self.param))


class TestConstParamShapeInControlFlow(unittest.TestCase):
    def test_param_shape_const(self):
        mymod = MyModule(in_channels=5)

        # Test that we go down the True branch in forward
        x = torch.randn(10, 5)
        torch.testing.assert_allclose(mymod(x), torch.mm(x, mymod.param))
        tracer = torch.fx.Tracer(param_shapes_constant=True)
        traced_graph = tracer.trace(mymod)

        # Make a new module with different parameter shape to go down the different
        # code path
        mymod2 = MyModule(in_channels=15)
        x = torch.randn(10, 15)
        torch.testing.assert_allclose(mymod2(x), torch.relu(torch.mm(x, mymod2.param)))

        tracer2 = torch.fx.Tracer(param_shapes_constant=True)
        traced_graph2 = tracer2.trace(mymod2)

        graph1_node_names = [n.name for n in traced_graph.nodes]
        graph2_node_names = [n.name for n in traced_graph2.nodes]

        # the second graph has an exta relu function call node
        assert 'mm' in graph1_node_names and 'mm' in graph2_node_names
        assert 'relu' not in graph1_node_names and 'relu' in graph2_node_names

if __name__ == '__main__':
    unittest.main()
