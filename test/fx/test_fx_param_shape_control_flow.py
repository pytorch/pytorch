import unittest

import torch
import torch.fx
import difflib


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
        tracer = torch.fx.Tracer(param_shapes_constant=True)
        traced_graph = tracer.trace(mymod)
        traced_mod = torch.fx.GraphModule(tracer.root, traced_graph)
        torch.testing.assert_allclose(mymod(x), torch.mm(x, mymod.param))
        # save the transformed Python code for comparison below
        code = traced_mod.code.split('\n')

        # Make a new module with different parameter shape to go down the different
        # code path
        mymod2 = MyModule(in_channels=15)
        x = torch.randn(10, 15)
        torch.testing.assert_allclose(mymod2(x), torch.relu(torch.mm(x, mymod2.param)))

        tracer = torch.fx.Tracer(param_shapes_constant=True)
        traced_graph = tracer.trace(mymod2)

        traced_mod = torch.fx.GraphModule(tracer.root, traced_graph)
        code2 = traced_mod.code.split('\n')

        # white box test (not just checking returned tensor values)
        # the diff between two versions of the transformed Python code should be like this:
        #   -    return mm
        #  +   relu = torch.relu(mm);  mm = None
        #  +   return relu
        for _, s in enumerate(difflib.unified_diff(code, code2)):
            if s[0] == '-' and s[1] != '-':
                self.assertEqual(s, '-    return mm')
                seen_delete = True
            elif s[0] == '+' and s[1] != '+':
                if seen_delete:
                    self.assertTrue(s.startswith('+    relu = torch.relu(mm);  mm = None'))
                    seen_delete = False
                else:
                    self.assertEqual(s, '+    return relu')


if __name__ == '__main__':
    unittest.main()
