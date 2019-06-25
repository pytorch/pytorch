from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import sys
import onnxruntime  # noqa


class TestONNXRuntime(unittest.TestCase):

    def test_onnxruntime_installed(self):
        self.assertTrue('onnxruntime' in sys.modules)


    def test_topk(self):
        class MyModule(Module):
            def forward(self, x):
                return torch.topk(x, 3)

        x = torch.arange(1., 6., requires_grad=True)
        model = MyModule()
        output = model(x)
        self.run_test(model, x, output)

    def test_maxpool(self):
        model = torch.nn.MaxPool1d(2, stride=1)
        x = torch.randn(20, 16, 50)
        output = model(x)
        self.run_test(model, x, output)

        # add test with dilations
        model = torch.nn.MaxPool1d(2, stride=1, dilation=2)
        x = torch.randn(20, 16, 50)
        output = model(x)
        self.run_test(model, x, output)

    def test_avgpool(self):
        model = torch.nn.AveragePool1d(2, stride=1)
        x = torch.randn(20, 16, 50)
        output = model(x)
        self.run_test(model, x, output)

        # add test with dilations
        model = torch.nn.AveragePool1d(2, stride=1, dilation=2)
        x = torch.randn(20, 16, 50)
        output = model(x)
        self.run_test(model, x, output)

    def test_slice_trace(self):
        class MyModule(Module):
            def forward(self, x):
                return x[0:1]

        x = torch.randn(3)
        model = MyModule()
        output = model(x)
        self.run_test(model, x, output)

    def test_slice_script(self):
        class DynamicSliceModel(torch.jit.ScriptModule):
            @torch.jit.script_method
            def forward(self, x):
                return x[1:x.size(0)] 

        model = DynamicSliceModel()
        x = torch.rand(1, 2)
        example_output = model(x)
        output = model(x)
        self.run_test(model, x, output)

    def test_flip(self):
        class MyModule(Module):
            def forward(self, x):
                return torch.flip(x, dims=[0])

        x = torch.tensor(numpy.arange(6.0).reshape(2, 3))
        model = MyModule()
        output = model(x)
        self.run_test(model, x, output)

    def test_interpolate(self):
        class MyModel(torch.nn.Module):
            def forward(self, x):
                size = [v * 2 for v in x.size()[2:]]
                return torch.nn.functional.interpolate(x,
                                                       size=size,
                                                       mode='nearest')

        x = torch.randn(1, 2, 3, 4, requires_grad=True)
        model = MyModule()
        output = model(x)
        self.run_test(model, x, output)

        class MyDynamicModel(torch.nn.Module):
            def forward(self, x):
                size = [v * 2 for v in x.size()[2:]]
                # work around for now: turn the dynamic sizes into constant
                size = [int(i) for i in size]
                return torch.nn.functional.interpolate(x,
                                                       size=size,
                                                       mode='nearest')

        x = torch.randn(20, 16, 50)
        model = MyModule()
        output = model(x)
        self.run_test(model, x, output)



if __name__ == '__main__':
    unittest.main()
