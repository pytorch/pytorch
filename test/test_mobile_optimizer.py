import unittest
import torch
import torch.utils.mobile_optimizer
from torch.nn import functional as F

FileCheck = torch._C.FileCheck

class TestOptimizer(unittest.TestCase):

    def test_optimize_for_mobile(self):
        batch_size = 2
        input_channels_per_group = 6
        height = 16
        width = 16
        output_channels_per_group = 6
        groups = 4
        kernel_h = kernel_w = 3
        stride_h = stride_w = 1
        pad_h = pad_w = 1
        dilation = 1
        input_channels = input_channels_per_group * groups
        output_channels = output_channels_per_group * groups
        kernels = (kernel_h, kernel_w)
        strides = (stride_h, stride_w)
        paddings = (pad_h, pad_w)
        dilations = (dilation, dilation)
        conv_weight_shape = (output_channels, input_channels_per_group, kernel_h, kernel_w)
        conv_bias_shape = (output_channels)

        input_data = torch.rand((batch_size, input_channels, height, width))
        conv_weight = torch.rand((output_channels, input_channels_per_group, kernel_h, kernel_w))
        conv_bias = torch.rand((output_channels))
        result = F.conv2d(input_data, conv_weight, conv_bias, strides, paddings, dilations, groups)
        weight_output_dim = 24
        linear_input_shape = result.shape[1]
        linear_weight_shape = (weight_output_dim, linear_input_shape)

        class MyTestModule(torch.nn.Module):
            def __init__(self, activation_fn=F.relu):
                super(MyTestModule, self).__init__()
                self.conv_weight = torch.nn.Parameter(torch.Tensor(torch.rand(conv_weight_shape)))
                self.conv_bias = torch.nn.Parameter(torch.Tensor(torch.rand((conv_bias_shape))))
                self.linear_weight = torch.nn.Parameter(torch.Tensor(torch.rand(linear_weight_shape)))
                self.linear_bias = torch.nn.Parameter(torch.Tensor(torch.rand((weight_output_dim))))
                self.strides = strides
                self.paddings = paddings
                self.dilations = dilations
                self.groups = groups
                self.activation_fn = activation_fn

            def forward(self, x):
                o = F.conv2d(x, self.conv_weight, self.conv_bias,
                             self.strides, self.paddings, self.dilations, self.groups)
                o = self.activation_fn(o)
                o = o.permute([0, 2, 3, 1])
                o = F.linear(o, self.linear_weight, self.linear_bias)
                return self.activation_fn(o)

        data_shape = (batch_size, input_channels, height, width)
        input_data = torch.normal(1, 20, size=data_shape)

        scripted_model = torch.jit.script(MyTestModule())
        scripted_model.eval()
        initial_result = scripted_model(input_data)

        optimized_scripted_model = torch.utils.mobile_optimizer.optimize_for_mobile(scripted_model)
        optimized_result = optimized_scripted_model(input_data)

        pattern_count_map = {"Tensor = aten::conv2d": -1,
                             "Tensor = prim::CallFunction": -1,
                             "prepacked::conv2d_clamp_prepack": -1,
                             "prepacked::conv2d_clamp_run": 1,
                             "prepacked::linear_clamp_prepack": -1,
                             "prepacked::linear_clamp_run": 1}
        for pattern, v in pattern_count_map.items():
            if (v == 0):
                FileCheck().check(pattern).run(optimized_scripted_model.graph)
            elif (v == -1):
                FileCheck().check_not(pattern).run(optimized_scripted_model.graph)
            else:
                FileCheck().check_count(pattern, v, exactly=True).run(optimized_scripted_model.graph)


        torch.testing.assert_allclose(initial_result, optimized_result, rtol=1e-2, atol=1e-3)

if __name__ == '__main__':
    unittest.main()
