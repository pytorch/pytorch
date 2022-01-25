# Owner(s): ["oncall: jit"]

from test_graph_rewrite_passes import TestGraphRewritePasses, FunctionalLinear, FunctionalConv2d
import torch
import torch._C

class TestGraphRewritePassesVulkan(TestGraphRewritePasses):
    def test_vulkan_insert_pre_packed_ops(self) -> None:
        x_1 = torch.rand(3)
        w_1 = torch.rand(5, 3)
        b_1 = torch.rand(5)
        model_1 = torch.jit.trace(FunctionalLinear(w_1, b_1), [x_1])
        check_pattern_count_map_1 = {"aten::matmul": 0, "aten::add_": 0, "aten::t": 0}
        self.check_single_replacement(
            "aten::matmul",
            "vulkan_prepack::linear_run",
            torch._C._jit_pass_vulkan_insert_prepacked_ops,
            model_1
        )
        self.check_op_presence(check_pattern_count_map_1, torch._C._jit_pass_vulkan_insert_prepacked_ops, model_1)
        model_1(x_1)  # make sure it runs

        conv2d_in_channels = 3
        conv2d_out_channels = 4
        conv2d_kernel = 2
        conv2d_weight = torch.rand(
            conv2d_out_channels,
            conv2d_in_channels,
            conv2d_kernel,
            conv2d_kernel,
        )
        conv2d_bias = torch.rand(conv2d_out_channels)
        x_2_shape = (3, 2, 5)
        x_2 = torch.rand(x_2_shape)
        model_2 = torch.jit.trace(FunctionalConv2d(conv2d_weight, conv2d_bias), [x_2])
        check_pattern_count_map_2 = {
            "aten::_convolution": 0,
            "vulkan_prepack::conv2d_clamp_run": 1,
            "vulkan_prepack::conv2d_clamp_prepack": 1
        }
        self.check_single_replacement(
            "aten::_convolution",
            "prim::Constant",
            torch._C._jit_pass_vulkan_insert_prepacked_ops,
            model_2
        )
        self.check_op_presence(
            check_pattern_count_map_2,
            torch._C._jit_pass_vulkan_insert_prepacked_ops,
            model_2
        )
        model_2(x_2)  # make sure it runs

# TODO: elimintate this code and move TestGraphRewritePassesVulkan inside test_graph_rewrite_passes.py
# after CI build is implemented that supports USE_VULKAN=1 
if __name__ == '__main__':
    test_case = TestGraphRewritePassesVulkan()
    test_case.test_vulkan_insert_pre_packed_ops()
