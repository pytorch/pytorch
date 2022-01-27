# Owner(s): ["oncall: mobile"]

import unittest
import torch
from torch.nn import functional as F

from torch.testing._internal.common_utils import TestCase, run_tests
from torch.testing import FileCheck
import io
from typing import Dict, List, Tuple

class ExampleConv2D(torch.nn.Module):
    def __init__(self):
        # convolutiopn parameters for unit tests
        self.batch_size = 2
        self.input_channels_per_group = 6
        self.height = 16
        self.width = 16
        self.output_channels_per_group = 6
        self.groups = 4
        self.kernel_h = 3
        self.kernel_w = self.kernel_h
        self.stride_h = 1
        self.stride_w = self.stride_h
        self.pad_h = 1
        self.pad_w = self.pad_h
        self.dilation = 1
        self.input_channels = self.input_channels_per_group * self.groups
        self.output_channels = self.output_channels_per_group * self.groups
        self.kernels = (self.kernel_h, self.kernel_w)
        self.strides = (self.stride_h, self.stride_w)
        self.paddings = (self.pad_h, self.pad_w)
        self.dilations = (self.dilation, self.dilation)
        self.conv_weight_shape = (self.output_channels, self.input_channels_per_group, self.kernel_h, self.kernel_w)
        self.conv_bias_shape = (self.output_channels)
        super(ExampleConv2D, self).__init__()
        self.weight = torch.nn.Parameter(torch.rand(self.conv_weight_shape), requires_grad=False)
        self.bias = torch.nn.Parameter(torch.rand(self.conv_bias_shape), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.weight, self.bias, self.strides, self.paddings, self.dilations, self.groups)

class ExampleConv2DRelu(ExampleConv2D):
    def __init__(self):
        super(ExampleConv2DRelu, self).__init__()
        self.weight = torch.nn.Parameter(torch.rand(self.conv_weight_shape), requires_grad=False)
        self.bias = torch.nn.Parameter(torch.rand(self.conv_bias_shape), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o = F.conv2d(x, self.weight, self.bias, self.strides, self.paddings, self.dilations, self.groups)
        o = F.relu(o)
        return o

class ExampleConv2DHardtanh(ExampleConv2D):
    def __init__(self):
        super(ExampleConv2DHardtanh, self).__init__()
        self.weight = torch.nn.Parameter(torch.rand(self.conv_weight_shape), requires_grad=False)
        self.bias = torch.nn.Parameter(torch.rand(self.conv_bias_shape), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o = F.conv2d(x, self.weight, self.bias, self.strides, self.paddings, self.dilations, self.groups)
        o = F.hardtanh(o)
        return o

@unittest.skipUnless(torch.is_vulkan_available(), "Vulkan backend must be available for these tests.")
class TestVulkanRewritePass(TestCase):
    def get_source_range(self, scripted_model, kind) -> str:
        for node in scripted_model.graph.nodes():
            if node.kind() == kind:
                source_range = node.sourceRange()
        return source_range

    def validate_transformed_module(
            self,
            model: torch.nn.Module,
            pattern_count_map: Dict[str, int],
            data_shape: Tuple[int, int, int, int],
            old_node_kinds: List[str] = None,
            new_node_kinds: List[str] = None,
            prepack_removal: bool = False,
            fuse_clamping_ops: bool = False) -> None:
        scripted_model = torch.jit.script(model)
        # store old source ranges
        if old_node_kinds or new_node_kinds:
            self.assertTrue(len(old_node_kinds) == len(new_node_kinds))
            old_source_ranges = len(old_node_kinds) * [None]
            new_source_ranges = len(new_node_kinds) * [None]
            for i in range(len(old_node_kinds)):
                old_source_ranges[i] = self.get_source_range(scripted_model, old_node_kinds[i])
        # evaluate non-rewritten model
        scripted_model.eval()
        input_data = torch.normal(1, 20, size=data_shape)
        ref_result = scripted_model(input_data)
        # rewrite model
        torch._C._jit_pass_vulkan_insert_prepacked_ops(scripted_model._c)
        if fuse_clamping_ops or prepack_removal:
            scripted_model._c = torch._C._freeze_module(scripted_model._c)
        if fuse_clamping_ops:
            torch._C._jit_pass_vulkan_fuse_clamp_w_prepacked_conv(scripted_model._c)
        if prepack_removal:
            torch._C._jit_pass_vulkan_fold_prepacking_ops(scripted_model._c)
        # validate source ranges
        if old_node_kinds or new_node_kinds:
            for i in range(len(old_node_kinds)):
                new_source_ranges[i] = self.get_source_range(scripted_model, new_node_kinds[i])
            for i in range(len(old_node_kinds)):
                self.assertTrue(old_source_ranges[i] == new_source_ranges[i])
        # validate rewrite
        buffer = io.BytesIO()
        torch.jit.save(scripted_model, buffer)
        buffer.seek(0)
        deserialized_scripted_model = torch.jit.load(buffer)
        for pattern, v in pattern_count_map.items():
            FileCheck().check_count(pattern, v, exactly=True).run(deserialized_scripted_model.graph)

        # TODO: evaluate rewritten model

    def test_conv2d_rewrite(self) -> None:
        model = ExampleConv2D()
        old_node_kinds = ["aten::conv2d"]
        new_node_kinds = ["vulkan_prepack::conv2d_clamp_run"]

        data_shape = (model.batch_size, model.input_channels, model.height, model.width)
        pattern_count_map = {"Tensor = aten::conv2d": 0,
                             "vulkan_prepack::conv2d_clamp_prepack": 1,
                             "vulkan_prepack::conv2d_clamp_run": 1}
        self.validate_transformed_module(model, pattern_count_map, data_shape, old_node_kinds, new_node_kinds)

    def test_conv2d_relu_rewrite(self) -> None:
        model = ExampleConv2DRelu()
        old_node_kinds = ["aten::conv2d"]
        new_node_kinds = ["vulkan_prepack::conv2d_clamp_run"]
        data_shape = (model.batch_size, model.input_channels, model.height, model.width)
        pattern_count_map = {"Tensor = aten::conv2d": 0,
                             "vulkan_prepack::conv2d_clamp_prepack": 1,
                             "vulkan_prepack::conv2d_clamp_run": 1}
        self.validate_transformed_module(model, pattern_count_map, data_shape, old_node_kinds, new_node_kinds)

        pattern_count_map["aten::relu"] = 1
        pattern_count_map["vulkan_prepack::conv2d_clamp_prepack"] = 0
        self.validate_transformed_module(model, pattern_count_map, data_shape, prepack_removal=True)

        pattern_count_map["aten::relu"] = 0
        self.validate_transformed_module(model, pattern_count_map, data_shape, prepack_removal=True, fuse_clamping_ops=True)

    def test_conv2d_hardtanh_rewrite(self) -> None:
        model = ExampleConv2DHardtanh()
        old_node_kinds = ["aten::conv2d"]
        new_node_kinds = ["vulkan_prepack::conv2d_clamp_run"]
        data_shape = (model.batch_size, model.input_channels, model.height, model.width)
        pattern_count_map = {"Tensor = aten::conv2d": 0,
                             "vulkan_prepack::conv2d_clamp_prepack": 1,
                             "vulkan_prepack::conv2d_clamp_run": 1}
        self.validate_transformed_module(model, pattern_count_map, data_shape, old_node_kinds, new_node_kinds)

        pattern_count_map["aten::hardtanh"] = 1
        pattern_count_map["vulkan_prepack::conv2d_clamp_prepack"] = 0
        self.validate_transformed_module(
            model,
            pattern_count_map,
            data_shape,
            prepack_removal=True)

        pattern_count_map["aten::hardtanh"] = 0
        self.validate_transformed_module(
            model,
            pattern_count_map,
            data_shape,
            prepack_removal=True,
            fuse_clamping_ops=True)

if __name__ == "__main__":
    run_tests()
