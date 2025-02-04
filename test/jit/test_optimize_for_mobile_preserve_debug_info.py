# Owner(s): ["oncall: mobile"]

import torch
import torch._C
import torch.nn.functional as F
from torch.testing._internal.common_utils import skipIfNoXNNPACK
from torch.testing._internal.jit_utils import JitTestCase


class TestOptimizeForMobilePreserveDebugInfo(JitTestCase):
    def check_replacement(
        self,
        model,
        replacements,
        jit_pass,
    ):
        """
        model: Model which optimization is performed on
        replacements: Dict mapping from nodes' kinds in the optimized model
            to the kinds of nodes they replaced in the original model
        jit_pass: Function to perform optimization
        """

        original_kinds = set(replacements.values())
        original_source_ranges = {
            node.kind(): node.sourceRange()
            for node in model.graph.nodes()
            if node.kind() in original_kinds
        }

        jit_pass(model._c)

        for node in model.graph.nodes():
            if node.kind() in replacements:
                self.assertEqual(
                    node.sourceRange(),
                    original_source_ranges[replacements[node.kind()]],
                )

    @skipIfNoXNNPACK
    def test_replace_conv1d_with_conv2d(self):
        class TestConv1d(torch.nn.Module):
            def __init__(self, weight, bias):
                super().__init__()
                self.weight = weight
                self.bias = bias

            def forward(self, x):
                return F.conv1d(x, self.weight, self.bias)

        self.check_replacement(
            model=torch.jit.script(
                TestConv1d(
                    weight=torch.rand(3, 3, 3),
                    bias=torch.rand(3),
                ),
            ),
            replacements={
                "prim::ListUnpack": "aten::conv1d",
                "prim::ListConstruct": "aten::conv1d",
                "aten::unsqueeze": "aten::conv1d",
                "aten::conv2d": "aten::conv1d",
                "aten::squeeze": "aten::conv1d",
            },
            jit_pass=torch._C._jit_pass_transform_conv1d_to_conv2d,
        )

    @skipIfNoXNNPACK
    def test_insert_pre_packed_linear_before_inline_and_conv_2d_op(self):
        class TestPrepackedLinearBeforeInlineAndConv2dOp(torch.nn.Module):
            def __init__(
                self,
                linear_weight,
                linear_bias,
                conv2d_weight,
                conv2d_bias,
                conv_transpose2d_weight,
                conv_transpose2d_bias,
            ):
                super(
                    TestPrepackedLinearBeforeInlineAndConv2dOp,
                    self,
                ).__init__()
                self.linear_weight = linear_weight.float()
                self.linear_bias = linear_bias.float()
                self.conv2d_weight = conv2d_weight.float()
                self.conv2d_bias = conv2d_bias.float()
                self.conv_transpose2d_weight = conv_transpose2d_weight.float()
                self.conv_transpose2d_bias = conv_transpose2d_bias.float()

            def forward(self, x):
                linear_res = F.linear(
                    x.float(),
                    self.linear_weight,
                    self.linear_bias,
                )
                conv2d_res = F.conv2d(
                    input=linear_res.unsqueeze(dim=0).float(),
                    weight=self.conv2d_weight,
                    bias=self.conv2d_bias,
                )
                return F.conv_transpose2d(
                    input=conv2d_res,
                    weight=self.conv_transpose2d_weight,
                    bias=self.conv_transpose2d_bias,
                )

        in_channels = 6
        iW = 5
        out_channels = 6
        kH = 2
        kW = 3

        self.check_replacement(
            model=torch.jit.script(
                TestPrepackedLinearBeforeInlineAndConv2dOp(
                    linear_weight=torch.rand(iW, 3),
                    linear_bias=torch.rand(iW),
                    conv2d_weight=torch.rand(out_channels, in_channels, kH, kW),
                    conv2d_bias=torch.rand(out_channels),
                    conv_transpose2d_weight=torch.rand(
                        out_channels,
                        in_channels,
                        kH,
                        kW,
                    ),
                    conv_transpose2d_bias=torch.rand(out_channels),
                ),
            ),
            replacements={
                "prepacked::linear_clamp_prepack": "aten::linear",
                "prepacked::linear_clamp_run": "aten::linear",
                "prepacked::conv2d_clamp_prepack": "aten::conv2d",
                "prepacked::conv2d_clamp_run": "aten::conv2d",
                "prepacked::conv2d_transpose_clamp_prepack": "aten::conv_transpose2d",
                "prepacked::conv2d_transpose_clamp_run": "aten::conv_transpose2d",
            },
            jit_pass=torch._C._jit_pass_insert_prepacked_ops,
        )

    @skipIfNoXNNPACK
    def test_insert_pre_packed_linear_op(self):
        self.check_replacement(
            model=torch.jit.trace(torch.nn.Linear(5, 4), torch.rand(3, 2, 5)),
            replacements={
                "prepacked::linear_clamp_prepack": "aten::linear",
                "prepacked::linear_clamp_run": "aten::linear",
            },
            jit_pass=torch._C._jit_pass_insert_prepacked_ops,
        )

    def run_test_fuse_activation_with_pack_ops_linear_conv2d(
        self,
        linear_activation,
        linear_activation_kind,
        conv2d_activation,
        conv2d_activation_kind,
    ):
        class TestFuseActivationLinearConv2d(torch.nn.Module):
            def __init__(
                self,
                linear_weight,
                linear_bias,
                conv2d_weight,
                conv2d_bias,
            ):
                super().__init__()
                self.linear_weight = linear_weight
                self.linear_bias = linear_bias
                self.conv2d_weight = conv2d_weight
                self.conv2d_bias = conv2d_bias

            def forward(self, x):
                x = F.linear(
                    input=x,
                    weight=self.linear_weight,
                    bias=self.linear_bias,
                )
                x = linear_activation(x)
                x = F.conv2d(
                    input=x.unsqueeze(dim=0),
                    weight=self.conv2d_weight,
                    bias=self.conv2d_bias,
                )
                return conv2d_activation(x)

        linear_in_features = 5
        linear_out_features = 4
        conv2d_in_channels = 3
        conv2d_out_channels = 4
        conv2d_kernel = 2
        x_shape = (3, 2, 5)

        model = torch.jit.trace(
            TestFuseActivationLinearConv2d(
                linear_weight=torch.nn.Parameter(
                    data=torch.rand(
                        linear_out_features,
                        linear_in_features,
                    ),
                    requires_grad=False,
                ),
                linear_bias=torch.nn.Parameter(
                    data=torch.rand(linear_out_features),
                    requires_grad=False,
                ),
                conv2d_weight=torch.rand(
                    conv2d_out_channels,
                    conv2d_in_channels,
                    conv2d_kernel,
                    conv2d_kernel,
                ),
                conv2d_bias=torch.rand(conv2d_out_channels),
            ),
            torch.rand(x_shape),
        )

        torch._C._jit_pass_insert_prepacked_ops(model._c)

        self.check_replacement(
            model=model,
            replacements={
                "prepacked::linear_clamp_prepack": "prepacked::linear_clamp_prepack",
                "prepacked::linear_clamp_run": linear_activation_kind,
                "prepacked::conv2d_clamp_prepack": "prepacked::conv2d_clamp_prepack",
                "prepacked::conv2d_clamp_run": conv2d_activation_kind,
            },
            jit_pass=torch._C._jit_pass_fuse_clamp_w_prepacked_linear_conv,
        )

    @skipIfNoXNNPACK
    def test_fuse_activation_with_pack_ops_linear_conv2d_1(self):
        self.run_test_fuse_activation_with_pack_ops_linear_conv2d(
            linear_activation=F.hardtanh,
            linear_activation_kind="aten::hardtanh",
            conv2d_activation=F.hardtanh_,
            conv2d_activation_kind="aten::hardtanh_",
        )

    @skipIfNoXNNPACK
    def test_fuse_activation_with_pack_ops_linear_conv2d_2(self):
        self.run_test_fuse_activation_with_pack_ops_linear_conv2d(
            linear_activation=F.hardtanh_,
            linear_activation_kind="aten::hardtanh_",
            conv2d_activation=F.hardtanh,
            conv2d_activation_kind="aten::hardtanh",
        )

    @skipIfNoXNNPACK
    def test_fuse_activation_with_pack_ops_linear_conv2d_3(self):
        self.run_test_fuse_activation_with_pack_ops_linear_conv2d(
            linear_activation=F.relu,
            linear_activation_kind="aten::relu",
            conv2d_activation=F.relu_,
            conv2d_activation_kind="aten::relu_",
        )

    @skipIfNoXNNPACK
    def test_fuse_activation_with_pack_ops_linear_conv2d_4(self):
        self.run_test_fuse_activation_with_pack_ops_linear_conv2d(
            linear_activation=F.relu_,
            linear_activation_kind="aten::relu_",
            conv2d_activation=F.relu,
            conv2d_activation_kind="aten::relu",
        )
