# Owner(s): ["oncall: pt2"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import ExitStack

import torch
import torch.nn as nn
import torch.utils._pytree as pytree
from torch._decomp import decomposition_table
from torch._dynamo.testing import normalize_gm
from torch._functorch._aot_autograd.descriptors import (
    BufferAOTInput,
    ParamAOTInput,
    PlainAOTInput,
    PlainAOTOutput,
)
from torch._functorch._aot_autograd.fx_utils import (
    get_all_input_and_grad_nodes,
    get_all_output_and_tangent_nodes,
    get_buffer_nodes,
    get_named_buffer_nodes,
    get_named_param_nodes,
    get_param_and_grad_nodes,
    get_param_nodes,
    get_plain_input_and_grad_nodes,
    get_plain_output_and_tangent_nodes,
)
from torch._functorch.aot_autograd import (
    aot_compile_joint_with_descriptors,
    aot_export_joint_with_descriptors,
)
from torch.testing._internal.common_utils import run_tests, TestCase


class TestAOTJointWithDescriptors(TestCase):
    def test_simple_linear_module(self):
        """Test basic linear module with aot_export_joint_with_descriptors"""

        class SimpleLinear(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(3, 2)

            def forward(self, x):
                return self.linear(x)

        model = SimpleLinear()
        inputs = (torch.randn(4, 3),)

        with ExitStack() as stack:
            # Export joint with descriptors
            joint_with_descriptors = aot_export_joint_with_descriptors(
                stack, model, inputs, decompositions=decomposition_table
            )

            # Test the exported graph structure
            graph_code = joint_with_descriptors.graph_module.print_readable(
                print_output=False, expanded_def=True
            )

            self.assertExpectedInline(
                normalize_gm(graph_code),
                """\
class inner_f(torch.nn.Module):
    def forward(
        self,
        primals,
        tangents,
    ):
        primals_1: "f32[2, 3]"  # ParamAOTInput(target='linear.weight')
        primals_2: "f32[2]"  # ParamAOTInput(target='linear.bias')
        primals_3: "f32[4, 3]"  # PlainAOTInput(idx=0)
        tangents_1: "f32[4, 2]"  # TangentAOTInput(output=PlainAOTOutput(idx=0))
        primals_1, primals_2, primals_3, tangents_1, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
        transpose: "f32[3, 2]" = torch.ops.prims.transpose.default(primals_1, [1, 0]);  primals_1 = None
        mm: "f32[4, 2]" = torch.ops.aten.mm.default(primals_3, transpose);  transpose = None
        mul: "f32[4, 2]" = torch.ops.prims.mul.default(mm, 1.0);  mm = None
        mul_1: "f32[2]" = torch.ops.prims.mul.default(primals_2, 1.0);  primals_2 = None
        broadcast_in_dim: "f32[4, 2]" = torch.ops.prims.broadcast_in_dim.default(mul_1, [4, 2], [1]);  mul_1 = None
        add: "f32[4, 2]" = torch.ops.prims.add.default(mul, broadcast_in_dim);  mul = broadcast_in_dim = None
        transpose_1: "f32[2, 4]" = torch.ops.prims.transpose.default(tangents_1, [1, 0])
        mm_1: "f32[2, 3]" = torch.ops.aten.mm.default(transpose_1, primals_3);  transpose_1 = primals_3 = None
        transpose_2: "f32[3, 2]" = torch.ops.prims.transpose.default(mm_1, [1, 0]);  mm_1 = None
        sum_1: "f32[2]" = torch.ops.prims.sum.default(tangents_1, [0]);  tangents_1 = None
        broadcast_in_dim_1: "f32[1, 2]" = torch.ops.prims.broadcast_in_dim.default(sum_1, [1, 2], [1]);  sum_1 = None
        as_strided: "f32[2]" = torch.ops.aten.as_strided.default(broadcast_in_dim_1, [2], [1]);  broadcast_in_dim_1 = None
        transpose_3: "f32[2, 3]" = torch.ops.prims.transpose.default(transpose_2, [1, 0]);  transpose_2 = None
        return pytree.tree_unflatten([
            add,  # PlainAOTOutput(idx=0)
            transpose_3,  # GradAOTOutput(grad_of=ParamAOTInput(target='linear.weight'))
            as_strided,  # GradAOTOutput(grad_of=ParamAOTInput(target='linear.bias'))
            None,  # None
        ], self._out_spec)
""",
            )

            # Compile the result
            parallel_model_fn = aot_compile_joint_with_descriptors(
                joint_with_descriptors
            )

        # Test functional correctness
        expected_output = model(*inputs)
        actual_output = parallel_model_fn(
            *dict(model.named_parameters()).values(), *inputs
        )
        self.assertEqual(expected_output, actual_output)

    def test_conv_bn_module(self):
        """Test convolutional + batch norm module"""

        class ConvBN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(1, 3, 3, padding=1)
                self.bn = nn.BatchNorm2d(3)

            def forward(self, x):
                x = self.conv(x)
                x = self.bn(x)
                return torch.relu(x)

        model = ConvBN()
        model.train()  # Important for batch norm
        inputs = (torch.randn(2, 1, 4, 4),)

        with ExitStack() as stack:
            # Export joint with descriptors
            joint_with_descriptors = aot_export_joint_with_descriptors(
                stack, model, inputs, decompositions=decomposition_table
            )

            # Test the exported graph structure
            graph_code = joint_with_descriptors.graph_module.print_readable(
                print_output=False, expanded_def=True
            )

            # Test parameter and buffer specs
            self.assertIn("conv.weight", joint_with_descriptors.params_spec)
            self.assertIn("conv.bias", joint_with_descriptors.params_spec)
            self.assertIn("bn.weight", joint_with_descriptors.params_spec)
            self.assertIn("bn.bias", joint_with_descriptors.params_spec)

            self.assertIn("bn.running_mean", joint_with_descriptors.buffers_spec)
            self.assertIn("bn.running_var", joint_with_descriptors.buffers_spec)
            self.assertIn("bn.num_batches_tracked", joint_with_descriptors.buffers_spec)

            # Expect test on the printed graph
            self.assertExpectedInline(
                normalize_gm(graph_code),
                """\
class inner_f(torch.nn.Module):
    def forward(
        self,
        primals,
        tangents,
    ):
        primals_1: "f32[3, 1, 3, 3]"  # ParamAOTInput(target='conv.weight')
        primals_2: "f32[3]"  # ParamAOTInput(target='conv.bias')
        primals_3: "f32[3]"  # ParamAOTInput(target='bn.weight')
        primals_4: "f32[3]"  # ParamAOTInput(target='bn.bias')
        primals_5: "f32[3]"  # BufferAOTInput(target='bn.running_mean')
        primals_6: "f32[3]"  # BufferAOTInput(target='bn.running_var')
        primals_7: "i64[]"  # BufferAOTInput(target='bn.num_batches_tracked')
        primals_8: "f32[2, 1, 4, 4]"  # PlainAOTInput(idx=0)
        tangents_1: "f32[2, 3, 4, 4]"  # TangentAOTInput(output=PlainAOTOutput(idx=0))
        primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, tangents_1, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
        convolution: "f32[2, 3, 4, 4]" = torch.ops.aten.convolution.default(primals_8, primals_1, primals_2, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  primals_2 = None
        add: "i64[]" = torch.ops.prims.add.default(primals_7, 1);  primals_7 = None
        var: "f32[3]" = torch.ops.prims.var.default(convolution, [0, 2, 3], 0.0)
        broadcast_in_dim: "f32[1, 3, 1, 1]" = torch.ops.prims.broadcast_in_dim.default(var, [1, 3, 1, 1], [1]);  var = None
        sum_1: "f32[3]" = torch.ops.prims.sum.default(convolution, [0, 2, 3])
        broadcast_in_dim_1: "f32[1, 3, 1, 1]" = torch.ops.prims.broadcast_in_dim.default(sum_1, [1, 3, 1, 1], [1]);  sum_1 = None
        div: "f32[1, 3, 1, 1]" = torch.ops.prims.div.default(broadcast_in_dim_1, 32.0);  broadcast_in_dim_1 = None
        add_1: "f32[1, 3, 1, 1]" = torch.ops.prims.add.default(broadcast_in_dim, 1e-05)
        rsqrt: "f32[1, 3, 1, 1]" = torch.ops.prims.rsqrt.default(add_1);  add_1 = None
        broadcast_in_dim_2: "f32[2, 3, 4, 4]" = torch.ops.prims.broadcast_in_dim.default(div, [2, 3, 4, 4], [0, 1, 2, 3])
        sub: "f32[2, 3, 4, 4]" = torch.ops.prims.sub.default(convolution, broadcast_in_dim_2);  broadcast_in_dim_2 = None
        broadcast_in_dim_3: "f32[2, 3, 4, 4]" = torch.ops.prims.broadcast_in_dim.default(rsqrt, [2, 3, 4, 4], [0, 1, 2, 3])
        mul: "f32[2, 3, 4, 4]" = torch.ops.prims.mul.default(sub, broadcast_in_dim_3);  sub = broadcast_in_dim_3 = None
        squeeze: "f32[1, 3, 1]" = torch.ops.prims.squeeze.default(div, [3]);  div = None
        squeeze_1: "f32[1, 3]" = torch.ops.prims.squeeze.default(squeeze, [2]);  squeeze = None
        squeeze_2: "f32[3]" = torch.ops.prims.squeeze.default(squeeze_1, [0]);  squeeze_1 = None
        squeeze_3: "f32[1, 3, 1]" = torch.ops.prims.squeeze.default(rsqrt, [3]);  rsqrt = None
        squeeze_4: "f32[1, 3]" = torch.ops.prims.squeeze.default(squeeze_3, [2]);  squeeze_3 = None
        squeeze_5: "f32[3]" = torch.ops.prims.squeeze.default(squeeze_4, [0]);  squeeze_4 = None
        mul_1: "f32[3]" = torch.ops.prims.mul.default(squeeze_2, 0.1)
        mul_2: "f32[3]" = torch.ops.prims.mul.default(primals_5, 0.9);  primals_5 = None
        add_2: "f32[3]" = torch.ops.prims.add.default(mul_1, mul_2);  mul_1 = mul_2 = None
        squeeze_6: "f32[1, 3, 1]" = torch.ops.prims.squeeze.default(broadcast_in_dim, [3]);  broadcast_in_dim = None
        squeeze_7: "f32[1, 3]" = torch.ops.prims.squeeze.default(squeeze_6, [2]);  squeeze_6 = None
        squeeze_8: "f32[3]" = torch.ops.prims.squeeze.default(squeeze_7, [0]);  squeeze_7 = None
        mul_3: "f32[3]" = torch.ops.prims.mul.default(squeeze_8, 1.032258064516129);  squeeze_8 = None
        mul_4: "f32[3]" = torch.ops.prims.mul.default(mul_3, 0.1);  mul_3 = None
        mul_5: "f32[3]" = torch.ops.prims.mul.default(primals_6, 0.9);  primals_6 = None
        add_3: "f32[3]" = torch.ops.prims.add.default(mul_4, mul_5);  mul_4 = mul_5 = None
        broadcast_in_dim_4: "f32[3, 1]" = torch.ops.prims.broadcast_in_dim.default(primals_3, [3, 1], [0])
        broadcast_in_dim_5: "f32[3, 1, 1]" = torch.ops.prims.broadcast_in_dim.default(broadcast_in_dim_4, [3, 1, 1], [0, 1]);  broadcast_in_dim_4 = None
        broadcast_in_dim_6: "f32[2, 3, 4, 4]" = torch.ops.prims.broadcast_in_dim.default(broadcast_in_dim_5, [2, 3, 4, 4], [1, 2, 3]);  broadcast_in_dim_5 = None
        mul_6: "f32[2, 3, 4, 4]" = torch.ops.prims.mul.default(mul, broadcast_in_dim_6);  mul = broadcast_in_dim_6 = None
        broadcast_in_dim_7: "f32[3, 1]" = torch.ops.prims.broadcast_in_dim.default(primals_4, [3, 1], [0]);  primals_4 = None
        broadcast_in_dim_8: "f32[3, 1, 1]" = torch.ops.prims.broadcast_in_dim.default(broadcast_in_dim_7, [3, 1, 1], [0, 1]);  broadcast_in_dim_7 = None
        broadcast_in_dim_9: "f32[2, 3, 4, 4]" = torch.ops.prims.broadcast_in_dim.default(broadcast_in_dim_8, [2, 3, 4, 4], [1, 2, 3]);  broadcast_in_dim_8 = None
        add_4: "f32[2, 3, 4, 4]" = torch.ops.prims.add.default(mul_6, broadcast_in_dim_9);  mul_6 = broadcast_in_dim_9 = None
        le: "b8[2, 3, 4, 4]" = torch.ops.prims.le.default(add_4, 0.0)
        where: "f32[2, 3, 4, 4]" = torch.ops.prims.where.default(le, 0.0, add_4);  le = add_4 = None
        view_of: "f32[2, 3, 4, 4]" = torch.ops.prims.view_of.default(where)
        view_of_1: "f32[2, 3, 4, 4]" = torch.ops.prims.view_of.default(view_of);  view_of = None
        view_of_2: "f32[2, 3, 4, 4]" = torch.ops.prims.view_of.default(view_of_1);  view_of_1 = None
        view_of_3: "f32[2, 3, 4, 4]" = torch.ops.prims.view_of.default(view_of_2);  view_of_2 = None
        le_1: "b8[2, 3, 4, 4]" = torch.ops.prims.le.default(view_of_3, 0.0);  view_of_3 = None
        where_1: "f32[2, 3, 4, 4]" = torch.ops.prims.where.default(le_1, 0.0, tangents_1);  le_1 = tangents_1 = None
        broadcast_in_dim_10: "f32[1, 3]" = torch.ops.prims.broadcast_in_dim.default(squeeze_2, [1, 3], [1]);  squeeze_2 = None
        broadcast_in_dim_11: "f32[1, 3, 1]" = torch.ops.prims.broadcast_in_dim.default(broadcast_in_dim_10, [1, 3, 1], [0, 1]);  broadcast_in_dim_10 = None
        broadcast_in_dim_12: "f32[1, 3, 1, 1]" = torch.ops.prims.broadcast_in_dim.default(broadcast_in_dim_11, [1, 3, 1, 1], [0, 1, 2]);  broadcast_in_dim_11 = None
        sum_2: "f32[3]" = torch.ops.prims.sum.default(where_1, [0, 2, 3])
        broadcast_in_dim_13: "f32[2, 3, 4, 4]" = torch.ops.prims.broadcast_in_dim.default(broadcast_in_dim_12, [2, 3, 4, 4], [0, 1, 2, 3])
        sub_1: "f32[2, 3, 4, 4]" = torch.ops.prims.sub.default(convolution, broadcast_in_dim_13);  broadcast_in_dim_13 = None
        mul_7: "f32[2, 3, 4, 4]" = torch.ops.prims.mul.default(where_1, sub_1);  sub_1 = None
        sum_3: "f32[3]" = torch.ops.prims.sum.default(mul_7, [0, 2, 3]);  mul_7 = None
        mul_8: "f32[3]" = torch.ops.prims.mul.default(sum_2, 0.03125)
        broadcast_in_dim_14: "f32[1, 3]" = torch.ops.prims.broadcast_in_dim.default(mul_8, [1, 3], [1]);  mul_8 = None
        broadcast_in_dim_15: "f32[1, 3, 1]" = torch.ops.prims.broadcast_in_dim.default(broadcast_in_dim_14, [1, 3, 1], [0, 1]);  broadcast_in_dim_14 = None
        broadcast_in_dim_16: "f32[1, 3, 1, 1]" = torch.ops.prims.broadcast_in_dim.default(broadcast_in_dim_15, [1, 3, 1, 1], [0, 1, 2]);  broadcast_in_dim_15 = None
        mul_9: "f32[3]" = torch.ops.prims.mul.default(sum_3, 0.03125)
        mul_10: "f32[3]" = torch.ops.prims.mul.default(squeeze_5, squeeze_5)
        mul_11: "f32[3]" = torch.ops.prims.mul.default(mul_9, mul_10);  mul_9 = mul_10 = None
        broadcast_in_dim_17: "f32[1, 3]" = torch.ops.prims.broadcast_in_dim.default(mul_11, [1, 3], [1]);  mul_11 = None
        broadcast_in_dim_18: "f32[1, 3, 1]" = torch.ops.prims.broadcast_in_dim.default(broadcast_in_dim_17, [1, 3, 1], [0, 1]);  broadcast_in_dim_17 = None
        broadcast_in_dim_19: "f32[1, 3, 1, 1]" = torch.ops.prims.broadcast_in_dim.default(broadcast_in_dim_18, [1, 3, 1, 1], [0, 1, 2]);  broadcast_in_dim_18 = None
        mul_12: "f32[3]" = torch.ops.prims.mul.default(squeeze_5, primals_3);  primals_3 = None
        broadcast_in_dim_20: "f32[1, 3]" = torch.ops.prims.broadcast_in_dim.default(mul_12, [1, 3], [1]);  mul_12 = None
        broadcast_in_dim_21: "f32[1, 3, 1]" = torch.ops.prims.broadcast_in_dim.default(broadcast_in_dim_20, [1, 3, 1], [0, 1]);  broadcast_in_dim_20 = None
        broadcast_in_dim_22: "f32[1, 3, 1, 1]" = torch.ops.prims.broadcast_in_dim.default(broadcast_in_dim_21, [1, 3, 1, 1], [0, 1, 2]);  broadcast_in_dim_21 = None
        broadcast_in_dim_23: "f32[2, 3, 4, 4]" = torch.ops.prims.broadcast_in_dim.default(broadcast_in_dim_12, [2, 3, 4, 4], [0, 1, 2, 3]);  broadcast_in_dim_12 = None
        sub_2: "f32[2, 3, 4, 4]" = torch.ops.prims.sub.default(convolution, broadcast_in_dim_23);  convolution = broadcast_in_dim_23 = None
        broadcast_in_dim_24: "f32[2, 3, 4, 4]" = torch.ops.prims.broadcast_in_dim.default(broadcast_in_dim_19, [2, 3, 4, 4], [0, 1, 2, 3]);  broadcast_in_dim_19 = None
        mul_13: "f32[2, 3, 4, 4]" = torch.ops.prims.mul.default(sub_2, broadcast_in_dim_24);  sub_2 = broadcast_in_dim_24 = None
        sub_3: "f32[2, 3, 4, 4]" = torch.ops.prims.sub.default(where_1, mul_13);  where_1 = mul_13 = None
        broadcast_in_dim_25: "f32[2, 3, 4, 4]" = torch.ops.prims.broadcast_in_dim.default(broadcast_in_dim_16, [2, 3, 4, 4], [0, 1, 2, 3]);  broadcast_in_dim_16 = None
        sub_4: "f32[2, 3, 4, 4]" = torch.ops.prims.sub.default(sub_3, broadcast_in_dim_25);  sub_3 = broadcast_in_dim_25 = None
        broadcast_in_dim_26: "f32[2, 3, 4, 4]" = torch.ops.prims.broadcast_in_dim.default(broadcast_in_dim_22, [2, 3, 4, 4], [0, 1, 2, 3]);  broadcast_in_dim_22 = None
        mul_14: "f32[2, 3, 4, 4]" = torch.ops.prims.mul.default(sub_4, broadcast_in_dim_26);  sub_4 = broadcast_in_dim_26 = None
        mul_15: "f32[3]" = torch.ops.prims.mul.default(sum_3, squeeze_5);  sum_3 = squeeze_5 = None
        convolution_backward = torch.ops.aten.convolution_backward.default(mul_14, primals_8, primals_1, [3], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [False, True, True]);  mul_14 = primals_8 = primals_1 = None
        getitem_1: "f32[3, 1, 3, 3]" = convolution_backward[1]
        getitem_2: "f32[3]" = convolution_backward[2];  convolution_backward = None
        return pytree.tree_unflatten([
            add_2,  # InputMutationAOTOutput(mutated_input=BufferAOTInput(target='bn.running_mean'))
            add_3,  # InputMutationAOTOutput(mutated_input=BufferAOTInput(target='bn.running_var'))
            add,  # InputMutationAOTOutput(mutated_input=BufferAOTInput(target='bn.num_batches_tracked'))
            where,  # PlainAOTOutput(idx=0)
            getitem_1,  # GradAOTOutput(grad_of=ParamAOTInput(target='conv.weight'))
            getitem_2,  # GradAOTOutput(grad_of=ParamAOTInput(target='conv.bias'))
            mul_15,  # GradAOTOutput(grad_of=ParamAOTInput(target='bn.weight'))
            sum_2,  # GradAOTOutput(grad_of=ParamAOTInput(target='bn.bias'))
            None,  # None
            None,  # None
            None,  # None
            None,  # None
        ], self._out_spec)
""",  # noqa: B950
            )

            # Compile the result
            parallel_model_fn = aot_compile_joint_with_descriptors(
                joint_with_descriptors
            )

        # Test functional correctness
        expected_output = model(*inputs)
        all_params_buffers = (
            *dict(model.named_parameters()).values(),
            *dict(model.named_buffers()).values(),
        )
        actual_output = parallel_model_fn(*all_params_buffers, *inputs)
        torch.testing.assert_close(expected_output, actual_output, rtol=1e-4, atol=1e-4)

    def test_module_with_kwargs(self):
        """Test module with keyword arguments"""

        class ModuleWithKwargs(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(3, 2)

            def forward(self, x, scale=1.0):
                return self.linear(x) * scale

        model = ModuleWithKwargs()
        inputs = (torch.randn(4, 3),)
        kwargs = {"scale": 2.0}

        with ExitStack() as stack:
            # Export joint with descriptors
            joint_with_descriptors = aot_export_joint_with_descriptors(
                stack, model, inputs, kwargs, decompositions=decomposition_table
            )

            # Test the exported graph structure
            graph_code = joint_with_descriptors.graph_module.print_readable(
                print_output=False, expanded_def=True
            )

            # Expect test on the printed graph
            self.assertExpectedInline(
                normalize_gm(graph_code),
                """\
class inner_f(torch.nn.Module):
    def forward(
        self,
        primals,
        tangents,
    ):
        primals_1: "f32[2, 3]"  # ParamAOTInput(target='linear.weight')
        primals_2: "f32[2]"  # ParamAOTInput(target='linear.bias')
        primals_3: "f32[4, 3]"  # PlainAOTInput(idx=0)
        tangents_1: "f32[4, 2]"  # TangentAOTInput(output=PlainAOTOutput(idx=0))
        primals_1, primals_2, primals_3, primals_4  , tangents_1, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
        transpose: "f32[3, 2]" = torch.ops.prims.transpose.default(primals_1, [1, 0]);  primals_1 = None
        mm: "f32[4, 2]" = torch.ops.aten.mm.default(primals_3, transpose);  transpose = None
        mul: "f32[4, 2]" = torch.ops.prims.mul.default(mm, 1.0);  mm = None
        mul_1: "f32[2]" = torch.ops.prims.mul.default(primals_2, 1.0);  primals_2 = None
        broadcast_in_dim: "f32[4, 2]" = torch.ops.prims.broadcast_in_dim.default(mul_1, [4, 2], [1]);  mul_1 = None
        add: "f32[4, 2]" = torch.ops.prims.add.default(mul, broadcast_in_dim);  mul = broadcast_in_dim = None
        mul_2: "f32[4, 2]" = torch.ops.prims.mul.default(add, 2.0);  add = None
        mul_3: "f32[4, 2]" = torch.ops.prims.mul.default(tangents_1, 2.0);  tangents_1 = None
        transpose_1: "f32[2, 4]" = torch.ops.prims.transpose.default(mul_3, [1, 0])
        mm_1: "f32[2, 3]" = torch.ops.aten.mm.default(transpose_1, primals_3);  transpose_1 = primals_3 = None
        transpose_2: "f32[3, 2]" = torch.ops.prims.transpose.default(mm_1, [1, 0]);  mm_1 = None
        sum_1: "f32[2]" = torch.ops.prims.sum.default(mul_3, [0]);  mul_3 = None
        broadcast_in_dim_1: "f32[1, 2]" = torch.ops.prims.broadcast_in_dim.default(sum_1, [1, 2], [1]);  sum_1 = None
        as_strided: "f32[2]" = torch.ops.aten.as_strided.default(broadcast_in_dim_1, [2], [1]);  broadcast_in_dim_1 = None
        transpose_3: "f32[2, 3]" = torch.ops.prims.transpose.default(transpose_2, [1, 0]);  transpose_2 = None
        return pytree.tree_unflatten([
            mul_2,  # PlainAOTOutput(idx=0)
            transpose_3,  # GradAOTOutput(grad_of=ParamAOTInput(target='linear.weight'))
            as_strided,  # GradAOTOutput(grad_of=ParamAOTInput(target='linear.bias'))
            None,  # None
            None,  # None
        ], self._out_spec)
""",
            )

            # Compile the result
            parallel_model_fn = aot_compile_joint_with_descriptors(
                joint_with_descriptors
            )

        # Test functional correctness
        expected_output = model(*inputs, **kwargs)
        actual_output = parallel_model_fn(
            *dict(model.named_parameters()).values(), *inputs, **kwargs
        )
        self.assertEqual(expected_output, actual_output)

    def test_multiple_outputs_module(self):
        """Test module with multiple outputs"""

        class MultiOutputModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(3, 2)
                self.linear2 = nn.Linear(3, 4)

            def forward(self, x):
                out1 = self.linear1(x)
                out2 = self.linear2(x)
                return out1, out2

        model = MultiOutputModule()
        inputs = (torch.randn(4, 3),)

        with ExitStack() as stack:
            # Export joint with descriptors
            joint_with_descriptors = aot_export_joint_with_descriptors(
                stack, model, inputs, decompositions=decomposition_table
            )

            # Test the exported graph structure
            graph_code = joint_with_descriptors.graph_module.print_readable(
                print_output=False, expanded_def=True
            )

            # Expect test on the printed graph
            self.assertExpectedInline(
                normalize_gm(graph_code),
                """\
class inner_f(torch.nn.Module):
    def forward(
        self,
        primals,
        tangents,
    ):
        primals_1: "f32[2, 3]"  # ParamAOTInput(target='linear1.weight')
        primals_2: "f32[2]"  # ParamAOTInput(target='linear1.bias')
        primals_3: "f32[4, 3]"  # ParamAOTInput(target='linear2.weight')
        primals_4: "f32[4]"  # ParamAOTInput(target='linear2.bias')
        primals_5: "f32[4, 3]"  # PlainAOTInput(idx=0)
        tangents_1: "f32[4, 2]"  # TangentAOTInput(output=PlainAOTOutput(idx=0))
        tangents_2: "f32[4, 4]"  # TangentAOTInput(output=PlainAOTOutput(idx=1))
        primals_1, primals_2, primals_3, primals_4, primals_5, tangents_1, tangents_2, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
        transpose: "f32[3, 2]" = torch.ops.prims.transpose.default(primals_1, [1, 0]);  primals_1 = None
        mm: "f32[4, 2]" = torch.ops.aten.mm.default(primals_5, transpose);  transpose = None
        mul: "f32[4, 2]" = torch.ops.prims.mul.default(mm, 1.0);  mm = None
        mul_1: "f32[2]" = torch.ops.prims.mul.default(primals_2, 1.0);  primals_2 = None
        broadcast_in_dim: "f32[4, 2]" = torch.ops.prims.broadcast_in_dim.default(mul_1, [4, 2], [1]);  mul_1 = None
        add: "f32[4, 2]" = torch.ops.prims.add.default(mul, broadcast_in_dim);  mul = broadcast_in_dim = None
        transpose_1: "f32[3, 4]" = torch.ops.prims.transpose.default(primals_3, [1, 0]);  primals_3 = None
        mm_1: "f32[4, 4]" = torch.ops.aten.mm.default(primals_5, transpose_1);  transpose_1 = None
        mul_2: "f32[4, 4]" = torch.ops.prims.mul.default(mm_1, 1.0);  mm_1 = None
        mul_3: "f32[4]" = torch.ops.prims.mul.default(primals_4, 1.0);  primals_4 = None
        broadcast_in_dim_1: "f32[4, 4]" = torch.ops.prims.broadcast_in_dim.default(mul_3, [4, 4], [1]);  mul_3 = None
        add_1: "f32[4, 4]" = torch.ops.prims.add.default(mul_2, broadcast_in_dim_1);  mul_2 = broadcast_in_dim_1 = None
        transpose_2: "f32[4, 4]" = torch.ops.prims.transpose.default(tangents_2, [1, 0])
        mm_2: "f32[4, 3]" = torch.ops.aten.mm.default(transpose_2, primals_5);  transpose_2 = None
        transpose_3: "f32[3, 4]" = torch.ops.prims.transpose.default(mm_2, [1, 0]);  mm_2 = None
        sum_1: "f32[4]" = torch.ops.prims.sum.default(tangents_2, [0]);  tangents_2 = None
        broadcast_in_dim_2: "f32[1, 4]" = torch.ops.prims.broadcast_in_dim.default(sum_1, [1, 4], [1]);  sum_1 = None
        as_strided: "f32[4]" = torch.ops.aten.as_strided.default(broadcast_in_dim_2, [4], [1]);  broadcast_in_dim_2 = None
        transpose_4: "f32[4, 3]" = torch.ops.prims.transpose.default(transpose_3, [1, 0]);  transpose_3 = None
        transpose_5: "f32[2, 4]" = torch.ops.prims.transpose.default(tangents_1, [1, 0])
        mm_3: "f32[2, 3]" = torch.ops.aten.mm.default(transpose_5, primals_5);  transpose_5 = primals_5 = None
        transpose_6: "f32[3, 2]" = torch.ops.prims.transpose.default(mm_3, [1, 0]);  mm_3 = None
        sum_2: "f32[2]" = torch.ops.prims.sum.default(tangents_1, [0]);  tangents_1 = None
        broadcast_in_dim_3: "f32[1, 2]" = torch.ops.prims.broadcast_in_dim.default(sum_2, [1, 2], [1]);  sum_2 = None
        as_strided_1: "f32[2]" = torch.ops.aten.as_strided.default(broadcast_in_dim_3, [2], [1]);  broadcast_in_dim_3 = None
        transpose_7: "f32[2, 3]" = torch.ops.prims.transpose.default(transpose_6, [1, 0]);  transpose_6 = None
        return pytree.tree_unflatten([
            add,  # PlainAOTOutput(idx=0)
            add_1,  # PlainAOTOutput(idx=1)
            transpose_7,  # GradAOTOutput(grad_of=ParamAOTInput(target='linear1.weight'))
            as_strided_1,  # GradAOTOutput(grad_of=ParamAOTInput(target='linear1.bias'))
            transpose_4,  # GradAOTOutput(grad_of=ParamAOTInput(target='linear2.weight'))
            as_strided,  # GradAOTOutput(grad_of=ParamAOTInput(target='linear2.bias'))
            None,  # None
        ], self._out_spec)
""",  # noqa: B950
            )

            # Compile the result
            parallel_model_fn = aot_compile_joint_with_descriptors(
                joint_with_descriptors
            )

        # Test functional correctness
        expected_output = model(*inputs)
        actual_output = parallel_model_fn(
            *dict(model.named_parameters()).values(), *inputs
        )

        # Check both outputs
        self.assertEqual(len(expected_output), len(actual_output))
        for exp, act in zip(expected_output, actual_output):
            self.assertEqual(exp, act)

    def test_in_out_specs(self):
        """Test that in_spec and out_spec are properly set"""

        class SimpleModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(3, 2)

            def forward(self, x):
                return self.linear(x)

        model = SimpleModule()
        inputs = (torch.randn(4, 3),)

        with ExitStack() as stack:
            # Export joint with descriptors
            joint_with_descriptors = aot_export_joint_with_descriptors(
                stack, model, inputs, decompositions=decomposition_table
            )

            # Test that specs are available
            self.assertIsNotNone(joint_with_descriptors.in_spec)
            self.assertIsNotNone(joint_with_descriptors.out_spec)
            self.assertIsNotNone(joint_with_descriptors.params_spec)
            self.assertIsNotNone(joint_with_descriptors.buffers_spec)

            # Test that they work with pytree operations
            flat_inputs, _ = pytree.tree_flatten((inputs, {}))
            self.assertTrue(len(flat_inputs) > 0)

            # Test parameter and buffer specs contain expected entries
            self.assertIn("linear.weight", joint_with_descriptors.params_spec)
            self.assertIn("linear.bias", joint_with_descriptors.params_spec)
            self.assertEqual(
                len(joint_with_descriptors.buffers_spec), 0
            )  # No buffers in simple linear

            # Compile the result to ensure everything works together
            parallel_model_fn = aot_compile_joint_with_descriptors(
                joint_with_descriptors
            )

        # Test functional correctness
        expected_output = model(*inputs)
        actual_output = parallel_model_fn(
            *dict(model.named_parameters()).values(), *inputs
        )
        self.assertEqual(expected_output, actual_output)

    def test_fx_utils_simple_linear(self):
        """Test FX utilities on a simple linear module"""

        class SimpleLinear(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(3, 2)

            def forward(self, x):
                return self.linear(x)

        model = SimpleLinear()
        inputs = (torch.randn(4, 3),)

        with ExitStack() as stack:
            joint_with_descriptors = aot_export_joint_with_descriptors(
                stack, model, inputs, decompositions=decomposition_table
            )

            graph = joint_with_descriptors.graph_module.graph

            # Test get_named_param_nodes
            named_params = get_named_param_nodes(graph)
            self.assertIn("linear.weight", named_params)
            self.assertIn("linear.bias", named_params)
            self.assertEqual(len(named_params), 2)

            # Test get_param_nodes
            param_nodes = get_param_nodes(graph)
            self.assertEqual(len(param_nodes), 2)

            # Test get_named_buffer_nodes (should be empty for simple linear)
            named_buffers = get_named_buffer_nodes(graph)
            self.assertEqual(len(named_buffers), 0)

            # Test get_buffer_nodes
            buffer_nodes = get_buffer_nodes(graph)
            self.assertEqual(len(buffer_nodes), 0)

            # Test get_all_input_and_grad_nodes
            input_grad_nodes = get_all_input_and_grad_nodes(graph)
            self.assertEqual(len(input_grad_nodes), 4)  # 2 params + 1 input + 1 tangent

            # Verify that parameters have gradients
            param_grads = get_param_and_grad_nodes(graph)
            self.assertEqual(len(param_grads), 2)
            for desc, (param_node, grad_node) in param_grads.items():
                self.assertIsInstance(desc, ParamAOTInput)
                self.assertIsNotNone(param_node)
                self.assertIsNotNone(grad_node)  # Should have gradients

            # Test get_plain_input_and_grad_nodes
            plain_input_grads = get_plain_input_and_grad_nodes(graph)
            self.assertEqual(len(plain_input_grads), 1)  # 1 plain input
            for desc, (input_node, grad_node) in plain_input_grads.items():
                self.assertIsInstance(desc, PlainAOTInput)
                self.assertIsNotNone(input_node)
                self.assertIsNone(grad_node)  # Plain inputs don't have gradients

            # Test get_all_output_and_tangent_nodes
            output_tangent_nodes = get_all_output_and_tangent_nodes(graph)
            self.assertEqual(len(output_tangent_nodes), 3)  # 1 output + 2 grad outputs

            # Test get_plain_output_and_tangent_nodes
            plain_output_tangents = get_plain_output_and_tangent_nodes(graph)
            self.assertEqual(len(plain_output_tangents), 1)
            for desc, (output_node, tangent_node) in plain_output_tangents.items():
                self.assertIsInstance(desc, PlainAOTOutput)
                self.assertIsNotNone(output_node)
                self.assertIsNotNone(
                    tangent_node
                )  # Should have tangents for backward pass

    def test_fx_utils_conv_bn_module(self):
        """Test FX utilities on a conv+batchnorm module with buffers"""

        class ConvBN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(1, 3, 3, padding=1)
                self.bn = nn.BatchNorm2d(3)

            def forward(self, x):
                x = self.conv(x)
                x = self.bn(x)
                return torch.relu(x)

        model = ConvBN()
        model.train()  # Important for batch norm
        inputs = (torch.randn(2, 1, 4, 4),)

        with ExitStack() as stack:
            joint_with_descriptors = aot_export_joint_with_descriptors(
                stack, model, inputs, decompositions=decomposition_table
            )

            graph = joint_with_descriptors.graph_module.graph

            # Test get_named_param_nodes
            named_params = get_named_param_nodes(graph)
            expected_params = ["conv.weight", "conv.bias", "bn.weight", "bn.bias"]
            for param_name in expected_params:
                self.assertIn(param_name, named_params)
            self.assertEqual(len(named_params), 4)

            # Test get_named_buffer_nodes
            named_buffers = get_named_buffer_nodes(graph)
            expected_buffers = [
                "bn.running_mean",
                "bn.running_var",
                "bn.num_batches_tracked",
            ]
            for buffer_name in expected_buffers:
                self.assertIn(buffer_name, named_buffers)
            self.assertEqual(len(named_buffers), 3)

            # Test get_buffer_nodes
            buffer_nodes = get_buffer_nodes(graph)
            self.assertEqual(len(buffer_nodes), 3)

            # Test that all inputs include params, buffers, and plain inputs
            input_grad_nodes = get_all_input_and_grad_nodes(graph)
            self.assertEqual(
                len(input_grad_nodes), 9
            )  # 4 params + 3 buffers + 1 input + 1 tangent

            # Verify buffer handling
            buffer_count = 0
            for desc, (node, grad_node) in input_grad_nodes.items():
                if isinstance(desc, BufferAOTInput):
                    buffer_count += 1
                    self.assertIsNotNone(node)
                    # Buffers typically don't have gradients unless they're trainable

            self.assertEqual(buffer_count, 3)

    def test_fx_utils_multiple_outputs(self):
        """Test FX utilities on a module with multiple outputs"""

        class MultiOutputModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(3, 2)
                self.linear2 = nn.Linear(3, 4)

            def forward(self, x):
                out1 = self.linear1(x)
                out2 = self.linear2(x)
                return out1, out2

        model = MultiOutputModule()
        inputs = (torch.randn(4, 3),)

        with ExitStack() as stack:
            joint_with_descriptors = aot_export_joint_with_descriptors(
                stack, model, inputs, decompositions=decomposition_table
            )

            graph = joint_with_descriptors.graph_module.graph

            # Test get_all_output_and_tangent_nodes
            output_tangent_nodes = get_all_output_and_tangent_nodes(graph)
            self.assertEqual(len(output_tangent_nodes), 6)  # 2 outputs + 4 grad outputs

            # Test get_plain_output_and_tangent_nodes
            plain_output_tangents = get_plain_output_and_tangent_nodes(graph)
            self.assertEqual(len(plain_output_tangents), 2)

            # Verify each output has a tangent
            for desc, (output_node, tangent_node) in plain_output_tangents.items():
                self.assertIsInstance(desc, PlainAOTOutput)
                self.assertIsNotNone(output_node)
                self.assertIsNotNone(tangent_node)

            # Test parameter handling with multiple outputs
            param_grads = get_param_and_grad_nodes(graph)
            self.assertEqual(len(param_grads), 4)  # 2 weights + 2 biases

            # All parameters should have gradients
            for desc, (param_node, grad_node) in param_grads.items():
                self.assertIsInstance(desc, ParamAOTInput)
                self.assertIsNotNone(param_node)
                self.assertIsNotNone(grad_node)

    def test_fx_utils_node_consistency(self):
        """Test that FX utilities return consistent node references"""

        class SimpleModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(3, 2)

            def forward(self, x):
                return self.linear(x)

        model = SimpleModule()
        inputs = (torch.randn(4, 3),)

        with ExitStack() as stack:
            joint_with_descriptors = aot_export_joint_with_descriptors(
                stack, model, inputs, decompositions=decomposition_table
            )

            graph = joint_with_descriptors.graph_module.graph

            # Get nodes through different APIs and verify consistency
            named_params = get_named_param_nodes(graph)
            param_nodes = get_param_nodes(graph)
            param_grads = get_param_and_grad_nodes(graph)
            all_input_grads = get_all_input_and_grad_nodes(graph)

            # Check that get_param_nodes returns the same nodes as get_named_param_nodes
            self.assertEqual(len(param_nodes), len(named_params))
            for node in param_nodes:
                self.assertIn(node, named_params.values())

            # Check that param_grads contains the same parameter nodes
            for desc, (param_node, grad_node) in param_grads.items():
                self.assertIn(param_node, param_nodes)
                self.assertEqual(param_node, named_params[desc.target])

            # Check that all_input_grads contains the parameter nodes
            param_count = 0
            for desc, (input_node, grad_node) in all_input_grads.items():
                if isinstance(desc, ParamAOTInput):
                    param_count += 1
                    self.assertIn(input_node, param_nodes)
                    self.assertEqual(input_node, named_params[desc.target])

            self.assertEqual(param_count, len(param_nodes))


if __name__ == "__main__":
    run_tests()
