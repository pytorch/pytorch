# Owner(s): ["oncall: fx"]

import unittest
from typing import Callable, List

import numpy as np
import torch
import torch.fx.experimental.fx_acc.acc_normalizer as acc_normalizer
import torch.fx.experimental.fx_acc.acc_ops as acc_ops
import torch.fx.experimental.fx_acc.acc_tracer as acc_tracer
import torch.fx.experimental.fx_acc.acc_utils as acc_utils
import torch.nn as nn
import torchvision
from parameterized import parameterized, param

torch.manual_seed(0)


class AccTracerTest(unittest.TestCase):
    def _make_model_unit_test(
        self,
        model,
        *args,
        input_shape=None,
        enable_allclose=False,
        **kwargs,
    ):
        """
        Test that the model can be traced correctly and is producing correct
        result.
        """
        if input_shape is None:
            input_shape = [1, 3, 224, 224]
        input = torch.randn(input_shape)
        traced = acc_tracer.trace(model, [input])
        if enable_allclose:
            torch.testing.assert_allclose(model(input), traced(input))
        else:
            self.assertTrue(torch.equal(model(input), traced(input)))

    def _make_acc_op_function_test(
        self,
        acc_op: Callable,
        torch_op,
        *args,
        input_shape=(2, 3),
        validate_same_kwargs=True,
        enable_allclose=False,
        **kwargs,
    ):
        """
        Test that acc_op is traced somewhat.
        """

        class TestModule(torch.nn.Module):
            def __init__(self, torch_op, args, kwargs):
                super().__init__()
                self._torch_op = torch_op
                self._args = args
                self._kwargs = kwargs

            def forward(self, a: torch.Tensor) -> torch.Tensor:
                return self._torch_op(a, *self._args, **self._kwargs)

        m = TestModule(torch_op, args, kwargs)

        a = torch.randn(*input_shape)
        traced = acc_tracer.trace(m, [a])
        ph_a = acc_op_node = None
        for node in traced.graph.nodes:
            if node.op == "placeholder":
                if str(node.target) == "a":
                    ph_a = node
            elif node.op == "call_function":
                self.assertEqual(node.target, acc_op)
                self.assertEqual(node.kwargs["input"], ph_a)
                if validate_same_kwargs:
                    for key, value in kwargs.items():
                        self.assertEqual(node.kwargs[key], value)
                acc_op_node = node
            elif node.op == "output":
                if acc_op is None:
                    # If we expect no new acc_op after graph building
                    # and found we have only output in traced graph
                    continue
                self.assertEqual(acc_op_node, node.args[0])
            else:
                self.fail(f"Unexpected node: {node.format_node()}")

        ref_outputs = m(a)
        outputs = traced(a)
        if isinstance(ref_outputs, torch.Tensor):
            ref_outputs = [ref_outputs]
            outputs = [outputs]

        for ref_output, output in zip(ref_outputs, outputs):
            if enable_allclose:
                torch.testing.assert_allclose(
                    torch.nan_to_num(ref_output), torch.nan_to_num(output)
                )
            else:
                self.assertTrue(
                    torch.equal(torch.nan_to_num(ref_output), torch.nan_to_num(output))
                )

    def test_sum(self):
        self._make_acc_op_function_test(acc_ops.sum, torch.sum)
        self._make_acc_op_function_test(acc_ops.sum, torch.sum, dim=(1,), keepdim=True)

    def test_mean(self):
        self._make_acc_op_function_test(acc_ops.mean, torch.mean)
        self._make_acc_op_function_test(acc_ops.mean, torch.mean, dim=(1,), keepdim=True)

    def test_pad(self):
        self._make_acc_op_function_test(acc_ops.pad, torch.nn.functional.pad, pad=(2, 0))

    def test_max(self):
        def torch_max(x, *args, **kwargs):
            return x.max(*args, **kwargs)

        self._make_acc_op_function_test(acc_ops.max_full_reduce, torch_max)
        self._make_acc_op_function_test(
            acc_ops.max_dim_reduce, torch_max, dim=1, keepdim=True
        )
        self._make_acc_op_function_test(
            acc_ops.max_dim_reduce, torch_max, input_shape=(1, 4), dim=1, keepdim=True
        )
        self._make_acc_op_function_test(
            acc_ops.max_dim_reduce, torch_max, input_shape=(3, 4, 3), dim=2
        )

    @parameterized.expand(
        [
            param("max_maximum", orig_op=torch.max, expected_op=acc_ops.maximum),
            param(
                "maximum_maximum", orig_op=torch.maximum, expected_op=acc_ops.maximum
            ),
            param("min_minimum", orig_op=torch.min, expected_op=acc_ops.minimum),
            param(
                "minimum_minimum", orig_op=torch.minimum, expected_op=acc_ops.minimum
            ),
        ]
    )
    def test_maximum_minimum(self, _: str, orig_op, expected_op):
        class TestModule(torch.nn.Module):
            def __init__(self, orig_op):
                super().__init__()
                self.orig_op = orig_op

            def forward(self, input: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
                return self.orig_op(input, other)

        m = TestModule(orig_op)
        input, other = torch.randn(2, 2), torch.randn(2, 2)
        traced = acc_tracer.trace(m, [input, other])

        ph_in = ph_oth = mxm = None
        for node in traced.graph.nodes:
            if node.op == "placeholder":
                if str(node.target) == "other":
                    ph_oth = node
                else:
                    self.assertTrue(str(node.target) == "input")
                    ph_in = node
            elif node.op == "call_function":
                if node.target == expected_op:
                    self.assertEqual(node.kwargs["input"], ph_in)
                    self.assertEqual(node.kwargs["other"], ph_oth)
                    mxm = node
            elif node.op == "output":
                self.assertEqual(mxm, node.args[0])
            else:
                self.fail(f"Unexpected node: {node.format_node()}")

        self.assertTrue(torch.equal(m(input, other), traced(input, other)))

    def test_conv(self):
        """
        Test that a conv is traced as expected.
        """

        class TestModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(8, 7, 3, stride=2)

            def forward(self, a: torch.Tensor) -> torch.Tensor:
                return self.conv(a)

        m = TestModule()
        input = torch.randn(3, 8, 10, 10)
        traced = acc_tracer.trace(m, [input])

        ph = weight_attr = bias_attr = conv = None
        for node in traced.graph.nodes:
            if node.op == "placeholder":
                self.assertEqual(str(node.target), "a")
                ph = node
            elif node.op == "get_attr" and node.target == "conv.weight":
                weight_attr = node
            elif node.op == "get_attr" and node.target == "conv.bias":
                bias_attr = node
            elif node.op == "call_function":
                self.assertEqual(node.target, acc_ops.conv2d)
                self.assertEqual(node.kwargs["input"], ph)
                self.assertEqual(node.kwargs["weight"], weight_attr)
                self.assertEqual(node.kwargs["bias"], bias_attr)
                self.assertEqual(node.kwargs["stride"], (2, 2))
                self.assertEqual(node.kwargs["padding"], (0, 0))
                self.assertEqual(node.kwargs["dilation"], (1, 1))
                self.assertEqual(node.kwargs["groups"], 1)
                conv = node
            elif node.op == "output":
                self.assertEqual(conv, node.args[0])
            else:
                self.fail(f"Unexpected node: {node.format_node()}")

        self.assertTrue(torch.equal(m(input), traced(input)))

    def test_quantized_conv2d(self):
        class TestModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.quantized.Conv2d(3, 3, 1)

            def forward(self, a: torch.Tensor) -> torch.Tensor:
                return self.conv(a)

        m = TestModule()
        input = torch.quantize_per_tensor(
            torch.randn(1, 3, 1, 1), scale=0.01, zero_point=3, dtype=torch.quint8
        )
        traced = acc_tracer.trace(m, [input])
        print(traced.graph)
        ph = weight_attr = bias_attr = conv = None
        for node in traced.graph.nodes:
            if node.op == "placeholder":
                self.assertEqual(str(node.target), "a")
                ph = node
            elif node.op == "get_attr" and node.target == "conv_weight":
                weight_attr = node
            elif node.op == "get_attr" and node.target == "conv_bias":
                bias_attr = node
            elif node.op == "call_function":
                self.assertEqual(node.target, acc_ops.quantized_conv2d)
                self.assertEqual(node.kwargs["input"], ph)
                self.assertEqual(node.kwargs["weight"], weight_attr)
                self.assertEqual(node.kwargs["bias"], bias_attr)
                conv = node
            elif node.op == "output":
                self.assertEqual(conv, node.args[0])
            else:
                self.fail(f"Unexpected node: {node.format_node()}")

        self.assertTrue(torch.equal(m(input), traced(input)))

    def test_quantized_convrelu2d(self):
        class TestModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.intrinsic.quantized.ConvReLU2d(3, 3, 1)

            def forward(self, a: torch.Tensor) -> torch.Tensor:
                return self.conv(a)

        m = TestModule()
        input = torch.quantize_per_tensor(
            torch.randn(1, 3, 1, 1), scale=0.01, zero_point=3, dtype=torch.quint8
        )
        traced = acc_tracer.trace(m, [input])
        ph = weight_attr = bias_attr = conv = relu = None
        for node in traced.graph.nodes:
            if node.op == "placeholder":
                self.assertEqual(str(node.target), "a")
                ph = node
            elif node.op == "get_attr" and node.target == "conv_weight":
                weight_attr = node
            elif node.op == "get_attr" and node.target == "conv_bias":
                bias_attr = node
            elif node.op == "call_function" and node.target == acc_ops.quantized_conv2d:
                self.assertEqual(node.target, acc_ops.quantized_conv2d)
                self.assertEqual(node.kwargs["input"], ph)
                self.assertEqual(node.kwargs["weight"], weight_attr)
                self.assertEqual(node.kwargs["bias"], bias_attr)
                conv = node
            elif node.op == "call_function" and node.target == acc_ops.relu:
                self.assertEqual(node.target, acc_ops.relu)
                self.assertEqual(node.kwargs["input"], conv)
                relu = node
            elif node.op == "output":
                self.assertEqual(relu, node.args[0])
            else:
                self.fail(f"Unexpected node: {node.format_node()}")

        self.assertTrue(torch.equal(m(input), traced(input)))

    def test_embedding_bag(self):
        """
        Test that an embedding_bag is traced as expected.
        """

        class TestModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.eb = nn.EmbeddingBag(10, 3, mode="sum", include_last_offset=True)

            def forward(self, inp: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
                return self.eb(inp, offsets)

        m = TestModule()
        inp = torch.LongTensor([1, 2, 4, 5, 4, 3, 2, 9])
        offsets = torch.LongTensor([0, 4])
        traced = acc_tracer.trace(m, [inp, offsets])

        inp_node = offsets_node = weight_attr = eb_node = None
        for node in traced.graph.nodes:
            if node.op == "placeholder":
                if str(node.target) == "inp":
                    inp_node = node
                elif str(node.target) == "offsets":
                    offsets_node = node
                else:
                    self.fail(f"Unexpected placeholder {node.target}.")
                continue
            elif node.op == "get_attr" and node.target == "eb.weight":
                weight_attr = node
            elif node.op == "call_function":
                self.assertEqual(node.target, acc_ops.embedding_bag)
                # Note: Normalization called from acc_tracer means we use all kwargs.
                self.assertEqual(node.kwargs["input"], inp_node)
                self.assertEqual(node.kwargs["offsets"], offsets_node)
                self.assertEqual(node.kwargs["weight"], weight_attr)
                self.assertEqual(node.kwargs["mode"], "sum")
                self.assertEqual(node.kwargs["include_last_offset"], True)
                # The rest of these were unspecified, so verify they fell back
                # to their respective default values thanks to normalization.
                self.assertEqual(node.kwargs["max_norm"], None)
                self.assertEqual(node.kwargs["norm_type"], 2.0)
                self.assertEqual(node.kwargs["scale_grad_by_freq"], False)
                self.assertEqual(node.kwargs["sparse"], False)
                self.assertEqual(node.kwargs["per_sample_weights"], None)
                eb_node = node
            elif node.op == "output":
                self.assertEqual(eb_node, node.args[0])

        self.assertTrue(torch.equal(m(inp, offsets), traced(inp, offsets)))

    def test_embedding_bag_byte_and_4bit_rowwise_offsets(self):
        """
        Test that 4 bit quantized embedding_bag is traced as expected.
        """

        class TestModule(nn.Module):
            def __init__(
                self,
                op,
                q_weights,
                per_index_weights,
            ):
                super().__init__()
                self.emb = op
                self.q_weights = q_weights
                self.per_index_weights = per_index_weights

            def forward(
                self,
                indices,
                offsets,
            ):
                return self.emb(
                    self.q_weights,
                    indices,
                    offsets,
                    mode=0,
                    per_sample_weights=self.per_index_weights,
                    include_last_offset=True,
                )

        def run_embedding_bag_test(is_4bit, use_weights):
            # generate random indices, offsets, and weights.
            num_embeddings = 16
            embedding_dim = 32
            num_lengths = 10

            weights = torch.from_numpy(
                (np.random.random_sample((num_embeddings, embedding_dim)) + 1).astype(
                    np.float32
                )
            )
            q_weights = (
                torch.ops.quantized.embedding_bag_4bit_prepack(weights)
                if is_4bit
                else torch.ops.quantized.embedding_bag_byte_prepack(weights)
            )
            np_lengths = np.random.randint(0, num_lengths, size=10).astype(np.int32)

            num_lengths = np.sum(np_lengths)
            indices = torch.from_numpy(
                np.random.randint(low=0, high=num_embeddings, size=num_lengths)
            ).int()

            lengths = torch.from_numpy(np_lengths)
            offsets = torch.cat([torch.zeros([1]), torch.cumsum(lengths, 0)]).int()

            weights = torch.randint(low=0, high=4, size=indices.size())
            per_sample_weights = weights.to(torch.float32)

            indices = indices.to(torch.int32)
            offsets = offsets.to(torch.int32)
            inputs = [
                indices,
                offsets,
            ]

            op = (
                torch.ops.quantized.embedding_bag_4bit_rowwise_offsets
                if is_4bit
                else torch.ops.quantized.embedding_bag_byte_rowwise_offsets
            )

            m = TestModule(
                op,
                q_weights,
                per_sample_weights,
            )

            traced = acc_tracer.trace(m, inputs)
            print(traced.graph)

            expected_target = (
                acc_ops.embedding_bag_4bit_rowwise_offsets
                if is_4bit
                else acc_ops.embedding_bag_byte_rowwise_offsets
            )

            for node in traced.graph.nodes:
                if node.op == "placeholder":
                    if str(node.target) == "indices":
                        inp_node = node
                    elif str(node.target) == "offsets":
                        offsets_node = node
                    else:
                        self.fail(f"Unexpected placeholder {node.target}.")
                    continue
                elif node.op == "get_attr" and node.target == "q_weights":
                    weight_attr = node
                elif node.op == "call_function":
                    self.assertEqual(node.target, expected_target)
                    # Note: Normalization called from acc_tracer means we use all kwargs.
                    self.assertEqual(node.kwargs["indices"], inp_node)
                    self.assertEqual(node.kwargs["offsets"], offsets_node)
                    self.assertEqual(node.kwargs["weight"], weight_attr)
                    self.assertEqual(node.kwargs["mode"], 0)
                    self.assertEqual(node.kwargs["include_last_offset"], True)
                    # The rest of these were unspecified, so verify they fell back
                    # to their respective default values thanks to normalization.
                    eb_node = node
                elif node.op == "output":
                    self.assertEqual(eb_node, node.args[0])
            self.assertTrue(torch.equal(m(indices, offsets), traced(indices, offsets)))

        # test 8-bit
        run_embedding_bag_test(is_4bit=False, use_weights=True)
        # test 4-bit
        run_embedding_bag_test(is_4bit=True, use_weights=True)

    def test_quantized_batch_norm2d(self):
        class TestModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.bn = nn.quantized.BatchNorm2d(3)

            def forward(self, a: torch.Tensor) -> torch.Tensor:
                return self.bn(a)

        m = TestModule()
        m.eval()
        input = torch.quantize_per_tensor(
            torch.randn(1, 3, 1, 1), scale=0.01, zero_point=3, dtype=torch.quint8
        )
        traced = acc_tracer.trace(m, [input])
        ph = weight_attr = bias_attr = bn_mean = bn_var = bn = None
        for node in traced.graph.nodes:
            if node.op == "placeholder":
                self.assertEqual(str(node.target), "a")
                ph = node
            elif node.op == "get_attr" and node.target == "bn.weight":
                weight_attr = node
            elif node.op == "get_attr" and node.target == "bn.bias":
                bias_attr = node
            elif node.op == "get_attr" and node.target == "bn.running_mean":
                bn_mean = node
            elif node.op == "get_attr" and node.target == "bn.running_var":
                bn_var = node
            elif node.op == "get_attr" and node.target == "bn.scale":
                bn_scale = node
            elif node.op == "get_attr" and node.target == "bn.zero_point":
                bn_zero_point = node
            elif node.op == "call_function":
                self.assertEqual(node.target, acc_ops.quantized_batch_norm2d)
                self.assertEqual(node.kwargs["input"], ph)
                self.assertEqual(node.kwargs["weight"], weight_attr)
                self.assertEqual(node.kwargs["bias"], bias_attr)
                self.assertEqual(node.kwargs["running_mean"], bn_mean)
                self.assertEqual(node.kwargs["running_var"], bn_var)
                self.assertEqual(node.kwargs["acc_out_ty"][6]["scale"], bn_scale)
                self.assertEqual(node.kwargs["acc_out_ty"][6]["zero_point"], bn_zero_point)
                bn = node
            elif node.op == "output":
                self.assertEqual(bn, node.args[0])
            else:
                self.fail(f"Unexpected node: {node.format_node()}")

        self.assertTrue(torch.equal(m(input), traced(input)))

    def test_linear(self):
        """
        Test that a linear is traced as expected, i.e. to the functional level and with
        kwarg normalization. Also verify that symbolic shape inference worked as part of
        the acc_tracer.
        """

        class TestModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(3, 5, bias=True)

            def forward(self, a: torch.Tensor) -> torch.Tensor:
                return self.linear(a)

        m = TestModule()
        test_input = torch.randn(1, 3)
        traced = acc_tracer.trace(m, test_input)
        ph = weight_attr = bias_attr = linear = None
        for node in traced.graph.nodes:
            if node.op == "placeholder":
                self.assertEqual(str(node.target), "a")
                ph = node
            elif node.op == "get_attr" and node.target == "linear.weight":
                weight_attr = node
            elif node.op == "get_attr" and node.target == "linear.bias":
                bias_attr = node
            elif node.op == "call_function":
                self.assertEqual(node.target, acc_ops.linear)
                self.assertEqual(node.kwargs["input"], ph)
                self.assertEqual(node.kwargs["weight"], weight_attr)
                self.assertEqual(node.kwargs["bias"], bias_attr)
                linear = node
            elif node.op == "output":
                self.assertEqual(linear, node.args[0])
            else:
                self.fail(f"Unexpected node: {node.format_node()}")
        self.assertTrue(torch.equal(m(test_input), traced(test_input)))

    def test_quantized_linear(self):
        class TestModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.quantized.Linear(3, 5)

            def forward(self, a: torch.Tensor) -> torch.Tensor:
                return self.linear(a)

        m = TestModule()
        input = torch.quantize_per_tensor(
            torch.randn(2, 3), scale=0.01, zero_point=3, dtype=torch.quint8
        )
        traced = acc_tracer.trace(m, [input])
        ph = weight_attr = bias_attr = linear = None
        for node in traced.graph.nodes:
            if node.op == "placeholder":
                self.assertEqual(str(node.target), "a")
                ph = node
            elif node.op == "get_attr" and node.target == "linear_weight":
                weight_attr = node
            elif node.op == "get_attr" and node.target == "linear_bias":
                bias_attr = node
            elif node.op == "call_function":
                self.assertEqual(node.target, acc_ops.quantized_linear)
                self.assertEqual(node.kwargs["input"], ph)
                self.assertEqual(node.kwargs["weight"], weight_attr)
                self.assertEqual(node.kwargs["bias"], bias_attr)
                linear = node
            elif node.op == "output":
                self.assertEqual(linear, node.args[0])
            else:
                self.fail(f"Unexpected node: {node.format_node()}")

        self.assertTrue(torch.equal(m(input), traced(input)))

    @parameterized.expand(
        [
            param("remove_exceptions_false", remove_exceptions=False),
            param("remove_exceptions_true", remove_exceptions=True),
        ]
    )
    def test_batch_norm(self, _, remove_exceptions):
        """
        Test that a batch norm is traced as expected, i.e. to the functional level
        and with kwarg normalization. Note that we also expect to see a
        ConditionalExceptionWrapper in the graph that the AST rewriter converted
        from `if x: raise y`.

        """

        class TestModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.bn = torch.nn.BatchNorm2d(2)

            def forward(self, a: torch.Tensor) -> torch.Tensor:
                return self.bn(a)

        m = TestModule()
        input = torch.randn(2, 2, 1, 1)
        # Note: Explicitly not removing exceptions so that we can check they
        # were found and exist below.
        traced = acc_tracer.trace(
            m,
            [input],
            remove_exceptions=remove_exceptions,
        )

        ph = exception_wrapper = weight = bias = mean = var = bn = None
        for node in traced.graph.nodes:
            if node.op == "placeholder":
                self.assertEqual(str(node.target), "a")
                ph = node
            elif node.op == "get_attr" and node.target == "bn.weight":
                weight = node
            elif node.op == "get_attr" and node.target == "bn.bias":
                bias = node
            elif node.op == "get_attr" and node.target == "bn.running_mean":
                mean = node
            elif node.op == "get_attr" and node.target == "bn.running_var":
                var = node
            elif node.op == "call_function" and node.target == acc_ops.batch_norm:
                # Note: Normalization called from acc_tracer means we use
                # all kwargs.
                self.assertEqual(node.kwargs["input"], ph)
                self.assertEqual(node.kwargs["weight"], weight)
                self.assertEqual(node.kwargs["bias"], bias)
                self.assertEqual(node.kwargs["running_mean"], mean)
                self.assertEqual(node.kwargs["running_var"], var)
                bn = node
            elif (
                node.op == "call_module"
                and node.target == "bn._conditional_exception_wrapper_ValueError"
            ):
                exception_wrapper = node
            elif node.op == "output":
                self.assertEqual(bn, node.args[0])

        self.assertTrue(remove_exceptions or exception_wrapper is not None)

        self.assertTrue(torch.equal(m(input), traced(input)))

    def test_remove_asserts(self):
        """
        Test that a Module with asserts has the asserts automatically removed, as
        well as calls to a class method that should be dead.
        """

        class TestModule(nn.Module):
            def __init__(self):
                super().__init__()

            def _test_method(self, a):
                return a

            def forward(self, a: torch.Tensor) -> torch.Tensor:
                assert torch.equal(self._test_method(a), a)
                return a

        m = TestModule()
        input = torch.randn(10)
        traced = acc_tracer.trace(m, [input], ast_rewriter_allow_list={TestModule})
        # Check we have no call_functions. If remove asserts didn't work
        # correctly we would see a call to torch._assert, _test_method, and
        # torch.equal.
        for node in traced.graph.nodes:
            self.assertFalse(node.op == "call_function")

        self.assertTrue(torch.equal(m(input), traced(input)))

    def test_sequential(self):
        """
        Test that the tracer works for torch.nn.Sequential.
        """

        class TestModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Sequential(nn.Sigmoid(), nn.ReLU())

            def forward(self, a: torch.Tensor) -> torch.Tensor:
                return self.model(a)

        m = TestModule()
        input = torch.randn(10)
        traced = acc_tracer.trace(m, [input])

        for node in traced.graph.nodes:
            if node.op == "call_function":
                is_sigmoid = node.target == acc_ops.sigmoid
                is_relu = node.target == acc_ops.relu
                self.assertTrue(is_sigmoid or is_relu)
            else:
                self.assertTrue(node.op == "placeholder" or node.op == "output")

        self.assertTrue(torch.equal(m(input), traced(input)))

    def test_unsqueeze(self):
        """
        Test that torch.unsqueeze is traced correctly.
        """
        self._make_acc_op_function_test(
            acc_ops.unsqueeze,
            torch.unsqueeze,
            validate_same_kwargs=False,
            dim=1,
        )

    def test_stack(self):
        """
        Test that torch.stack is traced correctly.
        """

        class TestModule(torch.nn.Module):
            def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                return torch.stack((a, b), dim=1)

        a, b = torch.randn(4, 5, 6), torch.randn(4, 5, 6)
        mod = TestModule()
        traced = acc_tracer.trace(mod, [a, b])
        self.assertTrue(torch.equal(mod(a, b), traced(a, b)))

        ph_a = ph_b = unsqueeze_a = unsqueeze_b = cat_node = None
        for node in traced.graph.nodes:
            if node.op == "placeholder":
                if str(node.target) == "a":
                    ph_a = node
                else:
                    self.assertTrue(str(node.target) == "b")
                    ph_b = node
            elif node.op == "call_function":
                if node.target == acc_ops.unsqueeze:
                    if node.kwargs["input"] is ph_a:
                        unsqueeze_a = node
                    else:
                        self.assertEqual(node.kwargs["input"], ph_b)
                        unsqueeze_b = node
                else:
                    self.assertEqual(node.target, acc_ops.cat)
                    self.assertEqual(node.kwargs["tensors"], [unsqueeze_a, unsqueeze_b])
                    cat_node = node
            elif node.op == "output":
                self.assertEqual(cat_node, node.args[0])
            else:
                self.fail(f"Unexpected node: {node.format_node()}")

    def test_no_raise(self):
        """
        self that we can trace `if x: raise y(msg)` when the raise isn't executed.
        """

        class TestModule(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, a, b):
                if torch.equal(a, b):
                    raise AssertionError("a equaled b!")
                return a

        m = TestModule()
        in_a, in_b = torch.randn(5), torch.randn(5)
        traced = acc_tracer.trace(
            m,
            [in_a, in_b],
            remove_exceptions=False,
            use_acc_normalization=False,
            ast_rewriter_allow_list={TestModule},
        )

        # Verify the structure of the graph, including the existence of the
        # exception_wrapper.
        ph_a = exception_wrapper = None
        for node in traced.graph.nodes:
            if node.op == "placeholder":
                if str(node.target) == "a":
                    ph_a = node
                else:
                    self.assertTrue(str(node.target) == "b")
            elif node.op == "call_module":
                self.assertEqual(
                    node.target, "_conditional_exception_wrapper_AssertionError"
                )
                exception_wrapper = node
            elif node.op == "output":
                self.assertEqual(ph_a, node.args[0])

        self.assertTrue(exception_wrapper is not None)

        self.assertTrue(torch.equal(m(in_a, in_b), traced(in_a, in_b)))

    def test_yes_raise(self):
        """
        Test that we can trace `if x: raise y(msg)` when the raise is executed.
        """
        err_str = "a equaled b!"

        class TestModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.err_str = err_str

            def forward(self, a, b):
                if torch.equal(a, b):
                    raise RuntimeError(self.err_str)
                return a

        m = TestModule()
        # Note: We must use different inputs here in order for shape_prop to work, as
        # otherwise the exception is thrown (as expected/checked below).
        in_a, in_b = torch.randn(5), torch.randn(5)
        traced = acc_tracer.trace(
            m,
            [in_a, in_b],
            remove_exceptions=False,
            ast_rewriter_allow_list={TestModule},
        )

        # Verify the structure of the graph, including the existence of the
        # exception_wrapper.
        ph_a = exception_wrapper = None
        for node in traced.graph.nodes:
            if node.op == "placeholder":
                if str(node.target) == "a":
                    ph_a = node
                else:
                    self.assertTrue(str(node.target) == "b")
            elif node.op == "call_module":
                self.assertEqual(
                    node.target, "_conditional_exception_wrapper_RuntimeError"
                )
                exception_wrapper = node
            elif node.op == "output":
                self.assertEqual(ph_a, node.args[0])

        self.assertTrue(exception_wrapper is not None)

        def test(mod):
            try:
                # Note: Use the same input here to ensure the exception is thrown.
                mod(in_a, in_a)
                self.fail("Shouldn't get here because exception should be thrown.")
            except RuntimeError as e:
                self.assertEqual(err_str, str(e))

        test(m)
        test(traced)

    def test_remove_raise(self):
        """
        Test that we can trace `if x: raise y(msg)` and then remove the exception_wrapper.
        """

        class TestModule(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, a, b):
                if torch.equal(a, b):
                    raise AssertionError("a equaled b!")
                return a

        m = TestModule()
        in_a, in_b = torch.randn(5), torch.randn(5)
        traced = acc_tracer.trace(
            m,
            [in_a, in_b],
            remove_exceptions=True,
            ast_rewriter_allow_list={TestModule},
        )

        # Verify the structure of the graph, including the existence of the
        # exception_wrapper.
        ph_a = None
        for node in traced.graph.nodes:
            if node.op == "placeholder":
                if str(node.target) == "a":
                    ph_a = node
                else:
                    self.assertTrue(str(node.target) == "b")
            elif node.op == "output":
                self.assertEqual(ph_a, node.args[0])
            else:
                # Should not encounter any call_modules, e.g. to the
                # exception_wrapper.
                self.assertFalse(node.op == "call_module")

        # Note: Using input in_a twice for the tracer version, which would
        # trigger the raise if it was still there.
        self.assertTrue(torch.equal(m(in_a, in_b), traced(in_a, in_a)))

    def test_raise_no_message(self):
        """
        Test that we can trace `if x: raise y` when `y` has no message.
        """

        class TestModule(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, a, b):
                if torch.equal(a, b):
                    raise AssertionError
                return a

        m = TestModule()
        in_a, in_b = torch.randn(5), torch.randn(5)
        traced = acc_tracer.trace(
            m,
            [in_a, in_b],
            remove_exceptions=False,
            use_acc_normalization=False,
            ast_rewriter_allow_list={TestModule},
        )

        # Verify the structure of the graph, including the existence of the
        # exception_wrapper.
        ph_a = exception_wrapper = None
        for node in traced.graph.nodes:
            if node.op == "placeholder":
                if str(node.target) == "a":
                    ph_a = node
                else:
                    self.assertTrue(str(node.target) == "b")
            elif node.op == "call_module":
                self.assertEqual(
                    node.target, "_conditional_exception_wrapper_AssertionError"
                )
                exception_wrapper = node
            elif node.op == "output":
                self.assertEqual(ph_a, node.args[0])

        self.assertTrue(exception_wrapper is not None)
        self.assertTrue(torch.equal(m(in_a, in_b), traced(in_a, in_b)))

    def test_quantized_add(self):
        """
        Test that a quantized_add and acc_ops.quantize_per_tensor are traced as expected,
        verifying the acc_out_tys are set as expected.
        """

        class TestModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_input = torch.nn.quantized.Quantize(
                    scale=1.0 / 128, zero_point=5, dtype=torch.quint8
                )
                self.q_other = torch.nn.quantized.Quantize(
                    scale=1.0 / 128, zero_point=10, dtype=torch.quint8
                )

            def forward(self, input: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
                return torch.ops.quantized.add(
                    self.q_input(input),
                    self.q_other(other),
                    scale=0.05,
                    zero_point=1,
                )

        m = TestModule()
        input, other = torch.randn(2, 3, 4), torch.randn(2, 3, 4)
        traced = acc_tracer.trace(m, [input, other])

        input_ph = other_ph = q_input = q_other = q_add = None
        for node in traced.graph.nodes:
            if node.op == "placeholder":
                if str(node.target) == "input":
                    input_ph = node
                else:
                    self.assertTrue(str(node.target) == "other")
                    other_ph = node
            elif (
                node.op == "call_function"
                and node.target == acc_ops.quantize_per_tensor
            ):
                qparams = {
                    "scale": 1.0 / 128,
                    "zero_point": 5,
                }
                expected_md = acc_utils.build_raw_tensor_meta(
                    dtype=torch.quint8,
                    qparams=qparams,
                )
                if node.kwargs["input"] == input_ph:
                    q_input = node
                else:
                    self.assertTrue(node.kwargs["input"] == other_ph)
                    q_other = node
                    qparams_copy = qparams.copy()
                    qparams_copy["zero_point"] = 10
                    expected_md = expected_md._replace(qparams=qparams_copy)
                self.assertEqual(node.kwargs["acc_out_ty"], expected_md)
            elif node.op == "call_function" and node.target == acc_ops.quantized_add:
                self.assertEqual(node.kwargs["input"], q_input)
                self.assertEqual(node.kwargs["other"], q_other)
                qparams = {
                    "scale": 0.05,
                    "zero_point": 1,
                }
                expected_md = acc_utils.build_raw_tensor_meta(qparams=qparams)
                self.assertEqual(node.kwargs["acc_out_ty"], expected_md)
                q_add = node
            elif node.op == "output":
                self.assertEqual(q_add, node.args[0])
            else:
                self.fail(f"Unexpected node: {node.format_node()}")

        self.assertTrue(torch.equal(m(input, other), traced(input, other)))

    def test_quantized_mul(self):
        """
        Test that a quantized_mul and acc_ops.quantize_per_tensor are traced as expected,
        verifying the acc_out_tys are set as expected.
        """

        class TestModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_input = torch.nn.quantized.Quantize(
                    scale=1.0 / 128, zero_point=5, dtype=torch.quint8
                )
                self.q_other = torch.nn.quantized.Quantize(
                    scale=1.0 / 128, zero_point=10, dtype=torch.quint8
                )

            def forward(self, input: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
                return torch.ops.quantized.mul(
                    self.q_input(input),
                    self.q_other(other),
                    scale=0.05,
                    zero_point=1,
                )

        m = TestModule()
        input, other = torch.randn(2, 3, 4), torch.randn(2, 3, 4)
        traced = acc_tracer.trace(m, [input, other])

        input_ph = other_ph = q_input = q_other = q_add = None
        for node in traced.graph.nodes:
            if node.op == "placeholder":
                if str(node.target) == "input":
                    input_ph = node
                else:
                    self.assertTrue(str(node.target) == "other")
                    other_ph = node
            elif (
                node.op == "call_function"
                and node.target == acc_ops.quantize_per_tensor
            ):
                qparams = {
                    "scale": 1.0 / 128,
                    "zero_point": 5,
                }
                expected_md = acc_utils.build_raw_tensor_meta(
                    dtype=torch.quint8,
                    qparams=qparams,
                )
                if node.kwargs["input"] == input_ph:
                    q_input = node
                else:
                    self.assertTrue(node.kwargs["input"] == other_ph)
                    q_other = node
                    qparams_copy = qparams.copy()
                    qparams_copy["zero_point"] = 10
                    expected_md = expected_md._replace(qparams=qparams_copy)
                self.assertEqual(node.kwargs["acc_out_ty"], expected_md)
            elif node.op == "call_function" and node.target == acc_ops.quantized_mul:
                self.assertEqual(node.kwargs["input"], q_input)
                self.assertEqual(node.kwargs["other"], q_other)
                qparams = {
                    "scale": 0.05,
                    "zero_point": 1,
                }
                expected_md = acc_utils.build_raw_tensor_meta(qparams=qparams)
                self.assertEqual(node.kwargs["acc_out_ty"], expected_md)
                q_add = node
            elif node.op == "output":
                self.assertEqual(q_add, node.args[0])
            else:
                self.fail(f"Unexpected node: {node.format_node()}")

        self.assertTrue(torch.equal(m(input, other), traced(input, other)))

    def test_cat(self):
        """
        Test that torch.cat is traced correctly.
        """

        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                return torch.cat([a, a, b], 0)

        m = TestModule()
        a, b = torch.randn(2, 2), torch.randn(2, 2)
        traced = acc_tracer.trace(m, (a, b))

        ph_a = ph_b = cat = None
        for node in traced.graph.nodes:
            if node.op == "placeholder":
                if str(node.target) == "a":
                    ph_a = node
                else:
                    self.assertTrue(str(node.target) == "b")
                    ph_b = node
            elif node.op == "call_function":
                self.assertEqual(node.target, acc_ops.cat)
                self.assertEqual(node.kwargs["tensors"][0], ph_a)
                self.assertEqual(node.kwargs["tensors"][1], ph_a)
                self.assertEqual(node.kwargs["tensors"][2], ph_b)
                self.assertEqual(node.kwargs["dim"], 0)
                cat = node
            elif node.op == "output":
                self.assertEqual(cat, node.args[0])
            else:
                self.fail(f"Unexpected node: {node.format_node()}")

        self.assertTrue(torch.equal(m(a, b), traced(a, b)))

    def test_square(self):
        """
        Test that torch.square is traced correctly.
        """
        self._make_acc_op_function_test(acc_ops.mul, torch.square)

    def test_reshape(self):
        """
        Test that torch.reshape is traced correctly.
        """
        self._make_acc_op_function_test(acc_ops.reshape, torch.reshape, (1, -1))
        # arg = (1, -1)
        self._make_acc_op_function_test(acc_ops.reshape, lambda x: x.reshape(1, -1))
        # arg = ((1, -1))
        self._make_acc_op_function_test(acc_ops.reshape, lambda x: x.reshape((1, -1)))

    def test_transpose(self):
        """
        Test that torch.transpose is traced correctly.
        """
        self._make_acc_op_function_test(
            acc_ops.permute, lambda x: torch.transpose(x, 1, 0)
        )

    def test_permute(self):
        """
        Test that torch.permute is traced correctly.
        """

        def torch_permute(a, *dim):
            return a.permute(*dim)

        self._make_acc_op_function_test(acc_ops.permute, torch_permute, 1, 0)

    def test_min_full_reduce(self):
        """
        Test that test_min_full_reduce is traced correctly.
        """
        self._make_acc_op_function_test(acc_ops.min_full_reduce, torch.min)

    def test_matmul(self):
        """
        Test that torch.matmul is traced correctly.
        """

        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                return torch.matmul(a, b)

        m = TestModule()
        a, b = torch.randn(2, 2), torch.randn(2, 2)
        traced = acc_tracer.trace(m, [a, b])

        ph_a = ph_b = matmul = None
        for node in traced.graph.nodes:
            if node.op == "placeholder":
                if str(node.target) == "a":
                    ph_a = node
                else:
                    self.assertTrue(str(node.target) == "b")
                    ph_b = node
            elif node.op == "call_function":
                self.assertEqual(node.target, acc_ops.matmul)
                self.assertEqual(node.kwargs["input"], ph_a)
                self.assertEqual(node.kwargs["other"], ph_b)
                matmul = node
            elif node.op == "output":
                self.assertEqual(matmul, node.args[0])
            else:
                self.fail(f"Unexpected node: {node.format_node()}")

        self.assertTrue(torch.equal(m(a, b), traced(a, b)))

    def test_bmm(self):
        self._make_acc_op_function_test(
            acc_ops.matmul, lambda x: torch.bmm(x, x), input_shape=(2, 4, 4)
        )

    def test_tile(self):
        return self._make_acc_op_function_test(
            acc_ops.tile, lambda x: torch.tile(x, (2, 1, 2)), input_shape=(1, 2)
        )

    def test_dropout(self):
        self._make_acc_op_function_test(
            None,
            lambda x: nn.functional.dropout(x, training=False),
            input_shape=(1, 2, 3),
        )

    def test_hardsigmoid(self):
        self._make_acc_op_function_test(
            acc_ops.hardsigmoid,
            lambda x: nn.functional.hardsigmoid(x),
            input_shape=(3, 4, 5),
        )

    def test_hardtanh(self):
        self._make_acc_op_function_test(
            acc_ops.hardtanh,
            lambda x: nn.functional.hardtanh(x),
            input_shape=(3, 4, 5),
        )

    def test_hardswish(self):
        class TestModule(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                y = nn.functional.hardswish(x)
                return y

        m = TestModule()
        x = torch.randn(3, 4, 5)
        traced = acc_tracer.trace(m, x)
        ph_x = hardsigmoid_y = res_y = None
        for node in traced.graph.nodes:
            if node.op == "placeholder":
                ph_x = node
            elif node.op == "call_function" and node.target == acc_ops.hardsigmoid:
                hardsigmoid_y = node
                self.assertEqual(node.kwargs["input"], ph_x)
            elif node.op == "call_function" and node.target == acc_ops.mul:
                res_y = node
                self.assertEqual(node.kwargs["input"], hardsigmoid_y)
                self.assertEqual(node.kwargs["other"], ph_x)
            elif node.op == "output":
                self.assertEqual(node.args[0], res_y)
            else:
                self.fail(f"Unexpected node: {node.format_node()}")

        ref = m(x)
        res = traced(x)
        torch.testing.assert_allclose(ref, res)

    def test_add_with_alpha(self):
        """
        Test that normalization works for torch add with alpha, which requires special
        normalization handling.
        """

        class TestModule(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                a1 = torch.add(a, b)
                a2 = torch.add(a, b, alpha=1.0)
                a3 = torch.add(a, b, alpha=0.5)
                return a1, a2, a3

        m = TestModule()
        input_a = torch.randn(2, 3)
        input_b = torch.randn(2, 3)
        traced = acc_tracer.trace(m, [input_a, input_b])

        ph_a = ph_b = add_1 = add_2 = add_3 = mul = None
        for node in traced.graph.nodes:
            if node.op == "placeholder":
                if str(node.target) == "a":
                    ph_a = node
                elif str(node.target) == "b":
                    ph_b = node
                else:
                    self.fail(f"Unexpected placeholder {node.target}.")
            elif node.op == "call_function" and node.target == acc_ops.mul:
                mul = node
                self.assertEqual(node.kwargs["input"], ph_b)
                self.assertEqual(node.kwargs["other"], 0.5)
            elif node.op == "call_function" and node.target == acc_ops.add:
                if add_1 is None:
                    add_1 = node
                    self.assertEqual(node.kwargs["input"], ph_a)
                    self.assertEqual(node.kwargs["other"], ph_b)
                elif add_2 is None:
                    add_2 = node
                    self.assertEqual(node.kwargs["input"], ph_a)
                    self.assertEqual(node.kwargs["other"], ph_b)
                elif add_3 is None:
                    add_3 = node
                    self.assertEqual(node.kwargs["input"], ph_a)
                    self.assertEqual(node.kwargs["other"], mul)
                else:
                    self.fail(f"Unexpected add: {node.format_node()}")
            elif node.op == "output":
                self.assertEqual(node.args[0][0], add_1)
                self.assertEqual(node.args[0][1], add_2)
                self.assertEqual(node.args[0][2], add_3)
            else:
                self.fail(f"Unexpected node: {node.format_node()}")

        ref = m(input_a, input_b)
        res = traced(input_a, input_b)
        self.assertTrue(torch.equal(ref[0], res[0]))
        self.assertTrue(torch.equal(ref[1], res[1]))
        self.assertTrue(torch.equal(ref[2], res[2]))

    def test_leaf_module_list(self):
        """
        Test leaf_module_list is working properly.
        """

        class LeafModule(nn.Module):
            def forward(self, x):
                return x

        class TestModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.mod = LeafModule()

            def forward(self, x):
                return self.mod(x)

        x = torch.randn(1, 1)
        mod = TestModule()
        acc_mod = acc_tracer.trace(
            mod,
            [x],
            leaf_module_list={LeafModule},
        )
        ph = leaf_module = None
        for node in acc_mod.graph.nodes:
            if node.op == "placeholder":
                ph = node
            elif node.op == "call_module":
                leaf_module = node
                self.assertEqual(leaf_module.target, "mod")
                self.assertEqual(leaf_module.args[0], ph)
            elif node.op == "output":
                self.assertEqual(node.args[0], leaf_module)
            else:
                self.fail(f"Unexpected node: {node.format_node()}")
        self.assertTrue(torch.equal(mod(x), acc_mod(x)))

    def test_sign(self):
        self._make_acc_op_function_test(acc_ops.sign, torch.sign)

    def test_relu(self):
        self._make_acc_op_function_test(acc_ops.relu, torch.relu)

    def test_leaky_relu(self):
        self._make_acc_op_function_test(acc_ops.leaky_relu, torch.nn.functional.leaky_relu)

    def test_elu(self):
        self._make_acc_op_function_test(acc_ops.elu, torch.nn.functional.elu)

    def test_selu(self):
        self._make_acc_op_function_test(acc_ops.selu, torch.nn.functional.selu)

    def test_softsign(self):
        self._make_acc_op_function_test(acc_ops.softsign, torch.nn.functional.softsign)

    def test_sigmoid(self):
        self._make_acc_op_function_test(acc_ops.sigmoid, torch.sigmoid)

    def test_sin(self):
        self._make_acc_op_function_test(acc_ops.sin, torch.sin)

    def test_cos(self):
        self._make_acc_op_function_test(acc_ops.cos, torch.cos)

    def test_tan(self):
        self._make_acc_op_function_test(acc_ops.tan, torch.tan)

    def test_sinh(self):
        self._make_acc_op_function_test(acc_ops.sinh, torch.sinh)

    def test_cosh(self):
        self._make_acc_op_function_test(acc_ops.cosh, torch.cosh)

    def test_tanh(self):
        self._make_acc_op_function_test(acc_ops.tanh, torch.tanh)

    def test_asin(self):
        self._make_acc_op_function_test(acc_ops.asin, torch.asin)

    def test_acos(self):
        self._make_acc_op_function_test(acc_ops.acos, torch.acos)

    def test_atan(self):
        self._make_acc_op_function_test(acc_ops.atan, torch.atan)

    def test_exp(self):
        self._make_acc_op_function_test(acc_ops.exp, torch.exp)

    def test_log(self):
        self._make_acc_op_function_test(acc_ops.log, torch.log)

    def test_sqrt(self):
        self._make_acc_op_function_test(acc_ops.sqrt, torch.sqrt)

    def test_reciprocal(self):
        self._make_acc_op_function_test(acc_ops.reciprocal, torch.reciprocal)

    def test_abs(self):
        self._make_acc_op_function_test(acc_ops.abs, torch.abs)

    def test_neg(self):
        self._make_acc_op_function_test(acc_ops.neg, torch.neg)

    def test_floor(self):
        self._make_acc_op_function_test(acc_ops.floor, torch.floor)

    def test_ceil(self):
        self._make_acc_op_function_test(acc_ops.ceil, torch.ceil)

    def test_softmax(self):
        self._make_acc_op_function_test(acc_ops.softmax, torch.nn.functional.softmax)

    def test_tensor_squeeze(self):
        self._make_acc_op_function_test(acc_ops.squeeze, lambda x: x.squeeze())

    def test_torch_squeeze(self):
        self._make_acc_op_function_test(acc_ops.squeeze, lambda x: torch.squeeze(x))

    def test_operator_mul(self):
        self._make_acc_op_function_test(acc_ops.mul, lambda x: x * 7)

    def test_torch_mul(self):
        self._make_acc_op_function_test(acc_ops.mul, lambda x: torch.mul(x, 7))

    def test_div(self):
        self._make_acc_op_function_test(acc_ops.div, lambda x: torch.div(x, 2))
        self._make_acc_op_function_test(acc_ops.div, lambda x: x / 2)

    def test_floor_div(self):
        self._make_acc_op_function_test(acc_ops.floor_div, lambda x: torch.div(x, 2, rounding_mode="floor"))

    def test_trunc_div(self):
        self._make_acc_op_function_test(acc_ops.trunc_div, lambda x: torch.div(x, 2, rounding_mode="trunc"))
        self._make_acc_op_function_test(acc_ops.trunc_div, lambda x: torch.floor_divide(x, 2))

    def test_view(self):
        """
        Test that Tensor.view is traced correctly.
        """

        self._make_acc_op_function_test(acc_ops.reshape, lambda x: x.view(1, -1))

    def test_narrow(self):
        """
        Test that torch.narrow is traced correctly.
        """
        return self._make_acc_op_function_test(
            acc_ops.slice_tensor,
            torch.narrow,
            validate_same_kwargs=False,
            dim=1,
            start=1,
            length=2,
        )

    def test_pow(self):
        self._make_acc_op_function_test(acc_ops.pow, torch.pow, exponent=2)

    def test_size(self):
        class TestModule(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, a):
                idx = a.size(1)
                return a.shape[idx]

        m = TestModule()
        a = torch.randn(2, 1, 4)
        traced = acc_tracer.trace(m, [a])

        ph_a = size_1 = size_2 = getitem_1 = getitem_2 = None
        for node in traced.graph.nodes:
            if node.op == "placeholder":
                self.assertTrue(node.target == "a")
                ph_a = node
            elif node.op == "call_function" and node.target == acc_ops.size:
                if size_1:
                    size_2 = node
                    self.assertTrue(size_2.kwargs["input"] is ph_a)
                else:
                    size_1 = node
                    self.assertTrue(size_1.kwargs["input"] is ph_a)
            elif node.op == "call_function" and node.target == acc_ops.getitem:
                if getitem_1:
                    getitem_2 = node
                    self.assertTrue(getitem_2.kwargs["idx"] == getitem_1)
                    self.assertTrue(getitem_2.kwargs["input"] == size_2)
                else:
                    getitem_1 = node
                    self.assertTrue(getitem_1.kwargs["idx"] == 1)
                    self.assertTrue(getitem_1.kwargs["input"] == size_1)
            elif node.op == "output":
                self.assertEqual(node.args[0], getitem_2)
            else:
                self.fail(f"Unexpected node: {node.format_node()}")

        ref = m(a)
        res = traced(a)
        self.assertEqual(ref, res)

    def test_flatten(self):
        """
        Test that torch.flatten is traced correctly.
        """
        self._make_acc_op_function_test(
            acc_ops.flatten, torch.flatten, start_dim=1, end_dim=1
        )
        self._make_acc_op_function_test(acc_ops.flatten, lambda x: x.flatten())

    def test_topk_multi_output(self):
        """
        Test that torch.topk multi outputs work.
        """

        class TestModule(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, a: torch.Tensor) -> torch.Tensor:
                return torch.topk(a, 3)[1]

        m = TestModule()
        input_a = torch.randn(10)
        traced = acc_tracer.trace(m, [input_a])

        ph_a = topk = getitem = None
        for node in traced.graph.nodes:
            if node.op == "placeholder" and str(node.target) == "a":
                ph_a = node
            elif node.op == "call_function" and node.target == acc_ops.topk:
                topk = node
                self.assertEqual(node.kwargs["input"], ph_a)
                self.assertEqual(node.kwargs["k"], 3)
            elif node.op == "call_function" and node.target == acc_ops.getitem:
                getitem = node
                self.assertEqual(node.kwargs["input"], topk)
                self.assertEqual(node.kwargs["idx"], 1)
            elif node.op == "output":
                self.assertEqual(node.args[0], getitem)
            else:
                self.fail(f"Unexpected node: {node.format_node()}")

        self.assertTrue(torch.equal(m(input_a), traced(input_a)))

    def test_addmm_with_alpha_beta(self):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(
                self, input: torch.Tensor, a: torch.Tensor, b: torch.Tensor
            ) -> torch.Tensor:
                return torch.addmm(input, a, b, alpha=1.2, beta=1.1)

        m = TestModule()
        input, a, b = torch.randn(2, 2), torch.randn(2, 2), torch.randn(2, 2)
        traced = acc_tracer.trace(m, [input, a, b])

        ph_in = ph_a = ph_b = mm = add = mm_mul = add_mul = None
        for node in traced.graph.nodes:
            if node.op == "placeholder":
                if str(node.target) == "a":
                    ph_a = node
                elif str(node.target) == "b":
                    ph_b = node
                else:
                    self.assertTrue(str(node.target) == "input")
                    ph_in = node
            elif node.op == "call_function":
                if node.target == acc_ops.matmul:
                    self.assertEqual(node.kwargs["input"], ph_a)
                    self.assertEqual(node.kwargs["other"], ph_b)
                    mm = node
                elif node.target == acc_ops.add:
                    self.assertEqual(node.kwargs["input"], mm_mul)
                    self.assertEqual(node.kwargs["other"], add_mul)
                    add = node
                elif mm_mul:
                    self.assertEqual(node.kwargs["input"], ph_in)
                    self.assertEqual(node.kwargs["other"], 1.1)
                    add_mul = node
                else:
                    self.assertEqual(node.kwargs["input"], mm)
                    self.assertEqual(node.kwargs["other"], 1.2)
                    mm_mul = node
            elif node.op == "output":
                self.assertEqual(add, node.args[0])
            else:
                self.fail(f"Unexpected node: {node.format_node()}")

        torch.testing.assert_allclose(m(input, a, b), traced(input, a, b))

    def test_log1p(self):
        class TestModule(torch.nn.Module):
            def forward(self, input: torch.Tensor) -> torch.Tensor:
                return torch.log1p(input)

        m = TestModule().eval()
        input = torch.tensor([[1.2, 0.3, -0.4]])
        traced = acc_tracer.trace(m, [input])

        ph_in = add = log = None
        for node in traced.graph.nodes:
            if node.op == "placeholder":
                self.assertTrue(str(node.target) == "input")
                ph_in = node
            elif node.op == "call_function":
                if node.target == acc_ops.add:
                    self.assertEqual(node.kwargs["input"], ph_in)
                    self.assertEqual(node.kwargs["other"], 1)
                    add = node
                else:
                    self.assertEqual(node.target, acc_ops.log)
                    self.assertEqual(node.kwargs["input"], add)
                    log = node
            elif node.op == "output":
                self.assertEqual(log, node.args[0])
            else:
                self.fail(f"Unexpected node: {node.format_node()}")

        torch.testing.assert_allclose(m(input), traced(input))

    def test_addmm(self):
        class TestModule(torch.nn.Module):
            def forward(
                self, input: torch.Tensor, a: torch.Tensor, b: torch.Tensor
            ) -> torch.Tensor:
                return torch.addmm(input, a, b)

        m = TestModule()
        input, a, b = torch.randn(2, 2), torch.randn(2, 2), torch.randn(2, 2)
        traced = acc_tracer.trace(m, [input, a, b])

        ph_in = ph_a = ph_b = mm = add = None
        for node in traced.graph.nodes:
            if node.op == "placeholder":
                if str(node.target) == "a":
                    ph_a = node
                elif str(node.target) == "b":
                    ph_b = node
                else:
                    self.assertTrue(str(node.target) == "input")
                    ph_in = node
            elif node.op == "call_function":
                if node.target == acc_ops.matmul:
                    self.assertEqual(node.kwargs["input"], ph_a)
                    self.assertEqual(node.kwargs["other"], ph_b)
                    mm = node
                else:
                    self.assertEqual(node.target, acc_ops.add)
                    self.assertEqual(node.kwargs["input"], mm)
                    self.assertEqual(node.kwargs["other"], ph_in)
                    add = node
            elif node.op == "output":
                self.assertEqual(add, node.args[0])
            else:
                self.fail(f"Unexpected node: {node.format_node()}")

        self.assertTrue(torch.equal(m(input, a, b), traced(input, a, b)))

    def test_gelu(self):
        return self._make_acc_op_function_test(acc_ops.gelu, torch.nn.functional.gelu)

    @parameterized.expand(
        [
            (1, True),
            (1, False),
            (None, False),
        ]
    )
    def test_argmin(self, dim, keepdim):
        class TestModule(torch.nn.Module):
            def __init__(self, dim, keepdim):
                super().__init__()
                self.dim = dim
                self.keepdim = keepdim

            def forward(self, input: torch.Tensor) -> torch.Tensor:
                return torch.argmin(input, dim=self.dim, keepdim=self.keepdim)

        m = TestModule(dim, keepdim)
        input = torch.randn(2, 2)
        traced = acc_tracer.trace(m, [input])

        ph_in = flatten = topk = getitem = squeeze = None
        for node in traced.graph.nodes:
            if node.op == "placeholder":
                self.assertTrue(str(node.target) == "input")
                ph_in = node
            elif node.op == "call_function":
                if node.target == acc_ops.flatten:
                    self.assertEqual(node.kwargs["input"], ph_in)
                    flatten = node
                elif node.target == acc_ops.topk:
                    self.assertEqual(
                        node.kwargs["input"], flatten if flatten else ph_in
                    )
                    topk = node
                elif node.target == acc_ops.getitem:
                    self.assertEqual(node.kwargs["input"], topk)
                    getitem = node
                elif node.target == acc_ops.squeeze:
                    self.assertEqual(node.kwargs["input"], getitem)
                    squeeze = node
            elif node.op == "output":
                self.assertEqual(squeeze if squeeze else getitem, node.args[0])
            else:
                self.fail(f"Unexpected node: {node.format_node()}")
        if dim is None:
            self.assertTrue(flatten is not None)
        if not keepdim:
            self.assertTrue(squeeze is not None)
        self.assertTrue(torch.equal(m(input), traced(input)))

    def test_t(self):
        """
        Test Tensor.t() is traced correctly.
        """
        self._make_acc_op_function_test(acc_ops.permute, lambda x: x.t())
        self._make_acc_op_function_test(
            acc_ops.permute, lambda x: x.t(), input_shape=(3,)
        )

    def test_split_size(self):
        self._make_acc_op_function_test(
            acc_ops.split,
            torch.split,
            validate_same_kwargs=False,
            split_size_or_sections=2,
            dim=1,
        )

    def test_split_sections(self):
        class TestModule(torch.nn.Module):
            def forward(self, input: torch.Tensor) -> torch.Tensor:
                return torch.split(input, [2, 5, 3], 1)

        m = TestModule()
        input = torch.randn(1, 10)
        traced = acc_tracer.trace(m, [input])

        ph_in = slice_node_0 = slice_node_1 = slice_node_2 = None
        tuple_construct_node = None
        for node in traced.graph.nodes:
            if node.op == "placeholder":
                self.assertTrue(str(node.target) == "input")
                ph_in = node
            elif node.op == "call_function":
                if node.target == acc_ops.slice_tensor:
                    self.assertEqual(node.kwargs["input"], ph_in)
                    if slice_node_0:
                        if slice_node_1:
                            slice_node_2 = node
                        else:
                            slice_node_1 = node
                    else:
                        slice_node_0 = node
                else:
                    self.assertEqual(node.target, acc_ops.tuple_construct)
                    self.assertEqual(
                        node.kwargs["tensors"],
                        (slice_node_0, slice_node_1, slice_node_2),
                    )
                    tuple_construct_node = node
            elif node.op == "output":
                self.assertEqual(tuple_construct_node, node.args[0])
            else:
                self.fail(f"Unexpected node: {node.format_node()}")

        ref_output = m(input)
        output = traced(input)
        for i, j in zip(ref_output, output):
            self.assertTrue(torch.equal(i, j))

    def test_list_input(self):
        """
        Test that list inputs are traced correctly.
        """

        class TestModule(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, a: List[torch.Tensor]) -> torch.Tensor:
                return a[0] + a[1]

        m = TestModule()
        input = [torch.randn(2, 3), torch.randn(2, 3)]
        traced = acc_tracer.trace(m, [input])

        ph = getitem_0 = getitem_1 = add = None
        for node in traced.graph.nodes:
            if node.op == "placeholder":
                self.assertEqual(str(node.target), "a")
                ph = node
            elif node.op == "call_function" and node.target == acc_ops.getitem:
                self.assertTrue(node.kwargs["idx"] == 0 or node.kwargs["idx"] == 1)
                if node.kwargs["idx"] == 0:
                    getitem_0 = node
                else:
                    getitem_1 = node
            elif node.op == "call_function":
                self.assertEqual(node.target, acc_ops.add)
                self.assertEqual(node.kwargs["input"], getitem_0)
                self.assertEqual(node.kwargs["other"], getitem_1)
                add = node
            elif node.op == "output":
                self.assertEqual(add, node.args[0])
            else:
                self.fail(f"Unexpected node: {node.format_node()}")

        # Check the tensor metadatas are correct given the input is a list.
        self.assertTrue(isinstance(ph.meta["tensor_meta"], list))
        self.assertEqual(len(ph.meta["tensor_meta"]), 2)
        self.assertEqual(getitem_0.meta["tensor_meta"], ph.meta["tensor_meta"][0])
        self.assertEqual(getitem_1.meta["tensor_meta"], ph.meta["tensor_meta"][1])

        self.assertTrue(torch.equal(m(input), traced(input)))

    def test_mobilenet_v3(self):
        """
        Test that we can trace mobilenet v3 small and run/compare against the untraced version.
        """
        m = torchvision.models.mobilenet_v3_small(pretrained=True)
        self._make_model_unit_test(m, enable_allclose=True)

    def test_mobilenet_v2(self):
        """
        Test that we can trace mobilenet v2 small and run/compare against the untraced version.
        """
        m = torchvision.models.mobilenet_v2(pretrained=True)
        self._make_model_unit_test(m)

    def test_vgg16(self):
        """
        Test that we can trace vgg16 and run/compare against the untraced version.
        """
        m = torchvision.models.vgg16(pretrained=True)
        self._make_model_unit_test(m)

    def test_resnet18(self):
        """
        Test that we can trace resnet18 and run/compare against the untraced version.
        """
        m = torchvision.models.resnet18(pretrained=True)
        self._make_model_unit_test(m)

    def test_resnext50_32x4d(self):
        """
        Test that we can trace resnext and run/compare against the untraced version.
        """
        m = torchvision.models.resnext50_32x4d(pretrained=True)
        self._make_model_unit_test(m)

    def test_cumsum(self):
        self._make_acc_op_function_test(acc_ops.cumsum, torch.cumsum, dim=1)
        self._make_acc_op_function_test(
            acc_ops.cumsum, torch.cumsum, dim=1, dtype=torch.float
        )

    def test_chunk(self):
        self._make_acc_op_function_test(acc_ops.chunk, torch.chunk, chunks=2, dim=0)

    def test_all_acc_ops_registered(self):
        self.assertEqual(
            acc_normalizer._acc_ops,
            {
                acc_ops.linear,
                acc_ops.max_pool2d,
                acc_ops.flatten,
                acc_ops.adaptive_avg_pool2d,
                acc_ops.avg_pool2d,
                acc_ops.add,
                acc_ops.min_full_reduce,
                acc_ops.min_dim_reduce,
                acc_ops.minimum,
                acc_ops.cat,
                acc_ops.softmax,
                acc_ops.sign,
                acc_ops.permute,
                acc_ops.matmul,
                acc_ops.quantize_per_tensor,
                acc_ops.quantize_per_channel,
                acc_ops.quantized_add,
                acc_ops.quantized_mul,
                acc_ops.dequantize,
                acc_ops.sub,
                acc_ops.mul,
                acc_ops.div,
                acc_ops.floor_div,
                acc_ops.trunc_div,
                acc_ops.pow,
                acc_ops.relu,
                acc_ops.leaky_relu,
                acc_ops.elu,
                acc_ops.selu,
                acc_ops.softsign,
                acc_ops.tuple_construct,
                acc_ops.unsqueeze,
                acc_ops.sigmoid,
                acc_ops.sum,
                acc_ops.max_full_reduce,
                acc_ops.max_dim_reduce,
                acc_ops.maximum,
                acc_ops.sinh,
                acc_ops.cosh,
                acc_ops.tanh,
                acc_ops.asin,
                acc_ops.acos,
                acc_ops.atan,
                acc_ops.exp,
                acc_ops.log,
                acc_ops.sqrt,
                acc_ops.reciprocal,
                acc_ops.abs,
                acc_ops.neg,
                acc_ops.floor,
                acc_ops.ceil,
                acc_ops.size,
                acc_ops.split,
                acc_ops.conv2d,
                acc_ops.batch_norm,
                acc_ops.embedding_bag,
                acc_ops.embedding_bag_byte_rowwise_offsets,
                acc_ops.embedding_bag_4bit_rowwise_offsets,
                acc_ops.contiguous,
                acc_ops.pad,
                acc_ops.sin,
                acc_ops.cos,
                acc_ops.tan,
                acc_ops.topk,
                acc_ops.getitem,
                acc_ops.squeeze,
                acc_ops.tile,
                acc_ops.reshape,
                acc_ops.quantized_linear,
                acc_ops.quantized_conv2d,
                acc_ops.quantized_batch_norm2d,
                acc_ops.to_dtype,
                acc_ops.clamp,
                acc_ops.layer_norm,
                acc_ops.linalg_norm,
                acc_ops.slice_tensor,
                acc_ops.hardsigmoid,
                acc_ops.mean,
                acc_ops.hardtanh,
                acc_ops.gelu,
                acc_ops.cumsum,
                acc_ops.chunk,
                acc_ops.rescale_quantize_per_tensor,
                acc_ops.rescale_quantize_per_channel,
            },
        )
