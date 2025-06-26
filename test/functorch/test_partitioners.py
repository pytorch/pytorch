# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from torch._functorch.partitioners import (
    should_quantize,
    statically_known_false,
    statically_known_true,
)
from torch.fx.experimental.proxy_tensor import make_fx


class TestPartitioners(unittest.TestCase):
    def setUp(self):
        # Save original config values to restore after tests
        self.original_config = {}
        if (
            hasattr(torch._inductor.config, "post_grad_fusion_options")
            and "activation_quantization_aten_pass"
            in torch._inductor.config.post_grad_fusion_options
        ):
            self.original_config = dict(
                torch._inductor.config.post_grad_fusion_options[
                    "activation_quantization_aten_pass"
                ]
            )

    def tearDown(self):
        # Restore original config values
        if (
            hasattr(torch._inductor.config, "post_grad_fusion_options")
            and "activation_quantization_aten_pass"
            in torch._inductor.config.post_grad_fusion_options
        ):
            torch._inductor.config.post_grad_fusion_options[
                "activation_quantization_aten_pass"
            ] = self.original_config

    def _create_test_node(self, shape=(10, 10), dtype=torch.bfloat16):
        """Helper to create a test node with meta data for testing should_quantize"""
        tensor = torch.randn(shape, dtype=dtype)

        # Create a dummy function to trace
        def dummy_fn(x):
            return x + 1

        # Use make_fx to create a traced function with meta data
        traced = make_fx(dummy_fn)(tensor)

        # Return the first node with meta data (should be the input node)
        for node in traced.graph.nodes:
            if hasattr(node, "meta") and "val" in node.meta:
                return node

        return None

    def test_should_quantize_basic(self):
        """Test the basic functionality of should_quantize"""
        # Setup config for testing
        if not hasattr(torch._inductor.config, "post_grad_fusion_options"):
            torch._inductor.config.post_grad_fusion_options = {}

        if (
            "activation_quantization_aten_pass"
            not in torch._inductor.config.post_grad_fusion_options
        ):
            torch._inductor.config.post_grad_fusion_options[
                "activation_quantization_aten_pass"
            ] = {}

        config = torch._inductor.config.post_grad_fusion_options[
            "activation_quantization_aten_pass"
        ]

        # Test with size below threshold
        config["size_in_mb"] = 100
        node = self._create_test_node(shape=(10, 10))  # Small tensor
        self.assertFalse(should_quantize(node))

        # Test with size above threshold
        node = self._create_test_node(shape=(5000, 5000))  # Large tensor
        self.assertTrue(should_quantize(node))

        # Test with unsupported dtype
        node = self._create_test_node(dtype=torch.int32)  # Unsupported dtype
        self.assertFalse(should_quantize(node))

    def test_should_quantize_skip_dynamo_guards(self):
        """Test the skip_dynamo_guards functionality in should_quantize"""
        # Setup config for testing
        if not hasattr(torch._inductor.config, "post_grad_fusion_options"):
            torch._inductor.config.post_grad_fusion_options = {}

        if (
            "activation_quantization_aten_pass"
            not in torch._inductor.config.post_grad_fusion_options
        ):
            torch._inductor.config.post_grad_fusion_options[
                "activation_quantization_aten_pass"
            ] = {}

        config = torch._inductor.config.post_grad_fusion_options[
            "activation_quantization_aten_pass"
        ]
        config["size_in_mb"] = 100

        # Create a test node
        node = self._create_test_node(shape=(5000, 5000))  # Large tensor

        # Test with skip_dynamo_guards=False (default behavior)
        config["skip_dynamo_guards"] = False
        self.assertTrue(should_quantize(node))

        # Test with skip_dynamo_guards=True, quantize_dynamic_shape=False
        config["skip_dynamo_guards"] = True
        config["quantize_dynamic_shape"] = False

        # Mock statically_known_true to return True for our test
        original_skt = torch.fx.experimental.symbolic_shapes.statically_known_true
        torch.fx.experimental.symbolic_shapes.statically_known_true = lambda x: True

        self.assertTrue(should_quantize(node))

        # Restore original function
        torch.fx.experimental.symbolic_shapes.statically_known_true = original_skt

    def test_should_quantize_with_dynamic_shapes(self):
        """Test should_quantize with dynamic shapes"""
        # Setup config for testing
        if not hasattr(torch._inductor.config, "post_grad_fusion_options"):
            torch._inductor.config.post_grad_fusion_options = {}

        if (
            "activation_quantization_aten_pass"
            not in torch._inductor.config.post_grad_fusion_options
        ):
            torch._inductor.config.post_grad_fusion_options[
                "activation_quantization_aten_pass"
            ] = {}

        config = torch._inductor.config.post_grad_fusion_options[
            "activation_quantization_aten_pass"
        ]
        config["size_in_mb"] = 100
        config["skip_dynamo_guards"] = True

        # Create a test node
        node = self._create_test_node(shape=(5000, 5000))  # Large tensor

        # Test case 1: Always quantize tensors with dynamic shapes
        config["quantize_dynamic_shape"] = True

        # Mock statically_known_true to return False
        original_skt = torch.fx.experimental.symbolic_shapes.statically_known_true
        torch.fx.experimental.symbolic_shapes.statically_known_true = lambda x: False

        # Mock statically_known_false to return False
        original_skf = torch.fx.experimental.symbolic_shapes.statically_known_false
        torch.fx.experimental.symbolic_shapes.statically_known_false = lambda x: False

        # Should return True because not statically_known_false(size_in_mb >= threshold) is True
        self.assertTrue(should_quantize(node))

        # Restore original functions
        torch.fx.experimental.symbolic_shapes.statically_known_true = original_skt
        torch.fx.experimental.symbolic_shapes.statically_known_false = original_skf

        # Test case 2: Always not quantize tensors with dynamic shapes
        config["quantize_dynamic_shape"] = False

        # Mock statically_known_true to return False
        original_skt = torch.fx.experimental.symbolic_shapes.statically_known_true
        torch.fx.experimental.symbolic_shapes.statically_known_true = lambda x: False

        # Should return False because statically_known_true(size_in_mb >= threshold) is False
        self.assertFalse(should_quantize(node))

        # Restore original function
        torch.fx.experimental.symbolic_shapes.statically_known_true = original_skt


if __name__ == "__main__":
    unittest.main()
