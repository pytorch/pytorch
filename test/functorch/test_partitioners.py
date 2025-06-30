# mypy: allow-untyped-defs

import unittest
import unittest.mock as mock

import torch
from torch._dynamo.utils import is_node_meta_valid
from torch._functorch.partitioners import calculate_tensor_size, should_quantize
from torch.fx.experimental.symbolic_shapes import (
    statically_known_false,
    statically_known_true,
)


class TestPartitioners(unittest.TestCase):
    def setUp(self):
        # Save original config to restore later
        self.original_config = {}
        if hasattr(torch._inductor, "config") and hasattr(
            torch._inductor.config, "post_grad_fusion_options"
        ):
            if (
                "activation_quantization_aten_pass"
                in torch._inductor.config.post_grad_fusion_options
            ):
                self.original_config = torch._inductor.config.post_grad_fusion_options[
                    "activation_quantization_aten_pass"
                ].copy()

        # Setup default config for tests
        if not hasattr(torch._inductor, "config"):
            torch._inductor.config = mock.MagicMock()

        torch._inductor.config.post_grad_fusion_options = {
            "activation_quantization_aten_pass": {
                "size_in_mb": 50,  # Threshold size
                "allowed_dtypes": "torch.bfloat16",
                "skip_dynamo_guards": False,
                "quantize_dynamic_shape": False,
            }
        }

    def tearDown(self):
        # Restore original config
        if self.original_config:
            torch._inductor.config.post_grad_fusion_options[
                "activation_quantization_aten_pass"
            ] = self.original_config

    @mock.patch("torch._functorch.partitioners.is_node_meta_valid")
    @mock.patch("torch._functorch.partitioners.calculate_tensor_size")
    def test_should_quantize_basic(
        self, mock_calculate_tensor_size, mock_is_node_meta_valid
    ):
        # Setup mocks
        mock_is_node_meta_valid.return_value = True
        mock_calculate_tensor_size.return_value = 60  # Greater than threshold (50)

        # Create a mock node
        mock_node = mock.MagicMock()
        mock_node.meta = {"val": mock.MagicMock()}
        mock_node.meta["val"].dtype = torch.bfloat16

        # Test the function
        result = should_quantize(mock_node)

        # Assert
        self.assertTrue(result)
        mock_is_node_meta_valid.assert_called_once_with(mock_node)
        mock_calculate_tensor_size.assert_called_once_with(mock_node.meta["val"])

    @mock.patch("torch._functorch.partitioners.is_node_meta_valid")
    @mock.patch("torch._functorch.partitioners.calculate_tensor_size")
    def test_should_quantize_below_threshold(
        self, mock_calculate_tensor_size, mock_is_node_meta_valid
    ):
        # Setup mocks
        mock_is_node_meta_valid.return_value = True
        mock_calculate_tensor_size.return_value = 40  # Less than threshold (50)

        # Create a mock node
        mock_node = mock.MagicMock()
        mock_node.meta = {"val": mock.MagicMock()}
        mock_node.meta["val"].dtype = torch.bfloat16

        # Test the function
        result = should_quantize(mock_node)

        # Assert
        self.assertFalse(result)

    @mock.patch("torch._functorch.partitioners.is_node_meta_valid")
    @mock.patch("torch._functorch.partitioners.calculate_tensor_size")
    @mock.patch("torch._functorch.partitioners.statically_known_true")
    def test_should_quantize_with_skip_dynamo_guards(
        self,
        mock_statically_known_true,
        mock_calculate_tensor_size,
        mock_is_node_meta_valid,
    ):
        # Setup config
        torch._inductor.config.post_grad_fusion_options[
            "activation_quantization_aten_pass"
        ]["skip_dynamo_guards"] = True

        # Setup mocks
        mock_is_node_meta_valid.return_value = True
        mock_calculate_tensor_size.return_value = 60  # Greater than threshold (50)
        mock_statically_known_true.return_value = True

        # Create a mock node
        mock_node = mock.MagicMock()
        mock_node.meta = {"val": mock.MagicMock()}
        mock_node.meta["val"].dtype = torch.bfloat16

        # Test the function
        result = should_quantize(mock_node)

        # Assert
        self.assertTrue(result)
        mock_statically_known_true.assert_called_once_with(
            mock_calculate_tensor_size.return_value >= 50
        )

    @mock.patch("torch._functorch.partitioners.is_node_meta_valid")
    @mock.patch("torch._functorch.partitioners.calculate_tensor_size")
    @mock.patch("torch._functorch.partitioners.statically_known_true")
    @mock.patch("torch._functorch.partitioners.statically_known_false")
    def test_should_quantize_with_dynamic_shape(
        self,
        mock_statically_known_false,
        mock_statically_known_true,
        mock_calculate_tensor_size,
        mock_is_node_meta_valid,
    ):
        # Setup config
        torch._inductor.config.post_grad_fusion_options[
            "activation_quantization_aten_pass"
        ]["skip_dynamo_guards"] = True
        torch._inductor.config.post_grad_fusion_options[
            "activation_quantization_aten_pass"
        ]["quantize_dynamic_shape"] = True

        # Setup mocks
        mock_is_node_meta_valid.return_value = True
        mock_calculate_tensor_size.return_value = 40  # Less than threshold (50)
        mock_statically_known_true.return_value = False
        mock_statically_known_false.return_value = (
            False  # Not statically known to be false
        )

        # Create a mock node
        mock_node = mock.MagicMock()
        mock_node.meta = {"val": mock.MagicMock()}
        mock_node.meta["val"].dtype = torch.bfloat16

        # Test the function
        result = should_quantize(mock_node)

        # Assert
        self.assertTrue(result)  # Should be True because not statically_known_false
        mock_statically_known_true.assert_called_once_with(
            mock_calculate_tensor_size.return_value >= 50
        )
        mock_statically_known_false.assert_called_once_with(
            mock_calculate_tensor_size.return_value >= 50
        )

    @mock.patch("torch._functorch.partitioners.is_node_meta_valid")
    @mock.patch("torch._functorch.partitioners.calculate_tensor_size")
    @mock.patch("torch._functorch.partitioners.statically_known_true")
    @mock.patch("torch._functorch.partitioners.statically_known_false")
    def test_should_quantize_without_dynamic_shape(
        self,
        mock_statically_known_false,
        mock_statically_known_true,
        mock_calculate_tensor_size,
        mock_is_node_meta_valid,
    ):
        # Setup config
        torch._inductor.config.post_grad_fusion_options[
            "activation_quantization_aten_pass"
        ]["skip_dynamo_guards"] = True
        torch._inductor.config.post_grad_fusion_options[
            "activation_quantization_aten_pass"
        ]["quantize_dynamic_shape"] = False

        # Setup mocks
        mock_is_node_meta_valid.return_value = True
        mock_calculate_tensor_size.return_value = 60  # Greater than threshold (50)
        mock_statically_known_true.return_value = True

        # Create a mock node
        mock_node = mock.MagicMock()
        mock_node.meta = {"val": mock.MagicMock()}
        mock_node.meta["val"].dtype = torch.bfloat16

        # Test the function
        result = should_quantize(mock_node)

        # Assert
        self.assertTrue(result)
        mock_statically_known_true.assert_called_once_with(
            mock_calculate_tensor_size.return_value >= 50
        )
        # statically_known_false should not be called in this case
        mock_statically_known_false.assert_not_called()


if __name__ == "__main__":
    unittest.main()
