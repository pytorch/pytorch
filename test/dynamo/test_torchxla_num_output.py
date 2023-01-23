# Owner(s): ["module: dynamo"]
import unittest

import torch
from torch import nn
from torch._dynamo.optimizations.torchxla_integration import GraphInputMatcher
from torch.utils._pytree import tree_map_only

try:
    from .test_torchxla_util import maybe_skip_torchxla_test
except ImportError:
    from test_torchxla_util import maybe_skip_torchxla_test

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
except ImportError:
    # tests using torch_xla will be skipped. It's fine to ignore the
    # importing error here.
    pass


class DirectReturnModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b, c):
        """
        The XLA graph will only return the first 2 items
        """
        return a + b, a + c, b

    def get_example_inputs(self):
        return (torch.rand(2), torch.rand(2), torch.rand(2))


class DirectReturnWithInplaceUpdateModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b, c):
        """
        Inplace update on b cause it to be returned in XLA graph
        """
        b.zero_()
        return a + b, a + c, b

    def get_example_inputs(self):
        return (torch.rand(2), torch.rand(2), torch.rand(2))


class DirectReturnWithDuplicatedInplaceUpdateModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b, c):
        """
        Even if we return b twice, the XLA graph only return b once.
        """
        b.zero_()
        return a + b, a + c, b, b

    def get_example_inputs(self):
        return (torch.rand(2), torch.rand(2), torch.rand(2))


class TestNumOutput(unittest.TestCase):
    def do_test(self, model_class, expected_num_output):
        xla_dev = xm.xla_device()
        model = model_class().to(device=xla_dev)
        inputs = tree_map_only(
            torch.Tensor, lambda x: x.to(device=xla_dev), model.get_example_inputs()
        )

        xm.mark_step()
        args_tensor_ids = [
            torch_xla._XLAC._xla_get_tensor_id(xla_arg) for xla_arg in inputs
        ]
        tensor_id_to_arg_idx = {
            tensor_id: i for i, tensor_id in enumerate(args_tensor_ids)
        }
        outputs = model(*inputs)
        xla_graph_hash = torch_xla._XLAC._get_graph_hash(outputs)

        (
            graph_input_tensor_ids,
            graph_input_xla_values,
        ) = torch_xla._XLAC._get_tensors_xla_device_data_node(outputs)

        graph_input_matcher = GraphInputMatcher(
            tensor_id_to_arg_idx, graph_input_tensor_ids, graph_input_xla_values
        )
        torch_xla._XLAC._xla_sync_multi(outputs, [])

        def run_cached_graph(*inputs):
            torch_xla._XLAC._xla_sync_multi(inputs, [])
            xla_graph_inputs = graph_input_matcher(inputs)
            xla_graph_outputs = torch_xla._XLAC._run_cached_graph(
                xla_graph_hash, xla_graph_inputs
            )
            return xla_graph_outputs

        test_inputs = tree_map_only(
            torch.Tensor, lambda x: x.to(device=xla_dev), model.get_example_inputs()
        )
        self.assertEqual(expected_num_output, len(run_cached_graph(*test_inputs)))

    @maybe_skip_torchxla_test
    def test_direct_return(self):
        self.do_test(DirectReturnModule, expected_num_output=2)

    @maybe_skip_torchxla_test
    def test_direct_return_with_inplace_update(self):
        self.do_test(DirectReturnWithInplaceUpdateModule, expected_num_output=3)

    @maybe_skip_torchxla_test
    def test_direct_return_with_duplicated_inplace_update(self):
        self.do_test(
            DirectReturnWithDuplicatedInplaceUpdateModule, expected_num_output=3
        )
