import copy
import itertools
from typing import Optional

import torch
import torch.nn as nn

from torch._dynamo.utils import detect_fake_mode
from torch.fx.experimental.optimization import replace_node_module
from torch.fx.experimental.symbolic_shapes import free_symbols
from torch.fx.passes.shape_prop import ShapeProp
from . import config


class PackedLSTM(nn.LSTM):
    def __init__(
        self,
        lstm: nn.Module,
        input_size: Optional[list],
    ):
        super().__init__(
            lstm.input_size,
            lstm.hidden_size,
            lstm.num_layers,
            lstm.bias,
            lstm.batch_first,
            lstm.dropout,
            lstm.bidirectional,
            lstm.proj_size,
            lstm.weight_ih_l0.device,
            lstm.weight_ih_l0.dtype,
        )
        self._update_module_params(lstm, input_size)
        self.forward_op = torch.ops.mkldnn._lstm

    def _update_module_params(self, lstm, input_size):
        self.__dict__ = copy.deepcopy(lstm.__dict__)
        packed_flat_weights = torch.ops.mkldnn._reorder_lstm_weight(
            self._flat_weights,
            self.input_size,
            self.hidden_size,
            self.bias,
            self.num_layers,
            self.bidirectional,
            self.batch_first,
            input_size,
        )
        assert len(packed_flat_weights) == len(self._flat_weights_names)
        for i, (name, tensor) in enumerate(
            zip(self._flat_weights_names, packed_flat_weights)
        ):
            setattr(
                self,
                name,
                torch.nn.Parameter(
                    tensor, requires_grad=self._flat_weights[i].requires_grad
                ),
            )


def packed_lstm_eval(lstm: nn.Module, input_size: Optional[list]):
    assert not (lstm.training), "Fusion only for eval!"
    return PackedLSTM(lstm, input_size)


def mkldnn_fuse_fx(gm: torch.fx.GraphModule, example_inputs):
    is_cpu = all(
        example_input.device == torch.device("cpu")
        for example_input in example_inputs
        if isinstance(example_input, torch.Tensor)
    )

    # make sure the autograd and autocast are disabled.
    if torch.is_grad_enabled() or torch.is_autocast_cpu_enabled():
        return gm
    if not (torch.backends.mkldnn.enabled and torch.backends.mkldnn.is_available()):
        return gm
    if not is_cpu:
        return gm
    fake_mode = detect_fake_mode(example_inputs)
    # NB: free_symbols test here is a BIG hammer.  ShapeProp doesn't
    # work with symbolic shapes though, see
    # https://github.com/pytorch/pytorch/pull/103512
    if config.cpp.weight_prepack:
        if not any(free_symbols(e) for e in example_inputs):
            ShapeProp(gm, fake_mode=fake_mode).propagate(*example_inputs)
        gm = pack_module(gm)
    return gm


def pack_module(gm: torch.fx.GraphModule):
    modules = dict(gm.named_modules())
    for node in gm.graph.nodes:
        if node.op == "call_module":
            assert isinstance(node.target, str)
            cur_module = modules[node.target]
            if type(cur_module) in computation_op_packed_map:
                if isinstance(cur_module, nn.LSTM):
                    devices = {w.device for w in cur_module._flat_weights}
                    assert (
                        len(devices) == 1
                    ), "Expect lstm weight to be on the same device"
                    device = devices.pop()

                    dtypes = {w.dtype for w in cur_module._flat_weights}
                    assert len(dtypes) == 1, "Expect lstm weight to be the same dtype"
                    dtype = dtypes.pop()

                    shapes = itertools.chain(
                        [w.shape for w in cur_module._flat_weights]
                    )
                else:
                    device = cur_module.weight.device
                    dtype = cur_module.weight.dtype
                    shapes = cur_module.weight.shape

                if (
                    device != torch.device("cpu")
                    or dtype not in [torch.bfloat16, torch.float32]
                    or any(size == 0 for size in shapes)
                ):
                    continue
                if cur_module.training:
                    continue
                if (
                    dtype == torch.bfloat16
                    and not torch.ops.mkldnn._is_mkldnn_bf16_supported()
                ):
                    continue
                if node.args[0].meta.get("tensor_meta") is None:
                    computation_node_input_size = None
                    # Conv2d and ConvTranspose2d weight format are dependent on input size,
                    # but ShapeProp may be failed to get the input size, so we skip them.
                    if type(cur_module) not in [torch.nn.LSTM]:
                        continue
                else:
                    computation_node_input_size = tuple(
                        int(x) for x in node.args[0].meta.get("tensor_meta").shape
                    )
                    if any(size == 0 for size in computation_node_input_size):
                        continue
                    if type(cur_module) in [nn.LSTM]:
                        # pack_padded_sequence input is not supported.
                        # For pack_padded_sequence input, the len(computation_node_input_size) == 4
                        if len(computation_node_input_size) not in [2, 3]:
                            continue
                new_module = computation_op_packed_map[type(cur_module)](
                    cur_module, computation_node_input_size
                )
                assert isinstance(new_module, nn.Module)
                replace_node_module(node, modules, new_module)
    gm.graph.lint()
    gm.recompile()
    return gm


computation_op_packed_map = {
    nn.LSTM: packed_lstm_eval,
}
