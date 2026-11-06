from collections import namedtuple

import torch
from torch import Tensor

from .cells import lstm_cell


# list[list[T]] -> list[T]
def flatten_list(lst):
    result = []
    for inner in lst:
        result.extend(inner)
    return result


"""
Define a creator as a function:
(options) -> (inputs, params, forward, backward_setup, backward)
inputs: the inputs to the returned 'forward'. One can call
    forward(*inputs) directly.
params: List[Tensor] all requires_grad=True parameters.
forward: function / graph executor / module
    One can call rnn(rnn_inputs) using the outputs of the creator.
backward_setup: backward_inputs = backward_setup(*outputs)
    Then, we pass backward_inputs to backward. If None, then it is assumed to
    be the identity function.
backward: Given `output = backward_setup(*forward(*inputs))`, performs
    backpropagation. If None, then nothing happens.

fastrnns.bench times the forward and backward invocations.
"""


ModelDef = namedtuple(
    "ModelDef", ["inputs", "params", "forward", "backward_setup", "backward"]
)


def lstm_backward_setup(lstm_outputs, seed=None):
    hx, _ = lstm_outputs
    return simple_backward_setup(hx, seed)


def simple_backward_setup(output, seed=None):
    if not isinstance(output, torch.Tensor):
        raise AssertionError(f"Expected output to be a Tensor, but got {type(output)}")
    if seed:
        torch.manual_seed(seed)
    grad_output = torch.randn_like(output)
    return output, grad_output


def simple_backward(output, grad_output, **kwargs):
    return output.backward(grad_output, **kwargs)


def pytorch_lstm_creator(**kwargs):
    input, hidden, _, module = lstm_inputs(return_module=True, **kwargs)
    return ModelDef(
        inputs=[input, hidden],
        params=flatten_list(module.all_weights),
        forward=module,
        backward_setup=lstm_backward_setup,
        backward=simple_backward,
    )


def lstm_creator(**kwargs):
    input, hidden, params, _ = lstm_inputs(return_module=False, **kwargs)
    inputs = [input, hidden] + params[0]
    return ModelDef(
        inputs=inputs,
        params=flatten_list(params),
        forward=lstm_factory(lstm_cell),
        backward_setup=lstm_backward_setup,
        backward=simple_backward,
    )


def imagenet_cnn_creator(arch):
    def creator(device="cuda", **kwargs):
        model = arch().to(device)
        x = torch.randn(32, 3, 224, 224, device=device)
        return ModelDef(
            inputs=(x,),
            params=list(model.parameters()),
            forward=model,
            backward_setup=simple_backward_setup,
            backward=simple_backward,
        )

    return creator


def varlen_lstm_inputs(
    minlen=30,
    maxlen=100,
    numLayers=1,
    inputSize=512,
    hiddenSize=512,
    miniBatch=64,
    return_module=False,
    device="cuda",
    seed=None,
    **kwargs,
):
    if seed is not None:
        torch.manual_seed(seed)
    lengths = torch.randint(
        low=minlen, high=maxlen, size=[miniBatch], dtype=torch.long, device=device
    )
    x = [torch.randn(length, inputSize, device=device) for length in lengths]
    hx = torch.randn(numLayers, miniBatch, hiddenSize, device=device)
    cx = torch.randn(numLayers, miniBatch, hiddenSize, device=device)
    lstm = torch.nn.LSTM(inputSize, hiddenSize, numLayers).to(device)

    if return_module:
        return x, lengths, (hx, cx), lstm.all_weights, lstm
    else:
        # NB: lstm.all_weights format:
        # w_ih, w_hh, b_ih, b_hh = lstm.all_weights[layer]
        return x, lengths, (hx, cx), lstm.all_weights, None


def varlen_lstm_backward_setup(forward_output, seed=None):
    if seed:
        torch.manual_seed(seed)
    rnn_utils = torch.nn.utils.rnn
    sequences = forward_output[0]
    padded = rnn_utils.pad_sequence(sequences)
    grad = torch.randn_like(padded)
    return padded, grad


def varlen_pytorch_lstm_creator(**kwargs):
    rnn_utils = torch.nn.utils.rnn
    sequences, _, hidden, _, module = varlen_lstm_inputs(return_module=True, **kwargs)

    def forward(sequences, hidden):
        packed = rnn_utils.pack_sequence(sequences, enforce_sorted=False)
        out, new_hidden = module(packed, hidden)
        padded, lengths = rnn_utils.pad_packed_sequence(out)
        # XXX: It's more efficient to store the output in its padded form,
        # but that might not be conducive to loss computation.
        # Un-padding the output also makes the backward pass 2x slower...
        # return [padded[:lengths[i], i, :] for i in range(lengths.size(0))]
        return padded, new_hidden

    return ModelDef(
        inputs=[sequences, hidden],
        params=flatten_list(module.all_weights),
        forward=forward,
        backward_setup=lstm_backward_setup,
        backward=simple_backward,
    )


def varlen_lstm_factory(cell):
    def dynamic_rnn(
        sequences: list[Tensor],
        hiddens: tuple[Tensor, Tensor],
        w_ih: Tensor,
        w_hh: Tensor,
        b_ih: Tensor,
        b_hh: Tensor,
    ) -> tuple[list[Tensor], tuple[list[Tensor], list[Tensor]]]:
        hx, cx = hiddens
        hxs = hx.unbind(1)
        cxs = cx.unbind(1)
        # List of: (output, hx, cx)
        outputs = []
        hx_outs = []
        cx_outs = []

        for batch in range(len(sequences)):
            output = []
            hy, cy = hxs[batch], cxs[batch]
            inputs = sequences[batch].unbind(0)

            for seq_idx in range(len(inputs)):
                hy, cy = cell(
                    inputs[seq_idx].unsqueeze(0), (hy, cy), w_ih, w_hh, b_ih, b_hh
                )
                output += [hy]
            outputs += [torch.stack(output)]
            hx_outs += [hy.unsqueeze(0)]
            cx_outs += [cy.unsqueeze(0)]

        return outputs, (hx_outs, cx_outs)

    return dynamic_rnn


def varlen_lstm_creator(**kwargs):
    if "script" in kwargs:
        raise TypeError("varlen_lstm_creator() does not support 'script'")
    sequences, _, hidden, params, _ = varlen_lstm_inputs(return_module=False, **kwargs)
    inputs = [sequences, hidden] + params[0]
    return ModelDef(
        inputs=inputs,
        params=flatten_list(params),
        forward=varlen_lstm_factory(lstm_cell),
        backward_setup=varlen_lstm_backward_setup,
        backward=simple_backward,
    )


# cudnn_layernorm_lstm: since cudnn does not have Layernorm LSTM, we cannot benchmark
# the lowerbound directly. Instead, we only benchmark the forward pass by mimicking the
# computation of a cudnn lstm + seq_len * 3 layernorm computation. This should serve
# as a perf lowerbound for the Layernorm LSTM forward pass(given that Layernorm itself
# is invariant), the lowerbound of backward pass is hard to get since we lose the
# intermediate results, we can still optimize the layernorm implementation to make
# a faster forward lowerbound though.
def layernorm_pytorch_lstm_creator(**kwargs):
    input, hidden, _, module = lstm_inputs(return_module=True, **kwargs)
    batch_size = kwargs["miniBatch"]
    hidden_size = kwargs["hiddenSize"]
    ln_i = torch.nn.LayerNorm(4 * hidden_size).cuda()
    ln_h = torch.nn.LayerNorm(4 * hidden_size).cuda()
    ln_c = torch.nn.LayerNorm(hidden_size).cuda()
    ln_input1 = torch.randn(batch_size, 4 * hidden_size, device="cuda")

    def forward(input, hidden):
        out, new_hidden = module(input, hidden)
        # plus (seq_len * three laynorm cell computation) to mimic the lower bound of
        # Layernorm cudnn LSTM in the forward pass
        seq_len = len(input.unbind(0))
        hy, cy = new_hidden
        for i in range(seq_len):
            ln_i(ln_input1)
            ln_h(ln_input1)
            cy = ln_c(cy)

        return out, (hy, cy)

    return ModelDef(
        inputs=[input, hidden],
        params=flatten_list(module.all_weights),
        forward=forward,
        backward_setup=lstm_backward_setup,
        backward=None,
    )


# input: lstm.all_weights format (w_ih, w_hh, b_ih, b_hh = lstm.all_weights[layer])
# output: packed_weights with format
# packed_weights[0] is w_ih with size (layer, 4*hiddenSize, inputSize)
# packed_weights[1] is w_hh with size (layer, 4*hiddenSize, hiddenSize)
# packed_weights[2] is b_ih with size (layer, 4*hiddenSize)
# packed_weights[3] is b_hh with size (layer, 4*hiddenSize)
def stack_weights(weights):
    def unzip_columns(mat):
        if not isinstance(mat, list):
            raise AssertionError(f"Expected mat to be a list, but got {type(mat)}")
        if not isinstance(mat[0], list):
            raise AssertionError(
                f"Expected mat[0] to be a list, but got {type(mat[0])}"
            )
        layers = len(mat)
        columns = len(mat[0])
        return [[mat[layer][col] for layer in range(layers)] for col in range(columns)]

    # XXX: script fns have problems indexing multidim lists, so we try to
    # avoid them by stacking tensors
    all_weights = weights
    packed_weights = [torch.stack(param) for param in unzip_columns(all_weights)]
    return packed_weights


# returns: x, (hx, cx), all_weights, lstm module with all_weights as params
def lstm_inputs(
    seqLength=100,
    numLayers=1,
    inputSize=512,
    hiddenSize=512,
    miniBatch=64,
    dropout=0.0,
    return_module=False,
    device="cuda",
    seed=None,
):
    if seed is not None:
        torch.manual_seed(seed)
    x = torch.randn(seqLength, miniBatch, inputSize, device=device)
    hx = torch.randn(numLayers, miniBatch, hiddenSize, device=device)
    cx = torch.randn(numLayers, miniBatch, hiddenSize, device=device)
    lstm = torch.nn.LSTM(inputSize, hiddenSize, numLayers, dropout=dropout)
    if "cuda" in device:
        lstm = lstm.cuda()

    if return_module:
        return x, (hx, cx), lstm.all_weights, lstm
    else:
        # NB: lstm.all_weights format:
        # w_ih, w_hh, b_ih, b_hh = lstm.all_weights[layer]
        return x, (hx, cx), lstm.all_weights, None


def lstm_factory(cell):
    def dynamic_rnn(
        input: Tensor,
        hidden: tuple[Tensor, Tensor],
        w_ih: Tensor,
        w_hh: Tensor,
        b_ih: Tensor,
        b_hh: Tensor,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        hx, cx = hidden
        outputs = []
        inputs = input.unbind(0)
        hy, cy = hx[0], cx[0]
        for seq_idx in range(len(inputs)):
            hy, cy = cell(inputs[seq_idx], (hy, cy), w_ih, w_hh, b_ih, b_hh)
            outputs += [hy]
        return torch.stack(outputs), (hy.unsqueeze(0), cy.unsqueeze(0))

    return dynamic_rnn


# premul: we're going to premultiply the inputs & weights


# premul: we're going to premultiply the inputs & weights, and add bias


# simple: flat inputs (no tuples), no list to accumulate outputs
#         useful mostly for benchmarking older JIT versions
