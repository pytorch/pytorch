from collections import namedtuple
from typing import List

import torch
from torch import Tensor

from .cells import flat_lstm_cell, lstm_cell, premul_lstm_cell, premul_lstm_cell_no_bias


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
    assert isinstance(output, torch.Tensor)
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


def lstm_creator(script=True, **kwargs):
    input, hidden, params, _ = lstm_inputs(return_module=False, **kwargs)
    inputs = [input, hidden] + params[0]
    return ModelDef(
        inputs=inputs,
        params=flatten_list(params),
        forward=lstm_factory(lstm_cell, script),
        backward_setup=lstm_backward_setup,
        backward=simple_backward,
    )


def lnlstm_creator(script=True, decompose_layernorm=False, **kwargs):
    assert script is True
    from .custom_lstms import script_lnlstm

    input_size = kwargs["inputSize"]
    hidden_size = kwargs["hiddenSize"]
    seq_len = kwargs["seqLength"]
    batch_size = kwargs["miniBatch"]
    ge = script_lnlstm(
        input_size, hidden_size, 1, decompose_layernorm=decompose_layernorm
    ).cuda()

    input = torch.randn(seq_len, batch_size, input_size, device="cuda")
    states = [
        (
            torch.randn(batch_size, hidden_size, device="cuda"),
            torch.randn(batch_size, hidden_size, device="cuda"),
        )
    ]

    return ModelDef(
        inputs=[input, states],
        params=ge.parameters(),
        forward=ge,
        backward_setup=lstm_backward_setup,
        backward=simple_backward,
    )


def dropoutlstm_creator(script=True, **kwargs):
    assert script is True
    from .custom_lstms import LSTMState, script_lstm

    input_size = kwargs["inputSize"]
    hidden_size = kwargs["hiddenSize"]
    seq_len = kwargs["seqLength"]
    batch_size = kwargs["miniBatch"]
    num_layers = kwargs["numLayers"]
    ge = script_lstm(input_size, hidden_size, num_layers, dropout=True).cuda()

    input = torch.randn(seq_len, batch_size, input_size, device="cuda")
    states = [
        LSTMState(
            torch.randn(batch_size, hidden_size, device="cuda"),
            torch.randn(batch_size, hidden_size, device="cuda"),
        )
        for _ in range(num_layers)
    ]
    return ModelDef(
        inputs=[input, states],
        params=ge.parameters(),
        forward=ge,
        backward_setup=lstm_backward_setup,
        backward=simple_backward,
    )


def lstm_premul_creator(script=True, **kwargs):
    input, hidden, params, _ = lstm_inputs(return_module=False, **kwargs)
    inputs = [input, hidden] + params[0]
    return ModelDef(
        inputs=inputs,
        params=flatten_list(params),
        forward=lstm_factory_premul(premul_lstm_cell, script),
        backward_setup=lstm_backward_setup,
        backward=simple_backward,
    )


def lstm_premul_bias_creator(script=True, **kwargs):
    input, hidden, params, _ = lstm_inputs(return_module=False, **kwargs)
    inputs = [input, hidden] + params[0]
    return ModelDef(
        inputs=inputs,
        params=flatten_list(params),
        forward=lstm_factory_premul_bias(premul_lstm_cell_no_bias, script),
        backward_setup=lstm_backward_setup,
        backward=simple_backward,
    )


def lstm_simple_creator(script=True, **kwargs):
    input, hidden, params, _ = lstm_inputs(return_module=False, **kwargs)
    inputs = [input] + [h[0] for h in hidden] + params[0]
    return ModelDef(
        inputs=inputs,
        params=flatten_list(params),
        forward=lstm_factory_simple(flat_lstm_cell, script),
        backward_setup=lstm_backward_setup,
        backward=simple_backward,
    )


def lstm_multilayer_creator(script=True, **kwargs):
    input, hidden, params, _ = lstm_inputs(return_module=False, **kwargs)
    inputs = [input, hidden, flatten_list(params)]
    return ModelDef(
        inputs=inputs,
        params=flatten_list(params),
        forward=lstm_factory_multilayer(lstm_cell, script),
        backward_setup=lstm_backward_setup,
        backward=simple_backward,
    )


def imagenet_cnn_creator(arch, jit=True):
    def creator(device="cuda", **kwargs):
        model = arch().to(device)
        x = torch.randn(32, 3, 224, 224, device=device)
        if jit:
            model = torch.jit.trace(model, x)
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
        # wih, whh, bih, bhh = lstm.all_weights[layer]
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


def varlen_lstm_factory(cell, script):
    def dynamic_rnn(
        sequences: List[Tensor],
        hiddens: tuple[Tensor, Tensor],
        wih: Tensor,
        whh: Tensor,
        bih: Tensor,
        bhh: Tensor,
    ) -> tuple[List[Tensor], tuple[List[Tensor], List[Tensor]]]:
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
                    inputs[seq_idx].unsqueeze(0), (hy, cy), wih, whh, bih, bhh
                )
                output += [hy]
            outputs += [torch.stack(output)]
            hx_outs += [hy.unsqueeze(0)]
            cx_outs += [cy.unsqueeze(0)]

        return outputs, (hx_outs, cx_outs)

    if script:
        cell = torch.jit.script(cell)
        dynamic_rnn = torch.jit.script(dynamic_rnn)

    return dynamic_rnn


def varlen_lstm_creator(script=False, **kwargs):
    sequences, _, hidden, params, _ = varlen_lstm_inputs(return_module=False, **kwargs)
    inputs = [sequences, hidden] + params[0]
    return ModelDef(
        inputs=inputs,
        params=flatten_list(params),
        forward=varlen_lstm_factory(lstm_cell, script),
        backward_setup=varlen_lstm_backward_setup,
        backward=simple_backward,
    )


# cudnn_layernorm_lstm: since cudnn does not have Layernorm LSTM, we cannot benchmark
# the lowerbound directly. Instead, we only benchmark the forward pass by mimicing the
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


# input: lstm.all_weights format (wih, whh, bih, bhh = lstm.all_weights[layer])
# output: packed_weights with format
# packed_weights[0] is wih with size (layer, 4*hiddenSize, inputSize)
# packed_weights[1] is whh with size (layer, 4*hiddenSize, hiddenSize)
# packed_weights[2] is bih with size (layer, 4*hiddenSize)
# packed_weights[3] is bhh with size (layer, 4*hiddenSize)
def stack_weights(weights):
    def unzip_columns(mat):
        assert isinstance(mat, list)
        assert isinstance(mat[0], list)
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
        # wih, whh, bih, bhh = lstm.all_weights[layer]
        return x, (hx, cx), lstm.all_weights, None


def lstm_factory(cell, script):
    def dynamic_rnn(
        input: Tensor,
        hidden: tuple[Tensor, Tensor],
        wih: Tensor,
        whh: Tensor,
        bih: Tensor,
        bhh: Tensor,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        hx, cx = hidden
        outputs = []
        inputs = input.unbind(0)
        hy, cy = hx[0], cx[0]
        for seq_idx in range(len(inputs)):
            hy, cy = cell(inputs[seq_idx], (hy, cy), wih, whh, bih, bhh)
            outputs += [hy]
        return torch.stack(outputs), (hy.unsqueeze(0), cy.unsqueeze(0))

    if script:
        cell = torch.jit.script(cell)
        dynamic_rnn = torch.jit.script(dynamic_rnn)

    return dynamic_rnn


# premul: we're going to premultiply the inputs & weights
def lstm_factory_premul(premul_cell, script):
    def dynamic_rnn(
        input: Tensor,
        hidden: tuple[Tensor, Tensor],
        wih: Tensor,
        whh: Tensor,
        bih: Tensor,
        bhh: Tensor,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        hx, cx = hidden
        outputs = []
        inputs = torch.matmul(input, wih.t()).unbind(0)
        hy, cy = hx[0], cx[0]
        for seq_idx in range(len(inputs)):
            hy, cy = premul_cell(inputs[seq_idx], (hy, cy), whh, bih, bhh)
            outputs += [hy]
        return torch.stack(outputs), (hy.unsqueeze(0), cy.unsqueeze(0))

    if script:
        premul_cell = torch.jit.script(premul_cell)
        dynamic_rnn = torch.jit.script(dynamic_rnn)

    return dynamic_rnn


# premul: we're going to premultiply the inputs & weights, and add bias
def lstm_factory_premul_bias(premul_cell, script):
    def dynamic_rnn(
        input: Tensor,
        hidden: tuple[Tensor, Tensor],
        wih: Tensor,
        whh: Tensor,
        bih: Tensor,
        bhh: Tensor,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        hx, cx = hidden
        outputs = []
        inpSize = input.size()
        # add bias for all timesteps instead of going step-by-step, results in a single reduction kernel in the backward
        # FIXME matmul(x,y) + bias currently goes through jit AD, and backward formula in AD is not optimized for this
        # case. Workaround with mm and views.
        inpSize = input.size()
        inputs = torch.mm(input.view(-1, inpSize[2]), wih.t()) + bih
        inputs = inputs.view(inpSize[0], inpSize[1], -1).unbind(0)
        hy, cy = hx[0], cx[0]
        for seq_idx in range(len(inputs)):
            hy, cy = premul_cell(inputs[seq_idx], (hy, cy), whh, bhh)
            outputs += [hy]
        return torch.stack(outputs), (hy.unsqueeze(0), cy.unsqueeze(0))

    if script:
        premul_cell = torch.jit.script(premul_cell)
        dynamic_rnn = torch.jit.script(dynamic_rnn)

    return dynamic_rnn


# simple: flat inputs (no tuples), no list to accumulate outputs
#         useful mostly for benchmarking older JIT versions
def lstm_factory_simple(cell, script):
    def dynamic_rnn(input, hx, cx, wih, whh, bih, bhh):
        hy = hx  # for scoping
        cy = cx  # for scoping
        inputs = input.unbind(0)
        for seq_idx in range(len(inputs)):
            hy, cy = cell(inputs[seq_idx], hy, cy, wih, whh, bih, bhh)
        return hy, cy

    if script:
        cell = torch.jit.script(cell)
        dynamic_rnn = torch.jit.script(dynamic_rnn)

    return dynamic_rnn


def lstm_factory_multilayer(cell, script):
    def dynamic_rnn(
        input: Tensor, hidden: tuple[Tensor, Tensor], params: List[Tensor]
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        params_stride = 4  # NB: this assumes that biases are there
        hx, cx = hidden
        hy, cy = hidden  # for scoping...
        inputs, outputs = input.unbind(0), []
        for layer in range(hx.size(0)):
            hy = hx[layer]
            cy = cx[layer]
            base_idx = layer * params_stride
            wih = params[base_idx]
            whh = params[base_idx + 1]
            bih = params[base_idx + 2]
            bhh = params[base_idx + 3]
            for seq_idx in range(len(inputs)):
                hy, cy = cell(inputs[seq_idx], (hy, cy), wih, whh, bih, bhh)
                outputs += [hy]
            inputs, outputs = outputs, []
        return torch.stack(inputs), (hy.unsqueeze(0), cy.unsqueeze(0))

    if script:
        cell = torch.jit.script(cell)
        dynamic_rnn = torch.jit.script(dynamic_rnn)

    return dynamic_rnn
