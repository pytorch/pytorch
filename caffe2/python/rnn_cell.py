## @package rnn_cell
# Module caffe2.python.rnn_cell
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import random

from caffe2.python.attention import (
    AttentionType,
    apply_regular_attention,
    apply_recurrent_attention,
)
from caffe2.python import core, recurrent, workspace
from caffe2.python.cnn import CNNModelHelper


class RNNCell(object):
    '''
    Base class for writing recurrent / stateful operations.

    One needs to implement 3 methods: _apply, prepare_input and get_state_names.
    As a result base class will provice apply_over_sequence method, which
    allows you to apply recurrent operations over a sequence of any length.
    '''
    def __init__(self, name):
        self.name = name
        self.recompute_blobs = []

    def scope(self, name):
        return self.name + '/' + name if self.name is not None else name

    def apply_over_sequence(
        self,
        model,
        inputs,
        seq_lengths,
        initial_states,
        outputs_with_grads=None,
    ):
        preprocessed_inputs = self.prepare_input(model, inputs)
        step_model = CNNModelHelper(name=self.name, param_model=model)
        input_t, timestep = step_model.net.AddScopedExternalInputs(
            'input_t',
            'timestep',
        )
        states_prev = step_model.net.AddScopedExternalInputs(*[
            s + '_prev' for s in self.get_state_names()
        ])
        states = self._apply(
            model=step_model,
            input_t=input_t,
            seq_lengths=seq_lengths,
            states=states_prev,
            timestep=timestep,
        )
        return recurrent.recurrent_net(
            net=model.net,
            cell_net=step_model.net,
            inputs=[(input_t, preprocessed_inputs)],
            initial_cell_inputs=zip(states_prev, initial_states),
            links=dict(zip(states_prev, states)),
            timestep=timestep,
            scope=self.name,
            outputs_with_grads=(
                outputs_with_grads
                if outputs_with_grads is not None
                else self.get_outputs_with_grads()
            ),
            recompute_blobs_on_backward=self.recompute_blobs,
        )

    def apply(self, model, input_t, seq_lengths, states, timestep):
        input_t = self.prepare_input(model, input_t)
        return self._apply(model, input_t, seq_lengths, states, timestep)

    def _apply(self, model, input_t, seq_lengths, states, timestep):
        '''
        A single step of a recurrent network.

        model: CNNModelHelper object new operators would be added to

        input_blob: single input with shape (1, batch_size, input_dim)

        seq_lengths: blob containing sequence lengths which would be passed to
        LSTMUnit operator

        states: previous recurrent states

        timestep: current recurrent iteration. Could be used together with
        seq_lengths in order to determine, if some shorter sequences
        in the batch have already ended.
        '''
        raise NotImplementedError('Abstract method')

    def prepare_input(self, model, input_blob):
        '''
        If some operations in _apply method depend only on the input,
        not on recurrent states, they could be computed in advance.

        model: CNNModelHelper object new operators would be added to

        input_blob: either the whole input sequence with shape
        (sequence_length, batch_size, input_dim) or a single input with shape
        (1, batch_size, input_dim).
        '''
        raise NotImplementedError('Abstract method')

    def get_state_names(self):
        '''
        Return the names of the recurrent states.
        It's required by apply_over_sequence method in order to allocate
        recurrent states for all steps with meaningful names.
        '''
        raise NotImplementedError('Abstract method')


class LSTMCell(RNNCell):

    def __init__(
        self,
        input_size,
        hidden_size,
        forget_bias,
        memory_optimization,
        name,
    ):
        super(LSTMCell, self).__init__(name)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.forget_bias = float(forget_bias)
        self.memory_optimization = memory_optimization

    def _apply(
        self,
        model,
        input_t,
        seq_lengths,
        states,
        timestep,
    ):
        hidden_t_prev, cell_t_prev = states
        gates_t = model.FC(
            hidden_t_prev,
            self.scope('gates_t'),
            dim_in=self.hidden_size,
            dim_out=4 * self.hidden_size,
            axis=2,
        )
        model.net.Sum([gates_t, input_t], gates_t)
        hidden_t, cell_t = model.net.LSTMUnit(
            [
                hidden_t_prev,
                cell_t_prev,
                gates_t,
                seq_lengths,
                timestep,
            ],
            list(self.get_state_names()),
            forget_bias=self.forget_bias,
        )
        model.net.AddExternalOutputs(hidden_t, cell_t)
        if self.memory_optimization:
            self.recompute_blobs = [gates_t]
        return hidden_t, cell_t

    def get_input_params(self):
        return {
            'weights': self.scope('i2h') + '_w',
            'biases': self.scope('i2h') + '_b',
        }

    def get_recurrent_params(self):
        return {
            'weights': self.scope('gates_t') + '_w',
            'biases': self.scope('gates_t') + '_b',
        }

    def prepare_input(self, model, input_blob):
        return model.FC(
            input_blob,
            self.scope('i2h'),
            dim_in=self.input_size,
            dim_out=4 * self.hidden_size,
            axis=2,
        )

    def get_state_names(self):
        return (self.scope('hidden_t'), self.scope('cell_t'))

    def get_outputs_with_grads(self):
        return [0]

    def get_output_size(self):
        return self.hidden_size


def LSTM(model, input_blob, seq_lengths, initial_states, dim_in, dim_out,
         scope, outputs_with_grads=(0,), return_params=False,
         memory_optimization=False, forget_bias=0.0):
    '''
    Adds a standard LSTM recurrent network operator to a model.

    model: CNNModelHelper object new operators would be added to

    input_blob: the input sequence in a format T x N x D
    where T is sequence size, N - batch size and D - input dimention

    seq_lengths: blob containing sequence lengths which would be passed to
    LSTMUnit operator

    initial_states: a tupple of (hidden_input_blob, cell_input_blob)
    which are going to be inputs to the cell net on the first iteration

    dim_in: input dimention

    dim_out: output dimention

    outputs_with_grads : position indices of output blobs which will receive
    external error gradient during backpropagation

    return_params: if True, will return a dictionary of parameters of the LSTM

    memory_optimization: if enabled, the LSTM step is recomputed on backward step
                   so that we don't need to store forward activations for each
                   timestep. Saves memory with cost of computation.
    '''
    cell = LSTMCell(
        input_size=dim_in,
        hidden_size=dim_out,
        forget_bias=forget_bias,
        memory_optimization=memory_optimization,
        name=scope,
    )
    result = cell.apply_over_sequence(
        model=model,
        inputs=input_blob,
        seq_lengths=seq_lengths,
        initial_states=initial_states,
        outputs_with_grads=outputs_with_grads,
    )
    if return_params:
        result = list(result) + [{
            'input': cell.get_input_params(),
            'recurrent': cell.get_recurrent_params(),
        }]
    return tuple(result)


def GetLSTMParamNames():
    weight_params = ["input_gate_w", "forget_gate_w", "output_gate_w", "cell_w"]
    bias_params = ["input_gate_b", "forget_gate_b", "output_gate_b", "cell_b"]
    return {'weights': weight_params, 'biases': bias_params}


def InitFromLSTMParams(lstm_pblobs, param_values):
    '''
    Set the parameters of LSTM based on predefined values
    '''
    weight_params = GetLSTMParamNames()['weights']
    bias_params = GetLSTMParamNames()['biases']
    for input_type in param_values.keys():
        weight_values = [param_values[input_type][w].flatten() for w in weight_params]
        wmat = np.array([])
        for w in weight_values:
            wmat = np.append(wmat, w)
        bias_values = [param_values[input_type][b].flatten() for b in bias_params]
        bm = np.array([])
        for b in bias_values:
            bm = np.append(bm, b)

        weights_blob = lstm_pblobs[input_type]['weights']
        bias_blob = lstm_pblobs[input_type]['biases']
        cur_weight = workspace.FetchBlob(weights_blob)
        cur_biases = workspace.FetchBlob(bias_blob)

        workspace.FeedBlob(
            weights_blob,
            wmat.reshape(cur_weight.shape).astype(np.float32))
        workspace.FeedBlob(
            bias_blob,
            bm.reshape(cur_biases.shape).astype(np.float32))


def cudnn_LSTM(model, input_blob, initial_states, dim_in, dim_out,
               scope, recurrent_params=None, input_params=None,
               num_layers=1, return_params=False):
    '''
    CuDNN version of LSTM for GPUs.
    input_blob          Blob containing the input. Will need to be available
                        when param_init_net is run, because the sequence lengths
                        and batch sizes will be inferred from the size of this
                        blob.
    initial_states      tuple of (hidden_init, cell_init) blobs
    dim_in              input dimensions
    dim_out             output/hidden dimension
    scope               namescope to apply
    recurrent_params    dict of blobs containing values for recurrent
                        gate weights, biases (if None, use random init values)
                        See GetLSTMParamNames() for format.
    input_params        dict of blobs containing values for input
                        gate weights, biases (if None, use random init values)
                        See GetLSTMParamNames() for format.
    num_layers          number of LSTM layers
    return_params       if True, returns (param_extract_net, param_mapping)
                        where param_extract_net is a net that when run, will
                        populate the blobs specified in param_mapping with the
                        current gate weights and biases (input/recurrent).
                        Useful for assigning the values back to non-cuDNN
                        LSTM.
    '''
    with core.NameScope(scope):
        weight_params = GetLSTMParamNames()['weights']
        bias_params = GetLSTMParamNames()['biases']

        input_weight_size = dim_out * dim_in
        upper_layer_input_weight_size = dim_out * dim_out
        recurrent_weight_size = dim_out * dim_out
        input_bias_size = dim_out
        recurrent_bias_size = dim_out

        def init(layer, pname, input_type):
            input_weight_size_for_layer = input_weight_size if layer == 0 else \
                upper_layer_input_weight_size
            if pname in weight_params:
                sz = input_weight_size_for_layer if input_type == 'input' \
                    else recurrent_weight_size
            elif pname in bias_params:
                sz = input_bias_size if input_type == 'input' \
                    else recurrent_bias_size
            else:
                assert False, "unknown parameter type {}".format(pname)
            return model.param_init_net.UniformFill(
                [],
                "lstm_init_{}_{}_{}".format(input_type, pname, layer),
                shape=[sz])

        # Multiply by 4 since we have 4 gates per LSTM unit
        first_layer_sz = input_weight_size + recurrent_weight_size + \
                         input_bias_size + recurrent_bias_size
        upper_layer_sz = upper_layer_input_weight_size + \
                         recurrent_weight_size + input_bias_size + \
                         recurrent_bias_size
        total_sz = 4 * (first_layer_sz + (num_layers - 1) * upper_layer_sz)

        weights = model.param_init_net.UniformFill(
            [], "lstm_weight", shape=[total_sz])

        model.params.append(weights)
        model.weights.append(weights)

        lstm_args = {
            'hidden_size': dim_out,
            'rnn_mode': 'lstm',
            'bidirectional': 0,  # TODO
            'dropout': 1.0,  # TODO
            'input_mode': 'linear',  # TODO
            'num_layers': num_layers,
            'engine': 'CUDNN'
        }

        param_extract_net = core.Net("lstm_param_extractor")
        param_extract_net.AddExternalInputs([input_blob, weights])
        param_extract_mapping = {}

        # Populate the weights-blob from blobs containing parameters for
        # the individual components of the LSTM, such as forget/input gate
        # weights and bises. Also, create a special param_extract_net that
        # can be used to grab those individual params from the black-box
        # weights blob. These results can be then fed to InitFromLSTMParams()
        for input_type in ['input', 'recurrent']:
            param_extract_mapping[input_type] = {}
            p = recurrent_params if input_type == 'recurrent' else input_params
            if p is None:
                p = {}
            for pname in weight_params + bias_params:
                for j in range(0, num_layers):
                    values = p[pname] if pname in p else init(j, pname, input_type)
                    model.param_init_net.RecurrentParamSet(
                        [input_blob, weights, values],
                        weights,
                        layer=j,
                        input_type=input_type,
                        param_type=pname,
                        **lstm_args
                    )
                    if pname not in param_extract_mapping[input_type]:
                        param_extract_mapping[input_type][pname] = {}
                    b = param_extract_net.RecurrentParamGet(
                        [input_blob, weights],
                        ["lstm_{}_{}_{}".format(input_type, pname, j)],
                        layer=j,
                        input_type=input_type,
                        param_type=pname,
                        **lstm_args
                    )
                    param_extract_mapping[input_type][pname][j] = b

        (hidden_input_blob, cell_input_blob) = initial_states
        output, hidden_output, cell_output, rnn_scratch, dropout_states = \
            model.net.Recurrent(
                [input_blob, cell_input_blob, cell_input_blob, weights],
                ["lstm_output", "lstm_hidden_output", "lstm_cell_output",
                 "lstm_rnn_scratch", "lstm_dropout_states"],
                seed=random.randint(0, 100000),  # TODO: dropout seed
                **lstm_args
            )
        model.net.AddExternalOutputs(
            hidden_output, cell_output, rnn_scratch, dropout_states)

    if return_params:
        param_extract = param_extract_net, param_extract_mapping
        return output, hidden_output, cell_output, param_extract
    else:
        return output, hidden_output, cell_output


class LSTMWithAttentionCell(RNNCell):

    def __init__(
        self,
        encoder_output_dim,
        encoder_outputs,
        decoder_input_dim,
        decoder_state_dim,
        name,
        attention_type,
        weighted_encoder_outputs,
        forget_bias,
        lstm_memory_optimization,
        attention_memory_optimization,
    ):
        super(LSTMWithAttentionCell, self).__init__(name)
        self.encoder_output_dim = encoder_output_dim
        self.encoder_outputs = encoder_outputs
        self.decoder_input_dim = decoder_input_dim
        self.decoder_state_dim = decoder_state_dim
        self.weighted_encoder_outputs = weighted_encoder_outputs
        self.encoder_outputs_transposed = None
        assert attention_type in [
            AttentionType.Regular,
            AttentionType.Recurrent,
        ]
        self.attention_type = attention_type
        self.lstm_memory_optimization = lstm_memory_optimization
        self.attention_memory_optimization = attention_memory_optimization

    def _apply(
        self,
        model,
        input_t,
        seq_lengths,
        states,
        timestep,
    ):
        (
            hidden_t_prev,
            cell_t_prev,
            attention_weighted_encoder_context_t_prev,
        ) = states

        gates_concatenated_input_t, _ = model.net.Concat(
            [hidden_t_prev, attention_weighted_encoder_context_t_prev],
            [
                self.scope('gates_concatenated_input_t'),
                self.scope('_gates_concatenated_input_t_concat_dims'),
            ],
            axis=2,
        )
        gates_t = model.FC(
            gates_concatenated_input_t,
            self.scope('gates_t'),
            dim_in=self.decoder_state_dim + self.encoder_output_dim,
            dim_out=4 * self.decoder_state_dim,
            axis=2,
        )
        model.net.Sum([gates_t, input_t], gates_t)

        hidden_t_intermediate, cell_t = model.net.LSTMUnit(
            [
                hidden_t_prev,
                cell_t_prev,
                gates_t,
                seq_lengths,
                timestep,
            ],
            ['hidden_t_intermediate', self.scope('cell_t')],
        )
        if self.attention_type == AttentionType.Recurrent:
            (
                attention_weighted_encoder_context_t,
                self.attention_weights_3d,
                attention_blobs,
            ) = apply_recurrent_attention(
                model=model,
                encoder_output_dim=self.encoder_output_dim,
                encoder_outputs_transposed=self.encoder_outputs_transposed,
                weighted_encoder_outputs=self.weighted_encoder_outputs,
                decoder_hidden_state_t=hidden_t_intermediate,
                decoder_hidden_state_dim=self.decoder_state_dim,
                scope=self.name,
                attention_weighted_encoder_context_t_prev=(
                    attention_weighted_encoder_context_t_prev
                ),
            )
        else:
            (
                attention_weighted_encoder_context_t,
                self.attention_weights_3d,
                attention_blobs,
            ) = apply_regular_attention(
                model=model,
                encoder_output_dim=self.encoder_output_dim,
                encoder_outputs_transposed=self.encoder_outputs_transposed,
                weighted_encoder_outputs=self.weighted_encoder_outputs,
                decoder_hidden_state_t=hidden_t_intermediate,
                decoder_hidden_state_dim=self.decoder_state_dim,
                scope=self.name,
            )
        hidden_t = model.Copy(hidden_t_intermediate, self.scope('hidden_t'))
        model.net.AddExternalOutputs(
            cell_t,
            hidden_t,
            attention_weighted_encoder_context_t,
        )
        if self.attention_memory_optimization:
            self.recompute_blobs.extend(attention_blobs)
        if self.lstm_memory_optimization:
            self.recompute_blobs.append(gates_t)

        return hidden_t, cell_t, attention_weighted_encoder_context_t

    def get_attention_weights(self):
        # [batch_size, encoder_length, 1]
        return self.attention_weights_3d

    def prepare_input(self, model, input_blob):
        if self.encoder_outputs_transposed is None:
            self.encoder_outputs_transposed = model.Transpose(
                self.encoder_outputs,
                self.scope('encoder_outputs_transposed'),
                axes=[1, 2, 0],
            )
        if self.weighted_encoder_outputs is None:
            self.weighted_encoder_outputs = model.FC(
                self.encoder_outputs,
                self.scope('weighted_encoder_outputs'),
                dim_in=self.encoder_output_dim,
                dim_out=self.encoder_output_dim,
                axis=2,
            )

        return model.FC(
            input_blob,
            self.scope('i2h'),
            dim_in=self.decoder_input_dim,
            dim_out=4 * self.decoder_state_dim,
            axis=2,
        )

    def get_state_names(self):
        return (
            self.scope('hidden_t'),
            self.scope('cell_t'),
            self.scope('attention_weighted_encoder_context_t'),
        )

    def get_outputs_with_grads(self):
        return [0, 4]

    def get_output_size(self):
        return self.decoder_state_dim + self.encoder_output_dim


def LSTMWithAttention(
    model,
    decoder_inputs,
    decoder_input_lengths,
    initial_decoder_hidden_state,
    initial_decoder_cell_state,
    initial_attention_weighted_encoder_context,
    encoder_output_dim,
    encoder_outputs,
    decoder_input_dim,
    decoder_state_dim,
    scope,
    attention_type=AttentionType.Regular,
    outputs_with_grads=(0, 4),
    weighted_encoder_outputs=None,
    lstm_memory_optimization=False,
    attention_memory_optimization=False,
    forget_bias=0.0,
):
    '''
    Adds a LSTM with attention mechanism to a model.

    The implementation is based on https://arxiv.org/abs/1409.0473, with
    a small difference in the order
    how we compute new attention context and new hidden state, similarly to
    https://arxiv.org/abs/1508.04025.

    The model uses encoder-decoder naming conventions,
    where the decoder is the sequence the op is iterating over,
    while computing the attention context over the encoder.

    model: CNNModelHelper object new operators would be added to

    decoder_inputs: the input sequence in a format T x N x D
    where T is sequence size, N - batch size and D - input dimention

    decoder_input_lengths: blob containing sequence lengths
    which would be passed to LSTMUnit operator

    initial_decoder_hidden_state: initial hidden state of LSTM

    initial_decoder_cell_state: initial cell state of LSTM

    initial_attention_weighted_encoder_context: initial attention context

    encoder_output_dim: dimension of encoder outputs

    encoder_outputs: the sequence, on which we compute the attention context
    at every iteration

    decoder_input_dim: input dimention (last dimension on decoder_inputs)

    decoder_state_dim: size of hidden states of LSTM

    attention_type: One of: AttentionType.Regular, AttentionType.Recurrent.
    Determines which type of attention mechanism to use.

    outputs_with_grads : position indices of output blobs which will receive
    external error gradient during backpropagation

    weighted_encoder_outputs: encoder outputs to be used to compute attention
    weights. In the basic case it's just linear transformation of
    encoder outputs (that the default, when weighted_encoder_outputs is None).
    However, it can be something more complicated - like a separate
    encoder network (for example, in case of convolutional encoder)

    lstm_memory_optimization: recompute LSTM activations on backward pass, so
                 we don't need to store their values in forward passes

    attention_memory_optimization: recompute attention for backward pass
    '''
    cell = LSTMWithAttentionCell(
        encoder_output_dim=encoder_output_dim,
        encoder_outputs=encoder_outputs,
        decoder_input_dim=decoder_input_dim,
        decoder_state_dim=decoder_state_dim,
        name=scope,
        attention_type=attention_type,
        weighted_encoder_outputs=weighted_encoder_outputs,
        forget_bias=forget_bias,
        lstm_memory_optimization=lstm_memory_optimization,
        attention_memory_optimization=attention_memory_optimization,
    )
    return cell.apply_over_sequence(
        model=model,
        inputs=decoder_inputs,
        seq_lengths=decoder_input_lengths,
        initial_states=(
            initial_decoder_hidden_state,
            initial_decoder_cell_state,
            initial_attention_weighted_encoder_context,
        ),
        outputs_with_grads=None,
    )


class MILSTMCell(LSTMCell):

    def _apply(
        self,
        model,
        input_t,
        seq_lengths,
        states,
        timestep,
    ):
        (
            hidden_t_prev,
            cell_t_prev,
        ) = states

        # hU^T
        # Shape: [1, batch_size, 4 * hidden_size]
        prev_t = model.FC(
            hidden_t_prev, self.scope('prev_t'), dim_in=self.hidden_size,
            dim_out=4 * self.hidden_size, axis=2)
        # defining MI parameters
        alpha = model.param_init_net.ConstantFill(
            [],
            [self.scope('alpha')],
            shape=[4 * self.hidden_size],
            value=1.0
        )
        beta1 = model.param_init_net.ConstantFill(
            [],
            [self.scope('beta1')],
            shape=[4 * self.hidden_size],
            value=1.0
        )
        beta2 = model.param_init_net.ConstantFill(
            [],
            [self.scope('beta2')],
            shape=[4 * self.hidden_size],
            value=1.0
        )
        b = model.param_init_net.ConstantFill(
            [],
            [self.scope('b')],
            shape=[4 * self.hidden_size],
            value=0.0
        )
        model.params.extend([alpha, beta1, beta2, b])
        # alpha * (xW^T * hU^T)
        # Shape: [1, batch_size, 4 * hidden_size]
        alpha_tdash = model.net.Mul(
            [prev_t, input_t],
            self.scope('alpha_tdash')
        )
        # Shape: [batch_size, 4 * hidden_size]
        alpha_tdash_rs, _ = model.net.Reshape(
            alpha_tdash,
            [self.scope('alpha_tdash_rs'), self.scope('alpha_tdash_old_shape')],
            shape=[-1, 4 * self.hidden_size],
        )
        alpha_t = model.net.Mul(
            [alpha_tdash_rs, alpha],
            self.scope('alpha_t'),
            broadcast=1,
            use_grad_hack=1
        )
        # beta1 * hU^T
        # Shape: [batch_size, 4 * hidden_size]
        prev_t_rs, _ = model.net.Reshape(
            prev_t,
            [self.scope('prev_t_rs'), self.scope('prev_t_old_shape')],
            shape=[-1, 4 * self.hidden_size],
        )
        beta1_t = model.net.Mul(
            [prev_t_rs, beta1],
            self.scope('beta1_t'),
            broadcast=1,
            use_grad_hack=1
        )
        # beta2 * xW^T
        # Shape: [batch_szie, 4 * hidden_size]
        input_t_rs, _ = model.net.Reshape(
            input_t,
            [self.scope('input_t_rs'), self.scope('input_t_old_shape')],
            shape=[-1, 4 * self.hidden_size],
        )
        beta2_t = model.net.Mul(
            [input_t_rs, beta2],
            self.scope('beta2_t'),
            broadcast=1,
            use_grad_hack=1
        )
        # Add 'em all up
        gates_tdash = model.net.Sum(
            [alpha_t, beta1_t, beta2_t],
            self.scope('gates_tdash')
        )
        gates_t = model.net.Add(
            [gates_tdash, b],
            self.scope('gates_t'),
            broadcast=1,
            use_grad_hack=1
        )
        # # Shape: [1, batch_size, 4 * hidden_size]
        gates_t_rs, _ = model.net.Reshape(
            gates_t,
            [self.scope('gates_t_rs'), self.scope('gates_t_old_shape')],
            shape=[1, -1, 4 * self.hidden_size],
        )

        hidden_t_intermediate, cell_t = model.net.LSTMUnit(
            [hidden_t_prev, cell_t_prev, gates_t_rs, seq_lengths, timestep],
            [self.scope('hidden_t_intermediate'), self.scope('cell_t')],
            forget_bias=self.forget_bias,
        )
        hidden_t = model.Copy(hidden_t_intermediate, self.scope('hidden_t'))
        model.net.AddExternalOutputs(
            cell_t,
            hidden_t,
        )
        if self.memory_optimization:
            self.recompute_blobs = [gates_t]
        return hidden_t, cell_t


def MILSTM(model, input_blob, seq_lengths, initial_states, dim_in, dim_out,
           scope, outputs_with_grads=(0,), memory_optimization=False,
           forget_bias=0.0):
    '''
    Adds MI flavor of standard LSTM recurrent network operator to a model.
    See https://arxiv.org/pdf/1606.06630.pdf

    model: CNNModelHelper object new operators would be added to

    input_blob: the input sequence in a format T x N x D
    where T is sequence size, N - batch size and D - input dimention

    seq_lengths: blob containing sequence lengths which would be passed to
    LSTMUnit operator

    initial_states: a tupple of (hidden_input_blob, cell_input_blob)
    which are going to be inputs to the cell net on the first iteration

    dim_in: input dimention

    dim_out: output dimention

    outputs_with_grads : position indices of output blobs which will receive
    external error gradient during backpropagation

    memory_optimization: if enabled, the LSTM step is recomputed on backward step
                   so that we don't need to store forward activations for each
                   timestep. Saves memory with cost of computation.
    '''
    cell = MILSTMCell(
        input_size=dim_in,
        hidden_size=dim_out,
        forget_bias=forget_bias,
        memory_optimization=memory_optimization,
        name=scope,
    )
    result = cell.apply_over_sequence(
        model=model,
        inputs=input_blob,
        seq_lengths=seq_lengths,
        initial_states=initial_states,
        outputs_with_grads=outputs_with_grads,
    )
    return tuple(result)


class MILSTMWithAttentionCell(LSTMWithAttentionCell):

    def _apply(
        self,
        model,
        input_t,
        seq_lengths,
        states,
        timestep,
    ):
        (
            hidden_t_prev,
            cell_t_prev,
            attention_weighted_encoder_context_t_prev,
        ) = states

        gates_concatenated_input_t, _ = model.net.Concat(
            [hidden_t_prev, attention_weighted_encoder_context_t_prev],
            [
                self.scope('gates_concatenated_input_t'),
                self.scope('_gates_concatenated_input_t_concat_dims'),
            ],
            axis=2,
        )
        # hU^T
        # Shape: [1, batch_size, 4 * hidden_size]
        prev_t = model.FC(
            gates_concatenated_input_t,
            self.scope('prev_t'),
            dim_in=self.decoder_state_dim + self.encoder_output_dim,
            dim_out=4 * self.decoder_state_dim,
            axis=2,
        )
        # defining MI parameters
        alpha = model.param_init_net.ConstantFill(
            [],
            [self.scope('alpha')],
            shape=[4 * self.decoder_state_dim],
            value=1.0
        )
        beta1 = model.param_init_net.ConstantFill(
            [],
            [self.scope('beta1')],
            shape=[4 * self.decoder_state_dim],
            value=1.0
        )
        beta2 = model.param_init_net.ConstantFill(
            [],
            [self.scope('beta2')],
            shape=[4 * self.decoder_state_dim],
            value=1.0
        )
        b = model.param_init_net.ConstantFill(
            [],
            [self.scope('b')],
            shape=[4 * self.decoder_state_dim],
            value=0.0
        )
        model.params.extend([alpha, beta1, beta2, b])
        # alpha * (xW^T * hU^T)
        # Shape: [1, batch_size, 4 * hidden_size]
        alpha_tdash = model.net.Mul(
            [prev_t, input_t],
            self.scope('alpha_tdash')
        )
        # Shape: [batch_size, 4 * hidden_size]
        alpha_tdash_rs, _ = model.net.Reshape(
            alpha_tdash,
            [self.scope('alpha_tdash_rs'), self.scope('alpha_tdash_old_shape')],
            shape=[-1, 4 * self.decoder_state_dim],
        )
        alpha_t = model.net.Mul(
            [alpha_tdash_rs, alpha],
            self.scope('alpha_t'),
            broadcast=1,
            use_grad_hack=1
        )
        # beta1 * hU^T
        # Shape: [batch_size, 4 * hidden_size]
        prev_t_rs, _ = model.net.Reshape(
            prev_t,
            [self.scope('prev_t_rs'), self.scope('prev_t_old_shape')],
            shape=[-1, 4 * self.decoder_state_dim],
        )
        beta1_t = model.net.Mul(
            [prev_t_rs, beta1],
            self.scope('beta1_t'),
            broadcast=1,
            use_grad_hack=1
        )
        # beta2 * xW^T
        # Shape: [batch_szie, 4 * hidden_size]
        input_t_rs, _ = model.net.Reshape(
            input_t,
            [self.scope('input_t_rs'), self.scope('input_t_old_shape')],
            shape=[-1, 4 * self.decoder_state_dim],
        )
        beta2_t = model.net.Mul(
            [input_t_rs, beta2],
            self.scope('beta2_t'),
            broadcast=1,
            use_grad_hack=1
        )
        # Add 'em all up
        gates_tdash = model.net.Sum(
            [alpha_t, beta1_t, beta2_t],
            self.scope('gates_tdash')
        )
        gates_t = model.net.Add(
            [gates_tdash, b],
            self.scope('gates_t'),
            broadcast=1,
            use_grad_hack=1
        )
        # # Shape: [1, batch_size, 4 * hidden_size]
        gates_t_rs, _ = model.net.Reshape(
            gates_t,
            [self.scope('gates_t_rs'), self.scope('gates_t_old_shape')],
            shape=[1, -1, 4 * self.decoder_state_dim],
        )

        hidden_t_intermediate, cell_t = model.net.LSTMUnit(
            [hidden_t_prev, cell_t_prev, gates_t_rs, seq_lengths, timestep],
            [self.scope('hidden_t_intermediate'), self.scope('cell_t')],
        )

        if self.attention_type == AttentionType.Recurrent:
            (
                attention_weighted_encoder_context_t,
                self.attention_weights_3d,
                self.recompute_blobs,
            ) = (
                apply_recurrent_attention(
                    model=model,
                    encoder_output_dim=self.encoder_output_dim,
                    encoder_outputs_transposed=self.encoder_outputs_transposed,
                    weighted_encoder_outputs=self.weighted_encoder_outputs,
                    decoder_hidden_state_t=hidden_t_intermediate,
                    decoder_hidden_state_dim=self.decoder_state_dim,
                    scope=self.name,
                    attention_weighted_encoder_context_t_prev=(
                        attention_weighted_encoder_context_t_prev
                    ),
                )
            )
        else:
            (
                attention_weighted_encoder_context_t,
                self.attention_weights_3d,
                self.recompute_blobs,
            ) = (
                apply_regular_attention(
                    model=model,
                    encoder_output_dim=self.encoder_output_dim,
                    encoder_outputs_transposed=self.encoder_outputs_transposed,
                    weighted_encoder_outputs=self.weighted_encoder_outputs,
                    decoder_hidden_state_t=hidden_t_intermediate,
                    decoder_hidden_state_dim=self.decoder_state_dim,
                    scope=self.name,
                )
            )
        hidden_t = model.Copy(hidden_t_intermediate, self.scope('hidden_t'))
        model.net.AddExternalOutputs(
            cell_t,
            hidden_t,
            attention_weighted_encoder_context_t,
        )
        return hidden_t, cell_t, attention_weighted_encoder_context_t
