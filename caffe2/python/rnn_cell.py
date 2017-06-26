## @package rnn_cell
# Module caffe2.python.rnn_cell
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import functools
import itertools
import logging
import numpy as np
import random
import six

from caffe2.proto import caffe2_pb2
from caffe2.python.attention import (
    AttentionType,
    apply_regular_attention,
    apply_recurrent_attention,
)
from caffe2.python import core, recurrent, workspace, brew, scope
from caffe2.python.modeling.parameter_sharing import ParameterSharing
from caffe2.python.model_helper import ModelHelper


class RNNCell(object):
    '''
    Base class for writing recurrent / stateful operations.

    One needs to implement 3 methods: _apply, prepare_input and get_state_names.
    As a result base class will provice apply_over_sequence method, which
    allows you to apply recurrent operations over a sequence of any length.
    '''
    def __init__(self, name, forward_only=False):
        self.name = name
        self.recompute_blobs = []
        self.forward_only = forward_only

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
        step_model = ModelHelper(name=self.name, param_model=model)
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

        if outputs_with_grads is None:
            outputs_with_grads = [self.get_output_state_index() * 2]

        # states_for_all_steps consists of combination of
        # states gather for all steps and final states. It looks like this:
        # (state_1_all, state_1_final, state_2_all, state_2_final, ...)
        states_for_all_steps = recurrent.recurrent_net(
            net=model.net,
            cell_net=step_model.net,
            inputs=[(input_t, preprocessed_inputs)],
            initial_cell_inputs=list(zip(states_prev, initial_states)),
            links=dict(zip(states_prev, states)),
            timestep=timestep,
            scope=self.name,
            outputs_with_grads=outputs_with_grads,
            recompute_blobs_on_backward=self.recompute_blobs,
        )

        output = self._prepare_output_sequence(
            model,
            states_for_all_steps,
        )
        return output, states_for_all_steps

    def apply(self, model, input_t, seq_lengths, states, timestep):
        input_t = self.prepare_input(model, input_t)
        states = self._apply(
            model, input_t, seq_lengths, states, timestep)
        output = self._prepare_output(model, states)
        return output, states

    def _apply(
        self,
        model,
        input_t,
        seq_lengths,
        states,
        timestep,
        extra_inputs,
    ):
        '''
        A single step of a recurrent network.

        model: ModelHelper object new operators would be added to

        input_t: single input with shape (1, batch_size, input_dim)

        seq_lengths: blob containing sequence lengths which would be passed to
        LSTMUnit operator

        states: previous recurrent states

        timestep: current recurrent iteration. Could be used together with
        seq_lengths in order to determine, if some shorter sequences
        in the batch have already ended.

        extra_inputs: list of tuples (input, dim). specifies additional input
        which is not subject to prepare_input(). (useful when a cell is a
        component of a larger recurrent structure, e.g., attention)
        '''
        raise NotImplementedError('Abstract method')

    def prepare_input(self, model, input_blob):
        '''
        If some operations in _apply method depend only on the input,
        not on recurrent states, they could be computed in advance.

        model: ModelHelper object new operators would be added to

        input_blob: either the whole input sequence with shape
        (sequence_length, batch_size, input_dim) or a single input with shape
        (1, batch_size, input_dim).
        '''
        return input_blob

    def get_output_state_index(self):
        '''
        Return index into state list of the "primary" step-wise output.
        '''
        return 0

    def get_state_names(self):
        '''
        Return the names of the recurrent states.
        It's required by apply_over_sequence method in order to allocate
        recurrent states for all steps with meaningful names.
        '''
        raise NotImplementedError('Abstract method')

    def get_output_dim(self):
        '''
        Specifies the dimension (number of units) of stepwise output.
        '''
        raise NotImplementedError('Abstract method')

    def _prepare_output(self, model, states):
        '''
        Allows arbitrary post-processing of primary output.
        '''
        return states[self.get_output_state_index()]

    def _prepare_output_sequence(self, model, state_outputs):
        '''
        Allows arbitrary post-processing of primary sequence output.

        (Note that state_outputs alternates between full-sequence and final
        output for each state, thus the index multiplier 2.)
        '''
        output_sequence_index = 2 * self.get_output_state_index()
        return state_outputs[output_sequence_index]


class LSTMCell(RNNCell):

    def __init__(
        self,
        input_size,
        hidden_size,
        forget_bias,
        memory_optimization,
        drop_states=False,
        **kwargs
    ):
        super(LSTMCell, self).__init__(**kwargs)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.forget_bias = float(forget_bias)
        self.memory_optimization = memory_optimization
        self.drop_states = drop_states

    def _apply(
        self,
        model,
        input_t,
        seq_lengths,
        states,
        timestep,
        extra_inputs=None,
    ):
        hidden_t_prev, cell_t_prev = states

        fc_input = hidden_t_prev
        fc_input_dim = self.hidden_size

        if extra_inputs is not None:
            extra_input_blobs, extra_input_sizes = zip(*extra_inputs)
            fc_input, _ = model.net.Concat(
                [hidden_t_prev] + list(extra_input_blobs),
                [
                    self.scope('gates_concatenated_input_t'),
                    self.scope('_gates_concatenated_input_t_concat_dims'),
                ],
                axis=2,
            )
            fc_input_dim += sum(extra_input_sizes)

        gates_t = brew.fc(
            model,
            fc_input,
            self.scope('gates_t'),
            dim_in=fc_input_dim,
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
            drop_states=self.drop_states,
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
        return brew.fc(
            model,
            input_blob,
            self.scope('i2h'),
            dim_in=self.input_size,
            dim_out=4 * self.hidden_size,
            axis=2,
        )

    def get_state_names(self):
        return (self.scope('hidden_t'), self.scope('cell_t'))

    def get_output_dim(self):
        return self.hidden_size


class MILSTMCell(LSTMCell):

    def _apply(
        self,
        model,
        input_t,
        seq_lengths,
        states,
        timestep,
        extra_inputs=None,
    ):
        hidden_t_prev, cell_t_prev = states

        fc_input = hidden_t_prev
        fc_input_dim = self.hidden_size

        if extra_inputs is not None:
            extra_input_blobs, extra_input_sizes = zip(*extra_inputs)
            fc_input, _ = model.net.Concat(
                [hidden_t_prev] + list(extra_input_blobs),
                [
                    self.scope('gates_concatenated_input_t'),
                    self.scope('_gates_concatenated_input_t_concat_dims'),
                ],
                axis=2,
            )
            fc_input_dim += sum(extra_input_sizes)

        prev_t = brew.fc(
            model,
            fc_input,
            self.scope('prev_t'),
            dim_in=fc_input_dim,
            dim_out=4 * self.hidden_size,
            axis=2,
        )

        # defining MI parameters
        alpha = model.param_init_net.ConstantFill(
            [],
            [self.scope('alpha')],
            shape=[4 * self.hidden_size],
            value=1.0,
        )
        beta_h = model.param_init_net.ConstantFill(
            [],
            [self.scope('beta1')],
            shape=[4 * self.hidden_size],
            value=1.0,
        )
        beta_i = model.param_init_net.ConstantFill(
            [],
            [self.scope('beta2')],
            shape=[4 * self.hidden_size],
            value=1.0,
        )
        b = model.param_init_net.ConstantFill(
            [],
            [self.scope('b')],
            shape=[4 * self.hidden_size],
            value=0.0,
        )
        model.params.extend([alpha, beta_h, beta_i, b])

        # alpha * input_t + beta_h
        # Shape: [1, batch_size, 4 * hidden_size]
        alpha_by_input_t_plus_beta_h = model.net.ElementwiseLinear(
            [input_t, alpha, beta_h],
            self.scope('alpha_by_input_t_plus_beta_h'),
            axis=2,
        )
        # (alpha * input_t + beta_h) * prev_t =
        # alpha * input_t * prev_t + beta_h * prev_t
        # Shape: [1, batch_size, 4 * hidden_size]
        alpha_by_input_t_plus_beta_h_by_prev_t = model.net.Mul(
            [alpha_by_input_t_plus_beta_h, prev_t],
            self.scope('alpha_by_input_t_plus_beta_h_by_prev_t')
        )
        # beta_i * input_t + b
        # Shape: [1, batch_size, 4 * hidden_size]
        beta_i_by_input_t_plus_b = model.net.ElementwiseLinear(
            [input_t, beta_i, b],
            self.scope('beta_i_by_input_t_plus_b'),
            axis=2,
        )
        # alpha * input_t * prev_t + beta_h * prev_t + beta_i * input_t + b
        # Shape: [1, batch_size, 4 * hidden_size]
        gates_t = model.net.Sum(
            [alpha_by_input_t_plus_beta_h_by_prev_t, beta_i_by_input_t_plus_b],
            self.scope('gates_t')
        )
        hidden_t, cell_t = model.net.LSTMUnit(
            [hidden_t_prev, cell_t_prev, gates_t, seq_lengths, timestep],
            [self.scope('hidden_t_intermediate'), self.scope('cell_t')],
            forget_bias=self.forget_bias,
            drop_states=self.drop_states,
        )
        model.net.AddExternalOutputs(
            cell_t,
            hidden_t,
        )
        if self.memory_optimization:
            self.recompute_blobs = [gates_t]
        return hidden_t, cell_t


class DropoutCell(RNNCell):
    '''
    Wraps arbitrary RNNCell, applying dropout to its output (but not to the
    recurrent connection for the corresponding state).
    '''

    def __init__(self, internal_cell, dropout_ratio=None, **kwargs):
        self.internal_cell = internal_cell
        self.dropout_ratio = dropout_ratio
        super(DropoutCell, self).__init__(**kwargs)

        self.prepare_input = internal_cell.prepare_input
        self.get_output_state_index = internal_cell.get_output_state_index
        self.get_state_names = internal_cell.get_state_names
        self.get_output_dim = internal_cell.get_output_dim

        self.mask = 0

    def _apply(
        self,
        model,
        input_t,
        seq_lengths,
        states,
        timestep,
        extra_inputs=None,
    ):
        return self.internal_cell._apply(
            model,
            input_t,
            seq_lengths,
            states,
            timestep,
            extra_inputs,
        )

    def _prepare_output(self, model, states):
        output = states[self.get_output_state_index()]
        if self.dropout_ratio is not None:
            output = self._apply_dropout(model, output)
        return output

    def _prepare_output_sequence(self, model, state_outputs):
        output_sequence_index = 2 * self.get_output_state_index()
        output = state_outputs[output_sequence_index]
        if self.dropout_ratio is not None:
            output = self._apply_dropout(model, output)
        return output

    def _apply_dropout(self, model, output):
        if self.dropout_ratio and not self.forward_only:
            with core.NameScope(self.name or ''):
                output, _ = model.net.Dropout(
                    output,
                    [
                        str(output) + '_with_dropout_mask{}'.format(self.mask),
                        str(output) + '_dropout_mask{}'.format(self.mask),
                    ],
                    ratio=float(self.dropout_ratio),
                )
                self.mask += 1
        return output


class MultiRNNCell(RNNCell):
    '''
    Multilayer RNN via the composition of RNNCell instance.

    It is the resposibility of calling code to ensure the compatibility
    of the successive layers in terms of input/output dimensiality, etc.,
    and to ensure that their blobs do not have name conflicts, typically by
    creating the cells with names that specify layer number.

    Assumes first state (recurrent output) for each layer should be the input
    to the next layer.
    '''

    def __init__(self, cells, residual_output_layers=None, **kwargs):
        '''
        cells: list of RNNCell instances, from input to output side.

        name: string designating network component (for scoping)

        residual_output_layers: list of indices of layers whose input will
        be added elementwise to their output elementwise. (It is the
        responsibility of the client code to ensure shape compatibility.)
        Note that layer 0 (zero) cannot have residual output because of the
        timing of prepare_input().

        forward_only: used to construct inference-only network.
        '''
        super(MultiRNNCell, self).__init__(**kwargs)
        self.cells = cells

        if residual_output_layers is None:
            self.residual_output_layers = []
        else:
            self.residual_output_layers = residual_output_layers

        output_index_per_layer = []
        base_index = 0
        for cell in self.cells:
            output_index_per_layer.append(
                base_index + cell.get_output_state_index(),
            )
            base_index += len(cell.get_state_names())

        self.output_connected_layers = []
        self.output_indices = []
        for i in range(len(self.cells) - 1):
            if (i + 1) in self.residual_output_layers:
                self.output_connected_layers.append(i)
                self.output_indices.append(output_index_per_layer[i])
            else:
                self.output_connected_layers = []
                self.output_indices = []
        self.output_connected_layers.append(len(self.cells) - 1)
        self.output_indices.append(output_index_per_layer[-1])

        self.state_names = []
        for cell in self.cells:
            self.state_names.extend(cell.get_state_names())

        if len(self.state_names) != len(set(self.state_names)):
            duplicates = {
                state_name for state_name in self.state_names
                if self.state_names.count(state_name) > 1
            }
            raise RuntimeError(
                'Duplicate state names in MultiRNNCell: {}'.format(
                    list(duplicates),
                ),
            )

    def prepare_input(self, model, input_blob):
        return self.cells[0].prepare_input(model, input_blob)

    def _apply(
        self,
        model,
        input_t,
        seq_lengths,
        states,
        timestep,
        extra_inputs=None,
    ):

        states_per_layer = [len(cell.get_state_names()) for cell in self.cells]
        assert len(states) == sum(states_per_layer)

        next_states = []
        states_index = 0

        layer_input = input_t
        for i, layer_cell in enumerate(self.cells):
            num_states = states_per_layer[i]
            layer_states = states[states_index:(states_index + num_states)]
            states_index += num_states

            if i > 0:
                prepared_input = layer_cell.prepare_input(model, layer_input)
            else:
                prepared_input = layer_input

            layer_next_states = layer_cell._apply(
                model,
                prepared_input,
                seq_lengths,
                layer_states,
                timestep,
                extra_inputs=(None if i > 0 else extra_inputs),
            )
            # Since we're using here non-public method _apply, instead of apply,
            # we have to manually extract output from states
            if i != len(self.cells) - 1:
                layer_output = layer_cell._prepare_output(
                    model,
                    layer_next_states,
                )
                if i > 0 and i in self.residual_output_layers:
                    layer_input = model.net.Sum(
                        [layer_output, layer_input],
                        self.scope('residual_output_{}'.format(i)),
                    )
                else:
                    layer_input = layer_output

            next_states.extend(layer_next_states)
        return next_states

    def get_state_names(self):
        return self.state_names

    def get_output_state_index(self):
        index = 0
        for cell in self.cells[:-1]:
            index += len(cell.get_state_names())
        index += self.cells[-1].get_output_state_index()
        return index

    def _prepare_output(self, model, states):
        connected_outputs = []
        state_index = 0
        for i, cell in enumerate(self.cells):
            num_states = len(cell.get_state_names())
            if i in self.output_connected_layers:
                layer_states = states[state_index:state_index + num_states]
                layer_output = cell._prepare_output(
                    model,
                    layer_states
                )
                connected_outputs.append(layer_output)
            state_index += num_states

        output = model.net.Sum(
            connected_outputs,
            self.scope('residual_output'),
        )
        return output

    def _prepare_output_sequence(self, model, states):
        connected_outputs = []
        state_index = 0
        for i, cell in enumerate(self.cells):
            num_states = 2 * len(cell.get_state_names())
            if i in self.output_connected_layers:
                layer_states = states[state_index:state_index + num_states]
                layer_output = cell._prepare_output_sequence(
                    model,
                    layer_states
                )
                connected_outputs.append(layer_output)
            state_index += num_states

        output = model.net.Sum(
            connected_outputs,
            self.scope('residual_output_sequence'),
        )
        return output


class AttentionCell(RNNCell):

    def __init__(
        self,
        encoder_output_dim,
        encoder_outputs,
        decoder_cell,
        decoder_state_dim,
        attention_type,
        weighted_encoder_outputs,
        attention_memory_optimization,
        **kwargs
    ):
        super(AttentionCell, self).__init__(**kwargs)
        self.encoder_output_dim = encoder_output_dim
        self.encoder_outputs = encoder_outputs
        self.decoder_cell = decoder_cell
        self.decoder_state_dim = decoder_state_dim
        self.weighted_encoder_outputs = weighted_encoder_outputs
        self.encoder_outputs_transposed = None
        assert attention_type in [
            AttentionType.Regular,
            AttentionType.Recurrent,
        ]
        self.attention_type = attention_type
        self.attention_memory_optimization = attention_memory_optimization

    def _apply(
        self,
        model,
        input_t,
        seq_lengths,
        states,
        timestep,
        extra_inputs=None,
    ):
        decoder_prev_states = states[:-1]
        attention_weighted_encoder_context_t_prev = states[-1]

        assert extra_inputs is None

        decoder_states = self.decoder_cell._apply(
            model,
            input_t,
            seq_lengths,
            decoder_prev_states,
            timestep,
            extra_inputs=[(
                attention_weighted_encoder_context_t_prev,
                self.encoder_output_dim,
            )],
        )

        self.hidden_t_intermediate = self.decoder_cell._prepare_output(
            model,
            decoder_states,
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
                decoder_hidden_state_t=self.hidden_t_intermediate,
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
                decoder_hidden_state_t=self.hidden_t_intermediate,
                decoder_hidden_state_dim=self.decoder_state_dim,
                scope=self.name,
            )

        if self.attention_memory_optimization:
            self.recompute_blobs.extend(attention_blobs)

        output = list(decoder_states) + [attention_weighted_encoder_context_t]
        output[self.decoder_cell.get_output_state_index()] = model.Copy(
            output[self.decoder_cell.get_output_state_index()],
            self.scope('hidden_t_external'),
        )
        model.net.AddExternalOutputs(*output)

        return output

    def get_attention_weights(self):
        # [batch_size, encoder_length, 1]
        return self.attention_weights_3d

    def prepare_input(self, model, input_blob):
        if self.encoder_outputs_transposed is None:
            self.encoder_outputs_transposed = brew.transpose(
                model,
                self.encoder_outputs,
                self.scope('encoder_outputs_transposed'),
                axes=[1, 2, 0],
            )
        if self.weighted_encoder_outputs is None:
            self.weighted_encoder_outputs = brew.fc(
                model,
                self.encoder_outputs,
                self.scope('weighted_encoder_outputs'),
                dim_in=self.encoder_output_dim,
                dim_out=self.encoder_output_dim,
                axis=2,
            )

        return self.decoder_cell.prepare_input(model, input_blob)

    def get_state_names(self):
        state_names = list(self.decoder_cell.get_state_names())
        state_names[self.get_output_state_index()] = self.scope(
            'hidden_t_external',
        )
        state_names.append(self.scope('attention_weighted_encoder_context_t'))
        return state_names

    def get_output_dim(self):
        return self.decoder_state_dim + self.encoder_output_dim

    def get_output_state_index(self):
        return self.decoder_cell.get_output_state_index()

    def _prepare_output(self, model, states):
        attention_context = states[-1]
        with core.NameScope(self.name or ''):
            output, _ = model.net.Concat(
                [self.hidden_t_intermediate, attention_context],
                [
                    'states_and_context_combination',
                    '_states_and_context_combination_concat_dims',
                ],
                axis=2,
            )

        return output

    def _prepare_output_sequence(self, model, state_outputs):
        decoder_output = self.decoder_cell._prepare_output_sequence(
            model,
            state_outputs[:-2],
        )
        attention_context_index = 2 * (len(self.get_state_names()) - 1)
        with core.NameScope(self.name or ''):
            output, _ = model.net.Concat(
                [
                    decoder_output,
                    state_outputs[attention_context_index],
                ],
                [
                    'states_and_context_combination',
                    '_states_and_context_combination_concat_dims',
                ],
                axis=2,
            )
        return output


class LSTMWithAttentionCell(AttentionCell):

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
        forward_only=False,
    ):
        decoder_cell = LSTMCell(
            input_size=decoder_input_dim,
            hidden_size=decoder_state_dim,
            forget_bias=forget_bias,
            memory_optimization=lstm_memory_optimization,
            name='{}/decoder'.format(name),
            forward_only=False,
            drop_states=False,
        )
        super(LSTMWithAttentionCell, self).__init__(
            encoder_output_dim=encoder_output_dim,
            encoder_outputs=encoder_outputs,
            decoder_cell=decoder_cell,
            decoder_state_dim=decoder_state_dim,
            name=name,
            attention_type=attention_type,
            weighted_encoder_outputs=weighted_encoder_outputs,
            attention_memory_optimization=attention_memory_optimization,
            forward_only=forward_only,
        )


class MILSTMWithAttentionCell(AttentionCell):

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
        forward_only=False,
    ):
        decoder_cell = MILSTMCell(
            input_size=decoder_input_dim,
            hidden_size=decoder_state_dim,
            forget_bias=forget_bias,
            memory_optimization=lstm_memory_optimization,
            name='{}/decoder'.format(name),
            forward_only=False,
            drop_states=False,
        )
        super(MILSTMWithAttentionCell, self).__init__(
            encoder_output_dim=encoder_output_dim,
            encoder_outputs=encoder_outputs,
            decoder_cell=decoder_cell,
            decoder_state_dim=decoder_state_dim,
            name=name,
            attention_type=attention_type,
            weighted_encoder_outputs=weighted_encoder_outputs,
            attention_memory_optimization=attention_memory_optimization,
            forward_only=forward_only,
        )


def _LSTM(
    cell_class,
    model,
    input_blob,
    seq_lengths,
    initial_states,
    dim_in,
    dim_out,
    scope,
    outputs_with_grads=(0,),
    return_params=False,
    memory_optimization=False,
    forget_bias=0.0,
    forward_only=False,
    drop_states=False,
    return_last_layer_only=True,
    static_rnn_unroll_size=None,
):
    '''
    Adds a standard LSTM recurrent network operator to a model.

    cell_class: LSTMCell or compatible subclass

    model: ModelHelper object new operators would be added to

    input_blob: the input sequence in a format T x N x D
            where T is sequence size, N - batch size and D - input dimension

    seq_lengths: blob containing sequence lengths which would be passed to
            LSTMUnit operator

    initial_states: a list of (2 * num_layers) blobs representing the initial
            hidden and cell states of each layer. If this argument is None,
            these states will be added to the model as network parameters.

    dim_in: input dimension

    dim_out: number of units per LSTM layer
            (use int for single-layer LSTM, list of ints for multi-layer)

    outputs_with_grads : position indices of output blobs for LAST LAYER which
            will receive external error gradient during backpropagation.
            These outputs are: (h_all, h_last, c_all, c_last)

    return_params: if True, will return a dictionary of parameters of the LSTM

    memory_optimization: if enabled, the LSTM step is recomputed on backward
            step so that we don't need to store forward activations for each
            timestep. Saves memory with cost of computation.

    forget_bias: forget gate bias (default 0.0)

    forward_only: whether to create a backward pass

    drop_states: drop invalid states, passed through to LSTMUnit operator

    return_last_layer_only: only return outputs from final layer
            (so that length of results does depend on number of layers)

    static_rnn_unroll_size: if not None, we will use static RNN which is
    unrolled into Caffe2 graph. The size of the unroll is the value of
    this parameter.
    '''
    if type(dim_out) is not list and type(dim_out) is not tuple:
        dim_out = [dim_out]
    num_layers = len(dim_out)

    cells = []
    for i in range(num_layers):
        name = '{}/layer_{}'.format(scope, i) if num_layers > 1 else scope
        cell = cell_class(
            input_size=(dim_in if i == 0 else dim_out[i - 1]),
            hidden_size=dim_out[i],
            forget_bias=forget_bias,
            memory_optimization=memory_optimization,
            name=name,
            forward_only=forward_only,
            drop_states=drop_states,
        )
        cells.append(cell)

    cell = MultiRNNCell(
        cells,
        name=scope,
        forward_only=forward_only,
    ) if num_layers > 1 else cells[0]

    cell = (
        cell if static_rnn_unroll_size is None
        else UnrolledCell(cell, static_rnn_unroll_size))

    if initial_states is None:
        initial_states = []
        for i in range(num_layers):
            with core.NameScope(scope):
                suffix = '_{}'.format(i) if num_layers > 1 else ''
                initial_hidden = model.param_init_net.ConstantFill(
                    [],
                    'initial_hidden_state' + suffix,
                    shape=[dim_out[i]],
                    value=0.0,
                )
                initial_cell = model.param_init_net.ConstantFill(
                    [],
                    'initial_cell_state' + suffix,
                    shape=[dim_out[i]],
                    value=0.0,
                )
                initial_states.extend([initial_hidden, initial_cell])
                model.params.extend([initial_hidden, initial_cell])

    assert len(initial_states) == 2 * num_layers, \
            "Incorrect initial_states, was expecting 2 * num_layers elements" \
            + " but had only {}".format(len(initial_states))

    # outputs_with_grads argument indexes into final layer
    outputs_with_grads = [4 * (num_layers - 1) + i for i in outputs_with_grads]
    _, result = cell.apply_over_sequence(
        model=model,
        inputs=input_blob,
        seq_lengths=seq_lengths,
        initial_states=initial_states,
        outputs_with_grads=outputs_with_grads,
    )

    if return_last_layer_only:
        result = result[4 * (num_layers - 1):]
    if return_params:
        result = list(result) + [{
            'input': cell.get_input_params(),
            'recurrent': cell.get_recurrent_params(),
        }]
    return tuple(result)


LSTM = functools.partial(_LSTM, LSTMCell)
MILSTM = functools.partial(_LSTM, MILSTMCell)


class UnrolledCell(RNNCell):
    def __init__(self, cell, T):
        self.T = T
        self.cell = cell

    def apply_over_sequence(
        self,
        model,
        inputs,
        seq_lengths,
        initial_states,
        outputs_with_grads=None,
    ):
        inputs = self.cell.prepare_input(model, inputs)

        # Now they are blob references - outputs of splitting the input sequence
        split_inputs = model.net.Split(
            inputs,
            [str(inputs) + "_timestep_{}".format(i)
             for i in range(self.T)],
            axis=0)
        if self.T == 1:
            split_inputs = [split_inputs]

        states = initial_states
        all_states = []
        for t in range(0, self.T):
            scope_name = "timestep_{}".format(t)
            # Parameters of all timesteps are shared
            with ParameterSharing({scope_name: ''}),\
                 scope.NameScope(scope_name):
                timestep = model.param_init_net.ConstantFill(
                    [], "timestep", value=t, shape=[1],
                    dtype=core.DataType.INT32,
                    device_option=core.DeviceOption(caffe2_pb2.CPU))
                states = self.cell._apply(
                    model=model,
                    input_t=split_inputs[t],
                    seq_lengths=seq_lengths,
                    states=states,
                    timestep=timestep,
                )
            all_states.append(states)

        all_states = zip(*all_states)
        all_states = [
            model.net.Concat(
                list(full_output),
                [
                    str(full_output[0])[len("timestep_0/"):] + "_concat",
                    str(full_output[0])[len("timestep_0/"):] + "_concat_info"

                ],
                axis=0)[0]
            for full_output in all_states
        ]
        outputs = tuple(
            six.next(it) for it in
            itertools.cycle([iter(all_states), iter(states)])
        )
        outputs_without_grad = set(range(len(outputs))) - set(
            outputs_with_grads)
        for i in outputs_without_grad:
            model.net.ZeroGradient(outputs[i], [])
        logging.debug("Added 0 gradients for blobs:",
                      [outputs[i] for i in outputs_without_grad])

        final_output = self.cell._prepare_output_sequence(model, outputs)

        return final_output, outputs


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
        weight_values = [
            param_values[input_type][w].flatten()
            for w in weight_params
        ]
        wmat = np.array([])
        for w in weight_values:
            wmat = np.append(wmat, w)
        bias_values = [
            param_values[input_type][b].flatten()
            for b in bias_params
        ]
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
    forward_only=False,
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

    model: ModelHelper object new operators would be added to

    decoder_inputs: the input sequence in a format T x N x D
    where T is sequence size, N - batch size and D - input dimension

    decoder_input_lengths: blob containing sequence lengths
    which would be passed to LSTMUnit operator

    initial_decoder_hidden_state: initial hidden state of LSTM

    initial_decoder_cell_state: initial cell state of LSTM

    initial_attention_weighted_encoder_context: initial attention context

    encoder_output_dim: dimension of encoder outputs

    encoder_outputs: the sequence, on which we compute the attention context
    at every iteration

    decoder_input_dim: input dimension (last dimension on decoder_inputs)

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

    forward_only: whether to create only forward pass
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
        forward_only=forward_only,
    )
    _, result = cell.apply_over_sequence(
        model=model,
        inputs=decoder_inputs,
        seq_lengths=decoder_input_lengths,
        initial_states=(
            initial_decoder_hidden_state,
            initial_decoder_cell_state,
            initial_attention_weighted_encoder_context,
        ),
        outputs_with_grads=outputs_with_grads,
    )
    return result


def _layered_LSTM(
        model, input_blob, seq_lengths, initial_states,
        dim_in, dim_out, scope, outputs_with_grads=(0,), return_params=False,
        memory_optimization=False, forget_bias=0.0, forward_only=False,
        drop_states=False, create_lstm=None):
    params = locals()  # leave it as a first line to grab all params
    params.pop('create_lstm')
    if not isinstance(dim_out, list):
        return create_lstm(**params)
    elif len(dim_out) == 1:
        params['dim_out'] = dim_out[0]
        return create_lstm(**params)

    assert len(dim_out) != 0, "dim_out list can't be empty"
    assert return_params is False, "return_params not supported for layering"
    for i, output_dim in enumerate(dim_out):
        params.update({
            'dim_out': output_dim
        })
        output, last_output, all_states, last_state = create_lstm(**params)
        params.update({
            'input_blob': output,
            'dim_in': output_dim,
            'initial_states': (last_output, last_state),
            'scope': scope + '_layer_{}'.format(i + 1)
        })
    return output, last_output, all_states, last_state


layered_LSTM = functools.partial(_layered_LSTM, create_lstm=LSTM)
