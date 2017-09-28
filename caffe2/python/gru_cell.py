# Copyright (c) 2016-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import functools
from caffe2.python import brew, rnn_cell


class GRUCell(rnn_cell.RNNCell):

    def __init__(
        self,
        input_size,
        hidden_size,
        forget_bias,  # Currently unused!  Values here will be ignored.
        memory_optimization,
        drop_states=False,
        **kwargs
    ):
        super(GRUCell, self).__init__(**kwargs)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.forget_bias = float(forget_bias)
        self.memory_optimization = memory_optimization
        self.drop_states = drop_states

    # Unlike LSTMCell, GRUCell needs the output of one gate to feed into another.
    # (reset gate -> output_gate)
    # So, much of the logic to calculate the reset gate output and modified
    # output gate input is set here, in the graph definition.
    # The remaining logic lives in in gru_unit_op.{h,cc}.
    def _apply(
        self,
        model,
        input_t,
        seq_lengths,
        states,
        timestep,
        extra_inputs=None,
    ):
        hidden_t_prev = states[0]

        # Split input tensors to get inputs for each gate.
        input_t_reset, input_t_update, input_t_output = model.net.Split(
            [
                input_t,
            ],
            [
                self.scope('input_t_reset'),
                self.scope('input_t_update'),
                self.scope('input_t_output'),
            ],
            axis=2,
        )

        # Fully connected layers for reset and update gates.
        reset_gate_t = brew.fc(
            model,
            hidden_t_prev,
            self.scope('reset_gate_t'),
            dim_in=self.hidden_size,
            dim_out=self.hidden_size,
            axis=2,
        )
        update_gate_t = brew.fc(
            model,
            hidden_t_prev,
            self.scope('update_gate_t'),
            dim_in=self.hidden_size,
            dim_out=self.hidden_size,
            axis=2,
        )

        # Calculating the modified hidden state going into output gate.
        reset_gate_t = model.net.Sum(
            [reset_gate_t, input_t_reset],
            self.scope('reset_gate_t')
        )
        reset_gate_t_sigmoid = model.net.Sigmoid(
            reset_gate_t,
            self.scope('reset_gate_t_sigmoid')
        )
        modified_hidden_t_prev = model.net.Mul(
            [reset_gate_t_sigmoid, hidden_t_prev],
            self.scope('modified_hidden_t_prev')
        )
        output_gate_t = brew.fc(
            model,
            modified_hidden_t_prev,
            self.scope('output_gate_t'),
            dim_in=self.hidden_size,
            dim_out=self.hidden_size,
            axis=2,
        )

        # Add input contributions to update and output gate.
        # We already (in-place) added input contributions to the reset gate.
        update_gate_t = model.net.Sum(
            [update_gate_t, input_t_update],
            self.scope('update_gate_t'),
        )
        output_gate_t = model.net.Sum(
            [output_gate_t, input_t_output],
            self.scope('output_gate_t'),
        )

        # Join gate outputs and add input contributions
        gates_t, _gates_t_concat_dims = model.net.Concat(
            [
                reset_gate_t,
                update_gate_t,
                output_gate_t,
            ],
            [
                self.scope('gates_t'),
                self.scope('_gates_t_concat_dims'),
            ],
            axis=2,
        )

        hidden_t = model.net.GRUUnit(
            [
                hidden_t_prev,
                gates_t,
                seq_lengths,
                timestep,
            ],
            list(self.get_state_names()),
            forget_bias=self.forget_bias,
            drop_states=self.drop_states,
        )
        model.net.AddExternalOutputs(hidden_t)
        return (hidden_t,)

    def prepare_input(self, model, input_blob):
        return brew.fc(
            model,
            input_blob,
            self.scope('i2h'),
            dim_in=self.input_size,
            dim_out=3 * self.hidden_size,
            axis=2,
        )

    def get_state_names(self):
        return (self.scope('hidden_t'),)


GRU = functools.partial(rnn_cell._LSTM, GRUCell)
