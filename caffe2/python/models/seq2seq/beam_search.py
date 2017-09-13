## @package beam_search
# Module caffe2.python.models.seq2seq.beam_search
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import namedtuple
from caffe2.python import core
import caffe2.python.models.seq2seq.seq2seq_util as seq2seq_util
from caffe2.python.models.seq2seq.seq2seq_model_helper import Seq2SeqModelHelper


class BeamSearchForwardOnly(object):
    """
    Class generalizing forward beam search for seq2seq models.

    Also provides types to specify the recurrent structure of decoding:

    StateConfig:
        initial_value: blob providing value of state at first step_model
        state_prev_link: LinkConfig describing how recurrent step receives
            input from global state blob in each step
        state_link: LinkConfig describing how step writes (produces new state)
            to global state blob in each step

    LinkConfig:
        blob: blob connecting global state blob to step application
        offset: offset from beginning of global blob for link in time dimension
        window: width of global blob to read/write in time dimension
    """

    LinkConfig = namedtuple('LinkConfig', ['blob', 'offset', 'window'])

    StateConfig = namedtuple(
        'StateConfig',
        ['initial_value', 'state_prev_link', 'state_link'],
    )

    def __init__(
        self,
        beam_size,
        model,
        eos_token_id,
        go_token_id=seq2seq_util.GO_ID,
    ):
        self.beam_size = beam_size
        self.model = model
        self.step_model = Seq2SeqModelHelper(
            name='step_model',
            param_model=self.model,
        )
        self.go_token_id = go_token_id
        self.eos_token_id = eos_token_id

        (
            self.timestep,
            self.scores_t_prev,
            self.tokens_t_prev,
            self.hypo_t_prev,
            self.attention_t_prev,
        ) = self.step_model.net.AddExternalInputs(
            'timestep',
            'scores_t_prev',
            'tokens_t_prev',
            'hypo_t_prev',
            'attention_t_prev',
        )
        tokens_t_prev_int32 = self.step_model.net.Cast(
            self.tokens_t_prev,
            'tokens_t_prev_int32',
            to=core.DataType.INT32,
        )
        self.tokens_t_prev_int32_flattened, _ = self.step_model.net.Reshape(
            [tokens_t_prev_int32],
            [tokens_t_prev_int32, 'input_t_int32_old_shape'],
            shape=[1, -1],
        )

    def get_step_model(self):
        return self.step_model

    def get_previous_tokens(self):
        return self.tokens_t_prev_int32_flattened

    def get_timestep(self):
        return self.timestep

    # TODO: make attentions a generic state
    def apply(
        self,
        inputs,
        length,
        log_probs,
        attentions,
        state_configs,
        word_rewards=None,
        possible_translation_tokens=None,
        go_token_id=None,
    ):
        # [beam_size, beam_size]
        best_scores_per_hypo, best_tokens_per_hypo = self.step_model.net.TopK(
            log_probs,
            ['best_scores_per_hypo', 'best_tokens_per_hypo_indices'],
            k=self.beam_size,
        )
        if possible_translation_tokens:
            # [beam_size, beam_size]
            best_tokens_per_hypo = self.step_model.net.Gather(
                [possible_translation_tokens, best_tokens_per_hypo],
                ['best_tokens_per_hypo']
            )

        # [beam_size]
        scores_t_prev_squeezed, _ = self.step_model.net.Reshape(
            self.scores_t_prev,
            ['scores_t_prev_squeezed', 'scores_t_prev_old_shape'],
            shape=[self.beam_size],
        )
        # [beam_size, beam_size]
        output_scores = self.step_model.net.Add(
            [best_scores_per_hypo, scores_t_prev_squeezed],
            'output_scores',
            broadcast=1,
            axis=0,
        )
        if word_rewards is not None:
            # [beam_size, beam_size]
            word_rewards_for_best_tokens_per_hypo = self.step_model.net.Gather(
                [word_rewards, best_tokens_per_hypo],
                'word_rewards_for_best_tokens_per_hypo',
            )
            # [beam_size, beam_size]
            output_scores = self.step_model.net.Add(
                [output_scores, word_rewards_for_best_tokens_per_hypo],
                'output_scores',
            )
        # [beam_size * beam_size]
        output_scores_flattened, _ = self.step_model.net.Reshape(
            [output_scores],
            [output_scores, 'output_scores_old_shape'],
            shape=[-1],
        )
        ZERO = self.model.param_init_net.ConstantFill(
            [],
            'ZERO',
            shape=[1],
            value=0,
            dtype=core.DataType.INT32,
        )
        SLICE_END = self._hack_get_slice_end(
            self.model,
            self.step_model,
            self.timestep,
        )
        # [current_beam_size * beam_size]
        output_scores_flattened_slice = self.step_model.net.Slice(
            [output_scores_flattened, ZERO, SLICE_END],
            'output_scores_flattened_slice',
        )
        # [1, current_beam_size * beam_size]
        output_scores_flattened_slice, _ = self.step_model.net.Reshape(
            output_scores_flattened_slice,
            [
                output_scores_flattened_slice,
                'output_scores_flattened_slice_old_shape',
            ],
            shape=[1, -1],
        )
        # [1, beam_size]
        scores_t, best_indices = self.step_model.net.TopK(
            output_scores_flattened_slice,
            ['scores_t', 'best_indices'],
            k=self.beam_size,
        )
        BEAM_SIZE = self.model.param_init_net.ConstantFill(
            [],
            'beam_size',
            shape=[1],
            value=self.beam_size,
            dtype=core.DataType.INT64,
        )
        # [1, beam_size]
        hypo_t_int32 = self.step_model.net.Div(
            [best_indices, BEAM_SIZE],
            'hypo_t_int32',
            broadcast=1,
        )
        hypo_t = self.step_model.net.Cast(
            hypo_t_int32,
            'hypo_t',
            to=core.DataType.FLOAT,
        )

        # [beam_size, encoder_length, 1]
        attention_t = self.step_model.net.Gather(
            [attentions, hypo_t_int32],
            'attention_t',
        )
        # [1, beam_size, encoder_length]
        attention_t, _ = self.step_model.net.Reshape(
            attention_t,
            [attention_t, 'attention_t_old_shape'],
            shape=[1, self.beam_size, -1],
        )
        # [beam_size * beam_size]
        best_tokens_per_hypo_flatten, _ = self.step_model.net.Reshape(
            best_tokens_per_hypo,
            [
                'best_tokens_per_hypo_flatten',
                'best_tokens_per_hypo_old_shape',
            ],
            shape=[-1],
        )
        tokens_t_int32 = self.step_model.net.Gather(
            [best_tokens_per_hypo_flatten, best_indices],
            'tokens_t_int32',
        )
        tokens_t = self.step_model.net.Cast(
            tokens_t_int32,
            'tokens_t',
            to=core.DataType.FLOAT,
        )

        def choose_state_per_hypo(state_config):
            state_flattened, _ = self.step_model.net.Reshape(
                state_config.state_link.blob,
                [
                    state_config.state_link.blob,
                    'state_old_shape_before_choosing_per_hypo',
                ],
                shape=[self.beam_size, -1],
            )
            state_chosen_per_hypo = self.step_model.net.Gather(
                [state_flattened, hypo_t_int32],
                str(state_config.state_link.blob) + '_chosen_per_hypo',
            )
            return self.StateConfig(
                initial_value=state_config.initial_value,
                state_prev_link=state_config.state_prev_link,
                state_link=self.LinkConfig(
                    blob=state_chosen_per_hypo,
                    offset=state_config.state_link.offset,
                    window=state_config.state_link.window,
                )
            )
        state_configs = [choose_state_per_hypo(c) for c in state_configs]
        initial_scores = self.model.param_init_net.ConstantFill(
            [],
            'initial_scores',
            shape=[1],
            value=0.0,
            dtype=core.DataType.FLOAT,
        )
        if go_token_id:
            initial_tokens = self.model.net.Copy(
                [go_token_id],
                'initial_tokens',
            )
        else:
            initial_tokens = self.model.param_init_net.ConstantFill(
                [],
                'initial_tokens',
                shape=[1],
                value=float(self.go_token_id),
                dtype=core.DataType.FLOAT,
            )

        initial_hypo = self.model.param_init_net.ConstantFill(
            [],
            'initial_hypo',
            shape=[1],
            value=-1.0,
            dtype=core.DataType.FLOAT,
        )
        encoder_inputs_flattened, _ = self.model.net.Reshape(
            inputs,
            ['encoder_inputs_flattened', 'encoder_inputs_old_shape'],
            shape=[-1],
        )
        init_attention = self.model.net.ConstantFill(
            encoder_inputs_flattened,
            'init_attention',
            value=0.0,
            dtype=core.DataType.FLOAT,
        )
        state_configs = state_configs + [
            self.StateConfig(
                initial_value=initial_scores,
                state_prev_link=self.LinkConfig(self.scores_t_prev, 0, 1),
                state_link=self.LinkConfig(scores_t, 1, 1),
            ),
            self.StateConfig(
                initial_value=initial_tokens,
                state_prev_link=self.LinkConfig(self.tokens_t_prev, 0, 1),
                state_link=self.LinkConfig(tokens_t, 1, 1),
            ),
            self.StateConfig(
                initial_value=initial_hypo,
                state_prev_link=self.LinkConfig(self.hypo_t_prev, 0, 1),
                state_link=self.LinkConfig(hypo_t, 1, 1),
            ),
            self.StateConfig(
                initial_value=init_attention,
                state_prev_link=self.LinkConfig(self.attention_t_prev, 0, 1),
                state_link=self.LinkConfig(attention_t, 1, 1),
            ),
        ]
        fake_input = self.model.net.ConstantFill(
            length,
            'beam_search_fake_input',
            input_as_shape=True,
            extra_shape=[self.beam_size, 1],
            value=0.0,
            dtype=core.DataType.FLOAT,
        )
        all_inputs = (
            [fake_input] +
            self.step_model.params +
            [state_config.initial_value for state_config in state_configs]
        )
        forward_links = []
        recurrent_states = []
        for state_config in state_configs:
            state_name = str(state_config.state_prev_link.blob) + '_states'
            recurrent_states.append(state_name)
            forward_links.append((
                state_config.state_prev_link.blob,
                state_name,
                state_config.state_prev_link.offset,
                state_config.state_prev_link.window,
            ))
            forward_links.append((
                state_config.state_link.blob,
                state_name,
                state_config.state_link.offset,
                state_config.state_link.window,
            ))
        link_internal, link_external, link_offset, link_window = (
            zip(*forward_links)
        )
        all_outputs = [
            str(s) + '_all'
            for s in [scores_t, tokens_t, hypo_t, attention_t]
        ]
        results = self.model.net.RecurrentNetwork(
            all_inputs,
            all_outputs + ['step_workspaces'],
            param=[all_inputs.index(p) for p in self.step_model.params],
            alias_src=[
                str(s) + '_states'
                for s in [
                    self.scores_t_prev,
                    self.tokens_t_prev,
                    self.hypo_t_prev,
                    self.attention_t_prev,
                ]
            ],
            alias_dst=all_outputs,
            alias_offset=[0] * 4,
            recurrent_states=recurrent_states,
            initial_recurrent_state_ids=[
                all_inputs.index(state_config.initial_value)
                for state_config in state_configs
            ],
            link_internal=[str(l) for l in link_internal],
            link_external=[str(l) for l in link_external],
            link_offset=link_offset,
            link_window=link_window,
            backward_link_internal=[],
            backward_link_external=[],
            backward_link_offset=[],
            step_net=str(self.step_model.net.Proto()),
            backward_step_net='',
            timestep=str(self.timestep),
            outputs_with_grads=[],
            enable_rnn_executor=1,
        )
        score_t_all, tokens_t_all, hypo_t_all, attention_t_all = results[:4]

        output_token_beam_list = self.model.net.Cast(
            tokens_t_all,
            'output_token_beam_list',
            to=core.DataType.INT32,
        )
        output_prev_index_beam_list = self.model.net.Cast(
            hypo_t_all,
            'output_prev_index_beam_list',
            to=core.DataType.INT32,
        )
        output_score_beam_list = self.model.net.Alias(
            score_t_all,
            'output_score_beam_list',
        )
        output_attention_weights_beam_list = self.model.net.Alias(
            attention_t_all,
            'output_attention_weights_beam_list',
        )

        return (
            output_token_beam_list,
            output_prev_index_beam_list,
            output_score_beam_list,
            output_attention_weights_beam_list,
        )

    def _max_int32(self, model, a_int32, b_int32, output_name):
        a_float = model.net.Cast(a_int32, 'a_float', to=core.DataType.FLOAT)
        b_float = model.net.Cast(b_int32, 'b_float', to=core.DataType.FLOAT)
        m_float = model.net.Max([a_float, b_float], output_name + '_float')
        m_int32 = model.net.Cast(m_float, output_name, to=core.DataType.INT32)
        return m_int32

    # Function returns (beam_size if timestep == 0 else -1)
    def _hack_get_slice_end(self, param_init_model, model, timestep):
        timestep_negative = model.net.Negative(
            timestep,
            'timestep_negative',
        )
        ONE_INT32 = param_init_model.param_init_net.ConstantFill(
            [],
            'ONE_INT32',
            value=1,
            shape=[1],
            dtype=core.DataType.INT32,
        )
        MINUS_ONE_INT32 = param_init_model.param_init_net.ConstantFill(
            [],
            'MINUS_ONE_INT32',
            value=-1,
            shape=[1],
            dtype=core.DataType.INT32,
        )
        zero_or_minus_one = self._max_int32(
            model=model,
            a_int32=timestep_negative,
            b_int32=MINUS_ONE_INT32,
            output_name='zero_or_minus_one',
        )
        BEAM_SIZE_PLUS_ONE = param_init_model.param_init_net.ConstantFill(
            [],
            'BEAM_SIZE_PLUS_ONE',
            value=self.beam_size + 1,
            shape=[1],
            dtype=core.DataType.INT32,
        )
        one_or_zero = model.net.Add(
            [zero_or_minus_one, ONE_INT32],
            'one_or_zero',
        )
        beam_size_plus_one_or_zero = model.net.Mul(
            [BEAM_SIZE_PLUS_ONE, one_or_zero],
            'beam_size_plus_one_or_zero',
        )
        beam_size_or_minus_one = model.net.Add(
            [beam_size_plus_one_or_zero, MINUS_ONE_INT32],
            'beam_size_or_minus_one'
        )
        return beam_size_or_minus_one
