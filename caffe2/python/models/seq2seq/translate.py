## @package translate
# Module caffe2.python.models.seq2seq.translate





from abc import ABCMeta, abstractmethod
import argparse
import logging
import numpy as np
import sys

from caffe2.python import core, rnn_cell, workspace
from caffe2.python.models.seq2seq.beam_search import BeamSearchForwardOnly
from caffe2.python.models.seq2seq.seq2seq_model_helper import Seq2SeqModelHelper
import caffe2.python.models.seq2seq.seq2seq_util as seq2seq_util


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stderr))


def _weighted_sum(model, values, weight, output_name):
    values_weights = zip(values, [weight] * len(values))
    values_weights_flattened = [x for v_w in values_weights for x in v_w]
    return model.net.WeightedSum(
        values_weights_flattened,
        output_name,
    )


class Seq2SeqModelCaffe2EnsembleDecoderBase(metaclass=ABCMeta):

    @abstractmethod
    def get_model_file(self, model):
        pass

    @abstractmethod
    def get_db_type(self):
        pass

    def build_word_rewards(self, vocab_size, word_reward, unk_reward):
        word_rewards = np.full([vocab_size], word_reward, dtype=np.float32)
        word_rewards[seq2seq_util.PAD_ID] = 0
        word_rewards[seq2seq_util.GO_ID] = 0
        word_rewards[seq2seq_util.EOS_ID] = 0
        word_rewards[seq2seq_util.UNK_ID] = word_reward + unk_reward
        return word_rewards

    def load_models(self):
        db_reader = 'reader'
        for model, scope_name in zip(
            self.models,
            self.decoder_scope_names,
        ):
            params_for_current_model = [
                param
                for param in self.model.GetAllParams()
                if str(param).startswith(scope_name)
            ]
            assert workspace.RunOperatorOnce(core.CreateOperator(
                'CreateDB',
                [], [db_reader],
                db=self.get_model_file(model),
                db_type=self.get_db_type())
            ), 'Failed to create db {}'.format(self.get_model_file(model))
            assert workspace.RunOperatorOnce(core.CreateOperator(
                'Load',
                [db_reader],
                params_for_current_model,
                load_all=1,
                add_prefix=scope_name + '/',
                strip_prefix='gpu_0/',
            ))
            logger.info('Model {} is loaded from a checkpoint {}'.format(
                scope_name, self.get_model_file(model)))


class Seq2SeqModelCaffe2EnsembleDecoder(Seq2SeqModelCaffe2EnsembleDecoderBase):

    def get_model_file(self, model):
        return model['model_file']

    def get_db_type(self):
        return 'minidb'

    def scope(self, scope_name, blob_name):
        return (
            scope_name + '/' + blob_name
            if scope_name is not None
            else blob_name
        )

    def _build_decoder(
        self,
        model,
        step_model,
        model_params,
        scope,
        previous_tokens,
        timestep,
        fake_seq_lengths,
    ):
        attention_type = model_params['attention']
        assert attention_type in ['none', 'regular']
        use_attention = (attention_type != 'none')

        with core.NameScope(scope):
            encoder_embeddings = seq2seq_util.build_embeddings(
                model=model,
                vocab_size=self.source_vocab_size,
                embedding_size=model_params['encoder_embedding_size'],
                name='encoder_embeddings',
                freeze_embeddings=False,
            )

        (
            encoder_outputs,
            weighted_encoder_outputs,
            final_encoder_hidden_states,
            final_encoder_cell_states,
            encoder_units_per_layer,
        ) = seq2seq_util.build_embedding_encoder(
            model=model,
            encoder_params=model_params['encoder_type'],
            num_decoder_layers=len(model_params['decoder_layer_configs']),
            inputs=self.encoder_inputs,
            input_lengths=self.encoder_lengths,
            vocab_size=self.source_vocab_size,
            embeddings=encoder_embeddings,
            embedding_size=model_params['encoder_embedding_size'],
            use_attention=use_attention,
            num_gpus=0,
            forward_only=True,
            scope=scope,
        )
        with core.NameScope(scope):
            if use_attention:
                # [max_source_length, beam_size, encoder_output_dim]
                encoder_outputs = model.net.Tile(
                    encoder_outputs,
                    'encoder_outputs_tiled',
                    tiles=self.beam_size,
                    axis=1,
                )

            if weighted_encoder_outputs is not None:
                weighted_encoder_outputs = model.net.Tile(
                    weighted_encoder_outputs,
                    'weighted_encoder_outputs_tiled',
                    tiles=self.beam_size,
                    axis=1,
                )

            decoder_embeddings = seq2seq_util.build_embeddings(
                model=model,
                vocab_size=self.target_vocab_size,
                embedding_size=model_params['decoder_embedding_size'],
                name='decoder_embeddings',
                freeze_embeddings=False,
            )
            embedded_tokens_t_prev = step_model.net.Gather(
                [decoder_embeddings, previous_tokens],
                'embedded_tokens_t_prev',
            )

        decoder_cells = []
        decoder_units_per_layer = []
        for i, layer_config in enumerate(model_params['decoder_layer_configs']):
            num_units = layer_config['num_units']
            decoder_units_per_layer.append(num_units)
            if i == 0:
                input_size = model_params['decoder_embedding_size']
            else:
                input_size = (
                    model_params['decoder_layer_configs'][i - 1]['num_units']
                )

            cell = rnn_cell.LSTMCell(
                forward_only=True,
                input_size=input_size,
                hidden_size=num_units,
                forget_bias=0.0,
                memory_optimization=False,
            )
            decoder_cells.append(cell)

        with core.NameScope(scope):
            if final_encoder_hidden_states is not None:
                for i in range(len(final_encoder_hidden_states)):
                    if final_encoder_hidden_states[i] is not None:
                        final_encoder_hidden_states[i] = model.net.Tile(
                            final_encoder_hidden_states[i],
                            'final_encoder_hidden_tiled_{}'.format(i),
                            tiles=self.beam_size,
                            axis=1,
                        )
            if final_encoder_cell_states is not None:
                for i in range(len(final_encoder_cell_states)):
                    if final_encoder_cell_states[i] is not None:
                        final_encoder_cell_states[i] = model.net.Tile(
                            final_encoder_cell_states[i],
                            'final_encoder_cell_tiled_{}'.format(i),
                            tiles=self.beam_size,
                            axis=1,
                        )
            initial_states = \
                seq2seq_util.build_initial_rnn_decoder_states(
                    model=model,
                    encoder_units_per_layer=encoder_units_per_layer,
                    decoder_units_per_layer=decoder_units_per_layer,
                    final_encoder_hidden_states=final_encoder_hidden_states,
                    final_encoder_cell_states=final_encoder_cell_states,
                    use_attention=use_attention,
                )

        attention_decoder = seq2seq_util.LSTMWithAttentionDecoder(
            encoder_outputs=encoder_outputs,
            encoder_output_dim=encoder_units_per_layer[-1],
            encoder_lengths=None,
            vocab_size=self.target_vocab_size,
            attention_type=attention_type,
            embedding_size=model_params['decoder_embedding_size'],
            decoder_num_units=decoder_units_per_layer[-1],
            decoder_cells=decoder_cells,
            weighted_encoder_outputs=weighted_encoder_outputs,
            name=scope,
        )
        states_prev = step_model.net.AddExternalInputs(*[
            '{}/{}_prev'.format(scope, s)
            for s in attention_decoder.get_state_names()
        ])
        decoder_outputs, states = attention_decoder.apply(
            model=step_model,
            input_t=embedded_tokens_t_prev,
            seq_lengths=fake_seq_lengths,
            states=states_prev,
            timestep=timestep,
        )

        state_configs = [
            BeamSearchForwardOnly.StateConfig(
                initial_value=initial_state,
                state_prev_link=BeamSearchForwardOnly.LinkConfig(
                    blob=state_prev,
                    offset=0,
                    window=1,
                ),
                state_link=BeamSearchForwardOnly.LinkConfig(
                    blob=state,
                    offset=1,
                    window=1,
                ),
            )
            for initial_state, state_prev, state in zip(
                initial_states,
                states_prev,
                states,
            )
        ]

        with core.NameScope(scope):
            decoder_outputs_flattened, _ = step_model.net.Reshape(
                [decoder_outputs],
                [
                    'decoder_outputs_flattened',
                    'decoder_outputs_and_contexts_combination_old_shape',
                ],
                shape=[-1, attention_decoder.get_output_dim()],
            )
            output_logits = seq2seq_util.output_projection(
                model=step_model,
                decoder_outputs=decoder_outputs_flattened,
                decoder_output_size=attention_decoder.get_output_dim(),
                target_vocab_size=self.target_vocab_size,
                decoder_softmax_size=model_params['decoder_softmax_size'],
            )
            # [1, beam_size, target_vocab_size]
            output_probs = step_model.net.Softmax(
                output_logits,
                'output_probs',
            )
            output_log_probs = step_model.net.Log(
                output_probs,
                'output_log_probs',
            )
            if use_attention:
                attention_weights = attention_decoder.get_attention_weights()
            else:
                attention_weights = step_model.net.ConstantFill(
                    [self.encoder_inputs],
                    'zero_attention_weights_tmp_1',
                    value=0.0,
                )
                attention_weights = step_model.net.Transpose(
                    attention_weights,
                    'zero_attention_weights_tmp_2',
                )
                attention_weights = step_model.net.Tile(
                    attention_weights,
                    'zero_attention_weights_tmp',
                    tiles=self.beam_size,
                    axis=0,
                )

        return (
            state_configs,
            output_log_probs,
            attention_weights,
        )

    def __init__(
        self,
        translate_params,
    ):
        self.models = translate_params['ensemble_models']
        decoding_params = translate_params['decoding_params']
        self.beam_size = decoding_params['beam_size']

        assert len(self.models) > 0
        source_vocab = self.models[0]['source_vocab']
        target_vocab = self.models[0]['target_vocab']
        for model in self.models:
            assert model['source_vocab'] == source_vocab
            assert model['target_vocab'] == target_vocab

        self.source_vocab_size = len(source_vocab)
        self.target_vocab_size = len(target_vocab)

        self.decoder_scope_names = [
            'model{}'.format(i) for i in range(len(self.models))
        ]

        self.model = Seq2SeqModelHelper(init_params=True)

        self.encoder_inputs = self.model.net.AddExternalInput('encoder_inputs')
        self.encoder_lengths = self.model.net.AddExternalInput(
            'encoder_lengths'
        )
        self.max_output_seq_len = self.model.net.AddExternalInput(
            'max_output_seq_len'
        )

        fake_seq_lengths = self.model.param_init_net.ConstantFill(
            [],
            'fake_seq_lengths',
            shape=[self.beam_size],
            value=100000,
            dtype=core.DataType.INT32,
        )

        beam_decoder = BeamSearchForwardOnly(
            beam_size=self.beam_size,
            model=self.model,
            go_token_id=seq2seq_util.GO_ID,
            eos_token_id=seq2seq_util.EOS_ID,
        )
        step_model = beam_decoder.get_step_model()

        state_configs = []
        output_log_probs = []
        attention_weights = []
        for model, scope_name in zip(
            self.models,
            self.decoder_scope_names,
        ):
            (
                state_configs_per_decoder,
                output_log_probs_per_decoder,
                attention_weights_per_decoder,
            ) = self._build_decoder(
                model=self.model,
                step_model=step_model,
                model_params=model['model_params'],
                scope=scope_name,
                previous_tokens=beam_decoder.get_previous_tokens(),
                timestep=beam_decoder.get_timestep(),
                fake_seq_lengths=fake_seq_lengths,
            )
            state_configs.extend(state_configs_per_decoder)
            output_log_probs.append(output_log_probs_per_decoder)
            if attention_weights_per_decoder is not None:
                attention_weights.append(attention_weights_per_decoder)

        assert len(attention_weights) > 0
        num_decoders_with_attention_blob = (
            self.model.param_init_net.ConstantFill(
                [],
                'num_decoders_with_attention_blob',
                value=1 / float(len(attention_weights)),
                shape=[1],
            )
        )
        # [beam_size, encoder_length, 1]
        attention_weights_average = _weighted_sum(
            model=step_model,
            values=attention_weights,
            weight=num_decoders_with_attention_blob,
            output_name='attention_weights_average',
        )

        num_decoders_blob = self.model.param_init_net.ConstantFill(
            [],
            'num_decoders_blob',
            value=1 / float(len(output_log_probs)),
            shape=[1],
        )
        # [beam_size, target_vocab_size]
        output_log_probs_average = _weighted_sum(
            model=step_model,
            values=output_log_probs,
            weight=num_decoders_blob,
            output_name='output_log_probs_average',
        )
        word_rewards = self.model.param_init_net.ConstantFill(
            [],
            'word_rewards',
            shape=[self.target_vocab_size],
            value=0.0,
            dtype=core.DataType.FLOAT,
        )
        (
            self.output_token_beam_list,
            self.output_prev_index_beam_list,
            self.output_score_beam_list,
            self.output_attention_weights_beam_list,
        ) = beam_decoder.apply(
            inputs=self.encoder_inputs,
            length=self.max_output_seq_len,
            log_probs=output_log_probs_average,
            attentions=attention_weights_average,
            state_configs=state_configs,
            data_dependencies=[],
            word_rewards=word_rewards,
        )

        workspace.RunNetOnce(self.model.param_init_net)
        workspace.FeedBlob(
            'word_rewards',
            self.build_word_rewards(
                vocab_size=self.target_vocab_size,
                word_reward=translate_params['decoding_params']['word_reward'],
                unk_reward=translate_params['decoding_params']['unk_reward'],
            )
        )

        workspace.CreateNet(
            self.model.net,
            input_blobs=[
                str(self.encoder_inputs),
                str(self.encoder_lengths),
                str(self.max_output_seq_len),
            ],
        )

        logger.info('Params created: ')
        for param in self.model.params:
            logger.info(param)

    def decode(self, numberized_input, max_output_seq_len):
        workspace.FeedBlob(
            self.encoder_inputs,
            np.array([
                [token_id] for token_id in reversed(numberized_input)
            ]).astype(dtype=np.int32),
        )
        workspace.FeedBlob(
            self.encoder_lengths,
            np.array([len(numberized_input)]).astype(dtype=np.int32),
        )
        workspace.FeedBlob(
            self.max_output_seq_len,
            np.array([max_output_seq_len]).astype(dtype=np.int64),
        )

        workspace.RunNet(self.model.net)

        num_steps = max_output_seq_len
        score_beam_list = workspace.FetchBlob(self.output_score_beam_list)
        token_beam_list = (
            workspace.FetchBlob(self.output_token_beam_list)
        )
        prev_index_beam_list = (
            workspace.FetchBlob(self.output_prev_index_beam_list)
        )

        attention_weights_beam_list = (
            workspace.FetchBlob(self.output_attention_weights_beam_list)
        )
        best_indices = (num_steps, 0)
        for i in range(num_steps + 1):
            for hyp_index in range(self.beam_size):
                if (
                    (
                        token_beam_list[i][hyp_index][0] ==
                        seq2seq_util.EOS_ID or
                        i == num_steps
                    ) and
                    (
                        score_beam_list[i][hyp_index][0] >
                        score_beam_list[best_indices[0]][best_indices[1]][0]
                    )
                ):
                    best_indices = (i, hyp_index)

        i, hyp_index = best_indices
        output = []
        attention_weights_per_token = []
        best_score = -score_beam_list[i][hyp_index][0]
        while i > 0:
            output.append(token_beam_list[i][hyp_index][0])
            attention_weights_per_token.append(
                attention_weights_beam_list[i][hyp_index]
            )
            hyp_index = prev_index_beam_list[i][hyp_index][0]
            i -= 1

        attention_weights_per_token = reversed(attention_weights_per_token)
        # encoder_inputs are reversed, see get_batch func
        attention_weights_per_token = [
            list(reversed(attention_weights))[:len(numberized_input)]
            for attention_weights in attention_weights_per_token
        ]
        output = list(reversed(output))
        return output, attention_weights_per_token, best_score


def run_seq2seq_beam_decoder(args, model_params, decoding_params):
    source_vocab = seq2seq_util.gen_vocab(
        args.source_corpus,
        args.unk_threshold,
    )
    logger.info('Source vocab size {}'.format(len(source_vocab)))
    target_vocab = seq2seq_util.gen_vocab(
        args.target_corpus,
        args.unk_threshold,
    )
    inversed_target_vocab = {v: k for (k, v) in target_vocab.items()}
    logger.info('Target vocab size {}'.format(len(target_vocab)))

    decoder = Seq2SeqModelCaffe2EnsembleDecoder(
        translate_params=dict(
            ensemble_models=[dict(
                source_vocab=source_vocab,
                target_vocab=target_vocab,
                model_params=model_params,
                model_file=args.checkpoint,
            )],
            decoding_params=decoding_params,
        ),
    )
    decoder.load_models()

    for line in sys.stdin:
        numerized_source_sentence = seq2seq_util.get_numberized_sentence(
            line,
            source_vocab,
        )
        translation, alignment, _ = decoder.decode(
            numerized_source_sentence,
            2 * len(numerized_source_sentence) + 5,
        )
        print(' '.join([inversed_target_vocab[tid] for tid in translation]))


def main():
    parser = argparse.ArgumentParser(
        description='Caffe2: Seq2Seq Translation',
    )
    parser.add_argument('--source-corpus', type=str, default=None,
                        help='Path to source corpus in a text file format. Each '
                        'line in the file should contain a single sentence',
                        required=True)
    parser.add_argument('--target-corpus', type=str, default=None,
                        help='Path to target corpus in a text file format',
                        required=True)
    parser.add_argument('--unk-threshold', type=int, default=50,
                        help='Threshold frequency under which token becomes '
                        'labeled unknown token')

    parser.add_argument('--use-bidirectional-encoder', action='store_true',
                        help='Set flag to use bidirectional recurrent network '
                        'in encoder')
    parser.add_argument('--use-attention', action='store_true',
                        help='Set flag to use seq2seq with attention model')
    parser.add_argument('--encoder-cell-num-units', type=int, default=512,
                        help='Number of cell units per encoder layer')
    parser.add_argument('--encoder-num-layers', type=int, default=2,
                        help='Number encoder layers')
    parser.add_argument('--decoder-cell-num-units', type=int, default=512,
                        help='Number of cell units in the decoder layer')
    parser.add_argument('--decoder-num-layers', type=int, default=2,
                        help='Number decoder layers')
    parser.add_argument('--encoder-embedding-size', type=int, default=256,
                        help='Size of embedding in the encoder layer')
    parser.add_argument('--decoder-embedding-size', type=int, default=512,
                        help='Size of embedding in the decoder layer')
    parser.add_argument('--decoder-softmax-size', type=int, default=None,
                        help='Size of softmax layer in the decoder')

    parser.add_argument('--beam-size', type=int, default=6,
                        help='Size of beam for the decoder')
    parser.add_argument('--word-reward', type=float, default=0.0,
                        help='Reward per each word generated.')
    parser.add_argument('--unk-reward', type=float, default=0.0,
                        help='Reward per each UNK token generated. '
                        'Typically should be negative.')

    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint', required=True)

    args = parser.parse_args()

    encoder_layer_configs = [
        dict(
            num_units=args.encoder_cell_num_units,
        ),
    ] * args.encoder_num_layers

    if args.use_bidirectional_encoder:
        assert args.encoder_cell_num_units % 2 == 0
        encoder_layer_configs[0]['num_units'] /= 2

    decoder_layer_configs = [
        dict(
            num_units=args.decoder_cell_num_units,
        ),
    ] * args.decoder_num_layers

    run_seq2seq_beam_decoder(
        args,
        model_params=dict(
            attention=('regular' if args.use_attention else 'none'),
            decoder_layer_configs=decoder_layer_configs,
            encoder_type=dict(
                encoder_layer_configs=encoder_layer_configs,
                use_bidirectional_encoder=args.use_bidirectional_encoder,
            ),
            encoder_embedding_size=args.encoder_embedding_size,
            decoder_embedding_size=args.decoder_embedding_size,
            decoder_softmax_size=args.decoder_softmax_size,
        ),
        decoding_params=dict(
            beam_size=args.beam_size,
            word_reward=args.word_reward,
            unk_reward=args.unk_reward,
        ),
    )


if __name__ == '__main__':
    main()
