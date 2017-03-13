from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import collections
import logging
import math
import numpy as np
import time
import argparse

from itertools import izip

from caffe2.python import core, workspace, recurrent
from caffe2.python.examples import seq2seq_util

logger = logging.getLogger(__name__)

Batch = collections.namedtuple('Batch', [
    'encoder_inputs',
    'encoder_lengths',
    'decoder_inputs',
    'decoder_lengths',
    'targets',
    'target_weights',
])

_PAD_ID = 0
_GO_ID = 1
_EOS_ID = 2
EOS = '<EOS>'
UNK = '<UNK>'
GO = '<GO>'
PAD = '<PAD>'


def prepare_batch(batch):
    encoder_lengths = [len(entry[0]) for entry in batch]
    max_encoder_length = max(encoder_lengths)
    decoder_lengths = []
    max_decoder_length = max([len(entry[1]) for entry in batch])

    batch_encoder_inputs = []
    batch_decoder_inputs = []
    batch_targets = []
    batch_target_weights = []

    for source_seq, target_seq in batch:
        encoder_pads = (
            [_PAD_ID] * (max_encoder_length - len(source_seq))
        )
        batch_encoder_inputs.append(
            list(reversed(source_seq)) + encoder_pads
        )

        decoder_pads = (
            [_PAD_ID] * (max_decoder_length - len(target_seq))
        )
        target_seq_with_go_token = [_GO_ID] + target_seq
        decoder_lengths.append(len(target_seq_with_go_token))
        batch_decoder_inputs.append(target_seq_with_go_token + decoder_pads)

        target_seq_with_eos = target_seq + [_EOS_ID]
        targets = target_seq_with_eos + decoder_pads
        batch_targets.append(targets)

        if len(source_seq) + len(target_seq) == 0:
            target_weights = [0] * len(targets)
        else:
            target_weights = [
                1 if target != _PAD_ID else 0
                for target in targets
            ]
        batch_target_weights.append(target_weights)

    return Batch(
        encoder_inputs=np.array(
            batch_encoder_inputs,
            dtype=np.int32,
        ).transpose(),
        encoder_lengths=np.array(encoder_lengths, dtype=np.int32),
        decoder_inputs=np.array(
            batch_decoder_inputs,
            dtype=np.int32,
        ).transpose(),
        decoder_lengths=np.array(decoder_lengths, dtype=np.int32),
        targets=np.array(
            batch_targets,
            dtype=np.int32,
        ).transpose(),
        target_weights=np.array(
            batch_target_weights,
            dtype=np.float32,
        ).transpose(),
    )


class Seq2SeqModelCaffe2:
    def _embedding_encoder(
        self,
        model,
        encoder_type,
        encoder_params,
        inputs,
        input_lengths,
        vocab_size,
        embedding_size,
        use_attention,
    ):
        # Initialize the word embeddings that will be learned during training.
        sqrt3 = math.sqrt(3)
        encoder_embeddings = model.param_init_net.UniformFill(
            [],
            'encoder_embeddings',
            shape=[vocab_size, embedding_size],
            min=-sqrt3,
            max=sqrt3,
        )
        model.params.append(encoder_embeddings)

        # Convert inputs to embedded inputs by the embeddings above.
        embedded_encoder_inputs = model.net.Gather(
            [encoder_embeddings, inputs],
            ['embedded_encoder_inputs'],
        )

        if encoder_type == 'rnn':
            encoder_num_units = (
                encoder_params['encoder_layer_configs'][0]['num_units']
            )

            encoder_initial_cell_state = model.param_init_net.ConstantFill(
                [],
                ['encoder_initial_cell_state'],
                shape=[encoder_num_units],
                value=0.0,
            )

            encoder_initial_hidden_state = model.param_init_net.ConstantFill(
                [],
                'encoder_initial_hidden_state',
                shape=[encoder_num_units],
                value=0.0,
            )

            # Choose corresponding rnn encoder function
            if encoder_params['use_bidirectional_encoder']:
                rnn_encoder_func = seq2seq_util.rnn_bidirectional_encoder
            else:
                rnn_encoder_func = seq2seq_util.rnn_unidirectional_encoder

            (
                encoder_outputs,
                final_encoder_hidden_state,
                final_encoder_cell_state,
            ) = rnn_encoder_func(
                model,
                embedded_encoder_inputs,
                input_lengths,
                encoder_initial_hidden_state,
                encoder_initial_cell_state,
                embedding_size,
                encoder_num_units,
                use_attention
            )

        else:
            raise ValueError('Unsupported encoder type {}'.format(encoder_type))

        return (
            encoder_outputs,
            final_encoder_hidden_state,
            final_encoder_cell_state,
        )

    def output_projection(
        self,
        model,
        decoder_outputs,
        decoder_output_size,
        target_vocab_size,
        decoder_softmax_size,
    ):
        if decoder_softmax_size is not None:
            decoder_outputs = model.FC(
                decoder_outputs,
                'decoder_outputs_scaled',
                dim_in=decoder_output_size,
                dim_out=decoder_softmax_size,
            )
            decoder_output_size = decoder_softmax_size

        output_projection_w = model.param_init_net.XavierFill(
            [],
            'output_projection_w',
            shape=[self.target_vocab_size, decoder_output_size],
        )

        output_projection_b = model.param_init_net.XavierFill(
            [],
            'output_projection_b',
            shape=[self.target_vocab_size],
        )
        model.params.extend([
            output_projection_w,
            output_projection_b,
        ])
        output_logits = model.net.FC(
            [
                decoder_outputs,
                output_projection_w,
                output_projection_b,
            ],
            ['output_logits'],
        )
        return output_logits

    def _build_model(
        self,
        init_params,
    ):
        model = seq2seq_util.ModelHelper(
            init_params=init_params,
        )

        self.encoder_inputs = model.net.AddExternalInput('encoder_inputs')
        self.encoder_lengths = model.net.AddExternalInput('encoder_lengths')
        self.decoder_inputs = model.net.AddExternalInput('decoder_inputs')
        self.decoder_lengths = model.net.AddExternalInput('decoder_lengths')
        self.targets = model.net.AddExternalInput('targets')
        self.target_weights = model.net.AddExternalInput('target_weights')

        optimizer_params = self.model_params['optimizer_params']
        attention_type = self.model_params['attention']
        assert attention_type in ['none', 'regular']

        self.learning_rate = model.AddParam(
            name='learning_rate',
            init_value=float(optimizer_params['learning_rate']),
            trainable=False,
        )
        self.global_step = model.AddParam(
            name='global_step',
            init_value=0,
            trainable=False,
        )
        self.start_time = model.AddParam(
            name='start_time',
            init_value=time.time(),
            trainable=False,
        )

        assert self.num_gpus < 2
        assert len(self.encoder_params['encoder_layer_configs']) == 1
        assert len(self.model_params['decoder_layer_configs']) == 1

        encoder_num_units = (
            self.encoder_params['encoder_layer_configs'][0]['num_units']
        )
        decoder_num_units = (
            self.model_params['decoder_layer_configs'][0]['num_units']
        )

        (
            encoder_outputs,
            final_encoder_hidden_state,
            final_encoder_cell_state,
        ) = self._embedding_encoder(
            model=model,
            encoder_type=self.encoder_type,
            encoder_params=self.encoder_params,
            inputs=self.encoder_inputs,
            input_lengths=self.encoder_lengths,
            vocab_size=self.source_vocab_size,
            embedding_size=self.model_params['encoder_embedding_size'],
            use_attention=(attention_type != 'none'),
        )

        # For bidirectional RNN, the num of units doubles after encodeing
        if (
            self.encoder_type == 'rnn' and
            self.encoder_params['use_bidirectional_encoder']
        ):
            encoder_num_units *= 2

        if attention_type == 'none':
            decoder_initial_hidden_state = model.FC(
                final_encoder_hidden_state,
                'decoder_initial_hidden_state',
                encoder_num_units,
                decoder_num_units,
                axis=2,
            )
            decoder_initial_cell_state = model.FC(
                final_encoder_cell_state,
                'decoder_initial_cell_state',
                encoder_num_units,
                decoder_num_units,
                axis=2,
            )
        else:
            decoder_initial_hidden_state = model.param_init_net.ConstantFill(
                [],
                'decoder_initial_hidden_state',
                shape=[decoder_num_units],
                value=0.0,
            )
            decoder_initial_cell_state = model.param_init_net.ConstantFill(
                [],
                'decoder_initial_cell_state',
                shape=[decoder_num_units],
                value=0.0,
            )
            initial_attention_weighted_encoder_context = (
                model.param_init_net.ConstantFill(
                    [],
                    'initial_attention_weighted_encoder_context',
                    shape=[encoder_num_units],
                    value=0.0,
                )
            )

        sqrt3 = math.sqrt(3)
        decoder_embeddings = model.AddParam(
            name='decoder_embeddings',
            init=('UniformFill', dict(
                shape=[
                    self.target_vocab_size,
                    self.model_params['decoder_embedding_size'],
                ],
                min=-sqrt3,
                max=sqrt3,
            )),
        )

        embedded_decoder_inputs = model.net.Gather(
            [decoder_embeddings, self.decoder_inputs],
            ['embedded_decoder_inputs'],
        )
        # seq_len x batch_size x decoder_embedding_size
        with core.NameScope('', reset=True):
            if attention_type == 'none':
                decoder_outputs, _, _, _ = recurrent.LSTM(
                    model=model,
                    input_blob=embedded_decoder_inputs,
                    seq_lengths=self.decoder_lengths,
                    initial_states=(
                        decoder_initial_hidden_state,
                        decoder_initial_cell_state,
                    ),
                    dim_in=self.model_params['decoder_embedding_size'],
                    dim_out=decoder_num_units,
                    scope='decoder',
                    outputs_with_grads=[0],
                )
                decoder_output_size = decoder_num_units
            else:
                (
                    decoder_outputs, _, _, _,
                    attention_weighted_encoder_contexts, _
                ) = recurrent.LSTMWithAttention(
                    model=model,
                    decoder_inputs=embedded_decoder_inputs,
                    decoder_input_lengths=self.decoder_lengths,
                    initial_decoder_hidden_state=decoder_initial_hidden_state,
                    initial_decoder_cell_state=decoder_initial_cell_state,
                    initial_attention_weighted_encoder_context=(
                        initial_attention_weighted_encoder_context
                    ),
                    encoder_output_dim=encoder_num_units,
                    encoder_outputs=encoder_outputs,
                    decoder_input_dim=self.model_params['decoder_embedding_size'],
                    decoder_state_dim=decoder_num_units,
                    # TODO: remove that later
                    batch_size=self.batch_size,
                    scope='decoder',
                    outputs_with_grads=[0, 4],
                )
                decoder_outputs, _ = model.net.Concat(
                    [decoder_outputs, attention_weighted_encoder_contexts],
                    [
                        'states_and_context_combination',
                        '_states_and_context_combination_concat_dims',
                    ],
                    axis=2,
                )
                decoder_output_size = decoder_num_units + encoder_num_units

        # we do softmax over the whole sequence
        # (max_length in the batch * batch_size) x decoder embedding size
        # -1 because we don't know max_length yet
        decoder_outputs_flattened, _ = model.net.Reshape(
            [decoder_outputs],
            [
                'decoder_outputs_flattened',
                'decoder_outputs_and_contexts_combination_old_shape',
            ],
            shape=[-1, decoder_output_size],
        )
        output_logits = self.output_projection(
            model=model,
            decoder_outputs=decoder_outputs_flattened,
            decoder_output_size=decoder_output_size,
            target_vocab_size=self.target_vocab_size,
            decoder_softmax_size=self.model_params['decoder_softmax_size'],
        )
        targets, _ = model.net.Reshape(
            [self.targets],
            ['targets', 'targets_old_shape'],
            shape=[-1],
        )
        target_weights, _ = model.net.Reshape(
            [self.target_weights],
            ['target_weights', 'target_weights_old_shape'],
            shape=[-1],
        )

        output_probs, loss_per_word = model.net.SoftmaxWithLoss(
            [output_logits, targets, target_weights],
            ['OutputProbs', 'loss_per_word'],
        )

        num_words = model.net.ReduceFrontSum(
            target_weights,
            'num_words',
        )
        self.total_loss_scalar = model.net.Mul(
            [loss_per_word, num_words],
            'total_loss_scalar',
        )
        self.forward_net = model.net.Clone(
            name=model.net.Name() + '_forward_only',
        )
        # print loss only in the forward net which evaluates loss after every
        # epoch
        self.forward_net.Print([self.total_loss_scalar], [])

        # Note: average over batch.
        # It is tricky because of two problems:
        # 1. ReduceFrontSum from 1-D tensor returns 0-D tensor
        # 2. If you want to multiply 0-D by 1-D tensor
        # (by scalar batch_size_inverse_tensor),
        # you need to use broadcasting. But gradient propogation
        # is broken for op with broadcasting.
        # total_loss_scalar, _ = model.net.Reshape(
        #     [total_loss_scalar],
        #     [total_loss_scalar, 'total_loss_scalar_old_shape'],
        #     shape=[1],
        # )
        batch_size_inverse_tensor = (
            model.param_init_net.ConstantFill(
                [],
                'batch_size_tensor',
                shape=[],
                value=1.0 / self.batch_size,
            )
        )
        total_loss_scalar_average = model.net.Mul(
            [self.total_loss_scalar, batch_size_inverse_tensor],
            ['total_loss_scalar_average'],
        )

        model.AddGradientOperators([
            total_loss_scalar_average,
        ])
        ONE = model.param_init_net.ConstantFill(
            [],
            'ONE',
            shape=[1],
            value=1.0,
        )
        logger.info('All trainable variables: ')

        for param in model.params:
            param_grad = model.param_to_grad[param]
            if param in model.param_to_grad:
                if isinstance(param_grad, core.GradientSlice):
                    param_grad_values = param_grad.values
                    param_grad_values = model.net.Clip(
                        [param_grad_values],
                        [param_grad_values],
                        min=0.0,
                        max=float(self.model_params['max_grad_value']),
                    )
                    model.net.ScatterWeightedSum(
                        [
                            param,
                            ONE,
                            param_grad.indices,
                            param_grad_values,
                            model.net.Negative(
                                [self.learning_rate],
                                'negative_learning_rate',
                            ),
                        ],
                        param,
                    )
                else:
                    param_grad = model.net.Clip(
                        [param_grad],
                        [param_grad],
                        min=0.0,
                        max=float(self.model_params['max_grad_value']),
                    )
                    model.net.WeightedSum(
                        [
                            param,
                            ONE,
                            param_grad,
                            model.net.Negative(
                                [self.learning_rate],
                                'negative_learning_rate',
                            ),
                        ],
                        param,
                    )
        self.model = model

    def _init_model(self):
        workspace.RunNetOnce(self.model.param_init_net)

        def create_net(net):
            workspace.CreateNet(
                net,
                input_blobs=map(str, net.external_inputs),
            )

        create_net(self.model.net)
        create_net(self.forward_net)

    def __init__(
        self,
        model_params,
        source_vocab_size,
        target_vocab_size,
        num_gpus=1,
        num_cpus=1,
    ):
        self.model_params = model_params
        self.encoder_type = 'rnn'
        self.encoder_params = model_params['encoder_type']
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.num_gpus = num_gpus
        self.num_cpus = num_cpus
        self.batch_size = model_params['batch_size']

        workspace.GlobalInit([
            'caffe2',
            # NOTE: modify log level for debugging purposes
            '--caffe2_log_level=0',
            # NOTE: modify log level for debugging purposes
            '--v=0',
            # Fail gracefully if one of the threads fails
            '--caffe2_handle_executor_threads_exceptions=1',
            '--caffe2_mkl_num_threads=' + str(self.num_cpus),
        ])

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        workspace.ResetWorkspace()

    def initialize_from_scratch(self):
        logger.info('Initializing Seq2SeqModelCaffe2 from scratch: Start')
        self._build_model(init_params=True)
        self._init_model()
        logger.info('Initializing Seq2SeqModelCaffe2 from scratch: Finish')

    def get_current_step(self):
        return workspace.FetchBlob(self.global_step)[0]

    def inc_current_step(self):
        workspace.FeedBlob(
            self.global_step,
            np.array([self.get_current_step() + 1]),
        )

    def step(
        self,
        batch,
        forward_only
    ):
        batch_obj = prepare_batch(batch)
        for batch_obj_name, batch_obj_value in izip(Batch._fields, batch_obj):
            workspace.FeedBlob(batch_obj_name, batch_obj_value)

        if forward_only:
            workspace.RunNet(self.forward_net)
        else:
            workspace.RunNet(self.model.net)
            self.inc_current_step()

        return workspace.FetchBlob(self.total_loss_scalar)


def gen_vocab(corpus, unk_threshold):
    vocab = collections.defaultdict(lambda: len(vocab))
    freqs = collections.defaultdict(lambda: 0)
    # Adding padding tokens to the vocabulary to maintain consistency with IDs
    vocab[PAD]
    vocab[GO]
    vocab[EOS]

    with open(corpus) as f:
        for sentence in f:
            tokens = sentence.strip().split()
            for token in tokens:
                freqs[token] += 1
    for token, freq in freqs.items():
        if freq > unk_threshold:
            # TODO: Add reverse lookup dict when it becomes necessary
            vocab[token]

    return vocab


def gen_batches(source_corpus, target_corpus, source_vocab, target_vocab,
                batch_size):
    batches = []
    with open(source_corpus) as source, open(target_corpus) as target:
        elems = 0
        batch = []
        for source_sentence, target_sentence in zip(source, target):

            # Convert sentence into list of vocabulary indices
            def get_numberized_sentence(sentence, vocab):
                sentence_with_id = []
                for token in sentence.strip().split():
                    if token in vocab:
                        sentence_with_id.append(vocab[token])
                    else:
                        sentence_with_id.append(vocab[UNK])
                sentence_with_id.append(vocab[EOS])
                return sentence_with_id

            batch.append(tuple([get_numberized_sentence(sentence, vocab)
                                 for vocab, sentence in
                                 zip([source_vocab, target_vocab],
                                     [source_sentence, target_sentence])
                                ]))
            elems += 1
            if elems >= batch_size:
                batches.append(batch)
                batch = []
                elems = 0
    if len(batch) > 0:
        while len(batch) < batch_size:
            batch.append(batch[-1])
        assert len(batch) == batch_size
        batches.append(batch)
    return batches


def run_seq2seq_model(args, model_params=None):
    source_vocab = gen_vocab(args.source_corpus, args.unk_threshold)
    target_vocab = gen_vocab(args.target_corpus, args.unk_threshold)
    batches = gen_batches(args.source_corpus, args.target_corpus, source_vocab,
                          target_vocab, model_params['batch_size'])

    logger.info('Source vocab size', len(source_vocab))
    logger.info('Target vocab size', len(target_vocab))

    batches_eval = gen_batches(args.source_corpus_eval, args.target_corpus_eval,
                               source_vocab, target_vocab,
                               model_params['batch_size'])

    with Seq2SeqModelCaffe2(
        model_params=model_params,
        source_vocab_size=len(source_vocab),
        target_vocab_size=len(target_vocab),
        num_gpus=0,
    ) as model_obj:
        model_obj.initialize_from_scratch()
        for _ in range(args.epochs):
            for batch in batches:
                model_obj.step(
                    batch=batch,
                    forward_only=False,
                )
            model_obj.step(
                # merge all eval batches for scalar loss computation
                batch=[entry for batch in batches_eval for entry in batch],
                forward_only=True,
            )


def run_seq2seq_rnn_unidirection_with_no_attention(args):
    run_seq2seq_model(args, model_params=dict(
        attention='none',
        decoder_layer_configs=[
            dict(
                num_units=args.decoder_cell_num_units,
            ),
        ],
        encoder_type=dict(
            encoder_layer_configs=[
                dict(
                    num_units=args.encoder_cell_num_units,
                ),
            ],
            use_bidirectional_encoder=args.use_bidirectional_encoder,
        ),
        batch_size=args.batch_size,
        optimizer_params=dict(
            learning_rate=args.learning_rate,
        ),
        encoder_embedding_size=args.encoder_embedding_size,
        decoder_embedding_size=args.decoder_embedding_size,
        decoder_softmax_size=args.decoder_softmax_size,
        max_grad_value=args.max_grad_value,
    ))


def main():
    parser = argparse.ArgumentParser(
        description='Caffe2: Seq2Seq Training'
    )
    parser.add_argument('--source-corpus', type=str, default=None,
                        help='Path to source corpus in a text file format. Each '
                        'line in the file should contain a single sentence',
                        required=True)
    parser.add_argument('--target-corpus', type=str, default=None,
                        help='Path to target corpus in a text file format',
                        required=True)
    parser.add_argument('--batch-size', type=int, default=500,
                        help='Training batch size')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of iterations over training data')
    parser.add_argument('--learning-rate', type=float, default=0.1,
                        help='Learning rate')
    parser.add_argument('--unk-threshold', type=int, default=5,
                        help='Threshold frequency under which token becomes '
                        'labelled unknown token')
    parser.add_argument('--max-grad-value', type=float, default=5.0,
                        help='Max clip value of gradients at the end of each '
                        'backward pass')
    parser.add_argument('--use-bidirectional-encoder', action='store_true',
                        help='Set flag to use bidirectional recurrent network '
                        'in encoder')
    parser.add_argument('--source-corpus-eval', type=str, default=None,
                        help='Path to source corpus for evaluation in a text '
                        'file format', required=True)
    parser.add_argument('--target-corpus-eval', type=str, default=None,
                        help='Path to target corpus for evaluation in a text '
                        'file format', required=True)
    parser.add_argument('--encoder-cell-num-units', type=int, default=25,
                        help='Number of cell units in the encoder layer')
    parser.add_argument('--decoder-cell-num-units', type=int, default=50,
                        help='Number of cell units in the decoder layer')
    parser.add_argument('--encoder-embedding-size', type=int, default=25,
                        help='Size of embedding in the encoder layer')
    parser.add_argument('--decoder-embedding-size', type=int, default=25,
                        help='Size of embedding in the decoder layer')
    parser.add_argument('--decoder-softmax-size', type=int, default=25,
                        help='Size of softmax layer in the decoder')

    args = parser.parse_args()

    run_seq2seq_rnn_unidirection_with_no_attention(args)


if __name__ == '__main__':
    main()
