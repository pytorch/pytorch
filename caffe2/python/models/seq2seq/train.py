## @package train
# Module caffe2.python.models.seq2seq.train





import argparse
import collections
import logging
import math
import numpy as np
import random
import time
import sys
import os

import caffe2.proto.caffe2_pb2 as caffe2_pb2
from caffe2.python import core, workspace, data_parallel_model
import caffe2.python.models.seq2seq.seq2seq_util as seq2seq_util
from caffe2.python.models.seq2seq.seq2seq_model_helper import Seq2SeqModelHelper


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stderr))

Batch = collections.namedtuple('Batch', [
    'encoder_inputs',
    'encoder_lengths',
    'decoder_inputs',
    'decoder_lengths',
    'targets',
    'target_weights',
])


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
            [seq2seq_util.PAD_ID] * (max_encoder_length - len(source_seq))
        )
        batch_encoder_inputs.append(
            list(reversed(source_seq)) + encoder_pads
        )

        decoder_pads = (
            [seq2seq_util.PAD_ID] * (max_decoder_length - len(target_seq))
        )
        target_seq_with_go_token = [seq2seq_util.GO_ID] + target_seq
        decoder_lengths.append(len(target_seq_with_go_token))
        batch_decoder_inputs.append(target_seq_with_go_token + decoder_pads)

        target_seq_with_eos = target_seq + [seq2seq_util.EOS_ID]
        targets = target_seq_with_eos + decoder_pads
        batch_targets.append(targets)

        if len(source_seq) + len(target_seq) == 0:
            target_weights = [0] * len(targets)
        else:
            target_weights = [
                1 if target != seq2seq_util.PAD_ID else 0
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

    def _build_model(
        self,
        init_params,
    ):
        model = Seq2SeqModelHelper(init_params=init_params)
        self._build_shared(model)
        self._build_embeddings(model)

        forward_model = Seq2SeqModelHelper(init_params=init_params)
        self._build_shared(forward_model)
        self._build_embeddings(forward_model)

        if self.num_gpus == 0:
            loss_blobs = self.model_build_fun(model)
            model.AddGradientOperators(loss_blobs)
            self.norm_clipped_grad_update(
                model,
                scope='norm_clipped_grad_update'
            )
            self.forward_model_build_fun(forward_model)

        else:
            assert (self.batch_size % self.num_gpus) == 0

            data_parallel_model.Parallelize_GPU(
                forward_model,
                input_builder_fun=lambda m: None,
                forward_pass_builder_fun=self.forward_model_build_fun,
                param_update_builder_fun=None,
                devices=list(range(self.num_gpus)),
            )

            def clipped_grad_update_bound(model):
                self.norm_clipped_grad_update(
                    model,
                    scope='norm_clipped_grad_update',
                )

            data_parallel_model.Parallelize_GPU(
                model,
                input_builder_fun=lambda m: None,
                forward_pass_builder_fun=self.model_build_fun,
                param_update_builder_fun=clipped_grad_update_bound,
                devices=list(range(self.num_gpus)),
            )
        self.norm_clipped_sparse_grad_update(
            model,
            scope='norm_clipped_sparse_grad_update',
        )
        self.model = model
        self.forward_net = forward_model.net

    def _build_shared(self, model):
        optimizer_params = self.model_params['optimizer_params']
        with core.DeviceScope(core.DeviceOption(caffe2_pb2.CPU)):
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

    def _build_embeddings(self, model):
        with core.DeviceScope(core.DeviceOption(caffe2_pb2.CPU)):
            sqrt3 = math.sqrt(3)
            self.encoder_embeddings = model.param_init_net.UniformFill(
                [],
                'encoder_embeddings',
                shape=[
                    self.source_vocab_size,
                    self.model_params['encoder_embedding_size'],
                ],
                min=-sqrt3,
                max=sqrt3,
            )
            model.params.append(self.encoder_embeddings)
            self.decoder_embeddings = model.param_init_net.UniformFill(
                [],
                'decoder_embeddings',
                shape=[
                    self.target_vocab_size,
                    self.model_params['decoder_embedding_size'],
                ],
                min=-sqrt3,
                max=sqrt3,
            )
            model.params.append(self.decoder_embeddings)

    def model_build_fun(self, model, forward_only=False, loss_scale=None):
        encoder_inputs = model.net.AddExternalInput(
            workspace.GetNameScope() + 'encoder_inputs',
        )
        encoder_lengths = model.net.AddExternalInput(
            workspace.GetNameScope() + 'encoder_lengths',
        )
        decoder_inputs = model.net.AddExternalInput(
            workspace.GetNameScope() + 'decoder_inputs',
        )
        decoder_lengths = model.net.AddExternalInput(
            workspace.GetNameScope() + 'decoder_lengths',
        )
        targets = model.net.AddExternalInput(
            workspace.GetNameScope() + 'targets',
        )
        target_weights = model.net.AddExternalInput(
            workspace.GetNameScope() + 'target_weights',
        )
        attention_type = self.model_params['attention']
        assert attention_type in ['none', 'regular', 'dot']

        (
            encoder_outputs,
            weighted_encoder_outputs,
            final_encoder_hidden_states,
            final_encoder_cell_states,
            encoder_units_per_layer,
        ) = seq2seq_util.build_embedding_encoder(
            model=model,
            encoder_params=self.encoder_params,
            num_decoder_layers=len(self.model_params['decoder_layer_configs']),
            inputs=encoder_inputs,
            input_lengths=encoder_lengths,
            vocab_size=self.source_vocab_size,
            embeddings=self.encoder_embeddings,
            embedding_size=self.model_params['encoder_embedding_size'],
            use_attention=(attention_type != 'none'),
            num_gpus=self.num_gpus,
        )

        (
            decoder_outputs,
            decoder_output_size,
        ) = seq2seq_util.build_embedding_decoder(
            model,
            decoder_layer_configs=self.model_params['decoder_layer_configs'],
            inputs=decoder_inputs,
            input_lengths=decoder_lengths,
            encoder_lengths=encoder_lengths,
            encoder_outputs=encoder_outputs,
            weighted_encoder_outputs=weighted_encoder_outputs,
            final_encoder_hidden_states=final_encoder_hidden_states,
            final_encoder_cell_states=final_encoder_cell_states,
            encoder_units_per_layer=encoder_units_per_layer,
            vocab_size=self.target_vocab_size,
            embeddings=self.decoder_embeddings,
            embedding_size=self.model_params['decoder_embedding_size'],
            attention_type=attention_type,
            forward_only=False,
            num_gpus=self.num_gpus,
        )

        output_logits = seq2seq_util.output_projection(
            model=model,
            decoder_outputs=decoder_outputs,
            decoder_output_size=decoder_output_size,
            target_vocab_size=self.target_vocab_size,
            decoder_softmax_size=self.model_params['decoder_softmax_size'],
        )
        targets, _ = model.net.Reshape(
            [targets],
            ['targets', 'targets_old_shape'],
            shape=[-1],
        )
        target_weights, _ = model.net.Reshape(
            [target_weights],
            ['target_weights', 'target_weights_old_shape'],
            shape=[-1],
        )
        _, loss_per_word = model.net.SoftmaxWithLoss(
            [output_logits, targets, target_weights],
            ['OutputProbs_INVALID', 'loss_per_word'],
            only_loss=True,
        )

        num_words = model.net.SumElements(
            [target_weights],
            'num_words',
        )
        total_loss_scalar = model.net.Mul(
            [loss_per_word, num_words],
            'total_loss_scalar',
        )
        total_loss_scalar_weighted = model.net.Scale(
            [total_loss_scalar],
            'total_loss_scalar_weighted',
            scale=1.0 / self.batch_size,
        )
        return [total_loss_scalar_weighted]

    def forward_model_build_fun(self, model, loss_scale=None):
        return self.model_build_fun(
            model=model,
            forward_only=True,
            loss_scale=loss_scale
        )

    def _calc_norm_ratio(self, model, params, scope, ONE):
        with core.NameScope(scope):
            grad_squared_sums = []
            for i, param in enumerate(params):
                logger.info(param)
                grad = (
                    model.param_to_grad[param]
                    if not isinstance(
                        model.param_to_grad[param],
                        core.GradientSlice,
                    ) else model.param_to_grad[param].values
                )
                grad_squared = model.net.Sqr(
                    [grad],
                    'grad_{}_squared'.format(i),
                )
                grad_squared_sum = model.net.SumElements(
                    grad_squared,
                    'grad_{}_squared_sum'.format(i),
                )
                grad_squared_sums.append(grad_squared_sum)

            grad_squared_full_sum = model.net.Sum(
                grad_squared_sums,
                'grad_squared_full_sum',
            )
            global_norm = model.net.Pow(
                grad_squared_full_sum,
                'global_norm',
                exponent=0.5,
            )
            clip_norm = model.param_init_net.ConstantFill(
                [],
                'clip_norm',
                shape=[],
                value=float(self.model_params['max_gradient_norm']),
            )
            max_norm = model.net.Max(
                [global_norm, clip_norm],
                'max_norm',
            )
            norm_ratio = model.net.Div(
                [clip_norm, max_norm],
                'norm_ratio',
            )
            return norm_ratio

    def _apply_norm_ratio(
        self, norm_ratio, model, params, learning_rate, scope, ONE
    ):
        for param in params:
            param_grad = model.param_to_grad[param]
            nlr = model.net.Negative(
                [learning_rate],
                'negative_learning_rate',
            )
            with core.NameScope(scope):
                update_coeff = model.net.Mul(
                    [nlr, norm_ratio],
                    'update_coeff',
                    broadcast=1,
                )
            if isinstance(param_grad, core.GradientSlice):
                param_grad_values = param_grad.values

                model.net.ScatterWeightedSum(
                    [
                        param,
                        ONE,
                        param_grad.indices,
                        param_grad_values,
                        update_coeff,
                    ],
                    param,
                )
            else:
                model.net.WeightedSum(
                    [
                        param,
                        ONE,
                        param_grad,
                        update_coeff,
                    ],
                    param,
                )

    def norm_clipped_grad_update(self, model, scope):

        if self.num_gpus == 0:
            learning_rate = self.learning_rate
        else:
            learning_rate = model.CopyCPUToGPU(self.learning_rate, 'LR')

        params = []
        for param in model.GetParams(top_scope=True):
            if param in model.param_to_grad:
                if not isinstance(
                    model.param_to_grad[param],
                    core.GradientSlice,
                ):
                    params.append(param)

        ONE = model.param_init_net.ConstantFill(
            [],
            'ONE',
            shape=[1],
            value=1.0,
        )
        logger.info('Dense trainable variables: ')
        norm_ratio = self._calc_norm_ratio(model, params, scope, ONE)
        self._apply_norm_ratio(
            norm_ratio, model, params, learning_rate, scope, ONE
        )

    def norm_clipped_sparse_grad_update(self, model, scope):
        learning_rate = self.learning_rate

        params = []
        for param in model.GetParams(top_scope=True):
            if param in model.param_to_grad:
                if isinstance(
                    model.param_to_grad[param],
                    core.GradientSlice,
                ):
                    params.append(param)

        ONE = model.param_init_net.ConstantFill(
            [],
            'ONE',
            shape=[1],
            value=1.0,
        )
        logger.info('Sparse trainable variables: ')
        norm_ratio = self._calc_norm_ratio(model, params, scope, ONE)
        self._apply_norm_ratio(
            norm_ratio, model, params, learning_rate, scope, ONE
        )

    def total_loss_scalar(self):
        if self.num_gpus == 0:
            return workspace.FetchBlob('total_loss_scalar')
        else:
            total_loss = 0
            for i in range(self.num_gpus):
                name = 'gpu_{}/total_loss_scalar'.format(i)
                gpu_loss = workspace.FetchBlob(name)
                total_loss += gpu_loss
            return total_loss

    def _init_model(self):
        workspace.RunNetOnce(self.model.param_init_net)

        def create_net(net):
            workspace.CreateNet(
                net,
                input_blobs=[str(i) for i in net.external_inputs],
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
        if self.num_gpus < 1:
            batch_obj = prepare_batch(batch)
            for batch_obj_name, batch_obj_value in zip(
                Batch._fields,
                batch_obj,
            ):
                workspace.FeedBlob(batch_obj_name, batch_obj_value)
        else:
            for i in range(self.num_gpus):
                gpu_batch = batch[i::self.num_gpus]
                batch_obj = prepare_batch(gpu_batch)
                for batch_obj_name, batch_obj_value in zip(
                    Batch._fields,
                    batch_obj,
                ):
                    name = 'gpu_{}/{}'.format(i, batch_obj_name)
                    if batch_obj_name in ['encoder_inputs', 'decoder_inputs']:
                        dev = core.DeviceOption(caffe2_pb2.CPU)
                    else:
                        dev = core.DeviceOption(workspace.GpuDeviceType, i)
                    workspace.FeedBlob(name, batch_obj_value, device_option=dev)

        if forward_only:
            workspace.RunNet(self.forward_net)
        else:
            workspace.RunNet(self.model.net)
            self.inc_current_step()

        return self.total_loss_scalar()

    def save(self, checkpoint_path_prefix, current_step):
        checkpoint_path = '{0}-{1}'.format(
            checkpoint_path_prefix,
            current_step,
        )

        assert workspace.RunOperatorOnce(core.CreateOperator(
            'Save',
            self.model.GetAllParams(),
            [],
            absolute_path=True,
            db=checkpoint_path,
            db_type='minidb',
        ))

        checkpoint_config_path = os.path.join(
            os.path.dirname(checkpoint_path_prefix),
            'checkpoint',
        )
        with open(checkpoint_config_path, 'w') as checkpoint_config_file:
            checkpoint_config_file.write(
                'model_checkpoint_path: "' + checkpoint_path + '"\n'
                'all_model_checkpoint_paths: "' + checkpoint_path + '"\n'
            )
            logger.info('Saved checkpoint file to ' + checkpoint_path)

        return checkpoint_path

def gen_batches(source_corpus, target_corpus, source_vocab, target_vocab,
                batch_size, max_length):
    with open(source_corpus) as source, open(target_corpus) as target:
        parallel_sentences = []
        for source_sentence, target_sentence in zip(source, target):
            numerized_source_sentence = seq2seq_util.get_numberized_sentence(
                source_sentence,
                source_vocab,
            )
            numerized_target_sentence = seq2seq_util.get_numberized_sentence(
                target_sentence,
                target_vocab,
            )
            if (
                len(numerized_source_sentence) > 0 and
                len(numerized_target_sentence) > 0 and
                (
                    max_length is None or (
                        len(numerized_source_sentence) <= max_length and
                        len(numerized_target_sentence) <= max_length
                    )
                )
            ):
                parallel_sentences.append((
                    numerized_source_sentence,
                    numerized_target_sentence,
                ))
    parallel_sentences.sort(key=lambda s_t: (len(s_t[0]), len(s_t[1])))

    batches, batch = [], []
    for sentence_pair in parallel_sentences:
        batch.append(sentence_pair)
        if len(batch) >= batch_size:
            batches.append(batch)
            batch = []
    if len(batch) > 0:
        while len(batch) < batch_size:
            batch.append(batch[-1])
        assert len(batch) == batch_size
        batches.append(batch)
    random.shuffle(batches)
    return batches


def run_seq2seq_model(args, model_params=None):
    source_vocab = seq2seq_util.gen_vocab(
        args.source_corpus,
        args.unk_threshold,
    )
    target_vocab = seq2seq_util.gen_vocab(
        args.target_corpus,
        args.unk_threshold,
    )
    logger.info('Source vocab size {}'.format(len(source_vocab)))
    logger.info('Target vocab size {}'.format(len(target_vocab)))

    batches = gen_batches(args.source_corpus, args.target_corpus, source_vocab,
                          target_vocab, model_params['batch_size'],
                          args.max_length)
    logger.info('Number of training batches {}'.format(len(batches)))

    batches_eval = gen_batches(args.source_corpus_eval, args.target_corpus_eval,
                               source_vocab, target_vocab,
                               model_params['batch_size'], args.max_length)
    logger.info('Number of eval batches {}'.format(len(batches_eval)))

    with Seq2SeqModelCaffe2(
        model_params=model_params,
        source_vocab_size=len(source_vocab),
        target_vocab_size=len(target_vocab),
        num_gpus=args.num_gpus,
        num_cpus=20,
    ) as model_obj:
        model_obj.initialize_from_scratch()
        for i in range(args.epochs):
            logger.info('Epoch {}'.format(i))
            total_loss = 0
            for batch in batches:
                total_loss += model_obj.step(
                    batch=batch,
                    forward_only=False,
                )
            logger.info('\ttraining loss {}'.format(total_loss))
            total_loss = 0
            for batch in batches_eval:
                total_loss += model_obj.step(
                    batch=batch,
                    forward_only=True,
                )
            logger.info('\teval loss {}'.format(total_loss))
            if args.checkpoint is not None:
                model_obj.save(args.checkpoint, i)


def main():
    random.seed(31415)
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
    parser.add_argument('--max-length', type=int, default=None,
                        help='Maximal lengths of train and eval sentences')
    parser.add_argument('--unk-threshold', type=int, default=50,
                        help='Threshold frequency under which token becomes '
                        'labeled unknown token')

    parser.add_argument('--batch-size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of iterations over training data')
    parser.add_argument('--learning-rate', type=float, default=0.5,
                        help='Learning rate')
    parser.add_argument('--max-gradient-norm', type=float, default=1.0,
                        help='Max global norm of gradients at the end of each '
                        'backward pass. We do clipping to match the number.')
    parser.add_argument('--num-gpus', type=int, default=0,
                        help='Number of GPUs for data parallel model')

    parser.add_argument('--use-bidirectional-encoder', action='store_true',
                        help='Set flag to use bidirectional recurrent network '
                        'for first layer of encoder')
    parser.add_argument('--use-attention', action='store_true',
                        help='Set flag to use seq2seq with attention model')
    parser.add_argument('--source-corpus-eval', type=str, default=None,
                        help='Path to source corpus for evaluation in a text '
                        'file format', required=True)
    parser.add_argument('--target-corpus-eval', type=str, default=None,
                        help='Path to target corpus for evaluation in a text '
                        'file format', required=True)
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

    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint')

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

    run_seq2seq_model(args, model_params=dict(
        attention=('regular' if args.use_attention else 'none'),
        decoder_layer_configs=decoder_layer_configs,
        encoder_type=dict(
            encoder_layer_configs=encoder_layer_configs,
            use_bidirectional_encoder=args.use_bidirectional_encoder,
        ),
        batch_size=args.batch_size,
        optimizer_params=dict(
            learning_rate=args.learning_rate,
        ),
        encoder_embedding_size=args.encoder_embedding_size,
        decoder_embedding_size=args.decoder_embedding_size,
        decoder_softmax_size=args.decoder_softmax_size,
        max_gradient_norm=args.max_gradient_norm,
    ))


if __name__ == '__main__':
    main()
