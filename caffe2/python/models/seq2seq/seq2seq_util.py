## @package seq2seq_util
# Module caffe2.python.examples.seq2seq_util
""" A bunch of util functions to build Seq2Seq models with Caffe2."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import collections
from future.utils import viewitems

import caffe2.proto.caffe2_pb2 as caffe2_pb2
from caffe2.python import attention, core, rnn_cell, brew


PAD_ID = 0
PAD = '<PAD>'
GO_ID = 1
GO = '<GO>'
EOS_ID = 2
EOS = '<EOS>'
UNK_ID = 3
UNK = '<UNK>'


def gen_vocab(corpus, unk_threshold):
    vocab = collections.defaultdict(lambda: len(vocab))
    freqs = collections.defaultdict(lambda: 0)
    # Adding padding tokens to the vocabulary to maintain consistency with IDs
    vocab[PAD]
    vocab[GO]
    vocab[EOS]
    vocab[UNK]

    with open(corpus) as f:
        for sentence in f:
            tokens = sentence.strip().split()
            for token in tokens:
                freqs[token] += 1
    for token, freq in viewitems(freqs):
        if freq > unk_threshold:
            vocab[token]

    return vocab


def get_numberized_sentence(sentence, vocab):
    numerized_sentence = []
    for token in sentence.strip().split():
        if token in vocab:
            numerized_sentence.append(vocab[token])
        else:
            numerized_sentence.append(vocab[UNK])
    return numerized_sentence


def rnn_unidirectional_layer(
    model,
    inputs,
    input_lengths,
    input_size,
    num_units,
    dropout_keep_prob,
    forward_only,
    return_sequence_output,
    return_final_state,
    scope=None,
):
    """ Unidirectional LSTM encoder."""
    with core.NameScope(scope):
        initial_cell_state = model.param_init_net.ConstantFill(
            [],
            'initial_cell_state',
            shape=[num_units],
            value=0.0,
        )
        initial_hidden_state = model.param_init_net.ConstantFill(
            [],
            'initial_hidden_state',
            shape=[num_units],
            value=0.0,
        )

    cell = rnn_cell.LSTMCell(
        input_size=input_size,
        hidden_size=num_units,
        forget_bias=0.0,
        memory_optimization=False,
        name=(scope + '/' if scope else '') + 'lstm',
        forward_only=forward_only,
    )

    dropout_ratio = (
        None if dropout_keep_prob is None else (1.0 - dropout_keep_prob)
    )
    if dropout_ratio is not None:
        cell = rnn_cell.DropoutCell(
            internal_cell=cell,
            dropout_ratio=dropout_ratio,
            name=(scope + '/' if scope else '') + 'dropout',
            forward_only=forward_only,
            is_test=False,
        )

    outputs_with_grads = []
    if return_sequence_output:
        outputs_with_grads.append(0)
    if return_final_state:
        outputs_with_grads.extend([1, 3])

    outputs, (_, final_hidden_state, _, final_cell_state) = (
        cell.apply_over_sequence(
            model=model,
            inputs=inputs,
            seq_lengths=input_lengths,
            initial_states=(initial_hidden_state, initial_cell_state),
            outputs_with_grads=outputs_with_grads,
        )
    )
    return outputs, final_hidden_state, final_cell_state


def rnn_bidirectional_layer(
    model,
    inputs,
    input_lengths,
    input_size,
    num_units,
    dropout_keep_prob,
    forward_only,
    return_sequence_output,
    return_final_state,
    scope=None,
):
    outputs_fw, final_hidden_fw, final_cell_fw = rnn_unidirectional_layer(
        model,
        inputs,
        input_lengths,
        input_size,
        num_units,
        dropout_keep_prob,
        forward_only,
        return_sequence_output,
        return_final_state,
        scope=(scope + '/' if scope else '') + 'fw',
    )
    with core.NameScope(scope):
        reversed_inputs = model.net.ReversePackedSegs(
            [inputs, input_lengths],
            ['reversed_inputs'],
        )
    outputs_bw, final_hidden_bw, final_cell_bw = rnn_unidirectional_layer(
        model,
        reversed_inputs,
        input_lengths,
        input_size,
        num_units,
        dropout_keep_prob,
        forward_only,
        return_sequence_output,
        return_final_state,
        scope=(scope + '/' if scope else '') + 'bw',
    )
    with core.NameScope(scope):
        outputs_bw = model.net.ReversePackedSegs(
            [outputs_bw, input_lengths],
            ['outputs_bw'],
        )

    # Concatenate forward and backward results
    if return_sequence_output:
        with core.NameScope(scope):
            outputs, _ = model.net.Concat(
                [outputs_fw, outputs_bw],
                ['outputs', 'outputs_dim'],
                axis=2,
            )
    else:
        outputs = None

    if return_final_state:
        with core.NameScope(scope):
            final_hidden_state, _ = model.net.Concat(
                [final_hidden_fw, final_hidden_bw],
                ['final_hidden_state', 'final_hidden_state_dim'],
                axis=2,
            )
            final_cell_state, _ = model.net.Concat(
                [final_cell_fw, final_cell_bw],
                ['final_cell_state', 'final_cell_state_dim'],
                axis=2,
            )
    else:
        final_hidden_state = None
        final_cell_state = None

    return outputs, final_hidden_state, final_cell_state


def build_embeddings(
    model,
    vocab_size,
    embedding_size,
    name,
    freeze_embeddings,
):
    embeddings = model.param_init_net.GaussianFill(
        [],
        name,
        shape=[vocab_size, embedding_size],
        std=0.1,
    )
    if not freeze_embeddings:
        model.params.append(embeddings)
    return embeddings


def get_layer_scope(scope, layer_type, i):
    prefix = (scope + '/' if scope else '') + layer_type
    return '{}/layer{}'.format(prefix, i)


def build_embedding_encoder(
    model,
    encoder_params,
    num_decoder_layers,
    inputs,
    input_lengths,
    vocab_size,
    embeddings,
    embedding_size,
    use_attention,
    num_gpus=0,
    forward_only=False,
    scope=None,
):
    with core.NameScope(scope or ''):
        if num_gpus == 0:
            embedded_encoder_inputs = model.net.Gather(
                [embeddings, inputs],
                ['embedded_encoder_inputs'],
            )
        else:
            with core.DeviceScope(core.DeviceOption(caffe2_pb2.CPU)):
                embedded_encoder_inputs_cpu = model.net.Gather(
                    [embeddings, inputs],
                    ['embedded_encoder_inputs_cpu'],
                )
            embedded_encoder_inputs = model.CopyCPUToGPU(
                embedded_encoder_inputs_cpu,
                'embedded_encoder_inputs',
            )

    layer_inputs = embedded_encoder_inputs
    layer_input_size = embedding_size
    encoder_units_per_layer = []
    final_encoder_hidden_states = []
    final_encoder_cell_states = []

    num_encoder_layers = len(encoder_params['encoder_layer_configs'])
    use_bidirectional_encoder = encoder_params.get(
        'use_bidirectional_encoder',
        False,
    )

    for i, layer_config in enumerate(encoder_params['encoder_layer_configs']):

        if use_bidirectional_encoder and i == 0:
            layer_func = rnn_bidirectional_layer
            output_dims = 2 * layer_config['num_units']
        else:
            layer_func = rnn_unidirectional_layer
            output_dims = layer_config['num_units']
        encoder_units_per_layer.append(output_dims)

        is_final_layer = (i == num_encoder_layers - 1)

        dropout_keep_prob = layer_config.get(
            'dropout_keep_prob',
            None,
        )

        return_final_state = i >= (num_encoder_layers - num_decoder_layers)
        (
            layer_outputs,
            final_layer_hidden_state,
            final_layer_cell_state,
        ) = layer_func(
            model=model,
            inputs=layer_inputs,
            input_lengths=input_lengths,
            input_size=layer_input_size,
            num_units=layer_config['num_units'],
            dropout_keep_prob=dropout_keep_prob,
            forward_only=forward_only,
            return_sequence_output=(not is_final_layer) or use_attention,
            return_final_state=return_final_state,
            scope=get_layer_scope(scope, 'encoder', i),
        )

        if not is_final_layer:
            layer_inputs = layer_outputs
            layer_input_size = output_dims
        final_encoder_hidden_states.append(final_layer_hidden_state)
        final_encoder_cell_states.append(final_layer_cell_state)

    encoder_outputs = layer_outputs
    weighted_encoder_outputs = None

    return (
        encoder_outputs,
        weighted_encoder_outputs,
        final_encoder_hidden_states,
        final_encoder_cell_states,
        encoder_units_per_layer,
    )


class LSTMWithAttentionDecoder(object):

    def scope(self, name):
        return self.name + '/' + name if self.name is not None else name

    def _get_attention_type(self, attention_type_as_string):
        if attention_type_as_string == 'regular':
            return attention.AttentionType.Regular
        elif attention_type_as_string == 'recurrent':
            return attention.AttentionType.Recurrent
        else:
            assert False, 'Unknown type ' + attention_type_as_string

    def __init__(
        self,
        encoder_outputs,
        encoder_output_dim,
        encoder_lengths,
        vocab_size,
        attention_type,
        embedding_size,
        decoder_num_units,
        decoder_cells,
        residual_output_layers=None,
        name=None,
        weighted_encoder_outputs=None,
    ):
        self.name = name
        self.num_layers = len(decoder_cells)
        if attention_type == 'none':
            self.cell = rnn_cell.MultiRNNCell(
                decoder_cells,
                name=self.scope('decoder'),
                residual_output_layers=residual_output_layers,
            )
            self.use_attention = False
            self.decoder_output_dim = decoder_num_units
            self.output_indices = self.cell.output_indices
        else:
            decoder_cell = rnn_cell.MultiRNNCell(
                decoder_cells,
                name=self.scope('decoder'),
                residual_output_layers=residual_output_layers,
            )
            self.cell = rnn_cell.AttentionCell(
                encoder_output_dim=encoder_output_dim,
                encoder_outputs=encoder_outputs,
                encoder_lengths=encoder_lengths,
                decoder_cell=decoder_cell,
                decoder_state_dim=decoder_num_units,
                name=self.scope('attention_decoder'),
                attention_type=self._get_attention_type(attention_type),
                weighted_encoder_outputs=weighted_encoder_outputs,
                attention_memory_optimization=True,
            )
            self.use_attention = True
            self.decoder_output_dim = decoder_num_units + encoder_output_dim

            self.output_indices = decoder_cell.output_indices
            self.output_indices.append(2 * self.num_layers)

    def get_state_names(self):
        return self.cell.get_state_names()

    def get_outputs_with_grads(self):
        # sequence (all) output locations are at twice their state index
        return [2 * i for i in self.output_indices]

    def get_output_dim(self):
        return self.decoder_output_dim

    def get_attention_weights(self):
        assert self.use_attention
        # [batch_size, encoder_length, 1]
        return self.cell.get_attention_weights()

    def apply(
        self,
        model,
        input_t,
        seq_lengths,
        states,
        timestep,
    ):
        return self.cell.apply(
            model=model,
            input_t=input_t,
            seq_lengths=seq_lengths,
            states=states,
            timestep=timestep,
        )

    def apply_over_sequence(
        self,
        model,
        inputs,
        seq_lengths,
        initial_states,
    ):
        return self.cell.apply_over_sequence(
            model=model,
            inputs=inputs,
            seq_lengths=seq_lengths,
            initial_states=initial_states,
            outputs_with_grads=self.get_outputs_with_grads(),
        )


def build_initial_rnn_decoder_states(
    model,
    encoder_units_per_layer,
    decoder_units_per_layer,
    final_encoder_hidden_states,
    final_encoder_cell_states,
    use_attention,
):
    num_encoder_layers = len(encoder_units_per_layer)
    num_decoder_layers = len(decoder_units_per_layer)
    if num_encoder_layers > num_decoder_layers:
        offset = num_encoder_layers - num_decoder_layers
    else:
        offset = 0

    initial_states = []
    for i, decoder_num_units in enumerate(decoder_units_per_layer):

        if (
            final_encoder_hidden_states and
            len(final_encoder_hidden_states) > (i + offset)
        ):
            final_encoder_hidden_state = final_encoder_hidden_states[i + offset]
        else:
            final_encoder_hidden_state = None

        if final_encoder_hidden_state is None:
            decoder_initial_hidden_state = model.param_init_net.ConstantFill(
                [],
                'decoder_initial_hidden_state_{}'.format(i),
                shape=[decoder_num_units],
                value=0.0,
            )
            model.params.append(decoder_initial_hidden_state)
        elif decoder_num_units != encoder_units_per_layer[i + offset]:
            decoder_initial_hidden_state = brew.fc(
                model,
                final_encoder_hidden_state,
                'decoder_initial_hidden_state_{}'.format(i),
                encoder_units_per_layer[i + offset],
                decoder_num_units,
                axis=2,
            )
        else:
            decoder_initial_hidden_state = final_encoder_hidden_state
        initial_states.append(decoder_initial_hidden_state)

        if (
            final_encoder_cell_states and
            len(final_encoder_cell_states) > (i + offset)
        ):
            final_encoder_cell_state = final_encoder_cell_states[i + offset]
        else:
            final_encoder_cell_state = None

        if final_encoder_cell_state is None:
            decoder_initial_cell_state = model.param_init_net.ConstantFill(
                [],
                'decoder_initial_cell_state_{}'.format(i),
                shape=[decoder_num_units],
                value=0.0,
            )
            model.params.append(decoder_initial_cell_state)
        elif decoder_num_units != encoder_units_per_layer[i + offset]:
            decoder_initial_cell_state = brew.fc(
                model,
                final_encoder_cell_state,
                'decoder_initial_cell_state_{}'.format(i),
                encoder_units_per_layer[i + offset],
                decoder_num_units,
                axis=2,
            )
        else:
            decoder_initial_cell_state = final_encoder_cell_state
        initial_states.append(decoder_initial_cell_state)

    if use_attention:
        initial_attention_weighted_encoder_context = (
            model.param_init_net.ConstantFill(
                [],
                'initial_attention_weighted_encoder_context',
                shape=[encoder_units_per_layer[-1]],
                value=0.0,
            )
        )
        model.params.append(initial_attention_weighted_encoder_context)
        initial_states.append(initial_attention_weighted_encoder_context)

    return initial_states


def build_embedding_decoder(
    model,
    decoder_layer_configs,
    inputs,
    input_lengths,
    encoder_lengths,
    encoder_outputs,
    weighted_encoder_outputs,
    final_encoder_hidden_states,
    final_encoder_cell_states,
    encoder_units_per_layer,
    vocab_size,
    embeddings,
    embedding_size,
    attention_type,
    forward_only,
    num_gpus=0,
    scope=None,
):
    with core.NameScope(scope or ''):
        if num_gpus == 0:
            embedded_decoder_inputs = model.net.Gather(
                [embeddings, inputs],
                ['embedded_decoder_inputs'],
            )
        else:
            with core.DeviceScope(core.DeviceOption(caffe2_pb2.CPU)):
                embedded_decoder_inputs_cpu = model.net.Gather(
                    [embeddings, inputs],
                    ['embedded_decoder_inputs_cpu'],
                )
            embedded_decoder_inputs = model.CopyCPUToGPU(
                embedded_decoder_inputs_cpu,
                'embedded_decoder_inputs',
            )

    decoder_cells = []
    decoder_units_per_layer = []
    for i, layer_config in enumerate(decoder_layer_configs):
        num_units = layer_config['num_units']
        decoder_units_per_layer.append(num_units)

        if i == 0:
            input_size = embedding_size
        else:
            input_size = decoder_cells[-1].get_output_dim()

        cell = rnn_cell.LSTMCell(
            forward_only=forward_only,
            input_size=input_size,
            hidden_size=num_units,
            forget_bias=0.0,
            memory_optimization=False,
        )

        dropout_keep_prob = layer_config.get('dropout_keep_prob', None)
        if dropout_keep_prob is not None:
            dropout_ratio = 1.0 - layer_config.dropout_keep_prob
            cell = rnn_cell.DropoutCell(
                internal_cell=cell,
                dropout_ratio=dropout_ratio,
                forward_only=forward_only,
                is_test=False,
                name=get_layer_scope(scope, 'decoder_dropout', i),
            )

        decoder_cells.append(cell)

    states = build_initial_rnn_decoder_states(
        model=model,
        encoder_units_per_layer=encoder_units_per_layer,
        decoder_units_per_layer=decoder_units_per_layer,
        final_encoder_hidden_states=final_encoder_hidden_states,
        final_encoder_cell_states=final_encoder_cell_states,
        use_attention=(attention_type != 'none'),
    )
    attention_decoder = LSTMWithAttentionDecoder(
        encoder_outputs=encoder_outputs,
        encoder_output_dim=encoder_units_per_layer[-1],
        encoder_lengths=encoder_lengths,
        vocab_size=vocab_size,
        attention_type=attention_type,
        embedding_size=embedding_size,
        decoder_num_units=decoder_units_per_layer[-1],
        decoder_cells=decoder_cells,
        weighted_encoder_outputs=weighted_encoder_outputs,
        name=scope,
    )
    decoder_outputs, _ = attention_decoder.apply_over_sequence(
        model=model,
        inputs=embedded_decoder_inputs,
        seq_lengths=input_lengths,
        initial_states=states,
    )

    # we do softmax over the whole sequence
    # (max_length in the batch * batch_size) x decoder embedding size
    # -1 because we don't know max_length yet
    decoder_outputs_flattened, _ = model.net.Reshape(
        [decoder_outputs],
        [
            'decoder_outputs_flattened',
            'decoder_outputs_and_contexts_combination_old_shape',
        ],
        shape=[-1, attention_decoder.get_output_dim()],
    )

    decoder_outputs = decoder_outputs_flattened
    decoder_output_dim = attention_decoder.get_output_dim()

    return (decoder_outputs, decoder_output_dim)


def output_projection(
    model,
    decoder_outputs,
    decoder_output_size,
    target_vocab_size,
    decoder_softmax_size,
):
    if decoder_softmax_size is not None:
        decoder_outputs = brew.fc(
            model,
            decoder_outputs,
            'decoder_outputs_scaled',
            dim_in=decoder_output_size,
            dim_out=decoder_softmax_size,
        )
        decoder_output_size = decoder_softmax_size

    output_projection_w = model.param_init_net.XavierFill(
        [],
        'output_projection_w',
        shape=[target_vocab_size, decoder_output_size],
    )

    output_projection_b = model.param_init_net.XavierFill(
        [],
        'output_projection_b',
        shape=[target_vocab_size],
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
