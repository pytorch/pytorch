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
from caffe2.python import core, rnn_cell


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


def rnn_unidirectional_encoder(
    model,
    embedded_inputs,
    input_lengths,
    initial_hidden_state,
    initial_cell_state,
    embedding_size,
    encoder_num_units,
    use_attention,
    scope=None,
):
    """ Unidirectional (forward pass) LSTM encoder."""

    outputs, final_hidden_state, _, final_cell_state = rnn_cell.LSTM(
        model=model,
        input_blob=embedded_inputs,
        seq_lengths=input_lengths,
        initial_states=(initial_hidden_state, initial_cell_state),
        dim_in=embedding_size,
        dim_out=encoder_num_units,
        scope=(scope + '/' if scope else '') + 'encoder',
        outputs_with_grads=([0] if use_attention else [1, 3]),
    )
    return outputs, final_hidden_state, final_cell_state


def rnn_bidirectional_encoder(
    model,
    embedded_inputs,
    input_lengths,
    initial_hidden_state,
    initial_cell_state,
    embedding_size,
    encoder_num_units,
    use_attention,
    scope=None,
):
    """ Bidirectional (forward pass and backward pass) LSTM encoder."""

    # Forward pass
    (
        outputs_fw,
        final_hidden_state_fw,
        _,
        final_cell_state_fw,
    ) = rnn_cell.LSTM(
        model=model,
        input_blob=embedded_inputs,
        seq_lengths=input_lengths,
        initial_states=(initial_hidden_state, initial_cell_state),
        dim_in=embedding_size,
        dim_out=encoder_num_units,
        scope=(scope + '/' if scope else '') + 'forward_encoder',
        outputs_with_grads=([0] if use_attention else [1, 3]),
    )

    # Backward pass
    reversed_embedded_inputs = model.net.ReversePackedSegs(
        [embedded_inputs, input_lengths],
        ['reversed_embedded_inputs'],
    )

    (
        outputs_bw,
        final_hidden_state_bw,
        _,
        final_cell_state_bw,
    ) = rnn_cell.LSTM(
        model=model,
        input_blob=reversed_embedded_inputs,
        seq_lengths=input_lengths,
        initial_states=(initial_hidden_state, initial_cell_state),
        dim_in=embedding_size,
        dim_out=encoder_num_units,
        scope=(scope + '/' if scope else '') + 'backward_encoder',
        outputs_with_grads=([0] if use_attention else [1, 3]),
    )

    outputs_bw = model.net.ReversePackedSegs(
        [outputs_bw, input_lengths],
        ['outputs_bw'],
    )

    # Concatenate forward and backward results
    outputs, _ = model.net.Concat(
        [outputs_fw, outputs_bw],
        ['outputs', 'outputs_dim'],
        axis=2,
    )

    final_hidden_state, _ = model.net.Concat(
        [final_hidden_state_fw, final_hidden_state_bw],
        ['final_hidden_state', 'final_hidden_state_dim'],
        axis=2,
    )

    final_cell_state, _ = model.net.Concat(
        [final_cell_state_fw, final_cell_state_bw],
        ['final_cell_state', 'final_cell_state_dim'],
        axis=2,
    )
    return outputs, final_hidden_state, final_cell_state


def build_embedding_encoder(
    model,
    encoder_params,
    inputs,
    input_lengths,
    vocab_size,
    embeddings,
    embedding_size,
    use_attention,
    num_gpus=0,
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

    assert len(encoder_params['encoder_layer_configs']) == 1
    encoder_num_units = (
        encoder_params['encoder_layer_configs'][0]['num_units']
    )
    with core.NameScope(scope or ''):
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
            rnn_encoder_func = rnn_bidirectional_encoder
            encoder_output_dim = 2 * encoder_num_units
        else:
            rnn_encoder_func = rnn_unidirectional_encoder
            encoder_output_dim = encoder_num_units

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
        use_attention,
        scope=scope,
    )
    weighted_encoder_outputs = None

    return (
        encoder_outputs,
        weighted_encoder_outputs,
        final_encoder_hidden_state,
        final_encoder_cell_state,
        encoder_output_dim,
    )


def build_initial_rnn_decoder_states(
    model,
    encoder_num_units,
    decoder_num_units,
    final_encoder_hidden_state,
    final_encoder_cell_state,
    use_attention,
):
    if use_attention:
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
        return (
            decoder_initial_hidden_state,
            decoder_initial_cell_state,
            initial_attention_weighted_encoder_context,
        )
    else:
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
        return (
            decoder_initial_hidden_state,
            decoder_initial_cell_state,
        )


def output_projection(
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
