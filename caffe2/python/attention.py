## @package attention
# Module caffe2.python.attention
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import brew


class AttentionType:
    Regular, Recurrent = tuple(range(2))


def s(scope, name):
    # We have to manually scope due to our internal/external blob
    # relationships.
    return "{}/{}".format(str(scope), str(name))


# c_i = \sum_j w_{ij}\textbf{s}_j
def _calc_weighted_context(
    model,
    encoder_outputs_transposed,
    encoder_output_dim,
    attention_weights_3d,
    scope,
):
    # [batch_size, encoder_output_dim, 1]
    attention_weighted_encoder_context = model.net.BatchMatMul(
        [encoder_outputs_transposed, attention_weights_3d],
        s(scope, 'attention_weighted_encoder_context'),
    )
    # [batch_size, encoder_output_dim]
    attention_weighted_encoder_context, _ = model.net.Reshape(
        attention_weighted_encoder_context,
        [
            attention_weighted_encoder_context,
            s(scope, 'attention_weighted_encoder_context_old_shape'),
        ],
        shape=[1, -1, encoder_output_dim],
    )
    return attention_weighted_encoder_context


# Calculate a softmax over the passed in attention energy logits
def _calc_attention_weights(
    model,
    attention_logits_transposed,
    scope,
):
    # TODO: we could try to force some attention weights to be zeros,
    # based on encoder_lengths.
    # [batch_size, encoder_length, 1]
    attention_weights_3d = brew.softmax(
        model,
        attention_logits_transposed,
        s(scope, 'attention_weights_3d'),
        engine='CUDNN',
        axis=1,
    )
    return attention_weights_3d


# e_{ij} = \textbf{v}^T tanh \alpha(\textbf{h}_{i-1}, \textbf{s}_j)
def _calc_attention_logits_from_sum_match(
    model,
    decoder_hidden_encoder_outputs_sum,
    encoder_output_dim,
    scope,
):
    # [encoder_length, batch_size, encoder_output_dim]
    decoder_hidden_encoder_outputs_sum = model.net.Tanh(
        decoder_hidden_encoder_outputs_sum,
        decoder_hidden_encoder_outputs_sum,
    )

    # [encoder_length, batch_size, 1]
    attention_logits = brew.fc(
        model,
        decoder_hidden_encoder_outputs_sum,
        s(scope, 'attention_logits'),
        dim_in=encoder_output_dim,
        dim_out=1,
        axis=2,
        freeze_bias=True,
    )

    # [batch_size, encoder_length, 1]
    attention_logits_transposed = brew.transpose(
        model,
        attention_logits,
        s(scope, 'attention_logits_transposed'),
        axes=[1, 0, 2],
    )
    return attention_logits_transposed


# \textbf{W}^\alpha used in the context of \alpha_{sum}(a,b)
def _apply_fc_weight_for_sum_match(
    model,
    input,
    dim_in,
    dim_out,
    scope,
    name,
):
    output = brew.fc(
        model,
        input,
        s(scope, name),
        dim_in=dim_in,
        dim_out=dim_out,
        axis=2,
    )
    output = model.net.Squeeze(
        output,
        output,
        dims=[0],
    )
    return output


# Implement RecAtt due to section 4.1 in http://arxiv.org/abs/1601.03317
def apply_recurrent_attention(
    model,
    encoder_output_dim,
    encoder_outputs_transposed,
    weighted_encoder_outputs,
    decoder_hidden_state_t,
    decoder_hidden_state_dim,
    attention_weighted_encoder_context_t_prev,
    scope,
):
    weighted_prev_attention_context = _apply_fc_weight_for_sum_match(
        model=model,
        input=attention_weighted_encoder_context_t_prev,
        dim_in=encoder_output_dim,
        dim_out=encoder_output_dim,
        scope=scope,
        name='weighted_prev_attention_context',
    )

    weighted_decoder_hidden_state = _apply_fc_weight_for_sum_match(
        model=model,
        input=decoder_hidden_state_t,
        dim_in=decoder_hidden_state_dim,
        dim_out=encoder_output_dim,
        scope=scope,
        name='weighted_decoder_hidden_state',
    )
    # [encoder_length, batch_size, encoder_output_dim]
    decoder_hidden_encoder_outputs_sum_tmp = model.net.Add(
        [
            weighted_prev_attention_context,
            weighted_decoder_hidden_state,
        ],
        s(scope, 'decoder_hidden_encoder_outputs_sum_tmp'),
    )
    # [encoder_length, batch_size, encoder_output_dim]
    decoder_hidden_encoder_outputs_sum = model.net.Add(
        [
            weighted_encoder_outputs,
            decoder_hidden_encoder_outputs_sum_tmp,
        ],
        s(scope, 'decoder_hidden_encoder_outputs_sum'),
        broadcast=1,
    )
    attention_logits_transposed = _calc_attention_logits_from_sum_match(
        model=model,
        decoder_hidden_encoder_outputs_sum=decoder_hidden_encoder_outputs_sum,
        encoder_output_dim=encoder_output_dim,
        scope=scope,
    )

    # [batch_size, encoder_length, 1]
    attention_weights_3d = _calc_attention_weights(
        model=model,
        attention_logits_transposed=attention_logits_transposed,
        scope=scope,
    )

    # [batch_size, encoder_output_dim, 1]
    attention_weighted_encoder_context = _calc_weighted_context(
        model=model,
        encoder_outputs_transposed=encoder_outputs_transposed,
        encoder_output_dim=encoder_output_dim,
        attention_weights_3d=attention_weights_3d,
        scope=scope,
    )
    return attention_weighted_encoder_context, attention_weights_3d, [
        decoder_hidden_encoder_outputs_sum,
    ]


def apply_regular_attention(
    model,
    encoder_output_dim,
    encoder_outputs_transposed,
    weighted_encoder_outputs,
    decoder_hidden_state_t,
    decoder_hidden_state_dim,
    scope,
):
    weighted_decoder_hidden_state = _apply_fc_weight_for_sum_match(
        model=model,
        input=decoder_hidden_state_t,
        dim_in=decoder_hidden_state_dim,
        dim_out=encoder_output_dim,
        scope=scope,
        name='weighted_decoder_hidden_state',
    )

    # [encoder_length, batch_size, encoder_output_dim]
    decoder_hidden_encoder_outputs_sum = model.net.Add(
        [weighted_encoder_outputs, weighted_decoder_hidden_state],
        s(scope, 'decoder_hidden_encoder_outputs_sum'),
        broadcast=1,
        use_grad_hack=1,
    )

    attention_logits_transposed = _calc_attention_logits_from_sum_match(
        model=model,
        decoder_hidden_encoder_outputs_sum=decoder_hidden_encoder_outputs_sum,
        encoder_output_dim=encoder_output_dim,
        scope=scope,
    )

    # [batch_size, encoder_length, 1]
    attention_weights_3d = _calc_attention_weights(
        model=model,
        attention_logits_transposed=attention_logits_transposed,
        scope=scope,
    )

    # [batch_size, encoder_output_dim, 1]
    attention_weighted_encoder_context = _calc_weighted_context(
        model=model,
        encoder_outputs_transposed=encoder_outputs_transposed,
        encoder_output_dim=encoder_output_dim,
        attention_weights_3d=attention_weights_3d,
        scope=scope,
    )
    return attention_weighted_encoder_context, attention_weights_3d, [
        decoder_hidden_encoder_outputs_sum,
    ]
