from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


def apply_regular_attention(
    model,
    encoder_output_dim,
    encoder_outputs_transposed,
    weighted_encoder_outputs,
    decoder_hidden_state_t,
    decoder_hidden_state_dim,
    # TODO: we need to provide batch_size for some reshape methods,
    # ideally, we should be able to not specify it
    batch_size,
    scope,
):
    def s(name):
        # We have to manually scope due to our internal/external blob
        # relationships.
        return "{}/{}".format(str(scope), str(name))

    # [1, batch_size, encoder_output_dim]
    weighted_decoder_hidden_state = model.FC(
        decoder_hidden_state_t,
        s('weighted_decoder_hidden_state'),
        dim_in=decoder_hidden_state_dim,
        dim_out=encoder_output_dim,
        axis=2,
    )
    # [batch_size, encoder_output_dim]
    weighted_decoder_hidden_state = model.net.Squeeze(
        weighted_decoder_hidden_state,
        weighted_decoder_hidden_state,
        dims=[0],
    )
    # TODO: remove that excessive when RecurrentNetwork supports
    # Sum op at the beginning of step_net
    weighted_encoder_outputs_copy = model.net.Copy(
        weighted_encoder_outputs,
        s('weighted_encoder_outputs_copy'),
    )
    # [encoder_length, batch_size, encoder_output_dim]
    decoder_hidden_encoder_outputs_sum = model.net.Add(
        [weighted_encoder_outputs_copy, weighted_decoder_hidden_state],
        s('decoder_hidden_encoder_outputs_sum'),
        broadcast=1,
        use_grad_hack=1,
    )
    # [encoder_length, batch_size, encoder_output_dim]
    decoder_hidden_encoder_outputs_sum = model.net.Tanh(
        decoder_hidden_encoder_outputs_sum,
        decoder_hidden_encoder_outputs_sum,
    )
    # [encoder_length * batch_size, encoder_output_dim]
    decoder_hidden_encoder_outputs_sum_tanh_2d, _ = model.net.Reshape(
        decoder_hidden_encoder_outputs_sum,
        [
            s('decoder_hidden_encoder_outputs_sum_tanh_2d'),
            s('decoder_hidden_encoder_outputs_sum_tanh_t_old_shape'),
        ],
        shape=[-1, encoder_output_dim],
    )
    attention_v = model.param_init_net.XavierFill(
        [],
        s('attention_v'),
        shape=[encoder_output_dim, 1],
    )
    model.add_param(attention_v)

    # [encoder_length * batch_size, 1]
    attention_logits = model.net.MatMul(
        [decoder_hidden_encoder_outputs_sum_tanh_2d, attention_v],
        s('attention_logits'),
    )
    # [encoder_length, batch_size]
    attention_logits, _ = model.net.Reshape(
        attention_logits,
        [
            attention_logits,
            s('attention_logits_old_shape'),
        ],
        shape=[-1, batch_size],
    )
    # [batch_size, encoder_length]
    attention_logits_transposed = model.net.Transpose(
        attention_logits,
        s('attention_logits_transposed'),
        axes=[1, 0],
    )
    # TODO: we could try to force some attention weights to be zeros,
    # based on encoder_lengths.
    # [batch_size, encoder_length]
    attention_weights = model.Softmax(
        attention_logits_transposed,
        s('attention_weights'),
    )
    # TODO: make this operation in-place
    # [batch_size, encoder_length, 1]
    attention_weights_3d = model.net.ExpandDims(
        attention_weights,
        s('attention_weights_3d'),
        dims=[2],
    )
    # [batch_size, encoder_output_dim, 1]
    attention_weighted_encoder_context = model.net.BatchMatMul(
        [encoder_outputs_transposed, attention_weights_3d],
        s('attention_weighted_encoder_context'),
    )
    # TODO: somehow I cannot use Squeeze in-place op here
    # [batch_size, encoder_output_dim]
    attention_weighted_encoder_context, _ = model.net.Reshape(
        attention_weighted_encoder_context,
        [
            attention_weighted_encoder_context,
            s('attention_weighted_encoder_context_old_shape')
        ],
        shape=[-1, encoder_output_dim],
    )
    return attention_weighted_encoder_context
