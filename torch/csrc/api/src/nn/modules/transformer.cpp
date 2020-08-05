#include <torch/nn/modules/transformer.h>
#include <torch/nn/init.h>

#include <torch/types.h>
#include <torch/utils.h>

#include <cmath>
#include <cstdint>

namespace F = torch::nn::functional;

namespace torch {
namespace nn {

// ========================TransformerDecoderLayerImpl=========================
  TransformerDecoderLayerImpl::TransformerDecoderLayerImpl(
    const TransformerDecoderLayerOptions& options_ )
    : options(options_),

    ///initialize self attention
    self_attn(register_module("self_attn", MultiheadAttention(
        MultiheadAttentionOptions(options.d_model(), options.nhead()).dropout(options.dropout())))),

    ///initialize Dropout, post self attention
    dropout1(register_module("dropout1", Dropout(DropoutOptions().p(options.dropout())))),

    ///initialize Normalization, post self attention
    norm1(register_module("norm1", LayerNorm(LayerNormOptions(std::vector<int64_t> {options.d_model()})))),

    ///initialize multihed attention
    multihead_attn(register_module("multihead_attn", MultiheadAttention(
        MultiheadAttentionOptions(options.d_model(), options.nhead()).dropout(options.dropout())))),

    ///initialize post multi-headed attention dropout layer
    dropout2(register_module("dropout2", Dropout(DropoutOptions().p(options.dropout())))),

   ///initialize post multi-headed attention Normalization
    norm2(register_module("norm2", LayerNorm(LayerNormOptions(std::vector<int64_t> {options.d_model()})))),

    ///Initialize Feed forward first linear layer
    linear1(register_module("linear1", Linear(LinearOptions(options.d_model(), options.dim_feedforward())))),

    ///initialize Feed forward dropout layer
    dropout(register_module("dropout", Dropout(DropoutOptions().p(options.dropout())))),

    ///initialize Feed forward second linear layer
    linear2(register_module("linear2", Linear(LinearOptions(options.dim_feedforward(), options.d_model())))),

    ///initialize dropout, post feed forward
    dropout3(register_module("dropout3",Dropout(DropoutOptions().p(options.dropout())))),

    ///initialize normalization, post feed forward
    norm3(register_module("norm3", LayerNorm(LayerNormOptions(std::vector<int64_t> {options.d_model()}))))
    {

    TORCH_CHECK(
    0 <= options.dropout() && options.dropout() <= 1,
    "dropout should be a number in range [0, 1] ",
    "representing the probability of an element being ",
    "zeroed");

    // ///initialize self attention
    // self_attn = MultiheadAttention(
    //     MultiheadAttentionOptions(
    //         options.d_model(),
    //         options.nhead()).
    //         dropout(options.dropout())
    //         );

    // ///initialize Dropout, post self attention
    // dropout1 = Dropout(DropoutOptions().p(options.dropout()));

    // ///initialize Normalization, post self attention
    // norm1 = LayerNorm(
    //     LayerNormOptions(std::vector<int64_t> {options.d_model()})
    //     );

    // ///initialize multihed attention
    // multihead_attn = MultiheadAttention(
    //     MultiheadAttentionOptions(
    //         options.d_model(),
    //         options.nhead()).
    //         dropout(options.dropout())
    //         );

    // ///initialize post multi-headed attention dropout layer
    // dropout2 = Dropout(
    //     DropoutOptions().p(options.dropout())
    //     );

    // ///initialize post multi-headed attention Normalization
    // norm2 = LayerNorm(
    //     LayerNormOptions(std::vector<int64_t> {options.d_model()})
    //     );

    // ///Initialize Feed forward first linear layer
    // linear1 = Linear(
    //     LinearOptions(options.d_model(), options.dim_feedforward())
    //     );

    // ///initialize Feed forward dropout layer
    // dropout = Dropout(
    //     DropoutOptions().p(options.dropout())
    //     );

    // ///initialize Feed forward second linear layer
    // linear2 = Linear(
    //     LinearOptions(options.dim_feedforward(), options.d_model())
    //     );

    // ///initialize dropout, post feed forward
    // dropout3 = Dropout(
    //     DropoutOptions().p(options.dropout())
    //     );

    // ///initialize normalization, post feed forward
    // norm3 = LayerNorm(
    //     LayerNormOptions(std::vector<int64_t> {options.d_model()})
    //     );

  }

  void TransformerDecoderLayerImpl::reset() {
    //TODO: need to need to unifiy once the python implementation is done

    self_attn->reset();
    dropout1->reset();
    norm1->reset();
    multihead_attn->reset();
    dropout2->reset();
    norm2->reset();
    linear1->reset();
    dropout->reset();
    linear2->reset();
    dropout3->reset();
    norm3->reset();

  }

  ///Pass the inputs (and mask) through the decoder layer.
  Tensor TransformerDecoderLayerImpl::forward(Tensor tgt, Tensor memory,
    Tensor tgt_mask,
    Tensor memory_mask,
    Tensor tgt_key_padding_mask,
    Tensor memory_key_padding_mask){

    Tensor  tgt2 = std::get<0>(self_attn(
        tgt, //query
        tgt, //key
        tgt, //value
        tgt_key_padding_mask, //key_padding_mask
        true, //need_weights
        tgt_mask)//attn_mask
        );
    tgt = tgt + dropout1(tgt2);
    tgt = norm1(tgt);

    tgt2 = std::get<0>(multihead_attn(
        tgt, //query
        memory, //key
        memory, //value
        memory_key_padding_mask, //key_padding_mask
        true, //need_weights
        memory_mask)//attn_mask
        );
    tgt = tgt + dropout2(tgt2);
    tgt = norm2(tgt);

    tgt2 = linear2(dropout(activation(linear1(tgt))));
    tgt = tgt + dropout3(tgt2);
    tgt = norm3(tgt);

    return tgt;
  }

  Tensor TransformerDecoderLayerImpl::activation(Tensor input){
    Tensor ret;
    if (c10::get_if<enumtype::kGELU>(&options.activation())) {
       ret = F::gelu(input);
    } else if (c10::get_if<enumtype::kReLU>(&options.activation())) {
       ret = F::relu(input);
    } else {
      TORCH_CHECK(false, "Unknown activation: ", torch::enumtype::get_enum_name(options.activation()));
    }
    return ret;
  }

} // namespace nn
} // namespace torch
