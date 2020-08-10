#include <torch/nn/modules/transformer.h>

namespace F = torch::nn::functional;

namespace torch {
namespace nn {

TransformerEncoderLayerImpl::TransformerEncoderLayerImpl(
  const TransformerEncoderLayerOptions& options_) : options(options_) {
  reset();
}

void TransformerEncoderLayerImpl::reset() {
  // NOTE: reset() is for initializing the model only, call reset() after the model is created
  // will cause throwing exceptions. Call reset_parameter() if the created model need a reset

  self_attn = this->register_module("self_attn",
    MultiheadAttention(MultiheadAttentionOptions(
      options.d_model(), options.nhead()).dropout(options.dropout())));

  linear1 = this->register_module("linear1", Linear(options.d_model(), options.dim_feedforward()));
  dropout = this->register_module("dropout", Dropout(options.dropout()));
  linear2 = this->register_module("linear2", Linear(options.dim_feedforward(), options.d_model()));

  norm1 = this->register_module("norm1", LayerNorm(LayerNormOptions({options.d_model()})));
  norm2 = this->register_module("norm2", LayerNorm(LayerNormOptions({options.d_model()})));

  dropout1 = this->register_module("dropout1", Dropout(options.dropout()));
  dropout2 = this->register_module("dropout2", Dropout(options.dropout()));

  reset_parameters();
}

void TransformerEncoderLayerImpl::reset_parameters() {

  // TODO xinyu: standardrize reset_parameters virtual funcs
  self_attn->_reset_parameters();

  linear1->reset_parameters();
  // dropout->reset_parameters();
  linear2->reset_parameters();

  norm1->reset_parameters();
  norm2->reset_parameters();

  // dropout1->reset_parameters();
  // dropout2->reset_parameters();
}

Tensor TransformerEncoderLayerImpl::forward(
  const Tensor& src,
  const Tensor& src_mask,
  const Tensor& src_key_padding_mask ) {


  // multihead attention
  Tensor src2 = std::get<0>(self_attn(src, src, src, src_key_padding_mask, /*need_weights=*/true, src_mask));
  // add & norm
  Tensor ret = norm1(src + dropout1(src2));

  // feedforward
  if (c10::get_if<enumtype::kGELU>(&options.activation())) {
    src2 = linear2(dropout(F::gelu(linear1(ret))));
  }
  else if (c10::get_if<enumtype::kReLU>(&options.activation())) {
    src2 = linear2(dropout(F::relu(linear1(ret))));
  }
  else {
    TORCH_CHECK(false, "activation should be kGELU/kReLU, not ", torch::enumtype::get_enum_name(options.activation()));
  }

  // add & norm
  return norm2(ret + dropout2(src2));
}



// ============================================================================

} // namespace nn
} // namespace torch
