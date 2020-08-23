#include <torch/nn/modules/transformerlayer.h>
#include <torch/nn/modules/transformercoder.h>

namespace F = torch::nn::functional;

namespace torch {
namespace nn {

// ========================TransformerEncoderLayerImpl=========================
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


// ========================TransformerDecoderLayerImpl=========================
TransformerDecoderLayerImpl::TransformerDecoderLayerImpl(
  const TransformerDecoderLayerOptions& options_ ) : options(options_) {
  reset();
}

void TransformerDecoderLayerImpl::reset() {
  ///initialize self attention
  self_attn = register_module(
    "self_attn",
    MultiheadAttention(
      MultiheadAttentionOptions(options.d_model(), options.nhead())
      .dropout(options.dropout())));

  ///initialize Dropout, post self attention
  dropout1 = register_module("dropout1",
    Dropout(DropoutOptions().p(options.dropout())));

  ///initialize Normalization, post self attention
  norm1 = register_module(
    "norm1",
    LayerNorm(LayerNormOptions(std::vector<int64_t> {options.d_model()})));

  ///initialize multihed attention
  multihead_attn = register_module(
    "multihead_attn",
    MultiheadAttention(
      MultiheadAttentionOptions(options.d_model(), options.nhead())
      .dropout(options.dropout())));

  ///initialize post multi-headed attention dropout layer
  dropout2 = register_module(
    "dropout2", Dropout(DropoutOptions().p(options.dropout())));

  ///initialize post multi-headed attention Normalization
  norm2 = register_module(
    "norm2", LayerNorm(
      LayerNormOptions(std::vector<int64_t> {options.d_model()})));

  ///Initialize Feed forward first linear layer
  linear1 = register_module(
    "linear1",
    Linear(LinearOptions(options.d_model(), options.dim_feedforward())));

  ///initialize Feed forward dropout layer
  dropout = register_module(
    "dropout",
    Dropout(DropoutOptions().p(options.dropout())));

  ///initialize Feed forward second linear layer
  linear2 = register_module(
    "linear2",
    Linear(LinearOptions(options.dim_feedforward(), options.d_model())));

  ///initialize dropout, post feed forward
  dropout3 = register_module(
    "dropout3",
    Dropout(DropoutOptions().p(options.dropout())));

  ///initialize normalization, post feed forward
  norm3 = register_module(
    "norm3",
    LayerNorm(LayerNormOptions(std::vector<int64_t> {options.d_model()})));
}

void TransformerDecoderLayerImpl::reset_parameters() {

  // TODO xinyu: standardrize reset_parameters virtual funcs
  self_attn->_reset_parameters();
  // dropout1->reset_parameters();
  norm1->reset_parameters();
  multihead_attn->_reset_parameters();
  // dropout2->reset_parameters();
  norm2->reset_parameters();
  linear1->reset_parameters();
  // dropout->reset_paramteres();
  linear2->reset_parameters();
  // dropout3->reset_paramteres();
  norm3->reset_parameters();
}

///Pass the inputs (and mask) through the decoder layer.
Tensor TransformerDecoderLayerImpl::forward(
  Tensor tgt,
  const Tensor& memory,
  const Tensor& tgt_mask,
  const Tensor& memory_mask,
  const Tensor& tgt_key_padding_mask,
  const Tensor& memory_key_padding_mask){

  Tensor  tgt2 = std::get<0>(self_attn(
    tgt, //query
    tgt, //key
    tgt, //value
    tgt_key_padding_mask, //key_padding_mask
    false, //need_weights
    tgt_mask)//attn_mask
  );
  tgt = tgt + dropout1(tgt2);
  tgt = norm1(tgt);

  tgt2 = std::get<0>(multihead_attn(
    tgt, //query
    memory, //key
    memory, //value
    memory_key_padding_mask, //key_padding_mask
    false, //need_weights
    memory_mask)//attn_mask
  );
  tgt = tgt + dropout2(tgt2);
  tgt = norm2(tgt);

  tgt2 = linear2(dropout(activation(linear1(tgt))));
  tgt = tgt + dropout3(tgt2);
  tgt = norm3(tgt);

  return tgt;
}

Tensor TransformerDecoderLayerImpl::activation(const Tensor& input){
  if (c10::get_if<enumtype::kGELU>(&options.activation())) {
    return F::gelu(input);
  } else if (c10::get_if<enumtype::kReLU>(&options.activation())) {
    return F::relu(input);
  } else {
    TORCH_CHECK(false,
      "Unknown activation: ",
      torch::enumtype::get_enum_name(options.activation()));
  }
}


// ========================TransformerEncoderImpl=========================
TransformerEncoderImpl::TransformerEncoderImpl(
  TransformerEncoderOptions options_) : options(std::move(options_)) {
  reset();
}

void TransformerEncoderImpl::reset() {
  layers = this->register_module("layers", ModuleList());
  for (int64_t i = 0; i < options.num_layers(); ++i) {
    layers->push_back(options.encoder_layer()->clone());
  }

  if (!options.norm().is_empty()) {
    norm = options.norm().clone();
    this->register_module("norm", norm.ptr());
  }
}

void TransformerEncoderImpl::reset_parameters() {
  TORCH_CHECK(
    layers->size() == options.num_layers(),
    "TransformerEncoder should have", options.num_layers(), " encoder layers, but got ", layers->size());

  size_t num_layers = layers->size();
  for (size_t i = 0; i < num_layers; ++i) {
    layers->at<TransformerEncoderLayerImpl>(i).reset_parameters();
  }
  // a. No way to know whether module in AnyModule has api to reset_parameters, so replace instead
  // b. Allow user to add/delete normalization module when reset parameters
  if (!norm.is_empty()) {
    this->unregister_module("norm");
    norm = AnyModule();
  }
  if (!options.norm().is_empty()) {
    norm = options.norm().clone();
    this->register_module("norm", norm.ptr());
  }
}

Tensor TransformerEncoderImpl::forward(
  const Tensor& src,
  const Tensor& src_mask,
  const Tensor& src_key_padding_mask ) {

  size_t num_layers = layers->size();
  Tensor output;
  if (num_layers > 0) {
    output = layers->at<TransformerEncoderLayerImpl>(0).forward(src, src_mask, src_key_padding_mask);
  }
  for (size_t i = 1; i < num_layers; ++i) {
    output = layers->at<TransformerEncoderLayerImpl>(i).forward(output, src_mask, src_key_padding_mask);
  }

  if (!norm.is_empty()) {
    output = norm.forward<Tensor>(num_layers == 0 ? src : output);
  }
  return output;
}

// ========================TransformerDecoderImpl=========================
TransformerDecoderImpl::TransformerDecoderImpl(
  TransformerDecoderOptions options_ ) : options(std::move(options_)){
  reset();
}

void TransformerDecoderImpl::reset() {

  layers = this->register_module("layers", ModuleList());
  for (int64_t i = 0; i < options.num_layers(); ++i) {
    layers->push_back(options.decoder_layer()->clone());
  }

  if (!options.norm().is_empty()) {
    norm = options.norm().clone();
    this->register_module("norm", norm.ptr());
  }
}

void TransformerDecoderImpl::reset_parameters() {

  TORCH_CHECK(layers->size() == options.num_layers(),
    "TransformerDecoder should have", options.num_layers(),
    " decoder layers, but got ", layers->size());

  size_t num_layers = layers->size();
  for (size_t i = 0; i < num_layers; ++i) {
    layers->at<TransformerDecoderLayerImpl>(i).reset_parameters();
  }
  // a. No way to know whether module in AnyModule has api to reset_parameters, so replace instead
  // b. Allow user to add/delete normalization module when reset parameters
  if (!norm.is_empty()) {
    this->unregister_module("norm");
    norm = AnyModule();
  }
  if (!options.norm().is_empty()) {
    norm = options.norm().clone();
    this->register_module("norm", norm.ptr());
  }

}

Tensor TransformerDecoderImpl::forward(
  const Tensor& tgt,
  const Tensor& memory,
  const Tensor& tgt_mask,
  const Tensor& memory_mask,
  const Tensor& tgt_key_padding_mask,
  const Tensor& memory_key_padding_mask){

  size_t num_layers = layers->size();
  Tensor output;
  if (num_layers > 0) {
    output = layers->at<TransformerDecoderLayerImpl>(0).forward(
      tgt,
      memory,
      tgt_mask,
      memory_mask,
      tgt_key_padding_mask,
      memory_key_padding_mask);
  }
  for (size_t i = 1; i < num_layers; ++i) {
    output = layers->at<TransformerDecoderLayerImpl>(i).forward(
      output,
      memory,
      tgt_mask,
      memory_mask,
      tgt_key_padding_mask,
      memory_key_padding_mask);
  }

  if (!norm.is_empty()) {
    output = norm.forward<Tensor>(num_layers == 0 ? tgt : output);
  }

  return output;
}

} // namespace nn
} // namespace torch
