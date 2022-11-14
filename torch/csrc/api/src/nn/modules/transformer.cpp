#include <c10/util/irange.h>
#include <torch/nn/init.h>
#include <torch/nn/modules/transformer.h>
#include <torch/nn/modules/transformercoder.h>
#include <torch/nn/modules/transformerlayer.h>

#include <limits>

namespace F = torch::nn::functional;

namespace torch {
namespace nn {

// ========================TransformerEncoderLayerImpl=========================
TransformerEncoderLayerImpl::TransformerEncoderLayerImpl(
    const TransformerEncoderLayerOptions& options_)
    : options(options_) {
  // NOLINTNEXTLINE(clang-analyzer-optin.cplusplus.VirtualCall)
  reset();
}

void TransformerEncoderLayerImpl::reset() {
  // NOTE: reset() is for initializing the model only, calling reset() after the
  // model is created will throw exceptionss. Call reset_parameter() if the
  // created model needs a reset

  self_attn = this->register_module(
      "self_attn",
      MultiheadAttention(
          MultiheadAttentionOptions(options.d_model(), options.nhead())
              .dropout(options.dropout())));

  linear1 = this->register_module(
      "linear1", Linear(options.d_model(), options.dim_feedforward()));
  dropout = this->register_module("dropout", Dropout(options.dropout()));
  linear2 = this->register_module(
      "linear2", Linear(options.dim_feedforward(), options.d_model()));

  norm1 = this->register_module(
      "norm1", LayerNorm(LayerNormOptions({options.d_model()})));
  norm2 = this->register_module(
      "norm2", LayerNorm(LayerNormOptions({options.d_model()})));

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
    const Tensor& src_key_padding_mask) {
  // multihead attention
  Tensor src2 = std::get<0>(self_attn(
      src, src, src, src_key_padding_mask, /*need_weights=*/true, src_mask));
  // add & norm
  Tensor ret = norm1(src + dropout1(src2));

  // feedforward
  if (c10::get_if<enumtype::kGELU>(&options.activation())) {
    src2 = linear2(dropout(F::gelu(linear1(ret))));
  } else if (c10::get_if<enumtype::kReLU>(&options.activation())) {
    src2 = linear2(dropout(F::relu(linear1(ret))));
  } else if (c10::get_if<std::function<Tensor(const Tensor&)>>(
                 &options.activation())) {
    auto callable_activation =
        *c10::get_if<std::function<Tensor(const Tensor&)>>(
            &options.activation());
    src2 = linear2(dropout(callable_activation(linear1(ret))));
  } else {
    TORCH_CHECK(false, "activation should be kGELU, kReLU, or a callable");
  }

  // add & norm
  return norm2(ret + dropout2(src2));
}

// ========================TransformerDecoderLayerImpl=========================
TransformerDecoderLayerImpl::TransformerDecoderLayerImpl(
    const TransformerDecoderLayerOptions& options_)
    : options(options_) {
  // NOLINTNEXTLINE(clang-analyzer-optin.cplusplus.VirtualCall)
  reset();
}

void TransformerDecoderLayerImpl::reset() {
  // NOTE: reset() is for initializing the model only, calling reset() after the
  // model is created will cause throwing exceptions. Call reset_parameter() if
  // the created model needs a reset.

  // initialize self attention
  self_attn = this->register_module(
      "self_attn",
      MultiheadAttention(
          MultiheadAttentionOptions(options.d_model(), options.nhead())
              .dropout(options.dropout())));

  // initialize multihed attention
  multihead_attn = this->register_module(
      "multihead_attn",
      MultiheadAttention(
          MultiheadAttentionOptions(options.d_model(), options.nhead())
              .dropout(options.dropout())));

  // Initialize Feed forward first linear layer
  linear1 = this->register_module(
      "linear1", Linear(options.d_model(), options.dim_feedforward()));
  // initialize Feed forward dropout layer
  dropout = this->register_module("dropout", Dropout(options.dropout()));
  // initialize Feed forward second linear layer
  linear2 = this->register_module(
      "linear2", Linear(options.dim_feedforward(), options.d_model()));

  // initialize Normalization, post self attention
  norm1 = this->register_module(
      "norm1", LayerNorm(LayerNormOptions({options.d_model()})));
  // initialize post multi-headed attention Normalization
  norm2 = this->register_module(
      "norm2", LayerNorm(LayerNormOptions({options.d_model()})));
  // initialize normalization, post feed forward
  norm3 = this->register_module(
      "norm3", LayerNorm(LayerNormOptions({options.d_model()})));

  // initialize Dropout, post self attention
  dropout1 = this->register_module("dropout1", Dropout(options.dropout()));
  // initialize post multi-headed attention dropout layer
  dropout2 = this->register_module("dropout2", Dropout(options.dropout()));
  // initialize dropout, post feed forward
  dropout3 = this->register_module("dropout3", Dropout(options.dropout()));
}

void TransformerDecoderLayerImpl::reset_parameters() {
  // TODO xinyu: standardrize reset_parameters virtual funcs
  self_attn->_reset_parameters();
  multihead_attn->_reset_parameters();

  linear1->reset_parameters();
  // dropout->reset_paramteres();
  linear2->reset_parameters();

  norm1->reset_parameters();
  norm2->reset_parameters();
  norm3->reset_parameters();
  // dropout1->reset_parameters();
  // dropout2->reset_parameters();
  // dropout3->reset_paramteres();
}

/// Pass the inputs (and mask) through the decoder layer.
Tensor TransformerDecoderLayerImpl::forward(
    Tensor tgt,
    const Tensor& memory,
    const Tensor& tgt_mask,
    const Tensor& memory_mask,
    const Tensor& tgt_key_padding_mask,
    const Tensor& memory_key_padding_mask) {
  Tensor tgt2 = std::get<0>(self_attn(
      tgt, // query
      tgt, // key
      tgt, // value
      tgt_key_padding_mask, // key_padding_mask
      false, // need_weights
      tgt_mask) // attn_mask
  );
  tgt = tgt + dropout1(tgt2);
  tgt = norm1(tgt);

  tgt2 = std::get<0>(multihead_attn(
      tgt, // query
      memory, // key
      memory, // value
      memory_key_padding_mask, // key_padding_mask
      false, // need_weights
      memory_mask) // attn_mask
  );
  tgt = tgt + dropout2(tgt2);
  tgt = norm2(tgt);

  tgt2 = linear2(dropout(activation(linear1(tgt))));
  tgt = tgt + dropout3(tgt2);
  tgt = norm3(tgt);

  return tgt;
}

Tensor TransformerDecoderLayerImpl::activation(const Tensor& input) {
  if (c10::get_if<enumtype::kGELU>(&options.activation())) {
    return F::gelu(input);
  } else if (c10::get_if<enumtype::kReLU>(&options.activation())) {
    return F::relu(input);
  } else if (c10::get_if<std::function<Tensor(const Tensor&)>>(
                 &options.activation())) {
    auto callable_activation =
        *c10::get_if<std::function<Tensor(const Tensor&)>>(
            &options.activation());
    return callable_activation(input);
  } else {
    TORCH_CHECK(false, "activation should be kGELU, kReLU, or a callable");
  }
}

// ========================TransformerEncoderImpl=========================
TransformerEncoderImpl::TransformerEncoderImpl(
    TransformerEncoderOptions options_)
    : options(std::move(options_)) {
  // NOLINTNEXTLINE(clang-analyzer-optin.cplusplus.VirtualCall)
  reset();
}

void TransformerEncoderImpl::reset() {
  layers = this->register_module("layers", ModuleList());
  for (const auto i : c10::irange(options.num_layers())) {
    (void)i; // Suppress unused variable warning
    layers->push_back(options.encoder_layer()->clone());
  }

  if (!options.norm().is_empty()) {
    norm = options.norm().clone();
    this->register_module("norm", norm.ptr());
  }
}

void TransformerEncoderImpl::reset_parameters() {
  TORCH_CHECK(
      layers->size() == static_cast<size_t>(options.num_layers()),
      "TransformerEncoder should have",
      options.num_layers(),
      " encoder layers, but got ",
      layers->size());

  size_t num_layers = layers->size();
  for (const auto i : c10::irange(num_layers)) {
    layers->at<TransformerEncoderLayerImpl>(i).reset_parameters();
  }
  // a. No way to know whether module in AnyModule has api to reset_parameters,
  // so replace instead b. Allow user to add/delete normalization module when
  // reset parameters
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
    const Tensor& src_key_padding_mask) {
  size_t num_layers = layers->size();
  Tensor output;
  if (num_layers > 0) {
    output = layers->at<TransformerEncoderLayerImpl>(0).forward(
        src, src_mask, src_key_padding_mask);
  }
  for (const auto i : c10::irange(1, num_layers)) {
    output = layers->at<TransformerEncoderLayerImpl>(i).forward(
        output, src_mask, src_key_padding_mask);
  }

  if (!norm.is_empty()) {
    output = norm.forward<Tensor>(num_layers == 0 ? src : output);
  }
  return output;
}

// ========================TransformerDecoderImpl=========================
TransformerDecoderImpl::TransformerDecoderImpl(
    TransformerDecoderOptions options_)
    : options(std::move(options_)) {
  // NOLINTNEXTLINE(clang-analyzer-optin.cplusplus.VirtualCall)
  reset();
}

void TransformerDecoderImpl::reset() {
  layers = this->register_module("layers", ModuleList());
  for (const auto i : c10::irange(options.num_layers())) {
    (void)i; // Suppress unused variable warning
    layers->push_back(options.decoder_layer()->clone());
  }

  if (!options.norm().is_empty()) {
    norm = options.norm().clone();
    this->register_module("norm", norm.ptr());
  }
}

void TransformerDecoderImpl::reset_parameters() {
  TORCH_CHECK(
      layers->size() == static_cast<size_t>(options.num_layers()),
      "TransformerDecoder should have",
      options.num_layers(),
      " decoder layers, but got ",
      layers->size());

  size_t num_layers = layers->size();
  for (const auto i : c10::irange(num_layers)) {
    layers->at<TransformerDecoderLayerImpl>(i).reset_parameters();
  }
  // a. No way to know whether module in AnyModule has api to reset_parameters,
  // so replace instead b. Allow user to add/delete normalization module when
  // reset parameters
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
    const Tensor& memory_key_padding_mask) {
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
  for (const auto i : c10::irange(1, num_layers)) {
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

// =======================================TransformerImpl================================
TransformerImpl::TransformerImpl(TransformerOptions options_)
    : options(std::move(options_)) {
  // NOLINTNEXTLINE(clang-analyzer-optin.cplusplus.VirtualCall)
  reset();
}

void TransformerImpl::reset() {
  // set up encoder
  if (options.custom_encoder().is_empty()) {
    LayerNorm norm(LayerNormOptions({options.d_model()}));
    TransformerEncoder trans_encoder(
        TransformerEncoderOptions(
            TransformerEncoderLayerOptions(options.d_model(), options.nhead())
                .dim_feedforward(options.dim_feedforward())
                .dropout(options.dropout())
                .activation(options.activation()),
            options.num_encoder_layers())
            .norm(AnyModule(norm)));

    this->encoder = AnyModule(trans_encoder);
  } else {
    this->encoder = options.custom_encoder().clone();
  }
  this->register_module("encoder", this->encoder.ptr());

  // set up decoder
  if (options.custom_decoder().is_empty()) {
    LayerNorm norm(LayerNormOptions({options.d_model()}));
    TransformerDecoder trans_decoder(
        TransformerDecoderOptions(
            TransformerDecoderLayerOptions(options.d_model(), options.nhead())
                .dim_feedforward(options.dim_feedforward())
                .dropout(options.dropout())
                .activation(options.activation()),
            options.num_decoder_layers())
            .norm(AnyModule(norm)));

    this->decoder = AnyModule(trans_decoder);
  } else {
    this->decoder = options.custom_decoder().clone();
  }
  this->register_module("decoder", this->decoder.ptr());

  reset_parameters();
}

void TransformerImpl::reset_parameters() {
  auto parameters = this->parameters();
  for (auto& param : parameters) {
    if (param.dim() > 1) {
      torch::nn::init::xavier_uniform_(param);
    }
  }
}

Tensor TransformerImpl::forward(
    const Tensor& src,
    const Tensor& tgt,
    const Tensor& src_mask,
    const Tensor& tgt_mask,
    const Tensor& memory_mask,
    const Tensor& src_key_padding_mask,
    const Tensor& tgt_key_padding_mask,
    const Tensor& memory_key_padding_mask) {
  TORCH_CHECK(
      src.dim() == 3 && tgt.dim() == 3,
      "src and tgt should have 3 dimensions, but got ",
      src.dim(),
      " and ",
      tgt.dim());

  TORCH_CHECK(
      src.size(1) == tgt.size(1),
      "src and tgt should have equal batch size (at dim 1), but got ",
      src.size(1),
      " and ",
      tgt.size(1));

  TORCH_CHECK(
      src.size(2) == options.d_model() && tgt.size(2) == options.d_model(),
      "src and tgt should have same feature size as d_model (at dim 2), but got ",
      src.size(2),
      " and ",
      tgt.size(2),
      " while d_model is ",
      options.d_model());

  Tensor memory =
      this->encoder.forward<Tensor>(src, src_mask, src_key_padding_mask);
  Tensor output = this->decoder.forward<Tensor>(
      tgt,
      memory,
      tgt_mask,
      memory_mask,
      tgt_key_padding_mask,
      memory_key_padding_mask);

  return output;
}

Tensor TransformerImpl::generate_square_subsequent_mask(int64_t sz) {
  // Treat 0 dim valid here
  TORCH_CHECK(
      sz >= 0,
      "Input size must be non-negative to generate a valid square subsequent mask, but got ",
      sz);

  // check IEEE754 support here since -inf is not guaranteed to be valid on non
  // IEEE754 platform
  if (std::numeric_limits<float>::is_iec559) {
    return torch::triu(
        torch::full({sz, sz}, -std::numeric_limits<float>::infinity()), 1);
  }
  // if IEEE754 is not supported, we use the smallest float number in current
  // platform
  else {
    TORCH_WARN_ONCE(
        "IEEE754 is not supported on this platform, generate_square_subsequent_mask will fill "
        "the mask with smallest float number on this platform instead of -inf");
    return torch::triu(
        torch::full({sz, sz}, std::numeric_limits<float>::lowest()), 1);
  }
}

} // namespace nn
} // namespace torch
