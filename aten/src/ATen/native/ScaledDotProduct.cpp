#include <ATen/ATen.h>

namespace at {
namespace native {

std::tuple<Tensor, Tensor> scaled_dot_product(
    const Tensor& _query, const Tensor& _key, const Tensor& _value, const Tensor& attn_mask,
    double dropout_p, bool training, bool batch_first) {
  // Transpose if necessary to make batch first.
  const auto query = batch_first ? _query : _query.transpose(-3, -2);
  const auto key = batch_first ? _key : _key.transpose(-3, -2);
  const auto value = batch_first ? _value : _value.transpose(-3, -2);

  // Validate dimensions.
  TORCH_CHECK((query.size(-1) == key.size(-1)) && (key.size(-1) == value.size(-1)),
      "The feature dim of query, key, value must be equal.");
  TORCH_CHECK(key.sizes() == value.sizes(), "Shape of key, value must match");
  const int64_t src_len = key.size(-2);
  const int64_t tgt_len = query.size(-2);
  const int64_t embed_dim = query.size(-1);
  const int64_t batch_size = std::max(query.size(-3), key.size(-3));
  if (attn_mask.defined()) {
      TORCH_CHECK((attn_mask.dim() == 3), "attn_mask must be a 3D tensor.");
      TORCH_CHECK((attn_mask.size(-1) == src_len) && (attn_mask.size(-2) == tgt_len) &&
          (attn_mask.size(-3) == 1 || attn_mask.size(-3) == batch_size),
          "The size of the attn_mask is not correct.");
  }

  // Dot product of scaled q, k.
  const auto scaled_query = query.mul(std::pow(static_cast<float>(embed_dim), -0.5));
  auto attn_output_weights = scaled_query.matmul(key.transpose(-2, -1));
  if (attn_mask.defined()) {
    if (attn_mask.dtype() == at::ScalarType::Bool) {
      attn_output_weights.masked_fill_(attn_mask, -std::numeric_limits<float>::infinity());
    } else if (attn_mask.dtype() == at::ScalarType::Byte) {
      TORCH_WARN("scaled_dot_product received a mask with dtype torch.uint8, this behavior is now deprecated;" \
          "please use a mask with dtype torch.bool instead.");
      attn_output_weights.masked_fill_(attn_mask.to(at::ScalarType::Bool), -std::numeric_limits<float>::infinity());
    } else {
      attn_output_weights.add_(attn_mask);
    }
  }
  attn_output_weights = attn_output_weights.softmax(-1);
  attn_output_weights = at::dropout(attn_output_weights, dropout_p, training);
  auto attn_output = attn_output_weights.matmul(value);

  if (!batch_first) {
    attn_output = attn_output.transpose(-3, -2);
  }
  return std::make_tuple(std::move(attn_output), std::move(attn_output_weights));
}

}
}
