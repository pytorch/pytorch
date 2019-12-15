#include <torch/nn/options/activation.h>

namespace torch {
namespace nn {

SELUOptions::SELUOptions(bool inplace) : inplace_(inplace) {}

GLUOptions::GLUOptions(int64_t dim) : dim_(dim) {}

HardshrinkOptions::HardshrinkOptions(double lambda) : lambda_(lambda) {}

SoftmaxOptions::SoftmaxOptions(int64_t dim) : dim_(dim) {}

SoftminOptions::SoftminOptions(int64_t dim) : dim_(dim) {}

LogSoftmaxOptions::LogSoftmaxOptions(int64_t dim) : dim_(dim) {}

ReLUOptions::ReLUOptions(bool inplace) : inplace_(inplace) {}

ReLU6Options::ReLU6Options(bool inplace) : inplace_(inplace) {}

SoftshrinkOptions::SoftshrinkOptions(double lambda) : lambda_(lambda) {}

namespace functional {

SoftmaxFuncOptions::SoftmaxFuncOptions(int64_t dim) : dim_(dim) {}

SoftminFuncOptions::SoftminFuncOptions(int64_t dim) : dim_(dim) {}

LogSoftmaxFuncOptions::LogSoftmaxFuncOptions(int64_t dim) : dim_(dim) {}

MultiheadAttentionForwardFuncOptions::MultiheadAttentionForwardFuncOptions(
    int64_t embed_dim_to_check, int64_t num_heads,
    Tensor in_proj_weight, Tensor in_proj_bias,
    Tensor bias_k, Tensor bias_v,
    bool add_zero_attn, double dropout_p,
    Tensor out_proj_weight, Tensor out_proj_bias
  ) : embed_dim_to_check_(std::move(embed_dim_to_check)),
      num_heads_(std::move(num_heads)),
      in_proj_weight_(std::move(in_proj_weight)),
      in_proj_bias_(std::move(in_proj_bias)),
      bias_k_(std::move(bias_k)), bias_v_(std::move(bias_v)),
      add_zero_attn_(std::move(add_zero_attn)),
      dropout_p_(std::move(dropout_p)),
      out_proj_weight_(std::move(out_proj_weight)),
      out_proj_bias_(std::move(out_proj_bias)) {}

} // namespace functional
} // namespace nn
} // namespace torch
