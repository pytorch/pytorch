#include <torch/nn/init.h>

#include <torch/tensor.h>
#include <torch/utils.h>

namespace torch {
namespace nn {
namespace init {

Tensor uniform_(Tensor tensor, double low, double high) {
  NoGradGuard guard;
  return tensor.uniform_(low, high);
}

Tensor normal_(Tensor tensor, double mean, double std) {
  NoGradGuard guard;
  return tensor.normal_(mean, std);
}

Tensor constant_(Tensor tensor, Scalar value) {
  NoGradGuard guard;
  return tensor.fill_(value);
}

Tensor ones_(Tensor tensor) {
  NoGradGuard guard;
  return tensor.fill_(1);
}

Tensor zeros_(Tensor tensor) {
  NoGradGuard guard;
  return tensor.zero_();
}

Tensor eye_(Tensor tensor) {
  NoGradGuard guard;
  return tensor.eye_();
}

Tensor dirac_(Tensor tensor) {
  NoGradGuard guard;
  return tensor.dirac_();
}

Tensor xavier_uniform_(Tensor tensor, double gain) {
  NoGradGuard guard;
  return tensor.xavier_uniform_(gain);
}

Tensor xavier_normal_(Tensor tensor, double gain) {
  NoGradGuard guard;
  return tensor.xavier_normal_(gain);
}

Tensor orthogonal_(Tensor tensor, double gain) {
  NoGradGuard guard;
  return tensor.orthogonal_(gain);
}

Tensor sparse_(Tensor tensor, double sparsity, double std) {
  NoGradGuard guard;
  return tensor.sparse_(sparsity, std);
}

} // namespace init
} // namespace nn
} // namespace torch
