#pragma once

#include <torch/nn/cloneable.h>
#include <torch/nn/pimpl.h>
#include <torch/tensor.h>

#include <cstddef>
#include <vector>

namespace torch {
namespace nn {
struct DropoutOptions {
  DropoutOptions(double rate);
  TORCH_ARG(double, rate) = 0.5;
};

namespace detail {
template <typename Derived>
class DropoutImplBase : public torch::nn::Cloneable<Derived> {
 public:
  explicit DropoutImplBase(DropoutOptions options_);

  void reset() override;
  Tensor forward(Tensor input);
  virtual Tensor noise_mask(Tensor input) const = 0;

  DropoutOptions options;
};
} // namespace detail

class DropoutImpl : public detail::DropoutImplBase<DropoutImpl> {
 public:
  using detail::DropoutImplBase<DropoutImpl>::DropoutImplBase;
  Tensor noise_mask(Tensor input) const override;
};

class Dropout2dImpl : public detail::DropoutImplBase<Dropout2dImpl> {
 public:
  using detail::DropoutImplBase<Dropout2dImpl>::DropoutImplBase;
  Tensor noise_mask(Tensor input) const override;
};

TORCH_MODULE(Dropout);
TORCH_MODULE(Dropout2d);
} // namespace nn
} // namespace torch
