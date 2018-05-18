#pragma once

#include <torch/nn/module.h>

#include <torch/csrc/autograd/variable.h>

#include <cstdint>

namespace torch { namespace nn {

namespace detail {
template <typename T>
class DropoutBase : public torch::nn::CloneableModule<T> {
 public:
  using nn::Module::is_training;

  explicit DropoutBase(double rate);

  void reset() override;

  variable_list forward(variable_list input) override;

  TORCH_ATTR(double, rate) = 0.5;

 protected:
  virtual Variable noise_mask(Variable input) const = 0;
};
} // namespace detail

class Dropout : public detail::DropoutBase<Dropout> {
 public:
  using detail::DropoutBase<Dropout>::DropoutBase;

 private:
  Variable noise_mask(Variable input) const override;
};

class Dropout2d : public detail::DropoutBase<Dropout2d> {
 public:
  using detail::DropoutBase<Dropout2d>::DropoutBase;

 private:
  Variable noise_mask(Variable input) const override;
};

}} // namespace torch::nn
