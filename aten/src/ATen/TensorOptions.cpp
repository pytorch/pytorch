#include <ATen/TensorOptions.h>

#include <ATen/DefaultTensorOptions.h>

namespace at {
TensorOptions::TensorOptions() {
  const auto default_options = DefaultTensorOptions::copy();
  this->dtype(default_options.dtype());
  this->device(default_options.device());
  this->layout(default_options.layout());
  this->requires_grad(default_options.requires_grad());
}
} // namespace at
