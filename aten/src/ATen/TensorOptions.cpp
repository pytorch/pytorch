#include <ATen/Device.h>
#include <ATen/Layout.h>
#include <ATen/ScalarType.h>
#include <ATen/TensorOptions.h>

#include <ATen/DefaultTensorOptions.h>

#include <mutex>

namespace at {
TensorOptions DefaultTensorOptions::options_(
    kFloat,
    Device::Type::CPU,
    kStrided,
    /*requires_grad=*/false);

std::mutex DefaultTensorOptions::mutex_;

TensorOptions::TensorOptions() {
  const auto default_options = DefaultTensorOptions::copy();
  this->dtype(default_options.dtype());
  this->device(default_options.device());
  this->layout(default_options.layout());
  this->requires_grad(default_options.requires_grad());
}
} // namespace at
