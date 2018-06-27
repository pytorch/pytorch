#include <ATen/TensorOptions.h>

#include <ATen/Device.h>
#include <ATen/Layout.h>
#include <ATen/OptionsGuard.h>
#include <ATen/ScalarType.h>
#include <ATen/optional.h>

namespace at {

TensorOptions::TensorOptions(bool use_thread_local_default_options) {
  if (use_thread_local_default_options) {
    this->dtype(DefaultTensorOptions::get().dtype());
    this->device(DefaultTensorOptions::get().device());
    this->layout(DefaultTensorOptions::get().layout());
    this->requires_grad(DefaultTensorOptions::get().requires_grad());
  }
}
} // namespace at
