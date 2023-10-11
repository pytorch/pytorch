#include <iostream>

#include <ATen/core/Generator.h>
#include <ATen/core/Tensor.h>
#include <c10/util/Exception.h>

#include <ATen/CPUGeneratorImpl.h>
#ifdef USE_CUDA
#include <ATen/cuda/CUDAGeneratorImpl.h>
#endif
#ifdef USE_MPS
#include <ATen/mps/MPSGeneratorImpl.h>
#endif

namespace at {

void Generator::set_state(const at::Tensor& new_state) {
  TORCH_CHECK(new_state.defined(), "Undefined tensor is not allowed");
  this->impl_->set_state(*new_state.unsafeGetTensorImpl());
}

at::Tensor Generator::get_state() const {
  return at::Tensor::wrap_tensor_impl(this->impl_->get_state());
}

Generator make_generator_for_device(c10::Device device, c10::optional<int64_t> seed) {
  if (device.is_cpu()) {
    if (seed.has_value()) {
      return at::detail::createCPUGenerator(seed.value());
    } else {
      return at::detail::createCPUGenerator();
    }
#ifdef USE_CUDA
  } else if (device.is_cuda()) {
    auto generator = at::cuda::detail::createCUDAGenerator(device.index());
    if (seed.has_value()) {
      generator.set_current_seed(seed.value());
    }
    return generator;
#endif
#ifdef USE_MPS
  } else if (device.is_mps()) {
    if (seed.has_value()) {
      return at::mps::detail::createMPSGenerator(seed.value());
    } else {
      return at::mps::detail::createMPSGenerator();
    }
#endif
  } else {
    AT_ERROR("Unsupported device for at::make_generator found: ", device.str());
  }
}

} // namespace at
