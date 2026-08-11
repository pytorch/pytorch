#include <torch/cuda.h>

#include <ATen/Context.h>
#include <c10/core/Device.h>
#include <c10/core/DeviceGuard.h>
#include <c10/util/irange.h>

namespace torch::cuda {

c10::DeviceIndex device_count() {
  return at::detail::getCUDAHooks().deviceCount();
}

bool is_available() {
  // NB: the semantics of this are different from at::globalContext().hasCUDA();
  // ATen's function tells you if you have a working driver and CUDA build,
  // whereas this function also tells you if you actually have any GPUs.
  // This function matches the semantics of at::cuda::is_available()
  return cuda::device_count() > 0;
}

bool cudnn_is_available() {
  return is_available() && at::detail::getCUDAHooks().hasCuDNN();
}

/// Sets the seed for the current GPU.
void manual_seed(uint64_t seed) {
  if (is_available()) {
    auto index = at::detail::getCUDAHooks().getCurrentDevice();
    auto gen = at::detail::getCUDAHooks().getDefaultGenerator(index);
    {
      // See Note [Acquire lock when using random generators]
      std::lock_guard<std::mutex> lock(gen.mutex());
      gen.set_current_seed(seed);
    }
  }
}

/// Sets the seed for all available GPUs.
void manual_seed_all(uint64_t seed) {
  auto num_gpu = device_count();
  for (const auto i : c10::irange(num_gpu)) {
    auto gen = at::detail::getCUDAHooks().getDefaultGenerator(i);
    {
      // See Note [Acquire lock when using random generators]
      std::lock_guard<std::mutex> lock(gen.mutex());
      gen.set_current_seed(seed);
    }
  }
}

void synchronize(int64_t device_index) {
  TORCH_CHECK(is_available(), "No CUDA GPUs are available");
  TORCH_CHECK(
      device_index >= -1 && device_index < c10::Device::MAX_NUM_DEVICES,
      "Device index ",
      device_index,
      " is out of range for DeviceIndex [",
      -1,
      ", ",
      c10::Device::MAX_NUM_DEVICES - 1,
      "]");
  auto num_gpus = cuda::device_count();
  TORCH_CHECK(
      device_index < 0 || device_index < num_gpus,
      "Device index out of range: ",
      device_index);
  at::detail::getCUDAHooks().deviceSynchronize(
      static_cast<c10::DeviceIndex>(device_index));
}

} // namespace torch::cuda
