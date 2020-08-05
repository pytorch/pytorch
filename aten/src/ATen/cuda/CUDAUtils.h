#pragma once

#include <ATen/cuda/CUDAContext.h>

namespace at { namespace cuda {

// Check if every tensor in a list of tensors matches the current
// device.
inline bool check_device(ArrayRef<Tensor> ts) {
  if (ts.empty()) {
    return true;
  }
  Device curDevice = Device(kCUDA, current_device());
  for (const Tensor& t : ts) {
    if (t.device() != curDevice) return false;
  }
  return true;
}

inline void manual_seed(uint64_t seed) {
  if (device_count() > 0) {
    auto cuda_gen =
        globalContext().defaultGenerator(Device(at::kCUDA, current_device()));
    {
      // See Note [Acquire lock when using random generators]
      std::lock_guard<std::mutex> lock(cuda_gen.mutex());
      cuda_gen.set_current_seed(seed);
    }
  }
}

inline void manual_seed_all(uint64_t seed) {
  // NB: Sometimes we build with CUDA, but we don't have any GPUs
  // available. In that case, we must not seed CUDA; it will fail!
  auto num_gpus = device_count();
  if (num_gpus > 0) {
    for (int i = 0; i < num_gpus; i++) {
      auto cuda_gen = globalContext().defaultGenerator(Device(at::kCUDA, i));
      {
        // See Note [Acquire lock when using random generators]
        std::lock_guard<std::mutex> lock(cuda_gen.mutex());
        cuda_gen.set_current_seed(seed);
      }
    }
  }
}

}} // namespace at::cuda
