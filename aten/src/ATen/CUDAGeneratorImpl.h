#pragma once

#include <c10/core/GeneratorImpl.h>
#include <ATen/core/Generator.h>

// TODO: this file should be in ATen/cuda, not top level

namespace at {

/*
philox_kernelarg_t and philox_cuda_state_t allow non-divergent use of
CUDAGeneratorImplHostState::philox_cuda_state() and
CUDAGeneratorImplDeviceState::philox_cuda_state()
in callers.

Intended usage (see e.g. native/cuda/Dropout.cu):

__global__ void kernel(..., philox_kernelarg_t philox_args) {
  // Provides std::pair<uint64_t, uint64_t>
  auto seeds = philox_args.get();
  auto seed = state.first;
  auto offset = state.second;
}

host_caller(...) {
  philox_cuda_state_t rng_engine_inputs;
  {
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(gen->mutex_);
    // gen could be HostState or DeviceState here!
    // No divergent code needed!
    rng_engine_inputs = gen->philox_cuda_state(counter_offset);
  }
  ...
  // rng_engine_inputs may contain device tensors, and extends the
  // lifetime of those tensors, so rng_engine_input
  // MUST REMAIN ALIVE on the host across the kernel launch.
  kernel<<<...>>>(..., rng_engine_inputs.to_kernel_arg());
}

We could rig implicit conversion constructors and type conversion operators
as alternatives to to_kernel_args() and get().  But i like the explicitness.
*/

struct philox_kernelarg_t {
  philox_kernelarg_t(uint64_t seed, uint64_t offset)
    : has_device_ptrs_{false} {
    state_ = std::make_pair(seed, offset);
  }
  philox_kernelarg_t(uint64_t* seed, uint64_t* offset)
    : has_device_ptrs_{true} {
    state_ptrs_ = std::make_pair(seed, offset);
  }
  std::pair<uint64_t, uint64_t> get() const {
    if (has_device_ptrs_) {
      return std::make_pair{*state_ptrs_.first, *state_ptrs_.second};
    } else {
      return state_;
    }
  }

  private:
  std::pair<uint64_t, uint64_t> state_;
  std::pair<uint64_t*, uint64_t*> state_ptrs_;
  const bool has_device_ptrs_;
}

struct philox_cuda_state_t {
  philox_cuda_state_t() {}
  philox_cuda_state_t(uint64_t seed, uint64_t offset)
    : has_device_tensors_{false} {
    state_ = std::make_pair(seed, offset);
  }
  philox_cuda_state_t(Tensor seed, Tensor offset)
    : has_device_tensors_{true} {
    state_tensors_ = std::make_pair(seed, offset);
  }
  philox_kernelarg_state_t to_kernel_arg() const {
    if (has_device_tensors_) {
      return philox_kernelarg_t{state_tensors_.first.data_ptr<uint64_t>(),
                                state_tensors_.second.data_ptr<uint64_t>()};
    } else {
      return philox_kernelarg_t{state_.first, state_.second};
    }
  }

  private:
  std::pair<uint64_t, uint64_t> state_;
  std::pair<Tensor, Tensor> state_tensors_;
  const bool has_device_tensors_;
}

struct TORCH_CUDA_API CUDAGeneratorImpl : public c10::GeneratorImpl {
  // Constructors
  ~CUDAGeneratorImpl() = 0;

  // CUDAGeneratorImpl methods
  std::shared_ptr<CUDAGeneratorImpl> clone() const = 0;
  void set_current_seed(uint64_t seed) override = 0;
  uint64_t current_seed() const override = 0;
  uint64_t seed() override = 0;
  void set_philox_offset_per_thread(uint64_t offset) = 0;
  uint64_t philox_offset_per_thread() = 0;
  philox_cuda_state_t philox_cuda_state(uint64_t increment) = 0;
  std::pair<uint64_t, uint64_t> philox_engine_inputs(uint64_t increment) = 0
  static DeviceType device_type() = 0;
  virtual bool state_on_device() = 0;
};

// Maintains philox state on the CPU.  Simple and fast, but not cuda graph-safe.
struct TORCH_CUDA_API CUDAGeneratorImplHostState : public c10::GeneratorImpl {
  // Constructors
  CUDAGeneratorImpl(DeviceIndex device_index = -1);
  ~CUDAGeneratorImpl() = default;

  // CUDAGeneratorImpl methods
  std::shared_ptr<CUDAGeneratorImpl> clone() const;
  void set_current_seed(uint64_t seed) override;
  uint64_t current_seed() const override;
  uint64_t seed() override;
  void set_philox_offset_per_thread(uint64_t offset);
  uint64_t philox_offset_per_thread();
  philox_cuda_state_t philox_cuda_state(uint64_t increment);
  std::pair<uint64_t, uint64_t> philox_engine_inputs(uint64_t increment);
  static DeviceType device_type();
  static state_on_device() { return false; }

  private:
  CUDAGeneratorImpl* clone_impl() const override = 0;
  uint64_t seed_ = default_rng_seed_val;
  uint64_t philox_offset_per_thread_ = 0;
};

// Maintains philox state on the GPU. More complex, but fully cuda graph-safe.
struct TORCH_CUDA_API CUDAGeneratorImplDeviceState : public c10::GeneratorImpl {
  // Constructors
  CUDAGeneratorImpl(DeviceIndex device_index = -1);
  ~CUDAGeneratorImpl() = default;

  // CUDAGeneratorImpl methods
  std::shared_ptr<CUDAGeneratorImpl> clone() const;
  void set_current_seed(uint64_t seed) override;
  uint64_t current_seed() const override;
  uint64_t seed() override;
  void set_philox_offset_per_thread(uint64_t offset);
  uint64_t philox_offset_per_thread();
  philox_cuda_state_t philox_cuda_state(uint64_t increment);
  std::pair<uint64_t, uint64_t> philox_engine_inputs(uint64_t increment);
  static DeviceType device_type();
  static state_on_device() { return true; }

  private:
  CUDAGeneratorImpl* clone_impl() const override;
  Tensor seed_ = default_rng_seed_val;
  Tensor philox_offset_per_thread_;
  c10::optional<c10::Stream>& state_update_stream_;
};


namespace cuda {
namespace detail {

  TORCH_CUDA_API const Generator& getDefaultCUDAGenerator(DeviceIndex device_index = -1);
  TORCH_CUDA_API Generator createCUDAGenerator(DeviceIndex device_index = -1);

} // namespace detail
} // namespace cuda
} // namespace at

