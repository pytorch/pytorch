#pragma once

#include <c10/core/GeneratorImpl.h>
#include <ATen/core/Generator.h>
#include <ATen/Tensor.h>
#include <ATen/Context.h>

// TODO: this file should be in ATen/cuda, not top level
// Should I move it to ATen/cuda as part of this PR?

namespace at {

/*
philox_kernelarg_t, philox_cuda_state_t, and unpack() in
cuda/StatefulCUDAOpsUtils.cuh allow non-divergent use of
CUDAGeneratorImplHostState::philox_cuda_state() and
CUDAGeneratorImplDeviceState::philox_cuda_state()
in callers without synchronization.

Intended usage (see e.g. native/cuda/Dropout.cu):

#include <ATen/cuda/philox_kernelarg_helper.h>

__global__ void kernel(..., philox_kernelarg_t philox_args) {
  // Provides std::pair<uint64_t, uint64_t>
  auto seeds = at::cuda::philox::unpack(philox_args);
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
*/

struct philox_kernelarg_t {
  philox_kernelarg_t(uint64_t seed, uint64_t offset)
    : has_device_ptrs_{false} {
    state_ = std::make_pair(seed, offset);
  }
  philox_kernelarg_t(int64_t* seed, int64_t* offset)
    : has_device_ptrs_{true} {
    state_ptrs_ = std::make_pair(seed, offset);
  }

  // Public members, directly accessible by at::cuda::philox::unpack.
  // If we made them private with getters/setters, the getters/setters
  // would have to be __device__, and we can't do that in ATen.
  std::pair<uint64_t, uint64_t> state_;
  // state_ptrs_, if present, are int64_t* (there's no such thing as
  // uint64_t tensors).
  std::pair<int64_t*, int64_t*> state_ptrs_;
  const bool has_device_ptrs_;
};

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

  // We could rig an a conversion "operator philox_kernelarg_t()"
  // instead of to_kernel_arg(). But i like the explicitness.
  philox_kernelarg_t to_kernel_arg() const {
    if (has_device_tensors_) {
      return philox_kernelarg_t{state_tensors_.first.data_ptr<int64_t>(),
                                state_tensors_.second.data_ptr<int64_t>()};
    } else {
      return philox_kernelarg_t{state_.first, state_.second};
    }
  }

  private:
  std::pair<uint64_t, uint64_t> state_;
  // Must be Tensor, not Tensor&.
  // Part of philox_cuda_state_t's job is to keep allocations alive.
  std::pair<Tensor, Tensor> state_tensors_;
  bool has_device_tensors_;
};

// Some callers cast to CUDAGeneratorImpl, so we need it as an interface.
struct TORCH_CUDA_API CUDAGeneratorImpl : public c10::GeneratorImpl {
  // Constructors
  CUDAGeneratorImpl(DeviceIndex device_index = -1);
  virtual ~CUDAGeneratorImpl() = 0;

  // CUDAGeneratorImpl methods
  static DeviceType device_type() { return DeviceType::CUDA; }
  uint64_t seed() override;
  std::pair<uint64_t, uint64_t> philox_engine_inputs(uint64_t increment);

  // Methods declared by GeneratorImpl base class, for reference:
  // virtual void set_current_seed(uint64_t seed) = 0;
  // virtual uint64_t current_seed() const = 0;
  // virtual uint64_t seed() = 0;
  // virtual GeneratorImpl* clone_impl() const = 0;

  // clone() WAS NOT declared virtual in GeneratorImpl.h:
  // c10::intrusive_ptr<GeneratorImpl> clone() const;
  // See "Simple Hierarchy: Covariance + Name hiding" in
  // https://www.fluentcpp.com/2017/09/12/how-to-return-a-smart-pointer-and-use-covariance/
  // Similarly declares clone() an ordinary nonvirtual function here:
  std::shared_ptr<CUDAGeneratorImpl> clone() const;

  // Adds methods specific to the CUDAGeneratorImpl interface:
  virtual void set_philox_offset_per_thread(uint64_t offset) = 0;
  virtual uint64_t philox_offset_per_thread() const = 0;
  virtual philox_cuda_state_t philox_cuda_state(uint64_t increment) = 0;
  virtual bool state_on_device() const = 0;
};

// Maintains philox state on the CPU.  Simple and fast, but not cuda graph-safe.
struct TORCH_CUDA_API CUDAGeneratorImplHostState : public CUDAGeneratorImpl {
  // Constructors
  CUDAGeneratorImplHostState(DeviceIndex device_index = -1);
  ~CUDAGeneratorImplHostState() = default;

  // CUDAGeneratorImplHostState methods
  void set_current_seed(uint64_t seed) override;
  uint64_t current_seed() const override;
  void set_philox_offset_per_thread(uint64_t offset) override;
  uint64_t philox_offset_per_thread() const override;
  philox_cuda_state_t philox_cuda_state(uint64_t increment) override;
  bool state_on_device() const override { return false; }

  std::shared_ptr<CUDAGeneratorImplHostState> clone() const;

  private:
  CUDAGeneratorImplHostState* clone_impl() const override;
  uint64_t seed_ = default_rng_seed_val;
  uint64_t philox_offset_per_thread_ = 0;
};

// Maintains philox state on the GPU. More complex, but fully cuda graph-safe.
struct TORCH_CUDA_API CUDAGeneratorImplDeviceState : public CUDAGeneratorImpl {
  // Constructors
  CUDAGeneratorImplDeviceState(DeviceIndex device_index = -1);
  ~CUDAGeneratorImplDeviceState() = default;

  // CUDAGeneratorImplDeviceState
  void set_current_seed(uint64_t seed) override;
  uint64_t current_seed() const override;
  void set_philox_offset_per_thread(uint64_t offset) override;
  uint64_t philox_offset_per_thread() const override;
  philox_cuda_state_t philox_cuda_state(uint64_t increment) override;
  bool state_on_device() const override { return false; }

  std::shared_ptr<CUDAGeneratorImplDeviceState> clone() const;

  private:
  CUDAGeneratorImplDeviceState* clone_impl() const override;
  Tensor seed_;
  Tensor philox_offset_per_thread_;
  c10::optional<c10::Stream> state_update_stream_;
};


namespace cuda {
namespace detail {

  TORCH_CUDA_API const Generator& getDefaultCUDAGenerator(DeviceIndex device_index = -1);
  TORCH_CUDA_API Generator createCUDAGenerator(DeviceIndex device_index = -1);

} // namespace detail
} // namespace cuda
} // namespace at

