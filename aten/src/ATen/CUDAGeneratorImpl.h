#pragma once

#include <c10/core/GeneratorImpl.h>
#include <ATen/core/Generator.h>
#include <ATen/Tensor.h>
#include <ATen/Context.h>

// TODO: this file should be in ATen/cuda, not top level
// Should I move it to ATen/cuda as part of this PR?

namespace at {

/*
philox_cuda_state_t and unpack() in
cuda/StatefulCUDAOpsUtils.cuh allow non-divergent use of
CUDAGeneratorImplHostState::philox_cuda_state() and
CUDAGeneratorImplDeviceState::philox_cuda_state()
in callers without synchronization.

Intended usage (see e.g. native/cuda/Dropout.cu):

#include <ATen/cuda/philox_kernelarg_helper.h>

__global__ void kernel(..., philox_cuda_state_t philox_args) {
  // Provides std::pair<uint64_t, uint64_t>
  auto seeds = at::cuda::philox::unpack(philox_args);
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
  kernel<<<...>>>(..., rng_engine_inputs);
}
*/

struct philox_cuda_state_t {
  philox_cuda_state_t() {}
  // Called by philox_cuda_state_t::to_kernel_arg if state lives on the CPU.
  philox_cuda_state_t(uint64_t seed,
                      uint64_t offset,
                      subseq_pool_start)
    : has_device_ptrs_{false},
      seed_{seed},
      offset_{offset},
      subseq_pool_start_{subseq_pool_start} {}
  // Called by philox_cuda_state_t::to_kernel_arg if state lives on the GPU.
  // Pointers are int64_t*, not uint64_t* (there's no such thing as uint64_t Tensors)
  philox_cuda_state_t(int64_t* seed,
                      int64_t* offset,
                      int64_t* next_offset,
                      uint64_t increment,
                      int subseq_pool_start)
    : has_device_ptrs_{true},
      seed_{seed},
      offset_{offset},
      next_offset_{next_offset},
      increment_{increment},
      subseq_pool_start_{subseq_start_this_stream} {}

  // Public members, directly accessible by at::cuda::philox::unpack.
  // If we made them private with getters/setters, the getters/setters
  // would have to be __device__, and we can't declare __device__ in ATen.
  // Deliberately not packed in pairs/tuples to make point-of-use less opaque.

  // Helps select a subsequence from the active stream's pool.
  // See Note [Per stream and device RNG states] in CUDAGeneratorImpl.cpp.
  int64_t subseq_pool_start_;

  // false if the state came from the CPU, true if it lives on the GPU.
  const bool has_device_ptrs_;

  // Contains the state if has_device_ptrs_ is false.
  uint64_t seed_;
  uintt64_t offset_;

  // The following are only populated and used by unpack() if has_device_ptrs_ is true.
  int64_t* seed_ptr_;
  int64_t* offset_ptr;
  int64_t* next_offset_ptr;

  // Added to this launch's offset to compute next launch's offset
  uint64_t increment_;
};

// Some callers cast to CUDAGeneratorImpl, so we need it as an interface.
struct TORCH_CUDA_API CUDAGeneratorImpl : public c10::GeneratorImpl {
  // Constructors
  CUDAGeneratorImpl(DeviceIndex device_index = -1);
  virtual ~CUDAGeneratorImpl() = 0;

  // CUDAGeneratorImpl methods
  static DeviceType device_type() { return DeviceType::CUDA; }
  uint64_t seed() override;

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
  virtual std::pair<uint64_t, uint64_t> philox_engine_inputs(uint64_t increment) = 0;
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

  // Temporarily accommodates call sites that use philox_engine_inputs.
  // Allows incremental refactor of call sites to use philox_cuda_state.
  std::pair<uint64_t, uint64_t> philox_engine_inputs(uint64_t increment) override;

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

  // CUDAGeneratorImplDeviceState methods
  void set_current_seed(uint64_t seed) override;
  uint64_t current_seed() const override;
  void set_philox_offset_per_thread(uint64_t offset) override;
  uint64_t philox_offset_per_thread() const override;
  philox_cuda_state_t philox_cuda_state(uint64_t increment) override;
  bool state_on_device() const override { return false; }

  // Throws an error at call sites that haven't been refactored to use philox_cuda_state.
  std::pair<uint64_t, uint64_t> philox_engine_inputs(uint64_t increment) override;

  std::shared_ptr<CUDAGeneratorImplDeviceState> clone() const;

  private:
  void clear_states();
  CUDAGeneratorImplDeviceState* clone_impl() const override;
  using per_stream_states_t = std::unordered_map<StreamId, philox_cuda_state_t>>;
  using live_refs_t = std::vector<std::tuple<Tensor, Tensor, Tensor, c10::Stream>>;
  void accept_clone_impl(const uint64_t&,
                         const uint64_t&,
                         const per_stream_states_t&,
                         const live_refs_t&);
  uint64_t init_seed_;
  uint64_t init_offset_;
  per_stream_states_t per_stream_states_;
  live_refs_t live_refs_;
};


namespace cuda {
namespace detail {

  TORCH_CUDA_API const Generator& getDefaultCUDAGenerator(DeviceIndex device_index = -1);
  TORCH_CUDA_API Generator createCUDAGenerator(DeviceIndex device_index = -1);

} // namespace detail
} // namespace cuda
} // namespace at

