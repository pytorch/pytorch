#pragma once

#include <c10/core/GeneratorImpl.h>
#include <ATen/core/Generator.h>
#include <ATen/Tensor.h>
#include <ATen/Context.h>
#include <limits>


// TODO: this file should be in ATen/cuda, not top level

namespace at {

/**
 * Note [Non-divergent use of HostState and DevState in callers]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * PhiloxCudaState in this file, and unpack() in
 * cuda/StatefulCUDAOpsUtils.cuh allow non-divergent use of
 * CUDAGeneratorImplHostState::philox_cuda_state() and
 * CUDAGeneratorImplDevState::philox_cuda_state()
 * in callers without synchronization.
 *
 * Each PhiloxCudaState instance should be used for one and only one
 * consumer kernel.
 *
 * Example (see e.g. native/cuda/Dropout.cu):
 *
 * #include <ATen/cuda/CUDAGeneratorImpl.h>
 * #include <ATen/cuda/StatefulCUDAOpsUtils.cuh>
 *
 * __global__ void kernel(..., PhiloxCudaState philox_args) {
 *   auto seeds = at::cuda::philox::unpack(philox_args);
 *   IndexType idx = blockIdx.x * blockDim.x + threadIdx.x;
 *   curandStatePhilox4_32_10_t state;
 *   curand_init(std::get<0>(seeds),       // seed
 *               std::get<1>(seeds) + idx, // per-thread subsequence
 *               std::get<2>(seeds),       // offset in subsequence
 *               &state);
 *   ...
 * }
 *
 * host_caller(...) {
 *   PhiloxCudaState rng_engine_inputs;
 *   {
 *     // See Note [Acquire lock when using random generators]
 *     std::lock_guard<std::mutex> lock(gen->mutex_);
 *
 *     // gen could be HostState or DevState here! No divergent code needed!
 *     rng_engine_inputs = gen->philox_cuda_state(offset_increment);
 *   }
 *   kernel<<<...>>>(..., rng_engine_inputs);
 * }
 *
 */


// Stores state values.  See Note [Non-divergent use...] above.
// Uses unions and some bitshifting to reduce size,
// which (hopefully) reduces register use in kernels.
struct PhiloxCudaState {
  PhiloxCudaState() = default;
  PhiloxCudaState(const PhiloxCudaState&) = default;
  // Called by CUDAGeneratorImplHostState
  PhiloxCudaState(uint64_t seed,
                  uint64_t offset,
                  StreamId stream_id) {
    seed_.val = seed;
    offset_.val = offset;
    TORCH_INTERNAL_ASSERT(stream_id <= std::numeric_limits<int>::max());
    is_on_device_and_seq_pool_id_ = uint32_t(stream_id);
  }
  // Called by CUDAGeneratorImplDevState.
  // Pointers are int64_t*, not uint64_t* (there's no such thing as uint64_t Tensors)
  PhiloxCudaState(int64_t* seed,
                  int64_t* offset,
                  StreamId stream_id) {
    seed_.ptr = seed;
    offset_.ptr = offset;
    TORCH_INTERNAL_ASSERT(stream_id <= std::numeric_limits<int>::max());
    // Uses first bit to indicate state is on device.
    is_on_device_and_seq_pool_id_ = (uint32_t(stream_id) | 0x80000000);
  }

  // Public members, directly accessible by at::cuda::philox::unpack.
  // If we made them private with getters/setters, the getters/setters
  // would have to be __device__, and we can't declare __device__ in ATen.

  union Payload {
    uint64_t val;
    int64_t* ptr;
  };

  Payload seed_;
  Payload offset_;

  int increment_;

  void set_increment(uint64_t increment) {
    TORCH_INTERNAL_ASSERT(increment <= std::numeric_limits<int>::max());
    increment_ = int(increment);
  }

  // Helps select a subsequence from the active stream's pool.
  // See Note [Per stream and device RNG states] in CUDAGeneratorImpl.cpp.
  // Also, the first bit is set if state lives on the device.
  uint32_t is_on_device_and_seq_pool_id_;
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
  virtual PhiloxCudaState philox_cuda_state(uint64_t increment) = 0;
  virtual bool state_on_device() const = 0;
  virtual std::pair<uint64_t, uint64_t> philox_engine_inputs(uint64_t increment) = 0;

  protected:
  bool seeded_ = false;
};

// Maintains philox state on the CPU.  Simple and fast, but not cuda graph-safe.
struct TORCH_CUDA_API CUDAGeneratorImplHostState final : public CUDAGeneratorImpl {
  // Constructors
  CUDAGeneratorImplHostState(DeviceIndex device_index = -1);
  ~CUDAGeneratorImplHostState() = default;

  // CUDAGeneratorImplHostState methods
  void set_current_seed(uint64_t seed) override;
  uint64_t current_seed() const override;
  void set_philox_offset_per_thread(uint64_t offset) override;
  uint64_t philox_offset_per_thread() const override;
  PhiloxCudaState philox_cuda_state(uint64_t increment) override;
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
struct TORCH_CUDA_API CUDAGeneratorImplDevState final : public CUDAGeneratorImpl {
  // Constructors
  CUDAGeneratorImplDevState(DeviceIndex device_index = -1);
  ~CUDAGeneratorImplDevState() = default;

  // CUDAGeneratorImplDevState methods
  void set_current_seed(uint64_t seed) override;
  uint64_t current_seed() const override;
  void set_philox_offset_per_thread(uint64_t offset) override;
  uint64_t philox_offset_per_thread() const override;
  PhiloxCudaState philox_cuda_state(uint64_t increment) override;
  bool state_on_device() const override { return false; }

  // Throws an error at call sites that haven't been refactored to use philox_cuda_state.
  std::pair<uint64_t, uint64_t> philox_engine_inputs(uint64_t increment) override;

  std::shared_ptr<CUDAGeneratorImplDevState> clone() const;

  private:
  using LiveRefs = std::tuple<Tensor, Tensor, c10::Stream>;
  using StateWithRefs = std::pair<PhiloxCudaState, LiveRefs>;
  using StreamStatesWithRefs = std::unordered_map<StreamId, StateWithRefs>;

  StateWithRefs& get_state_lazy_init();
  StateWithRefs& add_stream_state(Tensor,
                                  Tensor,
                                  c10::Stream);
  CUDAGeneratorImplDevState* clone_impl() const override;
  void accept_clone_impl(const uint64_t&,
                         const uint64_t&,
                         const StreamStatesWithRefs&);
  uint64_t init_seed_;
  uint64_t init_offset_;
  StreamStatesWithRefs stream_states_;
};


namespace cuda {
namespace detail {

  TORCH_CUDA_API const Generator& getDefaultCUDAGenerator(DeviceIndex device_index = -1);
  TORCH_CUDA_API Generator createCUDAGenerator(DeviceIndex device_index = -1);

} // namespace detail
} // namespace cuda
} // namespace at

