#pragma once

#include <c10/core/GeneratorImpl.h>
#include <ATen/core/Generator.h>
#include <ATen/Tensor.h>
#include <ATen/Context.h>
#include <limits>

// TODO: this file should be in ATen/cuda, not top level

namespace at {
/**
 * Note [CUDA Graph-safe RNG states]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *
 * Strategy:
 * ~~~~~~~~~
 * A CUDA graph containing multiple RNG ops behaves like a
 * single giant kernel from the perspective of ops external
 * to the graph.  During graph capture, logic below records
 * the total of all offset increments that occur in the graphed
 * region, and records the final total as the offset for the
 * entire graph.
 *
 * When the graph reruns, the logic that reruns it
 * increments this device's CUDA generator's offset
 * by that total.
 *
 * Meanwhile, within the graph, at capture time, instead of
 * populating PhiloxCudaStates with the uint64_t offset pulled
 * directly from the global state, PhiloxCudaState instead
 * holds a pointer to one-element stream-local int64_t device tensor
 * holding an initial offset value, and a uint64_t holding an
 * intra-graph offset. (The intra-graph offset starts from zero
 * when capture begins.)  In each consumer kernel,
 * at::cuda::philox::unpack computes the offset to use for this kernel
 * as intra-graph offset + *initial offset.
 *
 * When the graph reruns, the logic that reruns it first
 * fill_s the initial offset tensor with this device's
 * CUDA generator's current offset.
 *
 * The control flow above ensures graphed execution is bitwise
 * identical to eager execution as long as RNG ops are enqueued
 * from a single thread, even if RNG ops and graphs containing
 * RNG ops are enqueued and run simultaneously on multiple streams.
 *
 * Usage:
 * ~~~~~~
 * PhiloxCudaState in this file, and unpack() in
 * cuda/CUDAGraphsUtils.cuh allow non-divergent use of
 * CUDAGeneratorImpl whether graph capture is underway or not.
 *
 * Each PhiloxCudaState instance should be used for one and only one
 * consumer kernel.
 *
 * Example (see e.g. native/cuda/Dropout.cu):
 *
 * #include <ATen/cuda/CUDAGeneratorImpl.h>
 * #include <ATen/cuda/CUDAGraphsUtils.cuh>
 *
 * __global__ void kernel(..., PhiloxCudaState philox_args) {
 *   auto seeds = at::cuda::philox::unpack(philox_args);
 *   IndexType idx = blockIdx.x * blockDim.x + threadIdx.x;
 *   curandStatePhilox4_32_10_t state;
 *   curand_init(std::get<0>(seeds), // seed
 *               idx,                // per-thread subsequence
 *               std::get<1>(seeds), // offset in subsequence
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


// Stores state values. Passed as a kernel argument. See "Usage:" above.
struct PhiloxCudaState {
  PhiloxCudaState() = default;
  PhiloxCudaState(const PhiloxCudaState&) = default;
  // Called if graph capture is not underway
  PhiloxCudaState(uint64_t seed,
                  uint64_t offset) {
    seed_ = seed;
    offset_.val = offset;
  }
  // Called if graph capture is underway
  PhiloxCudaState(uint64_t seed,
                  int64_t* offset_extragraph,
                  uint32_t offset_intragraph) {
    seed_ = seed;
    offset_.ptr = offset_extragraph;
    offset_intragraph_ = offset_intragraph;
    captured_ = true;
  }

  // Public members, directly accessible by at::cuda::philox::unpack.
  // If we made them private with getters/setters, the getters/setters
  // would have to be __device__, and we can't declare __device__ in ATen.
  union Payload {
    uint64_t val;
    int64_t* ptr;
  };

  uint64_t seed_;
  Payload offset_;
  uint32_t offset_intragraph_;
  bool captured_ = false;
};

struct TORCH_CUDA_API CUDAGeneratorImpl : public c10::GeneratorImpl {
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
  void graph_prologue(int64_t* offset_extragraph);
  uint64_t graph_epilogue();
  PhiloxCudaState philox_cuda_state(uint64_t increment);

  // Temporarily accommodates call sites that use philox_engine_inputs.
  // Allows incremental refactor of call sites to use philox_cuda_state.
  std::pair<uint64_t, uint64_t> philox_engine_inputs(uint64_t increment);

  static DeviceType device_type();

private:
  CUDAGeneratorImpl* clone_impl() const override;
  uint64_t seed_ = default_rng_seed_val;
  uint64_t philox_offset_per_thread_ = 0;
  int64_t* offset_extragraph_;
  uint32_t offset_intragraph_ = 0;
};

namespace cuda {
namespace detail {

  TORCH_CUDA_API const Generator& getDefaultCUDAGenerator(DeviceIndex device_index = -1);
  TORCH_CUDA_API Generator createCUDAGenerator(DeviceIndex device_index = -1);

} // namespace detail
} // namespace cuda
} // namespace at

