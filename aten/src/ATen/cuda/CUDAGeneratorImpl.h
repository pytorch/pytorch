#pragma once

#include <ATen/Context.h>
#include <ATen/core/Generator.h>
#include <ATen/core/TensorBase.h>
#include <ATen/cuda/PhiloxCudaState.h>
#include <atomic>
#include <memory>
#include <unordered_set>
namespace at {

namespace cuda {
struct CUDAGraph;
}

/**
 * Note [CUDA Graph-safe RNG states]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *
 * Strategy:
 * ~~~~~~~~~
 * (It helps to look at
 * cuda/detail/PhiloxCudaStateRaw.cuh and
 * cuda/detail/UnpackRaw.cuh
 * while you read this.)
 *
 * A CUDA graph containing multiple RNG ops behaves like a
 * single giant kernel from the perspective of ops external
 * to the graph.  During graph capture, logic in CUDAGeneratorImpl
 * records the total of all offset increments that occur in the
 * graphed region, and records the final total as the offset for
 * the entire graph.
 *
 * When the graph reruns, the logic that reruns it
 * increments this device's CUDA generator's offset
 * by that total.
 *
 * Meanwhile, within the graph, at capture time, instead of
 * populating PhiloxCudaStates with the uint64_t offset pulled
 * directly from the global state, PhiloxCudaState uses a pointer
 * to a one-element stream-local int64_t device tensor
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

struct CUDAGeneratorState : public c10::intrusive_ptr_target {
  uint64_t seed_;
  uint64_t philox_offset_per_thread_;
  uint32_t offset_intragraph_;
  bool capturing_{};
  std::unordered_set<cuda::CUDAGraph*> registered_graphs_;
  at::TensorBase seed_extragraph_{};
  at::TensorBase offset_extragraph_{};

  CUDAGeneratorState(
      uint64_t seed = default_rng_seed_val,
      uint64_t philox_offset_per_thread = 0,
      uint32_t offset_intragraph = 0)
      : seed_(seed),
        philox_offset_per_thread_(philox_offset_per_thread),
        offset_intragraph_(offset_intragraph) {}

  void increase(uint64_t increment);

  void register_graph(cuda::CUDAGraph* graph);
  void unregister_graph(cuda::CUDAGraph* graph);

  void capture_prologue();
  // capture_epilogue returns the wholegraph_increment
  uint64_t capture_epilogue();
  void replay_prologue(uint64_t wholegraph_increment);
  c10::intrusive_ptr<CUDAGeneratorState> clone();
};

struct TORCH_CUDA_CPP_API CUDAGeneratorImpl : public c10::GeneratorImpl {
  // Constructors
  CUDAGeneratorImpl(DeviceIndex device_index = -1);
  CUDAGeneratorImpl(
      DeviceIndex device_index,
      c10::intrusive_ptr<CUDAGeneratorState> state_);
  ~CUDAGeneratorImpl() override = default;

  // CUDAGeneratorImpl methods
  std::shared_ptr<CUDAGeneratorImpl> clone() const;
  void set_current_seed(uint64_t seed) override;
  void set_offset(uint64_t offset) override;
  uint64_t get_offset() const override;
  uint64_t current_seed() const override;
  uint64_t seed() override;
  void set_state(const c10::TensorImpl& new_state) override;
  c10::intrusive_ptr<c10::TensorImpl> get_state() const override;
  void graphsafe_set_state(
      const c10::intrusive_ptr<GeneratorImpl>& state) override;
  c10::intrusive_ptr<c10::GeneratorImpl> graphsafe_get_state() const override;

  void set_philox_offset_per_thread(uint64_t offset);
  uint64_t philox_offset_per_thread() const;

  void register_graph(cuda::CUDAGraph* graph);
  void unregister_graph(cuda::CUDAGraph* graph);

  // Generates a PhiloxCudaState with a specified increment, and increment
  // current state
  PhiloxCudaState philox_cuda_state(uint64_t increment);

  bool reset_rnn_state() {
    return !no_reset_rnn_state_.test_and_set();
  }

  // Temporarily accommodates call sites that use philox_engine_inputs.
  // Allows incremental refactor of call sites to use philox_cuda_state.
  std::pair<uint64_t, uint64_t> philox_engine_inputs(uint64_t increment);

  static c10::DeviceType device_type();

 private:
  CUDAGeneratorImpl* clone_impl() const override;

  c10::intrusive_ptr<CUDAGeneratorState> state_;
  std::atomic_flag no_reset_rnn_state_{};
};

namespace cuda::detail {

TORCH_CUDA_CPP_API const Generator& getDefaultCUDAGenerator(
    DeviceIndex device_index = -1);
TORCH_CUDA_CPP_API Generator createCUDAGenerator(DeviceIndex device_index = -1);

} // namespace cuda::detail
} // namespace at
