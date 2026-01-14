#include <ATen/Functions.h>
#include <ATen/Tensor.h>
#include <ATen/Utils.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/CUDAGraph.h>
#include <ATen/cuda/CUDAGraphsUtils.cuh>
#include <c10/core/StreamGuard.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/CallOnce.h>
#include <iostream>
#include <deque>

namespace at {
namespace cuda::detail {

namespace {

// Total number of gpus in the system.
int64_t num_gpus;

// Ensures default_gens_cuda is initialized once.
std::deque<c10::once_flag> cuda_gens_init_flag;

// Default, global CUDA generators, one per GPU.
std::vector<Generator> default_gens_cuda;

/*
 * Populates the global variables related to CUDA generators
 * Warning: this function must only be called once!
 */
void initCUDAGenVector() {
  // Ensures we only call cudaGetDeviceCount only once.
  static bool num_gpu_init_flag [[maybe_unused]] = []() {
    num_gpus = static_cast<int32_t>(c10::cuda::device_count());
    cuda_gens_init_flag.resize(num_gpus);
    default_gens_cuda.resize(num_gpus);
    return true;
  }();
}

} // anonymous namespace

/**
 * PyTorch maintains a collection of default generators that get
 * initialized once. The purpose of these default generators is to
 * maintain a global running state of the pseudo random number generation,
 * when a user does not explicitly mention any generator.
 * getDefaultCUDAGenerator gets the default generator for a particular
 * cuda device.
 */
const Generator& getDefaultCUDAGenerator(DeviceIndex device_index) {
  initCUDAGenVector();
  DeviceIndex idx = device_index;
  if (idx == -1) {
    idx = c10::cuda::current_device();
  } else {
    TORCH_CHECK(idx >= 0 && idx < num_gpus);
  }
  c10::call_once(cuda_gens_init_flag[idx], [&] {
    default_gens_cuda[idx] = make_generator<CUDAGeneratorImpl>(idx);
    default_gens_cuda[idx].seed();
  });
  return default_gens_cuda[idx];
}

/**
 * Utility to create a CUDAGeneratorImpl. Returns a shared_ptr
 */
Generator createCUDAGenerator(DeviceIndex device_index) {
  initCUDAGenVector();
  DeviceIndex idx = device_index;
  if (idx == -1) {
    idx = c10::cuda::current_device();
  }
  TORCH_CHECK(idx >= 0 && idx < num_gpus, "The device_index is invalid.");
  auto gen = make_generator<CUDAGeneratorImpl>(idx);
  auto cuda_gen = check_generator<CUDAGeneratorImpl>(gen);
  cuda_gen->set_current_seed(default_rng_seed_val);
  cuda_gen->set_philox_offset_per_thread(0);
  return gen;
}

} // namespace cuda::detail

/**
 * Creates a clone of this CUDA Generator State.
 */
c10::intrusive_ptr<CUDAGeneratorState> CUDAGeneratorState::clone() {
  return make_intrusive<CUDAGeneratorState>(seed_, philox_offset_per_thread_);
}

/**
 * Function to increase the internal offset based on the specified increment.
 */
void CUDAGeneratorState::increase(uint64_t increment) {
  // Rounds increment up to the nearest multiple of 4 to meet alignment
  // requirements.
  // see Note [Why enforce RNG offset % 4 == 0?]
  increment = ((increment + 3) / 4) * 4;
  TORCH_INTERNAL_ASSERT(
      philox_offset_per_thread_ % 4 == 0,
      "RNG offset must be a multiple of 4.");
  TORCH_INTERNAL_ASSERT(
      philox_offset_per_thread_ <=
          std::numeric_limits<uint64_t>::max() - increment,
      "Increment causes overflow in the offset value.");
  philox_offset_per_thread_ += increment;
}

/**
 * Note [Explicit Registration of Generators to the CUDA Graph]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *
 * Ideally, it would be more user-friendly if the state could be exchanged and generators
 * could be registered with the CUDA graph implicitly. However, resetting GPU tensors during
 * the capture stage causes these reset operations to be recorded within the CUDA graph.
 * This behavior is undesirable because we do not want these tensors to be reset during
 * the replay stage of the graph.
 *
 * As of now, there is no available method to perform a CUDA operation during the graph's
 * recording phase without having that operation be included in the CUDA graph.
 * This limitation necessitates explicit user action to register generators with the graph.
 * By requiring users to manually register their generators, we can ensure that state resets
 * (capture_prologue) only occur before the graph capture begins, thus avoiding unintended
 * resets during the replay of the graph. See https://github.com/pytorch/pytorch/pull/114068.
 */

/**
 * Performs the prologue steps for capturing a CUDA graph state.
 * This method is intended to reset graph-related state variables before capturing begins.
 */
void CUDAGeneratorCaptureState::capture_prologue(
    const CUDAGeneratorState& base_state) {
  offset_intragraph_ = 0;
  seed_extragraph_.fill_(static_cast<int64_t>(base_state.seed_));
  offset_extragraph_.fill_(static_cast<int64_t>(base_state.philox_offset_per_thread_));
}

/**
 * Ends the capturing phase and resets related variables, returning the whole
 * graph increment.
 */
uint64_t CUDAGeneratorCaptureState::capture_epilogue() {
  return offset_intragraph_;
}

/**
 * Prepares the state for replay by setting initial state tensors and applying
 * total increment.
 */
void CUDAGeneratorCaptureState::replay_prologue(
    CUDAGeneratorState& base_state, uint64_t wholegraph_increment) {
  at::cuda::assertNotCapturing(
      "Cannot prepare for replay during capturing stage.");
  if (wholegraph_increment) {
    seed_extragraph_.fill_(static_cast<int64_t>(base_state.seed_));
    offset_extragraph_.fill_(
        static_cast<int64_t>(base_state.philox_offset_per_thread_));
    base_state.increase(wholegraph_increment);
  }
}

void CUDAGeneratorCaptureState::increase(uint64_t increment) {
  increment = ((increment + 3) / 4) * 4;
  TORCH_INTERNAL_ASSERT(
      offset_intragraph_ % 4 == 0, "RNG offset must be a multiple of 4.");
  TORCH_INTERNAL_ASSERT(
      offset_intragraph_ <= std::numeric_limits<uint64_t>::max() - increment,
      "Increment causes overflow in the offset value.");
  offset_intragraph_ += increment;
}

/**
 * Note [Why enforce RNG offset % 4 == 0?]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * Curand philox does allow offsets that aren't a multiple of 4.
 * But jit kernels don't use curand, they use a custom "Philox" class (see
 * torch/csrc/jit/tensorexpr/cuda_random.h or
 * torch/csrc/jit/codegen/cuda/runtime/random_numbers.cu).
 * The "Philox" constructor computes offset/4 (a uint64_t division) to locate its
 * internal start in its virtual bitstream viewed as 128-bit chunks, then, when called
 * in a thread, returns one 32-bit chunk at a time from that start in the bitstream.
 * In other words, if the incoming offset is not a multiple of 4, each thread
 * might repeat some previously-generated 32-bit values in the bitstream. See
 * https://github.com/pytorch/pytorch/pull/50169.
 */

/**
 * CUDAGeneratorImpl class implementation
 */
CUDAGeneratorImpl::CUDAGeneratorImpl(DeviceIndex device_index)
  : c10::GeneratorImpl{Device(DeviceType::CUDA, device_index),
          DispatchKeySet(c10::DispatchKey::CUDA)} {
  at::cuda::assertNotCapturing("Cannot construct a new CUDAGeneratorImpl");
  state_ = make_intrusive<CUDAGeneratorState>();
  no_reset_rnn_state_.clear();
}

CUDAGeneratorImpl::CUDAGeneratorImpl(
    DeviceIndex device_index,
    c10::intrusive_ptr<CUDAGeneratorState> state)
    : c10::
          GeneratorImpl{Device(DeviceType::CUDA, device_index), DispatchKeySet(c10::DispatchKey::CUDA)},
      state_(std::move(state)) {
  no_reset_rnn_state_.clear();
}

/**
 * Sets the seed to be used by curandStatePhilox4_32_10
 * Resets the philox_offset_per_thread_ to 0
 *
 * See Note [Acquire lock when using random generators]
 */
void CUDAGeneratorImpl::set_current_seed(uint64_t seed) {
  if (C10_LIKELY(at::cuda::currentStreamCaptureStatus() == at::cuda::CaptureStatus::None)) {
    state_->seed_ = seed;
    state_->philox_offset_per_thread_ = 0;
    no_reset_rnn_state_.clear();
  } else {
    TORCH_CHECK(state_->seed_ == seed, "CUDAGeneratorImpl::set_current_seed can be called during stream capture only if new seed is the same as the original seed.");
    // no-op case
  }
}

/**
 * Sets the offset to be used by curandStatePhilox4_32_10
 *
 * See Note [Acquire lock when using random generators]
 */
void CUDAGeneratorImpl::set_offset(uint64_t offset) {
  at::cuda::assertNotCapturing("Cannot call CUDAGeneratorImpl::set_offset");
  // the set function checks if the offset is a multiple of 4.
  set_philox_offset_per_thread(offset);
  no_reset_rnn_state_.clear();
}

/**
 * Gets the current offset of CUDAGeneratorImpl.
 */
uint64_t CUDAGeneratorImpl::get_offset() const {
  // Debatable if get_offset() should be allowed in captured regions.
  // Conservatively disallow it for now.
  at::cuda::assertNotCapturing("Cannot call CUDAGeneratorImpl::get_offset");
  return state_->philox_offset_per_thread_;
}

/**
 * Gets the current seed of CUDAGeneratorImpl.
 */
uint64_t CUDAGeneratorImpl::current_seed() const {
  return state_->seed_;
}

/**
 * Gets a nondeterministic random number from /dev/urandom or time,
 * seeds the CPUGeneratorImpl with it and then returns that number.
 *
 * FIXME: You can move this function to Generator.cpp if the algorithm
 * in getNonDeterministicRandom is unified for both CPU and CUDA
 */
uint64_t CUDAGeneratorImpl::seed() {
  at::cuda::assertNotCapturing("Cannot call CUDAGeneratorImpl::seed");
  auto random = c10::detail::getNonDeterministicRandom(true);
  this->set_current_seed(random);
  return random;
}

/**
 * Gets the current internal state of CUDAGeneratorImpl. The internal
 * state is returned as a CPU byte tensor.
 */
c10::intrusive_ptr<c10::TensorImpl> CUDAGeneratorImpl::get_state() const {
  // The RNG state comprises the seed, and an offset used for Philox.
  constexpr size_t seed_size = sizeof(uint64_t);
  constexpr size_t offset_size = sizeof(int64_t);
  constexpr size_t total_size = seed_size + offset_size;

  auto state_tensor = at::detail::empty_cpu({static_cast<int64_t>(total_size)}, ScalarType::Byte, std::nullopt, std::nullopt, std::nullopt, std::nullopt);
  auto rng_state = state_tensor.data_ptr<uint8_t>();
  auto current_seed = this->current_seed();
  auto offset = static_cast<int64_t>(this->philox_offset_per_thread()); // Note that old THCGeneratorState had offset as std::atomic<int64_t>
  memcpy(rng_state, &current_seed, seed_size);
  memcpy(rng_state + seed_size, &offset, offset_size);

  return state_tensor.getIntrusivePtr();
}

/**
 * Sets the internal state of CUDAGeneratorImpl. The new internal state
 * must be a strided CPU byte tensor and have appropriate size. See
 * comments of CUDAGeneratorImpl::state for information about the layout
 * and size of the internal state.
 */
void CUDAGeneratorImpl::set_state(const c10::TensorImpl& new_state) {
  constexpr size_t seed_size = sizeof(uint64_t);
  constexpr size_t offset_size = sizeof(int64_t);
  constexpr size_t total_size = seed_size + offset_size;

  detail::check_rng_state(new_state);

  bool no_philox_seed = false;
  auto new_state_size = new_state.numel();
  if (new_state_size == total_size - offset_size) {
    no_philox_seed = true;
  } else {
    TORCH_CHECK(new_state_size == total_size, "RNG state is wrong size");
  }

  uint64_t input_seed = 0;
  auto new_rng_state = new_state.data_dtype_initialized<uint8_t>();
  memcpy(&input_seed, new_rng_state, seed_size);
  this->set_current_seed(input_seed);
  int64_t philox_offset = 0;
  if (!no_philox_seed) {
    memcpy(&philox_offset, new_rng_state + seed_size, offset_size);
  }
  this->set_philox_offset_per_thread(static_cast<uint64_t>(philox_offset));
}

/**
 * Sets the generator's current state to <- Incomplete
 * This function allows switching between different registered states of
 * the generator.
 */
void CUDAGeneratorImpl::graphsafe_set_state(
    const c10::intrusive_ptr<GeneratorImpl>& gen) {
  c10::intrusive_ptr<CUDAGeneratorImpl> cuda_gen =
      dynamic_intrusive_pointer_cast<CUDAGeneratorImpl>(gen);
  TORCH_CHECK(cuda_gen, "Expected a CUDA Generator");
  // TODO: Should I do this only if the stream is not capturing?
  state_ = cuda_gen->state_;

  // capture_states_ = cuda_gen->capture_states_;
  capture_states_ = cuda_gen->capture_states_;

  // cudaStreamCaptureStatus status{};
  // unsigned long long capture_id{};
  // C10_CUDA_CHECK(
  //     cudaStreamGetCaptureInfo(at::cuda::getCurrentCUDAStream(), &status, &capture_id));
  // if (C10_UNLIKELY(at::cuda::CaptureStatus(status) != at::cuda::CaptureStatus::None)) {
  //   c10::intrusive_ptr<CUDAGeneratorCaptureState> capture_state;
  //   std::lock_guard<std::mutex> lock(capture_states_mutex_);
  //   // TODO: Should this be a deep-copy or a shallow copy? I think it
  //   // should be a deep-copy... Yes. definitely a deep copy.
  //   capture_states_.at(capture_id) = c10::intrusive_ptr<T>::make(cuda_gen->capture_states_.at(capture_id));
  // }
}

/**
 * Get the GeneratorImpl that point to current state_
 */
c10::intrusive_ptr<c10::GeneratorImpl> CUDAGeneratorImpl::graphsafe_get_state()
    const {
  // this is not enough. It's missing the
  auto gen = make_intrusive<CUDAGeneratorImpl>(device().index(), state_);
  return gen;
}

/**
 * Sets the philox_offset_per_thread_ to be used by curandStatePhilox4_32_10
 *
 * See Note [Acquire lock when using random generators]
 */
void CUDAGeneratorImpl::set_philox_offset_per_thread(uint64_t offset) {
  // see Note [Why enforce RNG offset % 4 == 0?]

  // Note: If you use CUDNN RNN's, calling
  // set_philox_offset_per_thread instead of set_offset will cause the
  // cudnn RNN rng state to become stale.
  TORCH_CHECK(offset % 4 == 0, "offset must be a multiple of 4");
  cudaStreamCaptureStatus status{};
  unsigned long long capture_id{};
  C10_CUDA_CHECK(
      cudaStreamGetCaptureInfo(at::cuda::getCurrentCUDAStream(), &status, &capture_id));
  if (C10_LIKELY(at::cuda::CaptureStatus(status) == at::cuda::CaptureStatus::None)) {
    state_->philox_offset_per_thread_ = offset;
  } else {
    c10::intrusive_ptr<CUDAGeneratorCaptureState> capture_state;
    {
      std::lock_guard<std::mutex> lock(capture_states_mutex_);
      auto it = capture_states_.find(capture_id);
      if (it != capture_states_.end()) {
        capture_state = it->second;
      }
    }
    TORCH_CHECK(
        capture_state,
        "CUDAGeneratorImpl::set_philox_offset_per_thread called during CUDA graph capture "
        "without a registered capture state.");
    capture_state->offset_intragraph_ = offset;
  }
}

/**
 * Gets the current philox_offset_per_thread_ of CUDAGeneratorImpl.
 */
uint64_t CUDAGeneratorImpl::philox_offset_per_thread() const {
  cudaStreamCaptureStatus status{};
  unsigned long long capture_id{};
  C10_CUDA_CHECK(
      cudaStreamGetCaptureInfo(at::cuda::getCurrentCUDAStream(), &status, &capture_id));
  if (C10_LIKELY(at::cuda::CaptureStatus(status) == at::cuda::CaptureStatus::None)) {
    return state_->philox_offset_per_thread_;
  }
  c10::intrusive_ptr<CUDAGeneratorCaptureState> capture_state;
  {
    std::lock_guard<std::mutex> lock(capture_states_mutex_);
    auto it = capture_states_.find(capture_id);
    if (it != capture_states_.end()) {
      capture_state = it->second;
    }
  }
  TORCH_CHECK(
      capture_state,
      "CUDAGeneratorImpl::philox_offset_per_thread called during CUDA graph capture "
      "without a registered capture state.");
  return capture_state->offset_intragraph_;
}

// /**
//  * Registers this state to a CUDA graph to manage within the graph.
//  */
// void CUDAGeneratorImpl::register_graph(cuda::CUDAGraph* graph) {
//   state_->register_graph(graph);
// }

// /**
//  * Unregisters a CUDA graph from the RNG state.
//  */
// void CUDAGeneratorImpl::unregister_graph(cuda::CUDAGraph* graph) {
//   state_->unregister_graph(graph);
// }

c10::intrusive_ptr<CUDAGeneratorCaptureState>
CUDAGeneratorImpl::create_capture_state() const {
  auto capture_state = make_intrusive<CUDAGeneratorCaptureState>();
  auto options = at::TensorOptions().device(device()).dtype(at::kLong);
  capture_state->seed_extragraph_ = at::empty({1}, options);
  capture_state->offset_extragraph_ = at::empty({1}, options);
  capture_state->capture_prologue(*state_);
  return capture_state;
}

  // So, if I go graphsafe_get_state after record_capture_state,
  // that's fine. Otherwise, things are not fine. IIUC. Actually,
  // anything returned by graphsafe_set_state is not actually
  // registered. So I need to
void CUDAGeneratorImpl::record_capture_state(
    CaptureId_t capture_id,
    const c10::intrusive_ptr<CUDAGeneratorCaptureState>& capture_state) {
  std::lock_guard<std::mutex> lock(capture_states_mutex_);
  std::cout << "GALVEZ: record_capture_state capture_id=" << capture_id << std::endl;
  TORCH_CHECK(
      capture_states_.find(capture_id) == capture_states_.end(),
      "Capture state was already recorded for this CUDA graph capture.");
  capture_states_.emplace(capture_id, capture_state);
}

uint64_t CUDAGeneratorImpl::release_capture_state(CaptureId_t capture_id) {
  c10::intrusive_ptr<CUDAGeneratorCaptureState> capture_state;
  {
    std::lock_guard<std::mutex> lock(capture_states_mutex_);
    auto it = capture_states_.find(capture_id);
    TORCH_CHECK(
        it != capture_states_.end(),
        "No capture state found when finishing CUDA graph capture.");
    capture_state = it->second;
    capture_states_.erase(it);
  }
  uint64_t wholegraph_increment = capture_state->capture_epilogue();
  return wholegraph_increment;
}

void CUDAGeneratorImpl::clear_capture_state(CaptureId_t capture_id) {
  // don't release_captue_state already
  std::lock_guard<std::mutex> lock(capture_states_mutex_);
  capture_states_.erase(capture_id);
}

/**
 * Gets the seed and philox offset value to be used in
 * curandStatePhilox4_32_10, in an opaque PhiloxCudaState that's safe
 * and can be used non-divergently in callers whether CUDA graph
 * capture is underway or not.  See
 * Note [CUDA Graph-safe RNG states]
 *
 * Each kernel using philox has to sensibly increment offset
 * for future users of philox. So it gets the "old" value for
 * itself (before add), and tells subsequent users which offset
 * they should use, since only the kernel knows how many randoms
 * it intends to generate.
 *
 * Increment should be at least the number of curand() random numbers used in
 * each thread. It is the user's responsibility to make sure the increment
 * for philox is never smaller than the number of curand() calls. Increment
 * value > the number of curand() calls won't harm but anything less would mean
 * that you would be reusing random values from previous calls.
 *
 * See Note [Acquire lock when using random generators]
 */
PhiloxCudaState CUDAGeneratorImpl::philox_cuda_state(uint64_t increment) {
  cudaStreamCaptureStatus status{};
  unsigned long long capture_id{};
  C10_CUDA_CHECK(cudaStreamGetCaptureInfo(at::cuda::getCurrentCUDAStream(), &status, &capture_id));
  if (C10_UNLIKELY(at::cuda::CaptureStatus(status) != at::cuda::CaptureStatus::None)) {
    std::cout << "GALVEZ: philox_cuda_state capture_id=" << capture_id << std::endl;
    c10::intrusive_ptr<CUDAGeneratorCaptureState> capture_state;
    {
      std::lock_guard<std::mutex> lock(capture_states_mutex_);
      auto it = capture_states_.find(capture_id);
      if (it != capture_states_.end()) {
        capture_state = it->second;
      }
    }
    TORCH_CHECK(
        capture_state,
        "CUDA RNG state not registered for the active graph capture.");
    uint64_t offset = capture_state->offset_intragraph_;
    capture_state->increase(increment);
    return PhiloxCudaState(
        capture_state->seed_extragraph_.data_ptr<int64_t>(),
        capture_state->offset_extragraph_.data_ptr<int64_t>(),
        offset);
  } else {
    uint64_t offset = state_->philox_offset_per_thread_;
    state_->increase(increment);
    return PhiloxCudaState(state_->seed_, offset);
  }
}

/**
 * Temporarily accommodates call sites that use philox_engine_inputs.
 * Allows incremental refactor of call sites to use philox_cuda_state.
 */
std::pair<uint64_t, uint64_t> CUDAGeneratorImpl::philox_engine_inputs(
    uint64_t increment) {
  at::cuda::assertNotCapturing(
      "Refactor this op to use CUDAGeneratorImpl::philox_cuda_state. Cannot call CUDAGeneratorImpl::philox_engine_inputs");
  uint64_t offset = state_->philox_offset_per_thread_;
  state_->increase(increment);
  return std::make_pair(state_->seed_, offset);
}

/*
 * Gets the DeviceType of CUDAGeneratorImpl.
 * Used for type checking during run time.
 */
DeviceType CUDAGeneratorImpl::device_type() {
  return DeviceType::CUDA;
}

/**
 * Public clone method implementation
 *
 * See Note [Acquire lock when using random generators]
 */
std::shared_ptr<CUDAGeneratorImpl> CUDAGeneratorImpl::clone() const {
  return std::shared_ptr<CUDAGeneratorImpl>(this->clone_impl());
}

/**
 * Private clone method implementation
 *
 * See Note [Acquire lock when using random generators]
 */
CUDAGeneratorImpl* CUDAGeneratorImpl::clone_impl() const {
  at::cuda::assertNotCapturing("Cannot call CUDAGeneratorImpl::clone_impl");
  auto gen = new CUDAGeneratorImpl(this->device().index(), state_->clone());
  return gen;
}

} // namespace at
