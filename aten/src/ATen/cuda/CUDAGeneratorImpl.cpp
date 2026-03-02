#include <ATen/Functions.h>
#include <ATen/Tensor.h>
#include <ATen/Utils.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/CUDAGraph.h>
#include <ATen/cuda/CUDAGraphsUtils.cuh>
#include <c10/core/StreamGuard.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/util/CallOnce.h>
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
 * Note [Lazy Registration of Generators to the CUDA Graph]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *
 * Generator states are registered lazily with CUDA graphs. When an RNG operation
 * is performed during graph capture, the generator state is automatically registered
 * with the capturing graph via get_graph_from_capture_id().
 *
 * Each (generator, capture_id) pair gets its own CUDAGeneratorCaptureState, which
 * holds the GPU tensors and offset tracking for that specific capture. This design
 * supports multiple concurrent graph captures using the same generator.
 *
 */

/**
 * Allocate tensors and initialize with seed value.
 *
 * The RNG state tensors must be allocated in the default memory pool (not the
 * graph pool) because they persist across graph replays and are managed
 * internally.
 *
 * We allocate on the default stream via StreamGuard so get_pool() returns the
 * default pool. CUDAStreamCaptureModeGuard is required because when the
 * current stream is default (not capturing), cudaMallocMaybeCapturing skips
 * the relaxed-mode guard, but cudaMalloc can still fail if another stream is
 * capturing.
 */
void CUDAGeneratorCaptureState::initialize(uint64_t seed) {
  if (is_initialized()) {
    return;
  }

  auto options = at::TensorOptions().device(at::kCUDA).dtype(at::kLong);
  c10::InferenceMode inference_guard(false);

  c10::cuda::CUDAStream default_stream = c10::cuda::getDefaultCUDAStream();
  c10::cuda::CUDAStreamCaptureModeGuard capture_mode_guard(
      cudaStreamCaptureModeRelaxed);
  c10::cuda::CUDAStreamGuard stream_guard(default_stream);

  rng_state_seed_extragraph_ = at::empty({1}, options);
  rng_state_offset_extragraph_ = at::empty({1}, options);
  offset_intragraph_ = 0;
}

/**
 * Increment offset during capture.
 */
void CUDAGeneratorCaptureState::increase(uint64_t increment) {
  // see Note [Why enforce RNG offset % 4 == 0?]
  TORCH_INTERNAL_ASSERT(
      offset_intragraph_ % 4 == 0, "RNG offset must be a multiple of 4.");
  TORCH_INTERNAL_ASSERT(
      offset_intragraph_ <= std::numeric_limits<uint64_t>::max() - increment,
      "Increment causes overflow in the offset value.");
  offset_intragraph_ += increment;
}

/**
 * Finalize capture: return wholegraph_increment and reset offset.
 */
uint64_t CUDAGeneratorCaptureState::finalize() {
  uint64_t result = offset_intragraph_;
  offset_intragraph_ = 0;
  return result;
}

/**
 * Prepare tensors for replay with current seed and offset values.
 * Also resets offset_intragraph_ to 0 so that the replay starts from the same
 * intra-graph offset as the original capture.
 */
void CUDAGeneratorCaptureState::setup_for_replay(uint64_t seed, uint64_t philox_offset) {
  TORCH_INTERNAL_ASSERT(is_initialized(),
      "Capture state not initialized");
  rng_state_seed_extragraph_.fill_(static_cast<int64_t>(seed));
  rng_state_offset_extragraph_.fill_(static_cast<int64_t>(philox_offset));
}

/**
 * Creates a clone of this CUDA Generator State.
 */
c10::intrusive_ptr<CUDAGeneratorState> CUDAGeneratorState::clone() {
  return make_intrusive<CUDAGeneratorState>(seed_, philox_offset_per_thread_);
}

/**
 * Get capture state for a capture ID
 * If create_if_not_found is true, create a new capture state if not found.
 * Otherwise, return nullptr if not found.
 *
 * Uses double-checked locking to avoid holding mutex during CUDA operations.
 */
CUDAGeneratorCaptureState* CUDAGeneratorState::get_capture_state(CaptureId_t capture_id, bool create_if_not_found) {
  uint64_t seed_for_init = 0;
  {
    std::lock_guard<std::mutex> lock(capture_states_mutex_);
    auto it = capture_states_.find(capture_id);
    if (it != capture_states_.end()) {
      return it->second.get();
    }
    // If not found and create_if_not_found is false, return nullptr
    if (!create_if_not_found) {
      return nullptr;
    }
    // Read seed_ under lock so set_current_seed() on another thread cannot race.
    seed_for_init = seed_;
  }

  // Get the graph to obtain device and mempool_id for initialize()
  auto* graph = cuda::get_graph_from_capture_id(capture_id);
  TORCH_CHECK(graph != nullptr,
      "RNG op during graph capture but could not find the CUDAGraph object. "
      "This should not happen.");

  // Create and initialize capture state.
  auto capture_state = make_intrusive<CUDAGeneratorCaptureState>();
  capture_state->initialize(seed_for_init);

  // Safe: CUDAGeneratorState is always managed by intrusive_ptr (e.g. from
  // CUDAGeneratorImpl::state_). reclaim_copy adds a reference for the graph.
  graph->register_generator_state(
      c10::intrusive_ptr<CUDAGeneratorState>::reclaim_copy(this));

  // Insert into map (with lock)
  // Double-check in case another thread created it concurrently
  {
    std::lock_guard<std::mutex> lock(capture_states_mutex_);
    auto it = capture_states_.find(capture_id);
    if (it != capture_states_.end()) {
      // Another thread created it - discard ours and use theirs.
      // Our tensors will be freed when capture_state goes out of scope.
      return it->second.get();
    }
    auto* ptr = capture_state.get();
    capture_states_[capture_id] = std::move(capture_state);
    return ptr;
  }
}

/**
 * Function to increase the internal offset based on the specified increment.
 */
void CUDAGeneratorState::increase(uint64_t increment) {
  // Rounds increment up to the nearest multiple of 4 to meet alignment
  // requirements.
  // see Note [Why enforce RNG offset % 4 == 0?]
  increment = ((increment + 3) / 4) * 4;

  auto capture_id = c10::cuda::currentStreamCaptureIdMayInitCtx();
  if (capture_id.has_value()) {
    // Get or create capture state for this capture
    auto* capture_state = get_capture_state(capture_id.value(), true);
    capture_state->increase(increment);
  } else {
    // Not capturing - update base offset
    TORCH_INTERNAL_ASSERT(
        philox_offset_per_thread_ % 4 == 0,
        "RNG offset must be a multiple of 4.");
    philox_offset_per_thread_ += increment;
  }
}

/**
 * Called by CUDAGraph::capture_end - returns wholegraph_increment for this capture.
 */
uint64_t CUDAGeneratorState::capture_epilogue(CaptureId_t capture_id) {
  auto capture_state = get_capture_state(capture_id, false);
  // If there is no captured state, return 0.
  if (capture_state) {
    return capture_state->finalize();
  } else {
    return 0;
  }
}

/**
 * Called by CUDAGraph::reset - removes capture state when graph is destroyed.
 * The RNG state tensors are allocated in the default pool (not the graph pool),
 * so they will be freed normally when the capture state is destroyed.
 */
void CUDAGeneratorState::remove_capture_state(CaptureId_t capture_id) {
  std::lock_guard<std::mutex> lock(capture_states_mutex_);
  auto it = capture_states_.find(capture_id);
  if (it != capture_states_.end()) {
    capture_states_.erase(it);
  }
}

/**
 * Called by CUDAGraph::replay - prepares capture state for replay.
 *
 * Thread safety: Multiple graphs may call replay concurrently if the same
 * generator state was used to capture multiple graphs. We hold the mutex
 * while reading seed_/philox_offset_per_thread_ and updating the offset
 * to prevent races.
 */
void CUDAGeneratorState::replay_prologue(CaptureId_t capture_id, uint64_t wholegraph_increment) {
  if (wholegraph_increment == 0) {
    return;
  }

  uint64_t replay_seed;
  uint64_t replay_offset;
  CUDAGeneratorCaptureState* capture_state = nullptr;
  {
    std::lock_guard<std::mutex> lock(capture_states_mutex_);
    auto it = capture_states_.find(capture_id);
    TORCH_INTERNAL_ASSERT(it != capture_states_.end(),
        "replay_prologue called but no capture state found for this capture_id");
    capture_state = it->second.get();
    replay_seed = seed_;
    replay_offset = philox_offset_per_thread_;
  }

  // Fill tensors without holding the mutex to avoid serializing CUDA work
  capture_state->setup_for_replay(replay_seed, replay_offset);

  {
    std::lock_guard<std::mutex> lock(capture_states_mutex_);
    philox_offset_per_thread_ += wholegraph_increment;
  }
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
  auto capture_id = c10::cuda::currentStreamCaptureIdMayInitCtx();
  if (C10_LIKELY(!capture_id.has_value())) {
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
 * Sets the generator's current state to
 * This function allows switching between different registered states of
 * the generator.
 */
void CUDAGeneratorImpl::graphsafe_set_state(
    const c10::intrusive_ptr<GeneratorImpl>& gen) {
  c10::intrusive_ptr<CUDAGeneratorImpl> cuda_gen =
      dynamic_intrusive_pointer_cast<CUDAGeneratorImpl>(gen);
  TORCH_CHECK(cuda_gen, "Expected a CUDA Generator");
  state_ = cuda_gen->state_;
}

/**
 * Get the GeneratorImpl that point to current state_
 */
c10::intrusive_ptr<c10::GeneratorImpl> CUDAGeneratorImpl::graphsafe_get_state()
    const {
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
  auto capture_id = c10::cuda::currentStreamCaptureIdMayInitCtx();
  if (C10_LIKELY(!capture_id.has_value())) {
    state_->philox_offset_per_thread_ = offset;
  } else {
    auto* capture_state = state_->get_capture_state(capture_id.value(), true);
    capture_state->offset_intragraph_ = offset;
  }
}

/**
 * Gets the current philox_offset_per_thread_ of CUDAGeneratorImpl.
 */
uint64_t CUDAGeneratorImpl::philox_offset_per_thread() const {
  auto capture_id = c10::cuda::currentStreamCaptureIdMayInitCtx();
  if (C10_LIKELY(!capture_id.has_value())) {
    return state_->philox_offset_per_thread_;
  } else {
    auto capture_state = state_->get_capture_state(capture_id.value(), true);
    return capture_state->offset_intragraph_;
  }
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
  auto capture_id = c10::cuda::currentStreamCaptureIdMayInitCtx();
  if (capture_id.has_value()) {
    // Get or create capture state (handles lazy initialization)
    auto* capture_state = state_->get_capture_state(capture_id.value(), true);

    // Get current offset before incrementing
    uint64_t offset = capture_state->offset_intragraph_;
    state_->increase(increment);

    return PhiloxCudaState(
        capture_state->rng_state_seed_extragraph_.data_ptr<int64_t>(),
        capture_state->rng_state_offset_extragraph_.data_ptr<int64_t>(),
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
