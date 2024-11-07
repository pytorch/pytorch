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

// Ensures we only call cudaGetDeviceCount only once.
static c10::once_flag num_gpu_init_flag;

// Total number of gpus in the system.
static int64_t num_gpus;

// Ensures default_gens_cuda is initialized once.
static std::deque<c10::once_flag> cuda_gens_init_flag;

// Default, global CUDA generators, one per GPU.
static std::vector<Generator> default_gens_cuda;

/*
 * Populates the global variables related to CUDA generators
 * Warning: this function must only be called once!
 */
static void initCUDAGenVector() {
  num_gpus = static_cast<int32_t>(c10::cuda::device_count());
  cuda_gens_init_flag.resize(num_gpus);
  default_gens_cuda.resize(num_gpus);
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
  c10::call_once(num_gpu_init_flag, initCUDAGenVector);
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
  c10::call_once(num_gpu_init_flag, initCUDAGenVector);
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
  return make_intrusive<CUDAGeneratorState>(
      seed_, philox_offset_per_thread_, offset_intragraph_);
}

/**
 * Function to increase the internal offset based on the specified increment.
 */
void CUDAGeneratorState::increase(uint64_t increment) {
  // Rounds increment up to the nearest multiple of 4 to meet alignment
  // requirements.
  // see Note [Why enforce RNG offset % 4 == 0?]
  increment = ((increment + 3) / 4) * 4;
  // Handling different behaviors based on whether capturing is active.
  if (at::cuda::currentStreamCaptureStatus() != at::cuda::CaptureStatus::None) {
    // Ensures that the state is actually capturing.
    TORCH_CHECK(
        capturing_,
        "Attempt to increase offset for a CUDA generator not in capture mode.");
    // Ensures the offset is a multiple of 4
    // see Note [Why enforce RNG offset % 4 == 0?]
    TORCH_INTERNAL_ASSERT(
        offset_intragraph_ % 4 == 0, "RNG offset must be a multiple of 4.");
    // Ensures the increment does not cause overflow.
    TORCH_INTERNAL_ASSERT(
        offset_intragraph_ <= std::numeric_limits<uint32_t>::max() - increment,
        "Increment causes overflow in the offset value.");
    offset_intragraph_ += increment;
  } else {
    // Checks that the increment is expected outside graph capturing.
    TORCH_CHECK(
        !capturing_,
        "Offset increment outside graph capture encountered unexpectedly.");
    // Ensures the offset is a multiple of 4
    // see Note [Why enforce RNG offset % 4 == 0?]
    TORCH_INTERNAL_ASSERT(
        philox_offset_per_thread_ % 4 == 0,
        "RNG offset must be a multiple of 4.");
    philox_offset_per_thread_ += increment;
  }
}

/**
 * Registers this state to a CUDA graph to manage within the graph.
 */
void CUDAGeneratorState::register_graph(cuda::CUDAGraph* graph) {
  // Ensures that the RNG state is not currently being captured.
  at::cuda::assertNotCapturing(
      "Cannot register the state during capturing stage.");

  // If this is the first graph to be registered, allocate memory for the seed
  // and offset on the GPU.
  if (registered_graphs_.empty()) {
    auto options = at::TensorOptions().device(at::kCUDA).dtype(at::kLong);
    seed_extragraph_ = at::empty({1}, options);
    offset_extragraph_ = at::empty({1}, options);
  }

  // Insert the graph into the set of registered graphs if it's not already
  // registered.
  if (registered_graphs_.find(graph) == registered_graphs_.end()) {
    registered_graphs_.insert(graph);
  }
}

/**
 * Unregisters a CUDA graph from the RNG state.
 */
void CUDAGeneratorState::unregister_graph(cuda::CUDAGraph* graph) {
  // Verify the graph was previously registered.
  TORCH_CHECK(
      registered_graphs_.find(graph) != registered_graphs_.end(),
      "The graph should be registered to the state");

  // Remove the graph from the set of registered graphs.
  registered_graphs_.erase(graph);

  // If no more graphs are registered, deallocate the GPU memory for the seed
  // and offset.
  if (registered_graphs_.empty()) {
    seed_extragraph_.reset();
    offset_extragraph_.reset();
  }
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
void CUDAGeneratorState::capture_prologue() {
  capturing_ = true;
  offset_intragraph_ = 0;
  seed_extragraph_.fill_(int64_t(seed_));
  offset_extragraph_.fill_(int64_t(0));
}

/**
 * Ends the capturing phase and resets related variables, returning the whole
 * graph increment.
 */
uint64_t CUDAGeneratorState::capture_epilogue() {
  capturing_ = false;
  return offset_intragraph_;
}

/**
 * Prepares the state for replay by setting initial state tensors and applying
 * total increment.
 */
void CUDAGeneratorState::replay_prologue(uint64_t wholegraph_increment) {
  // Ensures the generator is not in capturing mode.
  at::cuda::assertNotCapturing(
      "Cannot prepare for replay during capturing stage.");
  seed_extragraph_.fill_(int64_t(seed_));
  offset_extragraph_.fill_(int64_t(philox_offset_per_thread_));
  // Applies the total increment achieved during previous captures to update the
  // offset.
  increase(wholegraph_increment);
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
  at::cuda::assertNotCapturing(
      "Cannot call CUDAGeneratorImpl::set_current_seed");
  state_->seed_ = seed;
  state_->philox_offset_per_thread_ = 0;
  no_reset_rnn_state_.clear();
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
  // Debatable if current_seed() should be allowed in captured regions.
  // Conservatively disallow it for now.
  at::cuda::assertNotCapturing("Cannot call CUDAGeneratorImpl::current_seed");
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
  static const size_t seed_size = sizeof(uint64_t);
  static const size_t offset_size = sizeof(int64_t);
  static const size_t total_size = seed_size + offset_size;

  auto state_tensor = at::detail::empty_cpu({(int64_t)total_size}, ScalarType::Byte, std::nullopt, std::nullopt, std::nullopt, std::nullopt);
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
  at::cuda::assertNotCapturing(
      "Please ensure to utilize the CUDAGeneratorImpl::set_state_index method during capturing.");
  static const size_t seed_size = sizeof(uint64_t);
  static const size_t offset_size = sizeof(int64_t);
  static const size_t total_size = seed_size + offset_size;

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
  TORCH_CHECK(offset % 4 == 0, "offset must be a multiple of 4");
  state_->philox_offset_per_thread_ = offset;
}

/**
 * Gets the current philox_offset_per_thread_ of CUDAGeneratorImpl.
 */
uint64_t CUDAGeneratorImpl::philox_offset_per_thread() const {
  return state_->philox_offset_per_thread_;
}

/**
 * Registers this state to a CUDA graph to manage within the graph.
 */
void CUDAGeneratorImpl::register_graph(cuda::CUDAGraph* graph) {
  graph->register_generator_state(state_);
  state_->register_graph(graph);
}

/**
 * Unregisters a CUDA graph from the RNG state.
 */
void CUDAGeneratorImpl::unregister_graph(cuda::CUDAGraph* graph) {
  state_->unregister_graph(graph);
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
  if (at::cuda::currentStreamCaptureStatus() != at::cuda::CaptureStatus::None) {
    uint32_t offset = state_->offset_intragraph_;
    state_->increase(increment);
    return PhiloxCudaState(
        state_->seed_extragraph_.data_ptr<int64_t>(),
        state_->offset_extragraph_.data_ptr<int64_t>(),
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
