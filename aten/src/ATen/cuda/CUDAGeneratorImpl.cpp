#include <ATen/cuda/CUDAGeneratorImpl.h>
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
  num_gpus = c10::cuda::device_count();
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
    : c10::GeneratorImpl{
          Device(DeviceType::CUDA, device_index),
          DispatchKeySet(c10::DispatchKey::CUDA)} {
  at::cuda::assertNotCapturing("Cannot construct a new CUDAGeneratorImpl");
  current_state_id_ = states_.size();
  states_.emplace_back();
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
  state().seed_ = seed;
  state().philox_offset_per_thread_ = 0;
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
  return state().philox_offset_per_thread_;
}

#define CAPTURE_DEFAULT_GENS_MSG \
"In regions captured by CUDA graphs, you may only use the default CUDA RNG " \
"generator on the device that's current when capture begins. " \
"If you need a non-default (user-supplied) generator, or a generator on another " \
"device, please file an issue."

/**
 * Gets the current seed of CUDAGeneratorImpl.
 */
uint64_t CUDAGeneratorImpl::current_seed() const {
  // Debatable if current_seed() should be allowed in captured regions.
  // Conservatively disallow it for now.
  at::cuda::assertNotCapturing("Cannot call CUDAGeneratorImpl::current_seed");
  return state().seed_;
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

  auto state_tensor = at::detail::empty_cpu({(int64_t)total_size}, ScalarType::Byte, c10::nullopt, c10::nullopt, c10::nullopt, c10::nullopt);
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
 * Registers a new state with the CUDA generator and returns its index.
 * This function is used to manage multiple generator states, with each state
 * being identified by a unique index.
 */
size_t CUDAGeneratorImpl::register_state_with_index(
    const c10::TensorImpl& new_state) {
  at::cuda::assertNotCapturing(
      "Registration of new states is prohibited during capturing.");
  current_state_id_ = states_.size();
  states_.emplace_back();
  set_state(new_state);
  return current_state_id_;
}

/**
 * Sets the generator's current state to the one identified by the provided
 * index. This function allows switching between different registered states of
 * the generator.
 */
void CUDAGeneratorImpl::set_state_index(size_t index) {
  TORCH_CHECK(index < states_.size(), "The state index is invalid.");
  current_state_id_ = index;
}

/**
 * Retrieves the index of the generator's current state.
 * The index corresponds to a specific registered state currently in use.
 */
size_t CUDAGeneratorImpl::get_state_index() const {
  return current_state_id_;
}

/**
 * Generates and returns a list of seeds, each corresponding to a registered
 * state of the CUDA generator.
 */
std::vector<uint64_t> CUDAGeneratorImpl::seed_list() const {
  std::vector<uint64_t> seeds;
  for (auto& state : states_) {
    seeds.push_back(state.seed_);
  }

  return seeds;
}

/**
 * Sets the philox_offset_per_thread_ to be used by curandStatePhilox4_32_10
 *
 * See Note [Acquire lock when using random generators]
 */
void CUDAGeneratorImpl::set_philox_offset_per_thread(uint64_t offset) {
  // see Note [Why enforce RNG offset % 4 == 0?]
  TORCH_CHECK(offset % 4 == 0, "offset must be a multiple of 4");
  state().philox_offset_per_thread_ = offset;
}

/**
 * Gets the current philox_offset_per_thread_ of CUDAGeneratorImpl.
 */
uint64_t CUDAGeneratorImpl::philox_offset_per_thread() const {
  return state().philox_offset_per_thread_;
}

/**
 * Called by CUDAGraph to prepare this instance for a graph capture region.
 * offset_extragraph is the initial offset at the start of the graphed region.
 * offset_intragraph tracks the offset in the graphed region.
 */
void CUDAGeneratorImpl::capture_prologue(
    const std::vector<int64_t*>& seeds_device_ptr,
    const std::vector<int64_t*>& offsets_device_ptr) {
  // Assert that the size of seeds and offsets vectors are equal to the size of
  // states
  TORCH_INTERNAL_ASSERT(seeds_device_ptr.size() == states_.size());
  TORCH_INTERNAL_ASSERT(offsets_device_ptr.size() == states_.size());

  // Iterate over each state and update its properties
  for (size_t i = 0; i < states_.size(); i++) {
    states_[i].seed_extragraph_ = seeds_device_ptr[i];
    states_[i].offset_extragraph_ = offsets_device_ptr[i];
    states_[i].offset_intragraph_ = 0;
    states_[i].graph_expects_this_gen_ = true;
  }
}

/**
 * Called by CUDAGraph to finalize a graph capture region for this instance.
 */
std::vector<uint64_t> CUDAGeneratorImpl::capture_epilogue() {
  std::vector<uint64_t> wholegraph_increment_list_;

  for (auto&& state : states_) {
    state.graph_expects_this_gen_ = false;
    wholegraph_increment_list_.push_back(state.offset_intragraph_);
  }
  return wholegraph_increment_list_;
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
  // rounds increment up to the nearest multiple of 4
  increment = ((increment + 3) / 4) * 4;
  if (at::cuda::currentStreamCaptureStatus() != at::cuda::CaptureStatus::None) {
    TORCH_CHECK(
        state().graph_expects_this_gen_,
        "philox_cuda_state for an unexpected CUDA generator used during capture. " CAPTURE_DEFAULT_GENS_MSG);
    // see Note [Why enforce RNG offset % 4 == 0?]
    TORCH_INTERNAL_ASSERT(state().offset_intragraph_ % 4 == 0);
    uint32_t offset = state().offset_intragraph_;
    TORCH_INTERNAL_ASSERT(
        state().offset_intragraph_ <=
        std::numeric_limits<uint32_t>::max() - increment);
    state().offset_intragraph_ += increment;
    return PhiloxCudaState(
        state().seed_extragraph_, state().offset_extragraph_, offset);
  } else {
    TORCH_CHECK(
        !state().graph_expects_this_gen_,
        "CUDA generator expects graph capture to be underway, "
        "but the current stream is not capturing.");
    // see Note [Why enforce RNG offset % 4 == 0?]
    TORCH_INTERNAL_ASSERT(state().philox_offset_per_thread_ % 4 == 0);
    uint64_t offset = state().philox_offset_per_thread_;
    state().philox_offset_per_thread_ += increment;
    return PhiloxCudaState(state().seed_, offset);
  }
}

/**
 * Generates a list of Philox CUDA states, each adjusted according to a
 * corresponding increment value from the provided increment list.
 */
std::vector<PhiloxCudaState> CUDAGeneratorImpl::philox_cuda_state_list(
    std::vector<uint64_t> increment_list) {
  TORCH_INTERNAL_ASSERT(increment_list.size() == states_.size());
  size_t old_index = get_state_index();

  std::vector<PhiloxCudaState> state_list;
  for (size_t i = 0; i < states_.size(); i++) {
    set_state_index(i);
    state_list.push_back(philox_cuda_state(increment_list[i]));
  }
  set_state_index(old_index);
  return state_list;
}

/**
 * Temporarily accommodates call sites that use philox_engine_inputs.
 * Allows incremental refactor of call sites to use philox_cuda_state.
 */
std::pair<uint64_t, uint64_t> CUDAGeneratorImpl::philox_engine_inputs(
    uint64_t increment) {
  at::cuda::assertNotCapturing(
      "Refactor this op to use CUDAGeneratorImpl::philox_cuda_state. Cannot call CUDAGeneratorImpl::philox_engine_inputs");
  // rounds increment up to the nearest multiple of 4
  increment = ((increment + 3) / 4) * 4;
  // see Note [Why enforce RNG offset % 4 == 0?]
  TORCH_INTERNAL_ASSERT(state().philox_offset_per_thread_ % 4 == 0);
  uint64_t offset = state().philox_offset_per_thread_;
  state().philox_offset_per_thread_ += increment;
  return std::make_pair(state().seed_, offset);
}

/*
 * Gets the DeviceType of CUDAGeneratorImpl.
 * Used for type checking during run time.
 */
DeviceType CUDAGeneratorImpl::device_type() {
  return DeviceType::CUDA;
}

/**
 * Retrieves the current state of the CUDA generator as a constant reference.
 * This function provides read-only access to the generator's state.
 */
CUDAGeneratorState& CUDAGeneratorImpl::state() {
  TORCH_CHECK(
      current_state_id_ < states_.size(), "The state index is invalid.");
  return states_[current_state_id_];
}

const CUDAGeneratorState& CUDAGeneratorImpl::state() const {
  TORCH_CHECK(
      current_state_id_ < states_.size(), "The state index is invalid.");
  return states_[current_state_id_];
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
  auto gen = new CUDAGeneratorImpl(this->device().index());
  gen->set_current_seed(state().seed_);
  gen->set_philox_offset_per_thread(state().philox_offset_per_thread_);
  return gen;
}

} // namespace at
