#include <ATen/Utils.h>
#include <ATen/CUDAGeneratorImpl.h>
#include <ATen/cuda/StatefulCUDAOpsUtils.cuh>
#include <c10/core/StreamGuard.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>


namespace at {

// forward-declares full
namespace native {
  Tensor full(IntArrayRef size, Scalar fill_value, const TensorOptions& options);
}

namespace cuda { namespace detail {

namespace {
/**
 * Note [Per stream and device RNG states]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * CUDA generators assign a pool of subsequences (of the overall Philox
 * sequence) to each stream.
 * For a given RNG consumer kernel in a given stream, each thread
 * is assigned a subsequence from the active stream's pool.
 * max_threads_per_kernel is an estimated upper limit of the maximum
 * number of threads any kernel may possibly need.
 * In other words, it estimates how many subsequences we should assign
 * to each stream's pool.
 *
 * Curand philox offers 2^64 subsequences.  Since we assign 2^40 subsequences
 * to each stream, stream id values must be <= 2^24.
 * See Note [StreamId assignment] in c10/cuda/CUDAStream.cpp.
 */
constexpr uint64_t max_threads_per_kernel =  (uint64_t(1) << 40);

// Ensures we only call cudaGetDeviceCount only once.
std::once_flag num_gpu_init_flag;

// Total number of gpus in the system.
int64_t num_gpus;

// Ensures default_gens_cuda is initialized once.
std::deque<std::once_flag> cuda_gens_init_flag;

// Default, global CUDA generators, one per GPU.
std::vector<Generator> default_gens_cuda;

/*
* Populates the global variables related to CUDA generators
* Warning: this function must only be called once!
*/
void initCUDAGenVector(){
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
  std::call_once(num_gpu_init_flag, initCUDAGenVector);
  DeviceIndex idx = device_index;
  if (idx == -1) {
    idx = c10::cuda::current_device();
  } else {
    TORCH_CHECK(idx >= 0 && idx < num_gpus);
  }
  std::call_once(cuda_gens_init_flag[idx],
                 [&] {
                   if (at::globalContext().statefulCUDAOpStatesOnDevice()) {
                     default_gens_cuda[idx] = make_generator<CUDAGeneratorImplDeviceState>(idx);
                   } else {
                     default_gens_cuda[idx] = make_generator<CUDAGeneratorImplHostState>(idx);
                   }
                   default_gens_cuda[idx].seed();
                 });
  return default_gens_cuda[idx];
}

/**
 * Utility to create a CUDAGeneratorImpl. Returns a shared_ptr
 */
Generator createCUDAGenerator(DeviceIndex device_index) {
  std::call_once(num_gpu_init_flag, initCUDAGenVector);
  DeviceIndex idx = device_index;
  if (idx == -1) {
    idx = c10::cuda::current_device();
  }
  TORCH_CHECK(idx >= 0 && idx < num_gpus, "The device_index is invalid.");

  if (at::globalContext().statefulCUDAOpStatesOnDevice()) {
    auto gen = make_generator<CUDAGeneratorImplDeviceState>(idx);
    auto cuda_gen = check_generator<CUDAGeneratorImplDeviceState>(gen);
    cuda_gen->set_current_seed(default_rng_seed_val);
    cuda_gen->set_philox_offset_per_thread(0);
    return gen;
  } else {
    auto gen = make_generator<CUDAGeneratorImplHostState>(idx);
    auto cuda_gen = check_generator<CUDAGeneratorImplHostState>(gen);
    cuda_gen->set_current_seed(default_rng_seed_val);
    cuda_gen->set_philox_offset_per_thread(0);
    return gen;
  }
}

} // namespace detail
} // namespace cuda

/**
 * CUDAGeneratorImpl* class implementations
 */

/**
 * CUDAGeneratorImpl methods
 */

CUDAGeneratorImpl::CUDAGeneratorImpl(DeviceIndex device_index)
  : c10::GeneratorImpl{Device(DeviceType::CUDA, device_index),
                       DispatchKeySet(c10::DispatchKey::CUDA)} {}

// a pure virtual destructor must have a definition
CUDAGeneratorImpl::~CUDAGeneratorImpl() {}

/**
 * Gets a nondeterministic random number from /dev/urandom or time,
 * seeds the CPUGeneratorImpl with it and then returns that number.
 *
 * FIXME: You can move this function to Generator.cpp if the algorithm
 * in getNonDeterministicRandom is unified for both CPU and CUDA
 */
uint64_t CUDAGeneratorImpl::seed() {
  auto random = c10::detail::getNonDeterministicRandom(true);
  this->set_current_seed(random);
  return random;
}

/**
 * Public clone method implementation
 *
 * See Note [Acquire lock when using random generators]
 */
std::shared_ptr<CUDAGeneratorImpl> CUDAGeneratorImpl::clone() const {
  // The concrete type will always be CUDAGeneratorImplDeviceState or
  // CUDAGeneratorImplHostState, so safe to cast
  return std::shared_ptr<CUDAGeneratorImpl>(static_cast<CUDAGeneratorImpl*>(this->clone_impl()));
}

/**
 * CUDAGeneratorImplHostState methods
 */

CUDAGeneratorImplHostState::CUDAGeneratorImplHostState(DeviceIndex device_index)
  : CUDAGeneratorImpl(device_index) {}

/**
 * Sets the seed to be used by curandStatePhilox4_32_10
 * Resets the philox_offset_per_thread_ to 0
 *
 * See Note [Acquire lock when using random generators]
 */
void CUDAGeneratorImplHostState::set_current_seed(uint64_t seed) {
  seed_ = seed;
  philox_offset_per_thread_ = 0;
}

/**
 * Gets the current seed.
 */
uint64_t CUDAGeneratorImplHostState::current_seed() const {
  return seed_;
}

/**
 * Sets the philox_offset_per_thread_ to be used by curandStatePhilox4_32_10
 *
 * See Note [Acquire lock when using random generators]
 */
void CUDAGeneratorImplHostState::set_philox_offset_per_thread(uint64_t offset) {
  philox_offset_per_thread_ = offset;
}

/**
 * Gets the current philox_offset_per_thread_ of CUDAGeneratorImpl.
 */
uint64_t CUDAGeneratorImplHostState::philox_offset_per_thread() const {
  return philox_offset_per_thread_;
}

/**
 * Gets the seed and philox offset value to be used in
 * curandStatePhilox4_32_10
 *
 * Each kernel using philox has to sensibly increment offset
 * for future users of philox. So it gets the "old" value for
 * itself (before add), and tells subsequent users which offset
 * they should use, since only the kernel knows how many randoms
 * it intends to generate.
 *
 * Increment should be at least the number of curand() random numbers used in
 * each thread. It is the user's responsibility to make sure that the increment
 * for philox is never smaller than the number of curand() calls. Increment
 * value > the number of curand() calls won't harm but anything less would mean
 * that you would be reusing random values from previous calls.
 *
 * See Note [Acquire lock when using random generators]
 */
philox_cuda_state_t CUDAGeneratorImplHostState::philox_cuda_state(uint64_t increment) {
  // What's with all the "this->" explicitness?
  // Virtual calls in methods should work without "this->".
  uint64_t offset = this->philox_offset_per_thread_;
  this->philox_offset_per_thread_ += increment;
  return philox_cuda_state_t{this->seed_, offset};
}


/**
 * Temporary, allows incremental refactor of call sites to use philox_cuda_state.
 */
std::pair<uint64_t, uint64_t> CUDAGeneratorImplHostState::philox_engine_inputs(uint64_t increment) {
  return this->philox_cuda_state(increment).to_kernel_arg().state_;
}

std::shared_ptr<CUDAGeneratorImplHostState> CUDAGeneratorImplHostState::clone() const {
  return std::shared_ptr<CUDAGeneratorImplHostState>(this->clone_impl());
}

CUDAGeneratorImplHostState* CUDAGeneratorImplHostState::clone_impl() const {
  auto gen = new CUDAGeneratorImplHostState(this->device().index());
  gen->set_current_seed(this->seed_);
  gen->set_philox_offset_per_thread(this->philox_offset_per_thread_);
  return gen;
}

/**
 * CUDAGeneratorImplDeviceState methods
 * See descriptions of corresponding HostState methods.
 *
 * Some casts back and forth between uint64_t and int64_t occur because
 * there's no such thing as uint64_t Tensors, but DeviceState must match
 * HostState's uint64_t interface.
 */

CUDAGeneratorImplDeviceState::CUDAGeneratorImplDeviceState(DeviceIndex device_index)
  : CUDAGeneratorImpl(device_index) {}

void CUDAGeneratorImplDeviceState::clear_states() {
  per_stream_states_.clear();
  live_refs_.clear();
}

void CUDAGeneratorImplDeviceState::set_current_seed(uint64_t seed) {
  // Sets up this instance to lazily recreate per-stream states from the new seed.
  init_seed_ = seed;
  init_offset_ = 0;
  clear_states();
}

uint64_t CUDAGeneratorImplDeviceState::current_seed() const {
  // .item() syncs on the current stream.
  return static_cast<uint64_t>(seed_.item().to<int64_t>());
}

void CUDAGeneratorImplDeviceState::set_philox_offset_per_thread(uint64_t offset) {
  // Sets up this instance to lazily recreate per-stream states from the new offset.
  init_offset = 0;
  clear_states();
}

uint64_t CUDAGeneratorImplDeviceState::philox_offset_per_thread() const {
  // .item() syncs on the current stream.
  return static_cast<uint64_t>(philox_offset_per_thread_.item().to<int64_t>());
}

philox_cuda_state_t CUDAGeneratorImplDeviceState::philox_cuda_state(uint64_t increment) {
  auto stream = at::cuda::getCurrentCUDAStream();

  // State snapshot for the coming kernel.
  philox_cuda_state_t for_kernel;

  // Retrieves state for current stream, creates it if necessary
  auto state = per_stream_states.find(stream.id());
  if (state == per_stream_states.end()) {
    // State tensors for this stream don't exist yet, create them
    auto options = TensorOptions().device(at::kCUDA).dtype(at::kLong);
    auto seed = at::native::full({1}, static_cast<int64_t>(init_seed_), options);
    auto offset = at::native::full({1}, static_cast<int64_t>(init_offset_), options);
    auto next_offset = at::native::full({1}, static_cast<int64_t>(init_offset_), options);
    // The only reason we stash stream in live_refs is so accept_clone_impl
    // can clone tensors from a source DeviceState instance on the right streams.
    live_refs_.push_back{{seed,
                          offset,
                          next_offset,
                          stream}};
    philox_cuda_state_t new_state{seed.data_ptr<int64_t>(),
                                  offset.data_ptr<int64_t>(),
                                  next_offset.data_ptr<int64_t>(),
                                  increment,
                                  stream.id() * max_threads_per_kernel /*subseq_pool_start*/}
    for_kernel = new_state;
    // *next_offset_ptr will be updated by at::cuda::philox::unpack in the coming kernel.
    // After freezing for_kernel, swap next and current so "next" becomes "current" for
    // the kernel after the coming kernel.
    std::swap(state.offset_ptr, state.next_offset_ptr);
    per_stream_states[stream.id()] = new_state;
  } else {
    for_kernel = *state;
    std::swap(*state.offset_ptr, *state.next_offset_ptr);
  }

  return for_kernel;
}

/**
 * Unlike the HostState version, this version throws an error, so if we requested
 * DeviceState, it points out ops that need refactoring to use philox_cuda_state.
 */
std::pair<uint64_t, uint64_t> CUDAGeneratorImplDeviceState::philox_engine_inputs(uint64_t increment) {
  TORCH_CHECK(false,
              "An op called philox_engine_inputs, which is incompatible with maintaining "
              "cuda rng states on the device.  The op should be refactored to use "
              "philox_cuda_state instead.");
  return std::pair<uint64_t, uint64_t>{};
}

std::shared_ptr<CUDAGeneratorImplDeviceState> CUDAGeneratorImplDeviceState::clone() const {
  return std::shared_ptr<CUDAGeneratorImplDeviceState>(this->clone_impl());
}

// Tries to implement the following guarantee:
// If cloned instance is used identically to source instance in the future,
// it will behave identically to source instance.
CUDAGeneratorImplDeviceState* CUDAGeneratorImplDeviceState::clone_impl() const {
  auto gen = new CUDAGeneratorImplDeviceState(this->device().index());
  gen->accept_clone_impl(this->init_seed_,
                         this->init_offset,
                         this->per_stream_states,
                         this->live_refs);
  return gen;
}

void CUDAGeneratorImplDeviceState::accept_clone_impl(const uint64_t& init_seed,
                                                     const uint64_t& init_offset,
                                                     const per_stream_states_t per_stream_states,
                                                     const live_refs_t& live_refs) {
  for const auto&
}

} // namespace at
