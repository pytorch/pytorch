#include <ATen/Utils.h>
#include <ATen/CUDAGeneratorImpl.h>
#include <ATen/cuda/StatefulCUDAOpsUtils.cuh>
#include <c10/core/StreamGuard.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAFunctions.h>


namespace at {

// forward-declares
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
 *
 * max_kernel_threads, defined in StatefulCUDAOpsUtils.cuh as
 *   constexpr uint64_t max_kernel_threads =  (uint64_t(1) << 40);
 * is an estimated upper limit of the maximum
 * number of threads any kernel may possibly need.
 * In other words, it estimates how many subsequences we should assign
 * to each stream's pool.
 *
 * Curand philox offers 2^64 subsequences.  Since we assign 2^40 subsequences
 * to each stream, stream id values must be <= 2^24.
 * See Note [StreamId assignment] in c10/cuda/CUDAStream.cpp.
 */

// Ensures we only call cudaGetDeviceCount only once.
std::once_flag num_gpu_init_flag;

// Total number of gpus in the system.
int64_t num_gpus;

// Ensures default_gens* are initialized once.
std::deque<std::once_flag> cuda_gens_init_flag_host_state;
std::deque<std::once_flag> cuda_gens_init_flag_dev_state;

// Default, global CUDA generators, one per GPU.
std::vector<Generator> default_gens_host_state;
std::vector<Generator> default_gens_dev_state;

/*
* Populates the global variables related to CUDA generators
* Warning: this function must only be called once!
*/
void initCUDAGenVector(){
  num_gpus = c10::cuda::device_count();
  cuda_gens_init_flag_host_state.resize(num_gpus);
  cuda_gens_init_flag_dev_state.resize(num_gpus);
  default_gens_host_state.resize(num_gpus);
  default_gens_dev_state.resize(num_gpus);
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

  if (at::globalContext().statefulCUDAOpStatesOnDevice()) {
    std::call_once(cuda_gens_init_flag_dev_state[idx],
                   [&] {
                     default_gens_dev_state[idx] = make_generator<CUDAGeneratorImplDevState>(idx);
                     default_gens_dev_state[idx].seed();
                   });
    return default_gens_dev_state[idx];
  } else {
    std::call_once(cuda_gens_init_flag_host_state[idx],
                   [&] {
                     default_gens_host_state[idx] = make_generator<CUDAGeneratorImplHostState>(idx);
                     default_gens_host_state[idx].seed();
                   });
    return default_gens_host_state[idx];
  }
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
    auto gen = make_generator<CUDAGeneratorImplDevState>(idx);
    auto cuda_gen = check_generator<CUDAGeneratorImplDevState>(gen);
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
  // The concrete type will always be CUDAGeneratorImplDevState or
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
  seeded_ = true;
}

/**
 * Gets the current seed.
 */
uint64_t CUDAGeneratorImplHostState::current_seed() const {
  TORCH_CHECK(seeded_, "CUDAGeneratorImplHostState instance not yet seeded");
  return seed_;
}

/**
 * Sets the philox_offset_per_thread_ to be used by curandStatePhilox4_32_10
 *
 * See Note [Acquire lock when using random generators]
 */
void CUDAGeneratorImplHostState::set_philox_offset_per_thread(uint64_t offset) {
  TORCH_CHECK(seeded_, "CUDAGeneratorImplHostState instance not yet seeded");
  philox_offset_per_thread_ = offset;
}

/**
 * Gets the current philox_offset_per_thread_ of CUDAGeneratorImpl.
 */
uint64_t CUDAGeneratorImplHostState::philox_offset_per_thread() const {
  TORCH_CHECK(seeded_, "CUDAGeneratorImplHostState instance not yet seeded");
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
PhiloxCudaState CUDAGeneratorImplHostState::philox_cuda_state(uint64_t increment) {
  TORCH_CHECK(seeded_, "CUDAGeneratorImplHostState instance not yet seeded");
  // What's with all the "this->" explicitness?
  // Virtual calls in methods should work without "this->".
  uint64_t offset = this->philox_offset_per_thread_;
  this->philox_offset_per_thread_ += increment;
  return PhiloxCudaState{this->seed_, offset, 0};
}


/**
 * Temporary, allows incremental refactor of call sites to use philox_cuda_state.
 */
std::pair<uint64_t, uint64_t> CUDAGeneratorImplHostState::philox_engine_inputs(uint64_t increment) {
  TORCH_CHECK(seeded_, "CUDAGeneratorImplHostState instance not yet seeded");
  auto state = this->philox_cuda_state(increment);
  return std::make_pair(state.seed_.val, state.offset_.val);
}

std::shared_ptr<CUDAGeneratorImplHostState> CUDAGeneratorImplHostState::clone() const {
  TORCH_CHECK(seeded_, "CUDAGeneratorImplHostState instance not yet seeded");
  return std::shared_ptr<CUDAGeneratorImplHostState>(this->clone_impl());
}

CUDAGeneratorImplHostState* CUDAGeneratorImplHostState::clone_impl() const {
  TORCH_CHECK(seeded_, "CUDAGeneratorImplHostState instance not yet seeded");
  auto gen = new CUDAGeneratorImplHostState(this->device().index());
  gen->set_current_seed(this->seed_);
  gen->set_philox_offset_per_thread(this->philox_offset_per_thread_);
  return gen;
}

/**
 * CUDAGeneratorImplDevState methods
 * See descriptions of corresponding HostState methods.
 *
 * Some casts back and forth between uint64_t and int64_t occur because
 * there's no such thing as uint64_t Tensors, but DevState must match
 * HostState's uint64_t interface.
 */

CUDAGeneratorImplDevState::CUDAGeneratorImplDevState(DeviceIndex device_index)
  : CUDAGeneratorImpl(device_index) {}

void CUDAGeneratorImplDevState::set_current_seed(uint64_t seed) {
  // Sets up this instance to lazily recreate per-stream states from the new seed.
  init_seed_ = seed;
  init_offset_ = 0;
  stream_states_.clear();
  seeded_ = true;
}

uint64_t CUDAGeneratorImplDevState::current_seed() const {
  TORCH_CHECK(seeded_, "CUDAGeneratorImplDevState instance not yet seeded");
  auto state_iter = stream_states_.find(at::cuda::getCurrentCUDAStream().id());
  if (state_iter == stream_states_.end()) {
    // Returns init_seed_ instead of lazy-initing state, because this function is const.
    return init_seed_;
  } else {
    const auto& seed = std::get<0>((*state_iter).second.second);
    // .item() syncs on the current stream.
    return static_cast<uint64_t>(seed.item().to<int64_t>());
  }
}

void CUDAGeneratorImplDevState::set_philox_offset_per_thread(uint64_t offset) {
  TORCH_CHECK(seeded_, "CUDAGeneratorImplDevState instance not yet seeded");
  // Sets up this instance to lazily recreate per-stream states from the new offset.
  init_offset_ = 0;
  stream_states_.clear();
}

uint64_t CUDAGeneratorImplDevState::philox_offset_per_thread() const {
  TORCH_CHECK(seeded_, "CUDAGeneratorImplDevState instance not yet seeded");
  auto state_iter = stream_states_.find(at::cuda::getCurrentCUDAStream().id());
  if (state_iter == stream_states_.end()) {
    // Returns init_offset_ instead of lazy-initing state, because this function is const.
    return init_offset_;
  } else {
    const auto& offset = std::get<1>((*state_iter).second.second);
    // .item() syncs on the current stream.
    return static_cast<uint64_t>(offset.item().to<int64_t>());
  }
}

CUDAGeneratorImplDevState::StateWithRefs&
CUDAGeneratorImplDevState::add_stream_state(Tensor seed,
                                            Tensor offset,
                                            c10::Stream stream) {
  TORCH_CHECK(seeded_, "CUDAGeneratorImplDevState instance not yet seeded");
  LiveRefs new_refs{seed,
                    offset,
                    stream};
  PhiloxCudaState new_raw{seed.data_ptr<int64_t>(),
                          offset.data_ptr<int64_t>(),
                          stream.id()};
  // emplace state or assign if it already exists
  auto outcome = stream_states_.emplace(std::make_pair(stream.id(), std::make_pair(new_raw, new_refs)));
  if (!outcome.second) {
    (*outcome.first).second = std::make_pair(new_raw, new_refs);
  }
  return (*outcome.first).second;
}

CUDAGeneratorImplDevState::StateWithRefs&
CUDAGeneratorImplDevState::get_state_lazy_init() {
  TORCH_CHECK(seeded_, "CUDAGeneratorImplDevState instance not yet seeded");

  auto stream = at::cuda::getCurrentCUDAStream();
  auto state_iter = stream_states_.find(stream.id());

  if (state_iter == stream_states_.end()) {
    // State tensors for this stream don't exist yet, create them
    auto options = TensorOptions().device(at::kCUDA).dtype(at::kLong);
    auto seed = at::native::full({1}, static_cast<int64_t>(init_seed_), options);
    auto offset = at::native::full({1}, static_cast<int64_t>(init_offset_), options);
    auto& new_state = add_stream_state(seed,
                                       offset,
                                       stream);
    return new_state;
  } else {
    return (*state_iter).second;
  }
}

// Warning:  The retrieved PhiloxCudaState should be used for one and only one kernel.
PhiloxCudaState CUDAGeneratorImplDevState::philox_cuda_state(uint64_t increment) {
  TORCH_CHECK(seeded_, "CUDAGeneratorImplDevState instance not yet seeded");

  // Retrieves state for current stream, creates it if necessary
  auto& state = get_state_lazy_init().first;

  // Value-copied snapshot for the caller's kernel.
  PhiloxCudaState for_kernel = state;
  for_kernel.set_increment(increment);

  // for_kernel holds pointers to persistently stored states, so update_offset
  // increments the offset tensor in place.  We do it here so the caller isn't obligated
  // to request an update after their consumer kernel. The update here is "premature"
  // because it happens before the consumer kernel. The in-kernel "unpack" call
  // compensates by subtracting the current increment (see StatefulCUDAOpsUtils.cuh).
  at::cuda::philox::update_offset(for_kernel);

  return for_kernel;
}

/**
 * Unlike the HostState version, this version throws an error, so if we requested
 * DevState, it points out ops that need refactoring to use philox_cuda_state().
 */
std::pair<uint64_t, uint64_t> CUDAGeneratorImplDevState::philox_engine_inputs(uint64_t increment) {
  TORCH_CHECK(false,
              "An op called philox_engine_inputs, which is incompatible with maintaining "
              "cuda rng states on the device.  The op should be refactored to use "
              "PhiloxCudaState instead.");
  return std::pair<uint64_t, uint64_t>{};
}

std::shared_ptr<CUDAGeneratorImplDevState> CUDAGeneratorImplDevState::clone() const {
  TORCH_CHECK(seeded_, "CUDAGeneratorImplDevState instance not yet seeded");
  return std::shared_ptr<CUDAGeneratorImplDevState>(this->clone_impl());
}

// Tries to implement the following guarantee:
// If cloned instance is used identically to source instance in the future,
// it will behave identically to source instance.
CUDAGeneratorImplDevState* CUDAGeneratorImplDevState::clone_impl() const {
  TORCH_CHECK(seeded_, "CUDAGeneratorImplDevState instance not yet seeded");
  auto gen = new CUDAGeneratorImplDevState(this->device().index());
  gen->accept_clone_impl(this->init_seed_,
                         this->init_offset_,
                         this->stream_states_);
  return gen;
}

void CUDAGeneratorImplDevState::accept_clone_impl(const uint64_t& init_seed,
                                                  const uint64_t& init_offset,
                                                  const StreamStatesWithRefs& stream_states) {
  TORCH_CHECK(seeded_, "CUDAGeneratorImplDevState instance not yet seeded");
  init_seed_ = init_seed;
  init_offset_ = init_offset;
  stream_states_.clear();
  for (const auto& source_state : stream_states) {
    auto& source_refs = source_state.second.second;
    auto& source_stream = std::get<2>(source_refs);
    // Clones per-stream state tensors on their streams.
    c10::OptionalStreamGuard guard(source_stream);
    add_stream_state(std::get<0>(source_refs).clone(),
                     std::get<1>(source_refs).clone(),
                     source_stream);
  }
}

} // namespace at
