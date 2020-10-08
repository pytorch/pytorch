#include <ATen/CUDAGeneratorImpl.h>
#include <c10/cuda/CUDAFunctions.h>
#include <ATen/Utils.h>

namespace at {

namespace cuda { namespace detail {

// Ensures we only call cudaGetDeviceCount only once.
static std::once_flag num_gpu_init_flag;

// Total number of gpus in the system.
static int64_t num_gpus;

// Ensures default_gens_cuda is initialized once.
static std::deque<std::once_flag> cuda_gens_init_flag;

// Default, global CUDA generators, one per GPU.
static std::vector<Generator> default_gens_cuda;

/*
* Populates the global variables related to CUDA generators
* Warning: this function must only be called once!
*/
static void initCUDAGenVector(){
  num_gpus = c10::cuda::device_count();
  cuda_gens_init_flag.resize(num_gpus);
  default_gens_cuda.resize(num_gpus);
}

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
  std::call_once(cuda_gens_init_flag[idx], [&] {
    if (cuda_rng_state_on_device) {
      default_gens_cuda[idx] = make_generator<CUDAGeneratorImplHostState>(idx);
    } else {
      default_gens_cuda[idx] = make_generator<CUDAGeneratorImplDeviceState>(idx);
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

  if (cuda_rng_state_on_device) {
    auto gen = make_generator<CUDAGeneratorImplHostState>(idx);
    auto cuda_gen = check_generator<CUDAGeneratorImplHostState>(gen);
  } else {
    auto gen = make_generator<CUDAGeneratorImplDeviceState>(idx);
    auto cuda_gen = check_generator<CUDAGeneratorImplDeviceState>(gen);
  }
  cuda_gen->set_current_seed(default_rng_seed_val);
  cuda_gen->set_philox_offset_per_thread(0);
  return gen;
}

} // namespace detail
} // namespace cuda

/**
 * CUDAGeneratorImpl* class implementations
 */
CUDAGeneratorImplHostState::CUDAGeneratorImplHostState(DeviceIndex device_index)
  : c10::GeneratorImpl{Device(DeviceType::CUDA, device_index),
                       DispatchKeySet(c10::DispatchKey::CUDA)},
    state_on_device{false} {}

CUDAGeneratorImpl::CUDAGeneratorImplDeviceState(DeviceIndex device_index)
  : c10::GeneratorImpl{Device(DeviceType::CUDA, device_index),
                       DispatchKeySet(c10::DispatchKey::CUDA)},
    state_on_device{true} {
  state_update_stream_ = at::cuda::getStreamFromPool(/*isHighPriority=*/true,
                                                     /*index=*/device_index);
  c10::OptionalStreamGuard stream_guard{state_update_stream_};
  auto options = TensorOptions().device(at::kCUDA).dtype(at::kLong);
  seed_ = at::native::empty_cuda({1}, options);
  philox_offset_per_thread_ = at::native::empty_cuda({1}, options);
}

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

void CUDAGeneratorImplDeviceState::set_current_seed(uint64_t seed) {
  c10::OptionalStreamGuard stream_guard{state_update_stream_};
  seed_.fill_(seed);
  philox_offset_per_thread_.fill_(0);
}

/**
 * Gets the current seed of CUDAGeneratorImpl.
 */
uint64_t CUDAGeneratorImplHostState::current_seed() const {
  return seed_;
}

uint64_t CUDAGeneratorImplDeviceState::current_seed() const {
  // See Note: Device-side RNG state update ordering
  c10::OptionalStreamGuard stream_guard{state_update_stream_};
  // .item() syncs on the current stream.
  return seed_.item().to<uint64_t>();
}

/**
 * Gets a nondeterministic random number from /dev/urandom or time,
 * seeds the CPUGeneratorImpl with it and then returns that number.
 *
 * FIXME: You can move this function to Generator.cpp if the algorithm
 * in getNonDeterministicRandom is unified for both CPU and CUDA
 */
uint64_t CUDAGeneratorImplHostState::seed() {
  auto random = c10::detail::getNonDeterministicRandom(true);
  this->set_current_seed(random);
  return random;
}

uint64_t CUDAGeneratorImplDeviceState::seed() {
  auto random = c10::detail::getNonDeterministicRandom(true);
  this->set_current_seed(random);
  return random;
}

/**
 * Sets the philox_offset_per_thread_ to be used by curandStatePhilox4_32_10
 *
 * See Note [Acquire lock when using random generators]
 */
void CUDAGeneratorImplHostState::set_philox_offset_per_thread(uint64_t offset) {
  philox_offset_per_thread_ = offset;
}

void CUDAGeneratorImplDeviceState::set_philox_offset_per_thread(uint64_t offset) {
  // See Note: Device-side RNG state update ordering
  c10::OptionalStreamGuard stream_guard{state_update_stream_};
  philox_offset_per_thread_.fill_(offset);
}

/**
 * Gets the current philox_offset_per_thread_ of CUDAGeneratorImpl.
 */
uint64_t CUDAGeneratorImplHostState::philox_offset_per_thread() {
  return philox_offset_per_thread_;
}

uint64_t CUDAGeneratorImplDeviceState::philox_offset_per_thread() {
  // See Note: Device-side RNG state update ordering
  c10::OptionalStreamGuard stream_guard{state_update_stream_};
  // .item() syncs on the current stream.
  return philox_offset_per_thread_.item().to<uint64_t>();
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
  uint64_t offset = this->philox_offset_per_thread_;
  this->philox_offset_per_thread_ += increment;
  return philox_cuda_state_t{this->seed_, offset};
}

philox_cuda_state_t CUDAGeneratorImplDeviceState::philox_cuda_state(uint64_t increment) {
  auto ambient_stream = at::cuda::getCurrentCUDAStream();
  // See Note: Device-side RNG state update ordering
  c10::OptionalStreamGuard stream_guard{state_update_stream_};
  // Snapshots the current state of state_update_stream_.
  // Returns deep copies so the current caller gets its own frozen state,
  // and can sync on state_update_stream_ to use the frozen state.
  // If a subsequent caller enqueues some new call that changes the state
  // (set_current_seed, set_philox_offset_per_thread, or philox_cuda_state)
  // it won't affect the values the current caller's kernels are using.
  // This is equivalent to returning CPU-side states by value.

  // To discuss: what's with all the this-> here?  im imitating what i inherited but i don't
  // know why they were there in the first place.
  auto frozen_seed = this->seed_.clone();
  auto frozen_offset = this->philox_offset_per_thread_.clone();
  this->philox_offset_per_thread_.add_(increment);

  c10::cuda::CUDACachingAllocator::recordStream(frozen_seed.storage().data_ptr(),
                                                ambient_stream);
  c10::cuda::CUDACachingAllocator::recordStream(frozen_offset.storage().data_ptr(),
                                                ambient_stream);

  // Makes ambient thread wait for its state copies.
  auto event = c10::Event{c10::DeviceType::CUDA};
  event.record(*state_update_stream_);
  ambient_stream->wait(event);

  return philox_cuda_state_t{std::move(frozen_seed), std::move(frozen_offset)};
}

// TEMPORARY, ALLOWS JIT STUFF TO COMPILE, DO NOT MERGE WITH THIS,
// FIX JIT TO USE philox_cuda_state_t and philox_kernelarg_t
std::pair<uint64_t, uint64_t> CUDAGeneratorImplHostState::philox_engine_inputs(uint64_t increment) {
  return std::make_pair(current_seed(), philox_offset_per_thread());
}

std::pair<uint64_t, uint64_t> CUDAGeneratorImplDeviceState::philox_engine_inputs(uint64_t increment) {
  return std::make_pair(current_seed(), philox_offset_per_thread());
}

/*
 * Gets the DeviceType of CUDAGeneratorImpl.
 * Used for type checking during run time.
 */
DeviceType CUDAGeneratorImplHostState::device_type() {
  return DeviceType::CUDA;
}

DeviceType CUDAGeneratorImplDeviceState::device_type() {
  return DeviceType::CUDA;
}

/**
 * Public clone method implementation
 *
 * See Note [Acquire lock when using random generators]
 */
std::shared_ptr<CUDAGeneratorImplHostState> CUDAGeneratorImplHostState::clone() const {
  return std::shared_ptr<CUDAGeneratorImplHostState>(this->clone_impl());
}

std::shared_ptr<CUDAGeneratorImplDeviceState> CUDAGeneratorImplDeviceState::clone() const {
  return std::shared_ptr<CUDAGeneratorImplDeviceState>(this->clone_impl());
}

/**
 * Private clone method implementation
 *
 * See Note [Acquire lock when using random generators]
 */
CUDAGeneratorImplHostState* CUDAGeneratorImplHostState::clone_impl() const {
  auto gen = new CUDAGeneratorImplHostState(this->device().index());
  gen->set_current_seed(this->seed_);
  gen->set_philox_offset_per_thread(this->philox_offset_per_thread_);
  return gen;
}

CUDAGeneratorImplDeviceState* CUDAGeneratorImplDeviceState::clone_impl() const {
  auto gen = new CUDAGeneratorImplDeviceState(this->device().index());
  gen->set_current_seed(this->seed_);
  gen->set_philox_offset_per_thread(this->philox_offset_per_thread_);
  return gen;
}

} // namespace at
