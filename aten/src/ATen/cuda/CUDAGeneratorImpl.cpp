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
    default_gens_cuda[idx] = make_generator<CUDAGeneratorImpl>(idx);
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
  auto gen = make_generator<CUDAGeneratorImpl>(idx);
  auto cuda_gen = check_generator<CUDAGeneratorImpl>(gen);
  cuda_gen->set_current_seed(default_rng_seed_val);
  cuda_gen->set_philox_offset_per_thread(0);
  return gen;
}

} // namespace detail
} // namespace cuda

/**
 * CUDAGeneratorImpl class implementation
 */
CUDAGeneratorImpl::CUDAGeneratorImpl(DeviceIndex device_index)
  : c10::GeneratorImpl{Device(DeviceType::CUDA, device_index),
              DispatchKeySet(c10::DispatchKey::CUDA)} { }

/**
 * Sets the seed to be used by curandStatePhilox4_32_10
 * Resets the philox_offset_per_thread_ to 0
 * 
 * See Note [Acquire lock when using random generators]
 */
void CUDAGeneratorImpl::set_current_seed(uint64_t seed) {
  seed_ = seed;
  philox_offset_per_thread_ = 0;
}

/**
 * Gets the current seed of CUDAGeneratorImpl.
 */
uint64_t CUDAGeneratorImpl::current_seed() const {
  return seed_;
}

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
 * Sets the philox_offset_per_thread_ to be used by curandStatePhilox4_32_10
 * 
 * See Note [Acquire lock when using random generators]
 */
void CUDAGeneratorImpl::set_philox_offset_per_thread(uint64_t offset) {
  philox_offset_per_thread_ = offset;
}

/**
 * Gets the current philox_offset_per_thread_ of CUDAGeneratorImpl.
 */
uint64_t CUDAGeneratorImpl::philox_offset_per_thread() {
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
std::pair<uint64_t, uint64_t> CUDAGeneratorImpl::philox_engine_inputs(uint64_t increment) {
  uint64_t offset = this->philox_offset_per_thread_;
  this->philox_offset_per_thread_ += increment;
  return std::make_pair(this->seed_, offset);
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
  auto gen = new CUDAGeneratorImpl(this->device().index());
  gen->set_current_seed(this->seed_);
  gen->set_philox_offset_per_thread(this->philox_offset_per_thread_);
  return gen;
}

} // namespace at
