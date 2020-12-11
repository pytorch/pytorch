#include <ATen/Utils.h>
#include <ATen/CUDAGeneratorImpl.h>
#include <ATen/cuda/CUDAGraphsUtils.cuh>
#include <c10/core/StreamGuard.h>
#include <c10/cuda/CUDAFunctions.h>
#include <ATen/Utils.h>

namespace at {
namespace cuda {
namespace detail {

namespace {

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
              DispatchKeySet(c10::DispatchKey::CUDA)} {
  at::cuda::assertNotCapturing("Cannot construct a new CUDAGeneratorImpl");
}

/**
 * Sets the seed to be used by curandStatePhilox4_32_10
 * Resets the philox_offset_per_thread_ to 0
 *
 * See Note [Acquire lock when using random generators]
 */
void CUDAGeneratorImpl::set_current_seed(uint64_t seed) {
  at::cuda::assertNotCapturing("Cannot call CUDAGeneratorImpl::set_current_seed");
  seed_ = seed;
  philox_offset_per_thread_ = 0;
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
  at::cuda::assertNotCapturing("Cannot call CUDAGeneratorImpl::seed");
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
  at::cuda::assertNotCapturing("Cannot call CUDAGeneratorImpl::set_philox_offset_per_thread");
  philox_offset_per_thread_ = offset;
}

/**
 * Gets the current philox_offset_per_thread_ of CUDAGeneratorImpl.
 */
uint64_t CUDAGeneratorImpl::philox_offset_per_thread() {
  at::cuda::assertNotCapturing("Cannot call CUDAGeneratorImpl::philox_offset_per_thread");
  return philox_offset_per_thread_;
}

/**
 * Called by CUDAGraph to prepare this instance for a graph capture region.
 * offset_extragraph is the initial offset at the start of the graphed region.
 * offset_intragraph tracks the offset in the graphed region.
 */
void CUDAGeneratorImpl::capture_prologue(int64_t* offset_extragraph) {
  offset_extragraph_ = offset_extragraph;
  offset_intragraph_ = 0;
  graph_expects_this_gen_ = true;
}

/**
 * Called by CUDAGraph to finalize a graph capture region for this instance.
 */
uint64_t CUDAGeneratorImpl::capture_epilogue() {
  graph_expects_this_gen_ = false;
  return offset_intragraph_;
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
    TORCH_CHECK(graph_expects_this_gen_,
                "philox_cuda_state for an unexpected CUDA generator used during capture. "
                CAPTURE_DEFAULT_GENS_MSG);
    uint32_t offset = this->offset_intragraph_;
    TORCH_INTERNAL_ASSERT(this->offset_intragraph_ <=
                          std::numeric_limits<uint32_t>::max() - increment);
    this->offset_intragraph_ += increment;
    return PhiloxCudaState(this->seed_,
                           this->offset_extragraph_,
                           offset);
  } else {
    TORCH_CHECK(!graph_expects_this_gen_,
                "CUDA generator expects graph capture to be underway, "
                "but the current stream is not capturing.");
    uint64_t offset = this->philox_offset_per_thread_;
    this->philox_offset_per_thread_ += increment;
    return PhiloxCudaState(this->seed_, offset);
  }
}

/**
 * Temporarily accommodates call sites that use philox_engine_inputs.
 * Allows incremental refactor of call sites to use philox_cuda_state.
 */
std::pair<uint64_t, uint64_t> CUDAGeneratorImpl::philox_engine_inputs(uint64_t increment) {
  at::cuda::assertNotCapturing("Refactor this op to use CUDAGeneratorImpl::philox_cuda_state. "
                               "Cannot call CUDAGeneratorImpl::philox_engine_inputs");
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
  at::cuda::assertNotCapturing("Cannot call CUDAGeneratorImpl::clone_impl");
  auto gen = new CUDAGeneratorImpl(this->device().index());
  gen->set_current_seed(this->seed_);
  gen->set_philox_offset_per_thread(this->philox_offset_per_thread_);
  return gen;
}

} // namespace at
