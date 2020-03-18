#include <ATen/core/Generator.h>
#include <c10/cuda/CUDAFunctions.h>
#include <ATen/CPUGenerator.h>
#ifdef USE_CUDA
#include <ATen/CUDAGenerator.h>
#endif

namespace at {

namespace detail {

/**
 * PyTorch maintains a collection of default generators that get
 * initialized once. The purpose of these default generators is to
 * maintain a global running state of the pseudo random number generation,
 * when a user does not explicitly mention any generator.
 * getDefaultCPUGenerator gets the default generator for a particular
 * device.
 */
const Generator& getDefaultCPUGenerator() {
  static auto default_gen_cpu = createCPUGenerator(c10::detail::getNonDeterministicRandom());
  return default_gen_cpu;
}

/**
 * Utility to create a CPUGenerator. Returns a shared_ptr
 */
Generator createCPUGenerator(uint64_t seed_val) {
  return make_generator<CPUGenerator>(seed_val);
}

} // namespace detail

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
    default_gens_cuda[idx] = make_generator<at::CUDAGenerator>(idx);
    default_gens_cuda[idx]->seed();
  });
  return default_gens_cuda[idx];
}

/**
 * Utility to create a CUDAGenerator. Returns a shared_ptr
 */
Generator createCUDAGenerator(DeviceIndex device_index) {
  std::call_once(num_gpu_init_flag, initCUDAGenVector);
  DeviceIndex idx = device_index;
  if (idx == -1) {
    idx = c10::cuda::current_device();
  }
  TORCH_CHECK(idx >= 0 && idx < num_gpus, "The device_index is invalid.");
  auto gen = make_generator<at::CUDAGenerator>(idx);
  auto cuda_gen = check_generator<at::CUDAGenerator>(gen);
  cuda_gen->set_current_seed(default_rng_seed_val);
  cuda_gen->set_philox_offset_per_thread(0);
  return gen;
}

} // namespace detail
} // namespace cuda

} // namespace at
