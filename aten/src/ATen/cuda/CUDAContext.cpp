#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>

#include <ATen/cuda/CUDAConfig.h>
#include <mutex>
#include <deque>
#include <vector>

namespace at { namespace cuda {

namespace {

DeviceIndex num_gpus = -1;
std::once_flag init_flag;
std::deque<std::once_flag> device_flags;
std::vector<cudaDeviceProp> device_properties;

void initCUDAContextVectors() {
  num_gpus = c10::cuda::device_count();
  device_flags.resize(num_gpus);
  device_properties.resize(num_gpus);
}

void initDeviceProperty(DeviceIndex device_index) {
  cudaDeviceProp device_prop;
  AT_CUDA_CHECK(cudaGetDeviceProperties(&device_prop, device_index));
  device_properties[device_index] = device_prop;
}

} // anonymous namespace

// We need this function to force the linking against torch_cuda on Windows.
// If you need to modify this function, please specify a new function and apply the changes
// according to https://github.com/pytorch/pytorch/pull/34288.
// Related issue: https://github.com/pytorch/pytorch/issues/31611.
/* Device info */
int warp_size() {
  return getCurrentDeviceProperties()->warpSize;
}

cudaDeviceProp* getCurrentDeviceProperties() {
  auto device = c10::cuda::current_device();
  return getDeviceProperties(device);
}

cudaDeviceProp* getDeviceProperties(int64_t device) {
  std::call_once(init_flag, initCUDAContextVectors);
  if (device == -1) device = c10::cuda::current_device();
  AT_ASSERT(device >= 0 && device < num_gpus);
  std::call_once(device_flags[device], initDeviceProperty, device);
  return &device_properties[device];
}

Allocator* getCUDADeviceAllocator() {
  return c10::cuda::CUDACachingAllocator::get();
}

namespace detail {

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
