#include <torch/csrc/jit/fuser/cuda/fused_kernel.h>
#include <torch/csrc/jit/fuser/compiler.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/nvrtc_stub/ATenNVRTC.h>
#include <ATen/CUDAGenerator.h>
#include <THC/THC.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/csrc/jit/resource_guard.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

// See NOTE [ USE OF NVRTC AND DRIVER API ]
const at::cuda::NVRTC& nvrtc() {
  return at::globalContext().getNVRTC();
}

static void getMajorMinor(
    const cudaDeviceProp* const prop,
    int& major,
    int& minor) {
  int nvrtc_major, nvrtc_minor;
  AT_CUDA_NVRTC_CHECK(nvrtc().nvrtcVersion(&nvrtc_major, &nvrtc_minor));

  // Short-circuits if NVRTC version too low
  AT_ASSERT(nvrtc_major >= 6);

  // Major and minor is determined by device properties and
  // possibly "downcompiled" to a lower (compatible) compute architecture
  // based on the NVRTC version
  major = prop->major;
  minor = prop->minor;
  if (nvrtc_major <= 7 && prop->major > 5) { // 7 supports 2-5.x
    major = 5;
    minor = 0;
  } else if (nvrtc_major <= 8 && prop->major > 6) { // 8 supports 2-6.x
    major = 6;
    minor = 0;
  } else if (nvrtc_major <= 9 && prop->major >= 7) { // 9 supports 3-7.2
    major = 7;
    if (prop->major == 7 && prop->minor <= 2)
      minor = prop->minor;
    else
      minor = 0;
  } else if (nvrtc_major <= 10 && prop->major >= 7) { // 10 supports 3-7.5
    major = 7;
    if (prop->major == 7 && prop->minor <= 5)
      minor = prop->minor;
    else
      minor = 0;
  }
}

// Compiles the specified kernel and stores the metadata required to run it
FusedKernelCUDA::FusedKernelCUDA(
    int16_t device,
    std::string name,
    std::string code,
    std::vector<TensorDesc> input_desc,
    std::vector<TensorDesc> output_desc,
    std::vector<PartitionDesc> chunk_desc,
    std::vector<PartitionDesc> concat_desc,
    bool has_random)
    : FusedKernel(
          std::move(name),
          std::move(code),
          std::move(input_desc),
          std::move(output_desc),
          std::move(chunk_desc),
          std::move(concat_desc),
          has_random),
      device_(device) {
  // Initializes driver's API context (if necessary)
  CUcontext pctx = 0;
  AT_CUDA_DRIVER_CHECK(nvrtc().cuCtxGetCurrent(&pctx));
  if (!pctx) {
    std::unique_lock<std::mutex> cudaFreeMutexLock(
        *(c10::cuda::CUDACachingAllocator::getFreeMutex()));
    cudaFree(0);
  }

  // Note: hacked at::DeviceGuard since at::DeviceGuard was failing to work
  // properly in some scenarios
  const auto prior_device = at::cuda::current_device();
  at::cuda::set_device(device_);

  // Acquires device and NVRTC properties (for compile arch and occupancy
  // calculations)
  prop_ = at::cuda::getCurrentDeviceProperties();
  int major, minor;
  getMajorMinor(prop_, major, minor);

  // Creates the NVRTC program
  nvrtcProgram program;
  AT_CUDA_NVRTC_CHECK(nvrtc().nvrtcCreateProgram(
      &program, code_.c_str(), nullptr, 0, nullptr, nullptr));

  const std::string compute = "--gpu-architecture=compute_" +
      std::to_string(major) + std::to_string(minor);
  const std::vector<const char*> args = {
      "--std=c++11", compute.c_str(), "-default-device"};
  const auto result =
      nvrtc().nvrtcCompileProgram(program, args.size(), args.data());
  if (result == NVRTC_ERROR_COMPILATION) {
    size_t logsize;
    nvrtc().nvrtcGetProgramLogSize(program, &logsize);
    std::vector<char> log(logsize);
    nvrtc().nvrtcGetProgramLog(program, log.data());
    std::stringstream cu;
    cu << log.data();
    throw std::runtime_error(cu.str());
  }
  ResourceGuard holdProgram(
      [&] { AT_CUDA_NVRTC_CHECK(nvrtc().nvrtcDestroyProgram(&program)); });
  AT_CUDA_NVRTC_CHECK(result);
  size_t ptx_size;
  AT_CUDA_NVRTC_CHECK(nvrtc().nvrtcGetPTXSize(program, &ptx_size));
  ptx_.resize(ptx_size);
  AT_CUDA_NVRTC_CHECK(nvrtc().nvrtcGetPTX(program, ptx_.data()));

  AT_CUDA_DRIVER_CHECK(nvrtc().cuModuleLoadData(&module_, ptx_.data()));
  AT_CUDA_DRIVER_CHECK(
      nvrtc().cuModuleGetFunction(&function_, module_, name_.c_str()));

  // Computes max blocks
  AT_CUDA_DRIVER_CHECK(nvrtc().cuOccupancyMaxActiveBlocksPerMultiprocessor(
      &maxBlocks_, function_, 128, 0));
  maxBlocks_ *= prop_->multiProcessorCount;

  // Resets device (end of hacked at::DeviceGuard)
  at::cuda::set_device(prior_device);
}

static int ceilDiv(const int a, const int b) {
  return (a + b - 1) / b;
}

void FusedKernelCUDA::launch_raw(
    const uint32_t numel,
    std::vector<void*>& arguments) const {
  at::cuda::CUDAGuard{device_};
  // Hacked at::DeviceGuard (see note above)
  const auto prior_device = at::cuda::current_device();
  at::cuda::set_device(device_);

  const auto nBlocks = std::min(maxBlocks_, ceilDiv(numel, kBlockSize));

  // Adds random state to arguments if necessary
  // Note: philox_engine_inputs defined here so its lifetime extends to the launch
  std::pair<uint64_t, uint64_t> philox_engine_inputs;
  if (has_random_) {
    const auto rand_offset =
        4 * (std::ceil(numel / (4.0 * kBlockSize * nBlocks)) + 1);
    auto gen = at::cuda::detail::getDefaultCUDAGenerator();
    {
      // See Note [Acquire lock when using random generators]
      std::lock_guard<std::mutex> lock(gen->mutex_);
      philox_engine_inputs = gen->philox_engine_inputs(rand_offset);
    }
    arguments.push_back(&philox_engine_inputs.first);
    arguments.push_back(&philox_engine_inputs.second);
  }

  // Launches kernel on current stream (device was set by executor)
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_CUDA_DRIVER_CHECK(nvrtc().cuLaunchKernel(
      function_,
      nBlocks,
      1,
      1,
      kBlockSize,
      1,
      1,
      0,
      stream,
      arguments.data(),
      nullptr));

  // Resets device (see at::DeviceGuard notes above)
  at::cuda::set_device(prior_device);
}

FusedKernelCUDA::~FusedKernelCUDA() {
  nvrtc().cuModuleUnload(module_);
}

static std::shared_ptr<FusedKernel> createFusionKernel(
    int16_t device,
    std::string name,
    std::string code,
    std::vector<TensorDesc> input_desc,
    std::vector<TensorDesc> output_desc,
    std::vector<PartitionDesc> chunk_desc,
    std::vector<PartitionDesc> concat_desc,
    bool has_random) {
  return std::make_shared<FusedKernelCUDA>(
      device,
      std::move(name),
      std::move(code),
      std::move(input_desc),
      std::move(output_desc),
      std::move(chunk_desc),
      std::move(concat_desc),
      has_random);
}

RegisterFusionBackend reg(at::DeviceType::CUDA, createFusionKernel);

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
