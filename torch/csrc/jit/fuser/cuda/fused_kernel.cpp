#include "torch/csrc/jit/fuser/cuda/fused_kernel.h"

#include "ATen/cuda/CUDAContext.h"
#include "THC/THC.h"
#include "torch/csrc/cuda/cuda_check.h"
#include "torch/csrc/jit/resource_guard.h"

// Note: unclear why this forward declaration is necessary
#include "THC/THCTensorRandom.h"
#include "THC/THCGenerator.hpp"
THCGenerator* THCRandom_getGenerator(THCState* state);

#include "nvrtc.h"
#include "cuda.h"
#include "cuda_runtime.h"

#include <stdexcept>
#include <sstream>
#include <tuple>
#include <vector>
#include <algorithm>
#include <cmath>

namespace torch { namespace jit { namespace fuser { namespace cuda {

void checkCUDAVersion(const cudaDeviceProp& prop) {
  if ((prop.major >= 6 && CUDA_VERSION < 8000) ||
      (prop.major >= 7 && CUDA_VERSION < 9000)) {
    std::stringstream err_string;
    err_string << "In CUDAFusedKernel, PyTorch compiled with insufficient CUDA version: "
         << CUDA_VERSION << " for the current GPU device " << prop.name
         << " with device capability " << prop.major << "." << prop.minor;
    throw std::runtime_error(err_string.str());
  }
}

FusedKernelCUDA::FusedKernelCUDA(
  const int _device
, const std::string& _name
, const std::string& _code
, const std::vector<TensorDesc> _input_desc
, const std::vector<TensorDesc> _output_desc
, const std::vector<PartitionDesc> _chunk_desc
, const std::vector<PartitionDesc> _concat_desc
, const bool _has_random)
: FusedKernel{_name, _code, _input_desc, _output_desc, _chunk_desc, _concat_desc, _has_random}
, device_{_device} {
  // Acquires and validates device properties
  at::DeviceGuard device_guard(device_);
  TORCH_CUDA_CHECK(cudaGetDeviceProperties(&prop, device_));
  checkCUDAVersion(prop);

  nvrtcProgram program;
  TORCH_NVRTC_CHECK(nvrtcCreateProgram(
    &program
  , code_.c_str()
  , nullptr
  , 0
  , nullptr
  , nullptr));

  const std::string compute = "--gpu-architecture=compute_" + std::to_string(prop.major) + std::to_string(prop.minor);
  const std::vector<const char *> args = {"--std=c++11", compute.c_str(), "-default-device"};
  const auto result = nvrtcCompileProgram(program, args.size(), args.data());
  if (result == NVRTC_ERROR_COMPILATION) {
    size_t logsize;
    nvrtcGetProgramLogSize(program, &logsize);
    std::vector<char> log(logsize);
    nvrtcGetProgramLog(program, log.data());
    std::stringstream cu;
    cu << log.data();
    throw std::runtime_error(cu.str());
  }
  ResourceGuard holdProgram([&] {
    TORCH_NVRTC_CHECK(nvrtcDestroyProgram(&program));
  });
  TORCH_NVRTC_CHECK(result);

  size_t ptx_size;
  TORCH_NVRTC_CHECK(nvrtcGetPTXSize(program, &ptx_size));
  ptx.resize(ptx_size);
  TORCH_NVRTC_CHECK(nvrtcGetPTX(program, ptx.data()));
  CUcontext pctx = 0;
  TORCH_CU_CHECK(cuCtxGetCurrent(&pctx));
  if (!pctx) {
     std::unique_lock<std::mutex> cudaFreeMutexLock(
     *(THCCachingAllocator_getCudaFreeMutex()));
     cudaFree(0);
  }
  TORCH_CU_CHECK(cuModuleLoadData(&module, ptx.data()));
  TORCH_CU_CHECK(cuModuleGetFunction(&function, module, name_.c_str()));

  TORCH_CU_CHECK(cuOccupancyMaxActiveBlocksPerMultiprocessor(
    &maxBlocks, function, 128, 0));
  maxBlocks *= prop.multiProcessorCount;
}

static int ceilDiv(const int a, const int b) {
  return (a + b - 1) / b;
}

static int counter = 1;

void FusedKernelCUDA::launch_raw(
  const uint32_t numel
, std::vector<void*>& arguments) const {
  const auto nBlocks = std::min(maxBlocks, ceilDiv(numel, blockSize));

  // Adds random state to arguments if necessary
  // Note: offset defined here so its lifetime extends to the launch
  uint64_t offset;
  if (has_random_) {
    const auto rand_offset = 4 * (std::ceil(numel / (4.0 * blockSize * nBlocks)) + 1);
    auto gen = THCRandom_getGenerator(at::globalContext().getTHCState());
    offset = gen->state.philox_seed_offset.fetch_add(rand_offset);
    arguments.push_back(&gen->state.initial_seed);
    arguments.push_back(&offset);
  }

  // Initializes driver's API context (if necessary)
  CUcontext pctx = 0;
  TORCH_CU_CHECK(cuCtxGetCurrent(&pctx));
  if (!pctx) {
    std::unique_lock<std::mutex> 
      cudaFreeMutexLock{*(THCCachingAllocator_getCudaFreeMutex())};
    cudaFree(0);
  }

  // Launches kernel on current stream
  auto stream = at::cuda::getCurrentCUDAStream();
  TORCH_CU_CHECK(cuLaunchKernel(
    function,
    nBlocks, 1, 1,
    blockSize, 1, 1,
    0, stream,
    arguments.data(),
    nullptr));
}

} // namespace cuda
} // namespace fuser
} // namespace jit 
} // namespace torch
