#include "torch/csrc/jit/fusers/cuda/fused_kernel.h"

#include "torch/csrc/jit/resource_guard.h"

#include "ATen/cuda/CUDAContext.h"
#include "THC/THC.h"
#include "THC/THCGenerator.hpp"
#include "torch/csrc/cuda/cuda_check.h"

#include "nvrtc.h"
#include "cuda.h"
#include "cuda_runtime.h"

#include <stdexcept>
#include <sstream>
#include <tuple>
#include <vector>
#include <algorithm>

namespace torch { namespace jit { namespace cudafuser {

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

CUDAFusedKernel::CUDAFusedKernel(
  const std::string& name
, AnnotatedGraph& agraph)
: FusedKernel(name, agraph) {
  at::DeviceGuard device_guard(agraph.device);

  TORCH_CUDA_CHECK(cudaGetDeviceProperties(&prop, agraph.device));
  checkCUDAVersion(prop);

  std::stringstream cu;
  std::tie(chunk_desc, concat_desc, has_random) = emitCompilationUnit(cu, name, agraph, true);
  compilation_unit = cu.str();
  nvrtcProgram program;
  TORCH_NVRTC_CHECK(nvrtcCreateProgram(&program, compilation_unit.c_str(), nullptr, 0, nullptr, nullptr));

  std::string compute = "--gpu-architecture=compute_" + std::to_string(prop.major) + std::to_string(prop.minor);
  std::vector<const char *> args = {"--std=c++11", compute.c_str(), "-default-device"};
  nvrtcResult result = nvrtcCompileProgram(program, args.size(), args.data());
  if (result == NVRTC_ERROR_COMPILATION) {
    size_t logsize;
    nvrtcGetProgramLogSize(program, &logsize);
    std::vector<char> log(logsize);
    nvrtcGetProgramLog(program, log.data());
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
  TORCH_CU_CHECK(cuModuleGetFunction(&function, module, name.c_str()));

  TORCH_CU_CHECK(cuOccupancyMaxActiveBlocksPerMultiprocessor(
    &maxBlocks, function, 128, 0));
  maxBlocks *= prop.multiProcessorCount;
}

void CUDAFusedKernel::launch_raw(uint32_t numel, void** arguments) {
  int numBlocks = std::min(maxBlocks, ceilDiv(numel, blockSize));

     //std::cout << "maxBlocks = " << maxBlocks << " needed blocks: " << ceilDiv(numel,blockSize)
     //          << " numblocks =  " << numBlocks;

     // it is possible that this is the first cuda call on this thread
     // so make sure we initialize the Driver API's context
     // cudaFree(0) accomplishes this.
     CUcontext pctx = 0;
     TORCH_CU_CHECK(cuCtxGetCurrent(&pctx));
     if (!pctx) {
        std::unique_lock<std::mutex> cudaFreeMutexLock(
            *(THCCachingAllocator_getCudaFreeMutex()));
        cudaFree(0);
     }
     CUstream stream = at::cuda::getCurrentCUDAStream();
     TORCH_CU_CHECK(cuLaunchKernel(
       function,
       numBlocks, 1, 1,
       blockSize, 1, 1,
       0, stream,
       arguments,
       nullptr));
}

} // namespace cudafuser
} // namespace jit 
} // namespace torch
