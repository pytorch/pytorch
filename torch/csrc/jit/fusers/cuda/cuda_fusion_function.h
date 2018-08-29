#if USE_CUDA_FUSER
#pragma once

#include "torch/csrc/jit/fusers/common/common_fusion_function.h"
#include "torch/csrc/jit/fusers/common/annotated_graph.h"

#include "ATen/ATen.h"

#include "nvrtc.h"
#include "cuda.h"
#include "cuda_runtime.h"

#include <cstdint>
#include <vector>
#include <string>

namespace torch { namespace jit { namespace cudafuser {

struct CUDAFusionFunction : public ::torch::jit::CommonFusionFunction {
  CUDAFusionFunction(const std::string& name, AnnotatedGraph& agraph);

  virtual ~CUDAFusionFunction() override {
    cuModuleUnload(module);
  }

protected:
  virtual at::Backend backend() const override {
    return at::Backend::CUDA;
  }

  int ceilDiv(int a, int b) {
    return (a + b - 1) / b;
  }

  virtual uint64_t get_rand_offset(uint32_t numel) override {
     int numBlocks = std::min(maxBlocks, ceilDiv(numel, blockSize));
     return 4 * (ceil(numel/(4 * blockSize * numBlocks)) + 1);
  }

  virtual void launch_raw(uint32_t numel, void ** arguments) override;

  std::vector<char> ptx;
  CUmodule module;
  CUfunction function;

  // we record prop/device so if they are availiable for launch heuristics
  // querying at launch is too slow for device properties.
  int device;
  cudaDeviceProp prop;
  int blockSize = 128;
  int maxBlocks;
};

} // namespace cudafuser
} // namespace jit 
} // namespace torch

#endif // USE_CUDA_FUSER
