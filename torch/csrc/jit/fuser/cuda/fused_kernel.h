#pragma once
#include "torch/csrc/jit/fuser/config.h"
#if USE_CUDA_FUSER

#include "ATen/ATen.h"
#include "torch/csrc/jit/fuser/common/fused_kernel.h"
#include "torch/csrc/jit/fuser/common/annotated_graph.h"

#include "nvrtc.h"
#include "cuda.h"
#include "cuda_runtime.h"

#include <cstdint>
#include <vector>
#include <string>

namespace torch { namespace jit { namespace fuser { namespace cuda {

struct FusedKernelCUDA : public ::torch::jit::fuser::FusedKernel {
  FusedKernelCUDA(
    const int _device
  , const std::string& _name
  , const std::string& _code
  , const std::vector<TensorDesc> _input_desc
  , const std::vector<TensorDesc> _output_desc
  , const std::vector<PartitionDesc> _chunk_desc
  , const std::vector<PartitionDesc> _concat_desc
  , const bool _has_random);

  virtual ~FusedKernelCUDA() override {
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
     return 4 * (ceil(numel / (4.0 * blockSize * numBlocks)) + 1);
  }

  virtual void launch_raw(uint32_t numel, void ** arguments) override;

  std::vector<char> ptx;
  CUmodule module;
  CUfunction function;

  // we record prop/device so if they are availiable for launch heuristics
  // querying at launch is too slow for device properties.
  const int device_;
  cudaDeviceProp prop;
  int blockSize = 128;
  int maxBlocks;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch

#endif // USE_CUDA_FUSER
