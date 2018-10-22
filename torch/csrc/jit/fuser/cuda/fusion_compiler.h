#pragma once
#include "torch/csrc/jit/fuser/config.h"
#if USE_CUDA_FUSER

#include "ATen/ATen.h"
#include "torch/csrc/utils/disallow_copy.h"
#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/fuser/interface.h"
#include "torch/csrc/jit/fuser/common/fusion_handle_impl.h"
#include "torch/csrc/jit/fuser/kernel_spec.h"

#include <memory>
#include <vector>
#include <unordered_map>
#include <string>

namespace torch { namespace jit { namespace fuser { namespace cuda {

struct CUDAFusionCompiler {
  TH_DISALLOW_COPY_AND_ASSIGN(CUDAFusionCompiler);

  CUDAFusionCompiler() = default;

  ~CUDAFusionCompiler() = default;

  std::shared_ptr<FusionHandle> getFusionHandle(
    const KernelSpec& spec
  , const int device);
  
  std::vector<at::Tensor> debugLaunchGraph(
    Graph& graph
  , int device
  , at::ArrayRef<at::Tensor> inputs);

private:
  std::unordered_map<std::string, std::shared_ptr<FusionHandleImpl>> cache_map;
};

CUDAFusionCompiler& getFusionCompiler();

} // namespace cuda
} // namespace fuser
} // namespace jit 
} // namespace torch

#endif // USE_CUDA_FUSER
