#include "torch/csrc/jit/fusers/Config.h"
#if USE_CUDA_FUSER
#pragma once

#include "torch/csrc/jit/fusers/interface.h"
#include "torch/csrc/jit/fusers/common/fusion_handle_impl.h"

#include "torch/csrc/jit/ir.h"
#include "torch/csrc/utils/disallow_copy.h"

#include "ATen/ATen.h"

#include <memory>
#include <vector>
#include <unordered_map>
#include <string>

namespace torch { namespace jit { namespace cudafuser {

struct CUDAFusionCompiler {
  TH_DISALLOW_COPY_AND_ASSIGN(CUDAFusionCompiler);

  CUDAFusionCompiler() = default;

  ~CUDAFusionCompiler() = default;

  std::shared_ptr<FusionHandle> getFusionHandle(Node* fusion_group);
  
  std::vector<at::Tensor> debugLaunchGraph(
    Graph& graph
  , int device
  , at::ArrayRef<at::Tensor> inputs);

private:
  std::unordered_map<std::string, std::shared_ptr<FusionHandleImpl>> cache_map;
};

CUDAFusionCompiler& getFusionCompiler();

} // namespace cudafuser
} // namespace jit 
} // namespace torch

#endif // USE_CUDA_FUSER
