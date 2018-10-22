#pragma once
#include "torch/csrc/jit/fuser/config.h"
#if USE_CPU_FUSER

#include "ATen/ATen.h"
#include "torch/csrc/utils/disallow_copy.h"
#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/fuser/interface.h"
#include "torch/csrc/jit/fuser/common/fusion_handle_impl.h"

#include <memory>
#include <vector>
#include <unordered_map>
#include <string>

namespace torch { namespace jit { namespace fuser { namespace cpu {

struct CPUFusionCompilerConfig {
  std::string cxx = "g++"; // compiler location
  bool debug = false; // emit debugging information about fusions
  bool openmp = true;
};

struct CPUFusionCompiler {
  TH_DISALLOW_COPY_AND_ASSIGN(CPUFusionCompiler);

  CPUFusionCompiler();

  ~CPUFusionCompiler() = default;

  std::shared_ptr<FusionHandle> getFusionHandle(Node* fusion_group);
  
  std::vector<at::Tensor> debugLaunchGraph(
    Graph& graph
  , int device
  , at::ArrayRef<at::Tensor> inputs);

  CPUFusionCompilerConfig& getConfig() {
    return config_;
  }

private:
  CPUFusionCompilerConfig config_;
  std::unordered_map<std::string, std::shared_ptr<FusionHandleImpl>> cache_map;
};

CPUFusionCompiler& getFusionCompiler();

} // namespace cpu
} // namespace fuser
} // namespace jit 
} // namespace torch

#endif // USE_CPU_FUSER
