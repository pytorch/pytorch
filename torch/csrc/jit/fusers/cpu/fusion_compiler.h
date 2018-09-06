#include "torch/csrc/jit/fusers/Config.h"
#if USE_CPU_FUSER
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

namespace torch { namespace jit { namespace cpufuser {

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

} // namespace cudafuser
} // namespace jit 
} // namespace torch

#endif // USE_CPU_FUSER
