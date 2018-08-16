#if !(defined _WIN32)
#pragma once

#include "torch/csrc/jit/fusers/fuser_interface.h"

#include "torch/csrc/jit/fusers/cpu/tensor_desc.h"
#include "torch/csrc/jit/fusers/cpu/annotated_graph.h"
#include "torch/csrc/jit/fusers/cpu/concat_desc.h"
#include "torch/csrc/jit/fusers/cpu/temp_file.h"
#include "torch/csrc/jit/fusers/cpu/fusion_compiler_config.h"
#include "torch/csrc/jit/fusers/cpu/cpu_fusion_function.h"

#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/assertions.h"

#include "torch/csrc/WindowsTorchApiMacro.h"
#include "torch/csrc/utils/disallow_copy.h"

#include "ATen/ATen.h"

#include <string>
#include <algorithm>
#include <unordered_map>
#include <vector>
#include <utility>
#include <iostream>

namespace torch { namespace jit { namespace cpufuser {

TORCH_API struct CPUFusionCompiler {
  TH_DISALLOW_COPY_AND_ASSIGN(CPUFusionCompiler);

  CPUFusionCompiler();

  // ignores types in graph, and uses specific contiguity annotations
  std::shared_ptr<CPUFusionFunction> getOrCompile(
    AnnotatedGraph& agraph);

  // uses inputs/outputs as examples to infer continuity, does not run the graph
  std::shared_ptr<CPUFusionFunction> getOrCompile(
    Graph& graph
  , int device
  , at::ArrayRef<at::Tensor> inputs
  , at::ArrayRef<at::Tensor> outputs);

// debugging function that lets you do everything from compilation to execution
  // in one step.
  // this should not be used in the hot path of execution because it has to serialize
  // the graph each time
  void debugLaunchGraph(
    Graph& graph
  , int device
  , at::ArrayRef<at::Tensor> inputs
  , at::ArrayRef<at::Tensor> outputs);

  bool canCompileOnCPU() const { return config_.cxx.size() > 0; }

private:
  CPUFusionCompilerConfig config_;
  std::unordered_map<
    std::string
  , std::shared_ptr<CPUFusionFunction>> cache;
};

TORCH_API CPUFusionCompiler& getCompiler();

} // namespace cpufuser
} // namespace jit
} // namespace torch

#endif // !(defined _WIN32)
