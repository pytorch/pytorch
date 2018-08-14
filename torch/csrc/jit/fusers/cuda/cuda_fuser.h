#if defined USE_CUDA && !(defined _WIN32) && !(defined USE_ROCM)
#pragma once

#include "ATen/ATen.h"

#include "torch/csrc/jit/ir.h"
#include "torch/csrc/utils/disallow_copy.h"
#include "torch/csrc/jit/assertions.h"

#include "torch/csrc/jit/fusers/fuser_interface.h"

#include "torch/csrc/jit/fusers/cuda/tensor_desc.h"
#include "torch/csrc/jit/fusers/cuda/annotated_graph.h"
#include "torch/csrc/jit/fusers/cuda/concat_desc.h"
#include "torch/csrc/jit/fusers/cuda/cuda_fusion_function.h"
#include "torch/csrc/jit/fusers/cuda/tensor_info.h"
#include "torch/csrc/jit/fusers/cuda/temp_file.h"

#include <string>
#include <algorithm>
#include <unordered_map>
#include <vector>
#include <iostream>

namespace torch { namespace jit { namespace cudafuser {

std::pair<std::vector<ConcatDesc>, bool> emitCompilationUnit(
  std::ostream& out
, const std::string& name
, AnnotatedGraph& agraph
, bool use_cuda);

struct CUDAFusionCompilerConfig {
  std::string cxx = "g++"; // compiler location
  bool debug = false; // emit debugging information about fusions
  bool openmp = true;
};

// caching compiler
struct CUDAFusionCompiler {
  TH_DISALLOW_COPY_AND_ASSIGN(CUDAFusionCompiler);
  CUDAFusionCompiler();

  // ignores types in graph, and uses specific contiguity annotations
  std::shared_ptr<CUDAFusionFunction> getOrCompile(
    AnnotatedGraph& agraph);

  // uses inputs/outputs as examples to infer continuity, does not run the graph
  std::shared_ptr<CUDAFusionFunction> getOrCompile(
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
  
private:
  CUDAFusionCompilerConfig config_;
  std::unordered_map<
    std::string
  , std::shared_ptr<CUDAFusionFunction>> cache;
};


} // namespace cudafuser
} // namespace jit
} // namespace torch

#endif // defined USE_CUDA && !(defined _WIN32) && !(defined USE_ROCM)