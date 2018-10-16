#include "torch/csrc/jit/fusers/Config.h"
#if USE_CPU_FUSER || USE_CUDA_FUSER
#pragma once

#include "torch/csrc/jit/fusers/interface.h"
#include "torch/csrc/jit/fusers/common/fusion_arg_spec.h"
#include "torch/csrc/jit/fusers/common/fused_kernel.h"

#include "torch/csrc/jit/stack.h"
#include "torch/csrc/jit/interpreter.h"
#include "torch/csrc/jit/ir.h"

#include "ATen/ATen.h"

#include <memory>
#include <cstdint>
#include <vector>
#include <unordered_map>

namespace torch { namespace jit {

// FusionCompiler has very limited shape information available at the time getOrCompile
// is called, and this is why it can't really prepare the kernels at that time. Instead,
// it returns this object, which will take care of matching the run-time shapes to whatever
// kernels we have compiled already.
//
// Two configurations are considered eligible for the same fused kernel if:
//   - the shapes satisfy graph invariants for our fused code (e.g. that all intermediate shapes
//     are the same - see fusion_compiler.cpp for more details).
//   - their FusionArgSpecs compare equal
struct FusionHandleImpl : public FusionHandle {
  FusionHandleImpl(
    std::shared_ptr<Graph> _graph
  , int device);

  void run(Stack& inputs);

private:
  struct PartitionInfo {
    PartitionInfo(int64_t nsub, int64_t dim)
    : nSubtensors(nsub), dim(dim) { };

    int64_t nSubtensors;
    int64_t dim;
  };

  void runFallback(Stack& stack);
  void expandArgs(std::vector<at::Tensor>& args, std::vector<int64_t>& map_size);
  c10::optional<std::vector<int64_t>> canRunKernel(at::TensorList args);
  c10::optional<std::vector<int64_t>> getMapSize(
      at::TensorList args,
      at::IntList arg_subset);
  std::vector<std::vector<int64_t>> getInputBroadcastGroups();
  std::vector<PartitionInfo> getInputChunkDescriptors();
  std::unique_ptr<FusedKernel> compileSpec(
        const FusionArgSpec& spec, const std::vector<int64_t>& map_size);

  static std::atomic<size_t> next_kernel_id;

  int device;
  Code fallback_code;
  std::shared_ptr<Graph> graph;
  std::vector<std::vector<int64_t>> input_broadcast_groups;
  std::vector<PartitionInfo> input_chunks;
  std::unordered_map<
    FusionArgSpec
  , std::unique_ptr<FusedKernel>
  , torch::hash<FusionArgSpec>> kernels;
};

} // namespace jit 
} // namespace torch

#endif // USE_CPU_FUSER || USE_CUDA_FUSER
