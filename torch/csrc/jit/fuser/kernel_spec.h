#pragma once
#include "torch/csrc/jit/fuser/config.h"
#if USE_CPU_FUSER || USE_CUDA_FUSER

#include "ATen/ATen.h"
#include "c10/util/Optional.h"
#include "torch/csrc/jit/stack.h"
#include "torch/csrc/jit/interpreter.h"
#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/fuser/interface.h"
#include "torch/csrc/jit/fuser/arg_spec.h"
#include "torch/csrc/jit/fuser/fused_kernel.h"

#include <memory>
#include <cstdint>
#include <vector>
#include <unordered_map>

 namespace torch { namespace jit { namespace fuser {

struct PartitionInfo {
  PartitionInfo(
    const int64_t _nSubTensors
  , const int64_t _dim)
  : nSubTensors_{_nSubTensors}
  , dim_{_dim} 
  { };

  int64_t nSubTensors() const { return nSubTensors_; }
  int64_t dim() const { return dim_; }

private:
  int64_t nSubTensors_;
  int64_t dim_;
};
   
 // "Kernel Specification." - Contains device-independent fusion information.
struct KernelSpec {
  KernelSpec(
    const int64_t _key
  , std::shared_ptr<Graph> _graph)
  : key_{_key}
  , graph_{_graph} 
  , code_{_graph}
  , nInputs_{_graph->inputs().size()}
  , inputBroadcastGroups_{}
  , inputChunks_{}
  , kernels_{}
  { }

   // Getters
  int64_t key() const { return key_; }
  std::shared_ptr<Graph> graph() const { return graph_; }
  const Code& code() const { return code_; }
  int64_t nInputs() const { return nInputs_; }
  bool isFusable() const { return isFusable_; }
  std::vector<std::vector<int64_t>>& inputBroadcastGroups() { 
    return inputBroadcastGroups_; }
  const std::vector<std::vector<int64_t>>& inputBroadcastGroups() const { 
    return inputBroadcastGroups_; }
  std::vector<PartitionInfo>& inputChunks() { return inputChunks_; }
  const std::vector<PartitionInfo>& inputChunks() const { return inputChunks_; }

   // Setters
  void setFusable(const bool _isFusable) { isFusable_ = _isFusable; }

  // Cache functions
  c10::optional<std::shared_ptr<FusedKernel>> findKernel(const ArgSpec& arg_spec) const {
    const auto it = kernels_.find(arg_spec);
    if (it == kernels_.end()) return c10::nullopt;
    return it->second;
  }
  void cacheKernel(
    const ArgSpec& arg_spec
  , std::shared_ptr<FusedKernel> kernel) const {
    kernels_.emplace(arg_spec, kernel);
  }

private: 
  int64_t key_;
  std::shared_ptr<Graph> graph_;
  Code code_;
  int64_t nInputs_;
  bool isFusable_ = true;
  std::vector<std::vector<int64_t>> inputBroadcastGroups_;
  std::vector<PartitionInfo> inputChunks_;
  mutable std::unordered_map<
    ArgSpec
  , std::shared_ptr<FusedKernel>
  , torch::hash<ArgSpec>> kernels_;
};

} // namespace fuser
} // namespace jit 
} // namespace torch

#endif // USE_CPU_FUSER || USE_CUDA_FUSER
