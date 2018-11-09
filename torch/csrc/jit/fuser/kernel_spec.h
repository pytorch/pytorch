#pragma once
#include "torch/csrc/jit/fuser/config.h"
#if USE_CUDA_FUSER || USE_CPU_FUSER

#include "ATen/ATen.h"
#include "torch/csrc/WindowsTorchApiMacro.h"
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
#include <mutex>

namespace torch { namespace jit { namespace fuser {

// Helper struct containing partition information: the number of tensors
// created and the dimension the partitioning is performed on.
// Note: created during upfront compilation, once the tensors are known
// at runtime the partition info is logically combined with the tensor
// descriptions to create PartitionDesc objects.
struct TORCH_API PartitionInfo {
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
 // Each kernel specification contains a map of instantiated generated functions
 // that implement some or most of its functionality. Multiple generated
 // functions are needed by each abstract specification because of different
 // devices (cpu vs gpu, different gpus) and different inputs (int vs float,
 // contiguous vs discontiguous).
 // Note: uses a mutex to control access to its kernel store
 // Note: unordered containers do not invalidate references/pointers on
 //   rehashing, which is critical for thread-safety.
 // TODO: allow abstract kernels to use multiple generated kernels
 // TODO: allow abstract kernels to reuse generated kernels from common pool
struct TORCH_API KernelSpec {
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

  std::vector<std::vector<int64_t>>& inputBroadcastGroups() {
    return inputBroadcastGroups_;
  }
  const std::vector<std::vector<int64_t>>& inputBroadcastGroups() const {
    return inputBroadcastGroups_;
  }

  std::vector<PartitionInfo>& inputChunks() { return inputChunks_; }
  const std::vector<PartitionInfo>& inputChunks() const { return inputChunks_; }

  // Cache functions
  c10::optional<std::shared_ptr<FusedKernel>> findKernel(const ArgSpec& arg_spec) const {
    std::lock_guard<std::mutex> guard{mutex_};
    const auto it = kernels_.find(arg_spec);
    if (it == kernels_.end()) return c10::nullopt;
    return it->second;
  }
  void cacheKernel(
    const ArgSpec& arg_spec
  , std::shared_ptr<FusedKernel> kernel) const {
    std::lock_guard<std::mutex> guard{mutex_};
    kernels_.emplace(arg_spec, kernel);
  }

private:
  int64_t key_;
  std::shared_ptr<Graph> graph_;
  Code code_;
  uint64_t nInputs_;
  std::vector<std::vector<int64_t>> inputBroadcastGroups_;
  std::vector<PartitionInfo> inputChunks_;
  mutable std::mutex mutex_;
  mutable std::unordered_map<
    ArgSpec
  , std::shared_ptr<FusedKernel>
  , torch::hash<ArgSpec>> kernels_;
};

} // namespace fuser
} // namespace jit
} // namespace torch

#endif // USE_CPU_FUSER || USE_CUDA_FUSER
