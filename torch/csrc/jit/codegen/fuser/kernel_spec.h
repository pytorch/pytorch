#pragma once

#include <ATen/ATen.h>
#include <ATen/core/stack.h>
#include <torch/csrc/Export.h>
#include <torch/csrc/jit/codegen/fuser/arg_spec.h>
#include <torch/csrc/jit/codegen/fuser/fused_kernel.h>
#include <torch/csrc/jit/codegen/fuser/interface.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/runtime/interpreter.h>
#include <optional>

#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace torch::jit::fuser {

// Helper struct containing partition information: the number of tensors
// created and the dimension the partitioning is performed on.
// Note: created during upfront compilation, once the tensors are known
// at runtime the partition info is logically combined with the tensor
// descriptions to create PartitionDesc objects.
struct TORCH_API PartitionInfo {
  PartitionInfo(const int64_t _nSubTensors, const int64_t _dim)
      : nSubTensors_{_nSubTensors}, dim_{_dim} {}

  int64_t nSubTensors() const {
    return nSubTensors_;
  }
  int64_t dim() const {
    return dim_;
  }

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
  // Note: assumes the spec is a single block
  // Note: This is the appropriate place to generalize if you want to add other
  //  passes to upfront compilation that walk the graph.
  KernelSpec(const int64_t _key, const std::shared_ptr<Graph>& _graph)
      : key_{_key},
        graph_{_graph},
        code_{_graph, "<fused code>"},
        nInputs_{_graph->inputs().size()},

        inputBroadcastGroups_{},
        inputChunks_{},

        kernels_{} {
    // No need to iterate over reference since n is pointer
    for (const auto n : graph_->nodes()) {
      static_assert(std::is_pointer_v<decltype(n)>, "n must be a pointer");
      if (n->kind() == aten::rand_like) {
        has_random_ = true;
        break;
      }
    }
    nTensorInputs_ = std::count_if(
        graph_->inputs().begin(), graph_->inputs().end(), [](const Value* v) {
          return v->type()->isSubtypeOf(*TensorType::get());
        });
  }

  // Getters
  int64_t key() const {
    return key_;
  }
  std::shared_ptr<Graph> graph() const {
    return graph_;
  }
  const Code& code() const {
    return code_;
  }
  int64_t nInputs() const {
    return nInputs_;
  }
  int64_t nTensorInputs() const {
    return nTensorInputs_;
  }

  std::vector<std::vector<int64_t>>& inputBroadcastGroups() {
    return inputBroadcastGroups_;
  }
  const std::vector<std::vector<int64_t>>& inputBroadcastGroups() const {
    return inputBroadcastGroups_;
  }

  std::vector<PartitionInfo>& inputChunks() {
    return inputChunks_;
  }
  const std::vector<PartitionInfo>& inputChunks() const {
    return inputChunks_;
  }

  bool hasRandom() const {
    return has_random_;
  }

  // Cache functions
  std::optional<std::shared_ptr<FusedKernel>> findKernel(
      const ArgSpec& arg_spec) const {
    std::lock_guard<std::mutex> guard{mutex_};
    const auto it = kernels_.find(arg_spec);
    if (it == kernels_.end())
      return std::nullopt;
    return it->second;
  }
  void cacheKernel(
      const ArgSpec& arg_spec,
      const std::shared_ptr<FusedKernel>& kernel) const {
    std::lock_guard<std::mutex> guard{mutex_};
    kernels_.emplace(arg_spec, kernel);
  }

 private:
  int64_t key_;
  std::shared_ptr<Graph> graph_;
  Code code_;
  uint64_t nInputs_;
  uint64_t nTensorInputs_{};
  std::vector<std::vector<int64_t>> inputBroadcastGroups_;
  std::vector<PartitionInfo> inputChunks_;
  bool has_random_{false};
  mutable std::mutex mutex_;
  mutable std::
      unordered_map<ArgSpec, std::shared_ptr<FusedKernel>, c10::hash<ArgSpec>>
          kernels_;
};

} // namespace torch::jit::fuser
