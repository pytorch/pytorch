#include <torch/csrc/jit/codegen/cuda/executor.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/kernel_cache.h>
#include <torch/csrc/jit/codegen/cuda/manager.h>
#include <torch/csrc/jit/codegen/cuda/parser.h>
#include <torch/csrc/jit/codegen/cuda/scheduler.h>
#include <torch/csrc/jit/codegen/cuda/shape_inference.h>
#include <torch/csrc/jit/passes/canonicalize.h>
#include <torch/csrc/jit/passes/shape_analysis.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/csrc/jit/runtime/interpreter.h>

#include <ATen/DimVector.h>
#include <c10/core/DeviceType.h>
#include <c10/util/irange.h>

#include <unordered_map>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

//! [ Note -- cache entry indexing ]
//!
//! CudaFusionManager holds the cache and handles interfacing to CudaFusionGroup
//! node, including selection, construction and execution of FusionExecutors.
//!
//! CudaFusionManager bridges PyTorch IR node CudaFusionGroup to GraphCache.
//! Therefore, we want to cache on stringified graph. But it is expensive to
//! stringify and hash on a computational graph, we cache the hash of a
//! stringified graph on node via cache_id.
//!
//! CudaFusionGroup node stores:
//!     i.  a PyTorch IR in `attr::Subgraph`
//!     ii. an int in `attr::cache_id`, (a cached hash value of
//!     `attr::Subgraph`)
//!
//! We have 2 unordered_map at CudaFusionGroup:
//!   std::unordered_map<std::string, int32_t> graph_cache_ids_;
//!   std::unordered_map<int64_t, std::unique_ptr<GraphCache>> graph_cache_;
//!
//! Mapping from std::string to graph_cache_id ensures that we assign the same
//! cache_id to CudaFusionGroup with identical computational grah, allowing
//! kernel reuse; Direct mapping from cache_id to GraphCache allows efficient
//! graph_cache indexing;

namespace {

// CudaFusionManager is not thread safe!
// TODO: we should make the tradeoff here to use thread_local instead of global
// singleton;
// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
class CudaFusionManager {
 public:
  static CudaFusionManager& getManager() {
    static CudaFusionManager cuda_fusion_manager_;
    return cuda_fusion_manager_;
  };

  // TODO: I'm assuming we have stride information in `graph->toString`
  //       We need to make sure stride information is in the final string, as we
  //       want to AVOID kernel reuse between different fusion_node, unless they
  //       have identical contiguity information! (So identical stride + shape
  //       is even more restricting in a good way)
  int32_t registerOrGetCacheId(std::shared_ptr<Graph>& graph) {
    std::lock_guard<std::mutex> guard(mutex_);

    // prepare graph for lowering;
    // We should not call `EraseShapeInformation(graph);`, graph representation
    // does not incorporate static sizes, but just rank of input tensors, which
    // is exactly what we wanted.
    Canonicalize(graph, false);
    auto repr = graph->toString(false);

    // create new graph_cache_ids_ entry if none existed yet;
    if (graph_cache_ids_.count(repr) == 0) {
      int32_t kernel_id = getNextUniqueID();
      graph_cache_ids_[repr] = kernel_id;
      TORCH_CHECK(
          graph_cache_.emplace(kernel_id, std::make_unique<GraphCache>(graph))
              .second);
    }
    return graph_cache_ids_[repr];
  };

  std::vector<at::Tensor> runFusionNode(
      int32_t kernel_id,
      const at::ArrayRef<IValue> inputs) {
    std::lock_guard<std::mutex> guard(mutex_);
    return graph_cache_[kernel_id]->runGraphWithInputs(inputs);
  }

 private:
  // TODO: Dimension collapsing should be abstracted out and integrated into
  // graph caching.

  // Dimension collapsing only applicable to profiling executor at this moment
  bool graphHasReduction(const std::shared_ptr<Graph>& graph) {
    for (const auto& n : graph->nodes()) {
      if (isReductionNode(n)) {
        return true;
      }
    }
    return false;
  }

  TensorTypePtr mergeInputTensorType(const std::shared_ptr<Graph>& graph) {
    // run over inputs to extract common types;
    TensorTypePtr acc_type = TensorType::get();
    for (const auto& input : graph->inputs()) {
      // only check tensor types;
      if (auto input_type = input->type()->cast<TensorType>()) {
        if (!input_type->dim().has_value()) {
          // early termination when detecting undefined tensor;
          return TensorType::get()->withUndefined();
        }
        if (acc_type->dim().has_value()) {
          // TODO: I think merge cannot handle broadcast - Go verify it later;
          // TODO: Since we are only handling permutation here, we should just
          //       merge the stride_index_;
          acc_type = acc_type->merge(*input_type);
        } else {
          acc_type = input_type;
        }
      }
    }
    return acc_type;
  }

  // return a permutation order that would undo `permuted`
  at::DimVector restorePermutation(at::DimVector permuted) {
    int rank = static_cast<int>(permuted.size());
    at::DimVector permutation(rank, -1);
    for (const auto i : c10::irange(rank)) {
      permutation[permuted[i]] = i;
    }
    return permutation;
  }

  at::DimVector getSortStrideScheme(const TensorTypePtr& type) {
    // `permute_seq` is the returned permutation to achieve sorted stride;
    at::DimVector permute_seq;

    auto stride_properties = type->stride_properties().sizes();

    TORCH_INTERNAL_ASSERT(
        stride_properties.has_value(),
        "unknown sizes or stride_properties, collapsing shouldn't happen");

    // TODO: reuse this;
    const int rank = static_cast<int>(stride_properties->size());

    // stores axes with stride_index;
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    std::set<int> ordered_axes;

    // TODO: this does not support broadcast yet;
    for (const auto i : c10::irange(rank)) {
      if ((*stride_properties)[i].has_value() &&
          (*stride_properties)[i]->stride_index_.has_value()) {
        ordered_axes.insert((*stride_properties)[i]->stride_index_.value());
      }
    }

    int unallocated_axis = 0;
    // we push from slowest to fastest
    for (int i = rank - 1; i >= 0; i--) {
      if ((*stride_properties)[i].has_value() &&
          (*stride_properties)[i]->stride_index_.has_value()) {
        permute_seq.emplace_back(
            (*stride_properties)[i]->stride_index_.value());
      } else {
        // no designated axis for this slot, so we push an axis w/o designated
        // order;
        while (ordered_axes.count(unallocated_axis) != 0) {
          ++unallocated_axis;
        }
        permute_seq.emplace_back(unallocated_axis++);
      }
    }
    return permute_seq;
  }

 private:
  std::mutex mutex_;

  void runCudaKernel(
      int32_t key,
      const std::vector<int>& contiguity_tag,
      const c10::Device){};

  int32_t getNextUniqueID() {
    return next_unique_id_++;
  };

  std::unordered_map<std::string, int32_t> graph_cache_ids_;
  std::unordered_map<int64_t, std::unique_ptr<GraphCache>> graph_cache_;

  int32_t next_unique_id_ = 0;
};

} // namespace

void compileCudaFusionGroup(Node* fusion_node) {
  FUSER_PERF_SCOPE("compileCudaFusionGroup");

  TORCH_CHECK(
      fusion_node->kind() == prim::CudaFusionGroup,
      "Only prim::CudaFusionGroup can be compiled");
  if (fusion_node->hasAttribute(attr::cache_id)) {
    TORCH_WARN("Double registration of CudaFusionGroup on CudaFusionManager");
  }
  // This is not a critical code path, it's OK to do graph copy here;
  auto graph = fusion_node->g(attr::Subgraph)->copy();

  // type propagation is needed, as the protocol only requires scalar type on
  // input tensors.
  // Note that even for Profiling Executor, scalar type could still be missing,
  // especially for output tensor from a given node (as profiling node only
  // insert meta information after itself).
  TypePropagate(graph);

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int32_t fusion_cache_id =
      CudaFusionManager::getManager().registerOrGetCacheId(graph);
  fusion_node->i_(attr::cache_id, fusion_cache_id);
}

void runCudaFusionGroup(const Node* fusion_node, Stack& stack) {
  FUSER_PERF_SCOPE("runCudaFusionGroup");

  TORCH_CHECK(
      fusion_node->kind() == prim::CudaFusionGroup,
      "prim::CudaFusionGroup expected");
  // TODO: should we support runtime compilation with updated dynamic shape;
  //       shape inference would be needed so we can allocate output;
  TORCH_CHECK(
      fusion_node->hasAttribute(attr::cache_id),
      "node prim::CudaFusionGroup has not been compiled yet");
  int32_t kernel_id = fusion_node->i(attr::cache_id);

  // Currently we just construct I/O tensors for static graph;

  const auto nInputs = fusion_node->g(attr::Subgraph)->inputs().size();

  auto execute_lambda = [&]() {
    at::ArrayRef<IValue> inputs = last(stack, nInputs);

    auto outputs =
        CudaFusionManager::getManager().runFusionNode(kernel_id, inputs);

    drop(stack, inputs.size());
    stack.insert(
        stack.end(),
        std::make_move_iterator(outputs.begin()),
        std::make_move_iterator(outputs.end()));
  };

  const char* disable_fb_env = getenv("PYTORCH_CUDA_FUSER_DISABLE_FALLBACK");
  int disable_fb_flag = disable_fb_env ? atoi(disable_fb_env) : 0;
  if (disable_fb_flag) {
    execute_lambda();
  } else {
    try {
      execute_lambda();
    } catch (...) {
      TORCH_WARN(
          "FALLBACK path is taken. This is an indication that codegen"
          "Failed for some reason. To debug try disable codegen fallback path"
          "via setting the env variable"
          "`export PYTORCH_CUDA_FUSER_DISABLE_FALLBACK=1`");
      // copying graph here since we are eliminating shape information;
      auto copied_graph = fusion_node->g(attr::Subgraph)->copy();
      EraseShapeInformation(copied_graph);
      InterpreterState{Code(copied_graph, "fallback_cuda_fuser")}.run(stack);
    }
  }
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
