#include <torch/csrc/jit/codegen/cuda/executor.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/kernel_cache.h>
#include <torch/csrc/jit/codegen/cuda/parser.h>
#include <torch/csrc/jit/codegen/cuda/scheduler.h>
#include <torch/csrc/jit/codegen/cuda/shape_inference.h>
#include <torch/csrc/jit/codegen/cuda/utils.h>
#include <torch/csrc/jit/passes/canonicalize.h>
#include <torch/csrc/jit/passes/shape_analysis.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/csrc/jit/runtime/interpreter.h>

#include <unordered_map>

#include <ATen/DimVector.h>
#include <c10/core/DeviceType.h>

#include <torch/csrc/jit/codegen/cuda/manager.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {
c10::Device getDevice(const at::ArrayRef<IValue>& inputs) {
  // find device in inputs.
  for (const auto& input : inputs) {
    if (input.isTensor()) {
      auto dev = input.toTensor().device();
      TORCH_INTERNAL_ASSERT(
          dev.is_cuda(), "Could only fuser operations on cuda device");
      return dev;
    }
  }
  TORCH_INTERNAL_ASSERT(
      false, "Could not detect device of inputs to a fusion.");
}

// CudaFusionManager holds a FusionExecutor and handles all interfacing
// including compilation and execution.
//
// We cache two maps here:
//   a. string of graph -> kernel_id
//   b. kernel_id -> FusionExecutor
//
// This allows FusionExecutor reuse across nodes;
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

    // create new graph_cache_ entry;
    if (graph_cache_.count(repr) == 0) {
      int32_t kernel_id = getNextUniqueID();
      graph_cache_[repr] = kernel_id;
    }
    return graph_cache_[repr];
  };

  std::vector<at::Tensor> runFusionNode(
      int32_t kernel_id,
      std::shared_ptr<Graph>& graph,
      const at::ArrayRef<IValue> inputs) {
    std::lock_guard<std::mutex> guard(mutex_);

    auto inputs_vec = dimSortInputs(graph, inputs);
    const at::ArrayRef<IValue> inputs_ref = inputs_vec;

    FusionExecutor* fe;
    if (kernel_cache_.find(kernel_id) == kernel_cache_.end()) {
      // search kernel cache failed, we need to codegen new kernel for given
      // inputs;

      // we still need to permute input tensor type in the graph properly.
      auto copy = dimSortGraph(graph);
      auto fusion = parseJitIR(copy);

      // TODO: update the API to let `scheduleFusion` consume & return a fusion
      // magic scheduler updates fusion instance via transformation and setup
      // launch configurations;
      scheduleFusion(fusion.get(), inputs_ref);

      CompileOptions options;
      options.device = getDevice(inputs_ref);

      kernel_cache_[kernel_id] = std::make_unique<FusionExecutor>();
      kernel_cache_[kernel_id]->compileFusion(fusion.get(), options);
    }

    fe = kernel_cache_[kernel_id].get();
    return dimSortOutputs(graph, fe->runFusion(inputs_ref));
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
          acc_type = acc_type->merge(input_type);
        } else {
          acc_type = input_type;
        }
      }
    }
    return acc_type;
  }

  void debugPrint(const TensorTypePtr& type) {
    if (auto sizes = type->symbolic_sizes().sizes()) {
      // for (const auto& shape_symbol : sizes.value()) {
      int rank = static_cast<int>(sizes->size());
      for (int i = 0; i < rank; i++) {
        const auto& shape_symbol = sizes.value()[i];
        if (shape_symbol.is_static()) {
          printf("%ld, ", shape_symbol.static_size());
        } else {
          printf("s(%ld), ", *reinterpret_cast<const int64_t*>(&shape_symbol));
        }
      }
    } else {
      printf("no size available\n");
    }
    if (const auto& stride_properties = type->stride_properties().sizes()) {
      int rank = static_cast<int>(stride_properties->size());
      printf("\nstride: ");
      for (int i = 0; i < rank; i++) {
        if (auto val = (*stride_properties)[i]->stride_) {
          printf("%ld, ", val.value());
        } else {
          printf("?, ");
        }
      }
      printf("\nstride index: ");
      for (int i = 0; i < rank; i++) {
        if (auto val = (*stride_properties)[i]->stride_index_) {
          printf("%ld, ", val.value());
        } else {
          printf("?, ");
        }
      }
      printf("\ncontiguous: ");
      for (int i = 0; i < rank; i++) {
        if (auto val = (*stride_properties)[i]->contiguous_) {
          printf("%d, ", val.value());
        } else {
          printf("?, ");
        }
      }
    } else {
      printf("no stride properties available\n");
    }
  }

  // return a permutation order that would undo `permuted`
  at::DimVector restorePermutation(at::DimVector permuted) {
    int rank = static_cast<int>(permuted.size());
    at::DimVector permutation(rank, -1);
    for (int i; i < rank; i++) {
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
    std::set<int> ordered_axes;

    // TODO: this does not support broadcast yet;
    for (int i = 0; i < rank; i++) {
      if (auto index = (*stride_properties)[i]->stride_index_) {
        ordered_axes.insert(*index);
      }
    }

    int unallocated_axis = 0;
    // we push from slowest to fastest
    for (int i = rank - 1; i >= 0; i--) {
      if (auto index = (*stride_properties)[i]->stride_index_) {
        // pushing axis index to current entry in permute_seq;
        permute_seq.emplace_back(*index);
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

  std::vector<IValue> dimSortInputs(
      std::shared_ptr<Graph>& graph,
      const at::ArrayRef<IValue> inputs) {
    if (!IsNewExecutorEnabled() || graphHasReduction(graph)) {
      return inputs.vec();
    }
    auto acc_type = mergeInputTensorType(graph);

    if (!acc_type->dim().has_value()) {
      return inputs.vec();
    }

    auto strategy = getSortStrideScheme(acc_type);
    // TODO: early return if permutation is no-op;

    std::vector<IValue> permuted_inputs;
    for (const auto& input : inputs) {
      if (input.isTensor()) {
        permuted_inputs.emplace_back(input.toTensor().permute(strategy));
      } else {
        permuted_inputs.emplace_back(input);
      }
    }
    return permuted_inputs;
  }

  std::vector<at::Tensor> dimSortOutputs(
      const std::shared_ptr<Graph>& graph,
      const std::vector<at::Tensor>& outputs) {
    if (!IsNewExecutorEnabled() || graphHasReduction(graph)) {
      return outputs;
    }
    auto acc_type = mergeInputTensorType(graph);
    if (!acc_type->dim().has_value()) {
      return outputs;
    }

    auto strategy = getSortStrideScheme(acc_type);
    // TODO: early return if permutation is no-op;
    auto restore_strategy = restorePermutation(strategy);

    std::vector<at::Tensor> permuted_outputs;
    TORCH_INTERNAL_ASSERT(outputs.size() == graph->outputs().size());
    permuted_outputs.reserve(outputs.size());
    for (const auto& output : outputs) {
      permuted_outputs.emplace_back(output.permute(restore_strategy));
    }
    return permuted_outputs;
  }

  // two thing need to be adjusted:
  // 1. permutation of size_ -> so we declare broadcast for size-1 dimension
  //    properly;
  // 2. contiguity_ -> which is needed when we register input tensor to codegen
  //    to indicate dimension collapsing.
  std::shared_ptr<Graph> dimSortGraph(std::shared_ptr<Graph>& graph) {
    if (!IsNewExecutorEnabled() || graphHasReduction(graph)) {
      return graph->copy();
    }
    auto acc_type = mergeInputTensorType(graph);

    if (!acc_type->dim().has_value()) {
      return graph->copy();
    }

    auto strategy = getSortStrideScheme(acc_type);
    // TODO: early return if permutation is no-op;

    std::shared_ptr<Graph> copy = graph->copy();

    auto type_permute_fn = [&](const TensorTypePtr& type) {
      // std::vector<c10::ShapeSymbol> vec_shape_symbol =
      // type->symbolic_sizes().sizes().value();
      auto vec_shape_symbol = type->symbolic_sizes().sizes().value();
      // std::vector<c10::optional<c10::Stride>> vec_optional_stride =
      // type->stride_properties().sizes().value();
      auto vec_optional_stride = type->stride_properties().sizes().value();

      int rank = static_cast<int>(type->dim().value());

      std::vector<c10::ShapeSymbol> permuted_vec_ss;
      std::vector<c10::optional<c10::Stride>> permuted_vec_optional_stride;
      for (int i = 0; i < rank; i++) {
        // permuted_vec_ss.emplace_back(type->symbolic_sizes().sizes().value()[strategy[i]]);
        // permuted_vec_optional_stride.emplace_back(type->stride_properties().sizes().value()[strategy[i]]);
        permuted_vec_ss.emplace_back(vec_shape_symbol[strategy[i]]);
        permuted_vec_optional_stride.emplace_back(
            vec_optional_stride[strategy[i]]);
      }

      return TensorType::create(
          type->scalarType(),
          type->device(),
          permuted_vec_ss,
          permuted_vec_optional_stride,
          type->requires_grad());
    };

    for (auto input : copy->inputs()) {
      if (auto input_type = input->type()->cast<TensorType>()) {
        input->setType(type_permute_fn(input_type));
      }
    }
    return copy;
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

  std::unordered_map<std::string, int32_t> graph_cache_;
  std::unordered_map<int64_t, std::unique_ptr<FusionExecutor>> kernel_cache_;

  int32_t next_unique_id_ = 0;
};

} // namespace

void compileCudaFusionGroup(Node* fusion_node) {
  TORCH_CHECK(
      fusion_node->kind() == prim::CudaFusionGroup,
      "Only prim::CudaFusionGroup can be compiled");
  if (fusion_node->hasAttribute(attr::cache_id)) {
    TORCH_WARN("Double registration of CudaFusionGroup on CudaFusionManager");
  }
  int32_t fusion_cache_id =
      CudaFusionManager::getManager().registerOrGetCacheId(
          fusion_node->g(attr::Subgraph));
  fusion_node->i_(attr::cache_id, fusion_cache_id);
}

void runCudaFusionGroup(const Node* fusion_node, Stack& stack) {
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
  std::shared_ptr<Graph> graph = fusion_node->g(attr::Subgraph)->copy();

  auto execute_lambda = [&]() {
    const auto nInputs = graph->inputs().size();
    at::ArrayRef<IValue> inputs = last(stack, nInputs);

    // Only needed if we are doing codegen
    // if no shape information available, we feed current shape into the kernel;
    // This is needed because our current broadcast on size-1 dimension
    if (!IsNewExecutorEnabled()) {
      EraseShapeInformation(graph);
      for (size_t i = 0; i < nInputs; i++) {
        graph->inputs()[i]->setType(inputs[i].type());
      }
      // Type propagation that's here just to cover corner case, incase type
      // propagation failed in the original subgraph. We currently need output
      // types in order to support fp16, where we cast input to fp32 and output
      // back to fp16.
      TypePropagate(graph);
    }

    auto outputs =
        CudaFusionManager::getManager().runFusionNode(kernel_id, graph, inputs);

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
      EraseShapeInformation(graph);
      InterpreterState{Code(graph, "fallback_cuda_fuser")}.run(stack);
    }
  }
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
