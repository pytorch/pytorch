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

    FusionExecutor* fe;
    if (kernel_cache_.find(kernel_id) == kernel_cache_.end()) {
      // lower torch::jit::Graph to torch::jit::fuser::cuda::fusion
      // TODO: pass contiguity infor as well as size req, so we can apply proper
      //       transform to computation
      auto fusion = parseJitIR(graph);

      // TODO: update the API to let `scheduleFusion` consume & return a fusion
      // magic scheduler updates fusion instance via transformation and setup
      // launch configurations;
      scheduleFusion(fusion.get(), inputs);

      CompileOptions options;
      options.device = getDevice(inputs);

      kernel_cache_[kernel_id] = std::make_unique<FusionExecutor>();
      kernel_cache_[kernel_id]->compileFusion(fusion.get(), options);
    }
    fe = kernel_cache_[kernel_id].get();
    return fe->runFusion(inputs);
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
    if (!IsNewExecutorEnabled()) {
      EraseShapeInformation(graph);
      for (size_t i = 0; i < nInputs; i++) {
        graph->inputs()[i]->setType(inputs[i].type());
      }
      ShapeTypePropagate(graph);
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
