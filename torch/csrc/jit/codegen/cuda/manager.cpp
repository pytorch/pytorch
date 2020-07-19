#include <torch/csrc/jit/codegen/cuda/manager.h>
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

#include <torch/csrc/jit/codegen/cuda/executor.h>
#include <unordered_map>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {
// CudaFusionManager holds compiled `CudaKernel` and handles all interfacing
// including compilation and execution.
//
// We cache two maps here:
//   a. string of graph -> kernel_id
//   b. kernel_id -> CudaKernel
//
// This allows CudaKernel reuse across nodes;
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

      // create entry for cached kernel;
      // Note: use make_pair instead of uniform initialization list here since
      //       it doesn't work under some env that we still support.
      //       eg. cuda9.2 + gcc5.4
      kernel_cache_.insert(std::make_pair(kernel_id, CudaKernelCache()));

      // TODO: we should compile here using profiled information:
      //       size (range) / stride (contiguity)
    }
    return graph_cache_[repr];
  };

  void runFusionNode(
      int32_t kernel_id,
      std::shared_ptr<Graph>& graph,
      const at::ArrayRef<IValue> inputs,
      const std::vector<at::Tensor>& outputs) {
    std::lock_guard<std::mutex> guard(mutex_);
    TORCH_CHECK(
        kernel_cache_.count(kernel_id) != 0, "kernel id not recognized");

    if (auto cuda_kernel_opt = kernel_cache_[kernel_id].getKernelPtr(inputs)) {
      // TODO: update launch config for specific sizes;
      //       maybe we should store it in CudaKernel and compute it later
      runKernel(*cuda_kernel_opt, inputs, outputs);
    } else {
      // TODO: this should somehow be done after kernel compilation.
      //       we will want compileKernel to return a heuristic
      auto cuda_kernel = kernel_cache_[kernel_id].allocateKernelInCache(inputs);

      // lower torch::jit::Graph to torch::jit::fuser::cuda::fusion
      // TODO: pass contiguity infor as well as size req, so we can apply proper
      //       transform to computation
      cuda_kernel->setFusionPtr(parseJitIR(graph));
      TORCH_INTERNAL_ASSERT(
          cuda_kernel->fusion() != nullptr,
          "parser failed to construct a fusion from PyTorch JIT graph\n");

      // TODO: update the API to let `scheduleFusion` consume & return a fusion
      // magic scheduler updates fusion instance via transformation and setup
      // launch configurations;
      scheduleFusion(cuda_kernel->fusion(), inputs);

      // find device in inputs.
      for (const auto& input : inputs) {
        if (input.isTensor()) {
          const auto& device = input.toTensor().device();
          TORCH_INTERNAL_ASSERT(
              device.is_cuda(), "Could only fuser operations on cuda device");
          cuda_kernel->setDevice(device.index());
          break;
        }
      }

      // NVRTC compile kernel
      compileKernel(cuda_kernel);

      runKernel(cuda_kernel, inputs, outputs);
    }
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
  std::unordered_map<int64_t, CudaKernelCache> kernel_cache_;

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
    /*
    // TODO: Delete the shape inference here once we switch to
    //       ExpressionEvaluator to allocate outputs
    // shape inference in graph to allocate outputs
    // update shape information per the new inputs;
    EraseShapeInformation(shape_inf_graph);
    for (size_t i = 0; i < nInputs; i++) {
      shape_inf_graph->inputs()[i]->setType(inputs[i].type());
    }
    // shape inference
    ShapeTypePropagate(shape_inf_graph);

    // we need to construct outputs;
    std::vector<at::Tensor> outputs;
    for (const auto* output : shape_inf_graph->outputs()) {
      const auto type = output->type()->expect<TensorType>();
      // Expect output to be tensor;
      TORCH_CHECK(
          type && type->isComplete(),
          "Complete TensorType for output is expected.");

      const auto device = *(type->device());
      const auto scalar_type = *(type->scalarType());

      auto options = at::TensorOptions()
                         .dtype(scalar_type)
                         .layout(at::kStrided)
                         .device(device)
                         .requires_grad(type->requires_grad());

      // TODO: We should infer output shape from `inputs`
      const auto sizes = extractSizes(type);
      const auto strides = extractStrides(type);

      const auto tensor = at::empty_strided(sizes, strides, options);
      outputs.push_back(tensor);
    }
    CudaFusionManager::getManager().runFusionNode(
        kernel_id, graph, inputs, outputs);
    */
    FusionExecutor executor;
    auto fusion = parseJitIR(graph);
    scheduleFusion(fusion.get(), inputs);
    executor.compileFusion(fusion.get());
    auto outputs = executor.runFusion(inputs);

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
