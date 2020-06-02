#include <torch/csrc/jit/codegen/cuda/manager.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/kernel_cache.h>
#include <torch/csrc/jit/codegen/cuda/parser.h>
#include <torch/csrc/jit/codegen/cuda/shape_inference.h>
#include <torch/csrc/jit/codegen/cuda/tensor_meta.h>
#include <torch/csrc/jit/codegen/cuda/utils.h>
#include <torch/csrc/jit/passes/canonicalize.h>
#include <torch/csrc/jit/passes/shape_analysis.h>
#include <torch/csrc/jit/runtime/interpreter.h>

#include <unordered_map>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {
std::unique_ptr<KernelArgsReq> makePWKernelSupport(
    const at::ArrayRef<IValue>& inputs) {
  auto req_ptr = std::make_unique<NaivePWKernelArgsReq>();
  for (const auto& input : inputs) {
    req_ptr->dims_.push_back(input.isTensor() ? input.toTensor().dim() : -1);
  }
  return req_ptr;
}

// TODO: contiguity could be used for better kernel launch config.
TensorContiguity infer_contiguity_from_tensor_type(
    const std::shared_ptr<c10::TensorType>& tensor_type) {
  TORCH_INTERNAL_ASSERT(tensor_type->isComplete());
  return TensorContiguity(
      *(tensor_type->sizes().concrete_sizes()),
      *(tensor_type->strides().concrete_sizes()));
}

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
    // TODO: this is needed. Otherwise caching on tensor size would not work, as
    //       different tensor size would result in unique string representation.
    EraseShapeInformation(graph);
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
      std::vector<at::Tensor> outputs) {
    std::lock_guard<std::mutex> guard(mutex_);
    TORCH_CHECK(
        kernel_cache_.count(kernel_id) != 0, "kernel id not recognized");

    // TODO: temporary hack
    auto cuda_kernel = kernel_cache_[kernel_id].getKernelPtr(inputs);
    if (cuda_kernel) {
      // TODO: update launch config for specific sizes;
      //       maybe we should store it in CudaKernel and compute it later
      runKernel(*cuda_kernel, inputs, outputs);
    } else {
      // TODO: this should somehow be done after kernel compilation.
      //       we will want compileKernel to return a heuristic
      cuda_kernel = kernel_cache_[kernel_id].allocateKernelInCache(
          makePWKernelSupport(inputs));

      // lower torch::jit::Graph to torch::jit::fuser::cuda::fusion
      Fusion fusion;
      // TODO: pass contiguity infor as well as size req, so we can apply proper
      //       transform to computation
      // we should propagate more information back:
      //   1. device;
      //   2. launch config;
      parseJitIR(graph, fusion, cuda_kernel.value());

      // find device in inputs.
      for (const auto& input : inputs) {
        if (input.isTensor()) {
          const auto& device = input.toTensor().device();
          TORCH_INTERNAL_ASSERT(
              device.is_cuda(), "Could only fuser operations on cuda device");
          cuda_kernel.value()->device_ = device.index();
          break;
        }
      }

      // NVRTC compile kernel
      compileKernel(fusion, cuda_kernel.value());

      runKernel(*cuda_kernel, inputs, outputs);
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

void runCudaFusionGroup(const Node* const fusion_node, Stack& stack) {
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
    auto nInputs = graph->inputs().size();
    at::ArrayRef<IValue> inputs = last(stack, nInputs);

    // shape inference in graph
    // update shape information per the new inputs;
    EraseShapeInformation(graph);
    for (decltype(nInputs) i = 0; i < nInputs; i++) {
      graph->inputs()[i]->setType(inputs[i].type());
    }
    // shape inference
    ShapeTypePropagate(graph);

    // we need to construct outputs;
    std::vector<at::Tensor> outputs;
    for (const auto* const output : graph->outputs()) {
      auto type = output->type()->expect<TensorType>();
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

      auto tensor = at::empty_strided(sizes, strides, options);
      outputs.push_back(tensor);
    }
    CudaFusionManager::getManager().runFusionNode(
        kernel_id, graph, inputs, outputs);
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
