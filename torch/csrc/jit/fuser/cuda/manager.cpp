#include <torch/csrc/jit/fuser/cuda/manager.h>
#include <torch/csrc/jit/fuser/cuda/parser.h>
#include <torch/csrc/jit/fuser/common/tensor.h>
#include <torch/csrc/jit/fuser/common/fusion.h>
#include <torch/csrc/jit/passes/canonicalize.h>

#include <ATen/cuda/CUDAContext.h>

#include <unordered_map>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {

struct CudaKernelEntry {
  int16_t device_;
  CUmodule module_;
  CUfunction function_;

  // TODO: we don't need to keep the whole Fusion around after compilation.
  Fusion fusion_;

  void debugPrint() {
    FusionGuard fg(&fusion_);
    std::cout << "fusion group: \n"  << &fusion_ << std::endl;
  }
};

// The reason for two unordered_map here is to cache `torch::jit::Graph` lowering
class CudaFusionManager {
public:
  static CudaFusionManager& getManager() {
    static CudaFusionManager cuda_fusion_manager_;
    return cuda_fusion_manager_;
  };

  // TODO: discuss whether or not we want reuse codegen.
  // TODO: I'm assuming we have stride information in `graph->toString`
  //       We need to make sure stride information is in the final string, as we
  //       want to AVOID kernel reuse between different fusion_node, unless they
  //       have identical contiguity information! (So identical stride + shape
  //       is even more restricting in a good way)
  int32_t registerOrGetCacheId(std::shared_ptr<Graph>& graph) {
    std::lock_guard<std::mutex> guard(mutex_);
    // prepare graph for lowering;
    Canonicalize(graph, false);
    //EraseShapeInformation(graph);
    auto repr = graph->toString(false);

    // create new graph_cache_ entry;
    if (graph_cache_.count(repr) == 0) {
      int32_t kernel_id = getNextUniqueID();

      graph_cache_[repr] = kernel_id;

      // default constructor via accessing empty key;
      // lower torch::jit::Graph to torch::jit::fuser::cuda::fusion
      parseJitIR(graph, kernel_cache_[kernel_id].fusion_);

      // TODO: compile and blablabla;
      //compileJitIR(fusion, kernel_cache_[kernel_id]);

      return kernel_id;
    } else {
      return graph_cache_[repr];
    }
  };

  // TODO: IO construction should go outside;
  void runFusionNode(
      int32_t kernel_id,
      Stack& stack) {
    CudaKernelEntry& cuda_kernel_entry = kernel_cache_[kernel_id];
    const Fusion& fusion = cuda_kernel_entry.fusion_;
    at::ArrayRef<IValue> inputs = last(stack, fusion.inputs().size());
    std::vector<at::Tensor> outputs;

    cuda_kernel_entry.debugPrint();

    for (const auto* const output : fusion.outputs()) {
      assert(output->getValType() == ValType::Tensor);
      /*
      auto type = output->type()->expect<TensorType>();

      auto options = at::TensorOptions()
          .dtype(*(type->scalar_type()))
          .layout(at::kStrided)
          .device(*(type->device()))
          .requires_grad(type->requires_grad());
      // TODO: shape inference needed, we should generate output shape based on
      //       input shape.
      auto sizes = extractSizes(type);
      auto strides = extractStrides(type);
      auto tensor = at::empty_strided(sizes, strides, options);
       */
      auto tensor = at::empty({5});

      outputs.push_back(tensor);
      printf("push one tensor back\n");
    }

    // TODO: execute kernel with inputs/outputs;
    //cuda_kernel_entry.launch(launch_config, inputs, outputs);

    // Modify stack AFTER execution.
    drop(stack, inputs.size());
    stack.insert(stack.end(),
        std::make_move_iterator(outputs.begin()),
        std::make_move_iterator(outputs.end()));
  }

protected:

  std::mutex mutex_;

  void runCudaKernel(
    int32_t key,
    const std::vector<int>& contiguity_tag,
    const c10::Device) {
  };

  int32_t getNextUniqueID() {
    return next_unique_id_++;
  };

  std::unordered_map<std::string, int32_t> graph_cache_;
  std::unordered_map<int64_t, CudaKernelEntry> kernel_cache_;

private:

  int32_t next_unique_id_ = 0;
};


} // namespace

void compileCudaFusionGroup(Node* fusion_node) {
  assert(fusion_node->kind() == prim::FusionGroup);
  if (fusion_node->hasAttribute(attr::cache_id)) {
    // TODO: maybe we should error out here;
    AT_WARN("Double registration of CudaFusionGroup on CudaFusionManager");
  }
  int32_t fusion_cache_id = 
      CudaFusionManager::getManager().registerOrGetCacheId(fusion_node->g(attr::Subgraph));
  fusion_node->i_(attr::cache_id, fusion_cache_id);
}

void runCudaFusionGroup(const Node* const fusion_node, Stack& stack) {
  assert(fusion_node->kind() == prim::FusionGroup);
  /*
  if (!fusion_node->hasAttribute(attr::cache_id)) {
    AT_WARN("runCudaFusionGroup called before registration/compilation");
    int32_t fusion_cache_id = 
        CudaFusionManager::getManager().registerOrGetCacheId(fusion_node->g(attr::Subgraph));
    fusion_node->i_(attr::cache_id, fusion_cache_id);
  }
  */
  assert(fusion_node->hasAttribute(attr::cache_id));
  int32_t kernel_id = fusion_node->i(attr::cache_id);

  printf("run it through fusion manager: %d\n", kernel_id);
  // TODO: we need to construct outputs;
  CudaFusionManager::getManager().runFusionNode(kernel_id, stack);
}

}}}} // namespace torch::jit::fuser::cuda
