#include <oneapi/dnnl/dnnl_graph.hpp>
#include <torch/csrc/jit/codegen/onednn/decompose_silu.h>
#include <torch/csrc/jit/codegen/onednn/defer_size_check.h>
#include <torch/csrc/jit/codegen/onednn/graph_fuser.h>
#include <torch/csrc/jit/codegen/onednn/guard_shape.h>
#include <torch/csrc/jit/codegen/onednn/interface.h>
#include <torch/csrc/jit/codegen/onednn/kernel.h>
#include <torch/csrc/jit/codegen/onednn/layout_propagation.h>
#include <torch/csrc/jit/codegen/onednn/prepare_binary.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/decompose_ops.h>
#include <torch/csrc/jit/passes/pass_manager.h>
#include <torch/csrc/jit/passes/remove_mutation.h>
#include <torch/csrc/jit/passes/tensorexpr_fuser.h>
#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/csrc/jit/runtime/operator_options.h>

namespace torch::jit {
namespace fuser::onednn {

void fuseGraph(std::shared_ptr<Graph>& g) {
  // Follow the process of the tensorexpr_fuser in profiling mode:
  // Remove prim::profile nodes and embed the profile info directly in the
  // IR in value types to avoid breaking the fusion patterns.
  // Will add shape guard after LLGA optimization passes and
  // wipe the tensor type information from the IR, so that it's not
  // accidentally used by any other pass.

  // We rely on the shape specialization and shape guard to ensure the validity
  // of the cached compilation in the kernel, thus only support profiling mode.
  // TODO: add check on oneDNNFusionGroup to ensure allShapesAreKnown on nodes
  // to fuse: torch/csrc/jit/passes/tensorexpr_fuser.cpp: allShapesAreKnown
  if (getProfilingMode()) {
    GRAPH_DUMP(
        "Before RemoveProfileNodesAndSpecializeTypes. Beginning of LLGA "
        "optimization pass",
        g);
    RemoveProfileNodesAndSpecializeTypes(g);
    GRAPH_DUMP(
        "After RemoveProfileNodesAndSpecializeTypes. Before mutation removal",
        g);

    RemoveTensorMutation(g, [](Node* nodeToFunctionalize) {
      static std::unordered_set<Symbol> supportedOps = {
          aten::add_,
          aten::mul_,
          aten::tanh_,
          aten::elu_,
          aten::relu_,
          aten::relu6_,
          aten::gelu_,
          aten::sqrt_,
          aten::sigmoid_,
          aten::hardtanh_,
          aten::abs_,
          aten::square_,
          aten::pow_,
          aten::leaky_relu_,
          aten::round_,
          aten::exp_,
          aten::abs_,
          aten::hardswish_,
          aten::silu_};
      return supportedOps.count(nodeToFunctionalize->kind()) != 0;
    });
    RemoveListMutation(g);
    GRAPH_DUMP("After mutation removal. Before DecomposeSiluForLlga", g);
    DecomposeSiluForLLGA(g);
    GRAPH_DUMP("After DecomposeSiluForLlga. Before PrepareBinaryForLLGA", g);
    PrepareBinaryForLLGA(g);
    GRAPH_DUMP("After PrepareBinaryForLLGA. Before DeferSizeCheck", g);
    DeferSizeCheck(g);
    GRAPH_DUMP("After DeferSizeCheck. Before CreateLlgaSubgraphs", g);
    dnnl::graph::set_constant_tensor_cache(true);
    CreateLlgaSubgraphs(g);
    GRAPH_DUMP("After CreateLlgaSubgraphs. Before PropagateLayout", g);
    PropagateLayout(g);
    GRAPH_DUMP(
        "After PropagateLayout. Before prepareFusionGroupAndGuardOutputs", g);

    // Add shape guard for profiling mode and wipe the tensor type information
    // from the IR
    prepareFusionGroupAndGuardOutputs(g->block());
    GRAPH_DUMP(
        "After prepareFusionGroupAndGuardOutputs. Before "
        "RemoveTensorTypeSpecializations",
        g);
    RemoveTensorTypeSpecializations(g);
    GRAPH_DUMP(
        "After RemoveTensorTypeSpecializations. End of LLGA optimization pass",
        g);
  }
}

} // namespace fuser::onednn

static Operation createLlgaKernel(const Node* node) {
  auto kernel = std::make_shared<fuser::onednn::LlgaKernel>(node);
  return [kernel](Stack& stack) {
    RECORD_FUNCTION(kernel->debugName(), std::vector<c10::IValue>());
    kernel->run(stack);
    return 0;
  };
}

RegisterOperators oneDNNFusionGroupOp({
    torch::jit::Operator(
        prim::oneDNNFusionGroup,
        createLlgaKernel,
        AliasAnalysisKind::INTERNAL_SPECIAL_CASE),
});

// Currently, we convert some scalar inputs, such as the second argument of
// binary ops to a 1D tensor. Other scalar inputs are prim::Constant nodes.
// But if we have any scalar inputs to guard in the future, some logic here
// would have to be changed.
static Operation createLlgaGuardKernel(const Node* node) {
  return [node](Stack& stack) {
#ifdef GRAPH_DEBUG_ENABLED
    GRAPH_DEBUG("Guarding node: ", node->kind().toQualString());
#endif
    std::vector<TypePtr> types = node->tys(attr::types);
    const auto num_inputs = types.size();
#ifdef GRAPH_DEBUG_ENABLED
    GRAPH_DEBUG("num_inputs to guard: ", num_inputs);
#endif
    for (size_t i = 0; i < num_inputs; i++) {
#ifdef GRAPH_DEBUG_ENABLED
      GRAPH_DEBUG("checking input ", i);
#endif
      auto& input = peek(stack, i, num_inputs);
      const c10::TensorTypePtr& guard_tensor_type =
          types[i]->cast<TensorType>();

      if (!input.isTensor()) {
#ifdef GRAPH_DEBUG_ENABLED
        GRAPH_DEBUG("input ", i, " is not a tensor, return false");
#endif
        push(stack, IValue(false));
        return;
      }
      const at::Tensor& tensor = input.toTensor();

      // If input tensor is of mkldnn, it's originated from an upstream
      // LLGA partition that has passed the check on input shapes.
      // It is valid to continue here as long as the output shapes from
      // oneDNN graph partitions are determined by the input shapes.
      if (tensor.is_mkldnn()) {
#ifdef GRAPH_DEBUG_ENABLED
        GRAPH_DEBUG("input ", i, " is_mkldnn, continue");
#endif
        continue;
      }

      if (!guard_tensor_type->matchTensor(tensor)) {
#ifdef GRAPH_DEBUG_ENABLED
        GRAPH_DEBUG("input ", i, " check failed, return false");
#endif
        push(stack, IValue(false));
        return;
      }
    }
#ifdef GRAPH_DEBUG_ENABLED
    GRAPH_DEBUG("all check done, return true");
#endif
    push(stack, IValue(true));
    return;
  };
}

RegisterOperators oneDNNGuardOp({
    torch::jit::Operator(
        prim::oneDNNFusionGuard,
        createLlgaGuardKernel,
        AliasAnalysisKind::FROM_SCHEMA),
});
} // namespace torch::jit
