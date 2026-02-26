#include <torch/nativert/graph/passes/pass_manager/GraphPasses.h>

#include <torch/nativert/graph/passes/SubgraphRewriter.h>
#include <torch/nativert/graph/passes/pass_manager/GraphPassRegistry.h>

namespace torch::nativert {

void register_base_passes() {
  GraphPassRegistry::add_pass("EmptyPass", [](Graph*) { return false; });

  GraphPassRegistry::add_pass(
      "LinearDynamicFp16UnpackedWeight", [](Graph* graph) {
        std::string p = R"(
    graph(%i, %w, %b):
    %out_0 = torch.ops.aten.linear.default(input=%i, weight=%w, bias=%b)
    return (%out_0))";

        std::string p_1 = R"(
    graph(%i, %w, %b):
    %out_0 = torch.ops.quantized.linear_dynamic_fp16_unpacked_weight.default(X=%i, weight=%w, bias=%b)
    return (%out_0))";

        std::string p_new = R"(
    graph(%i, %w, %b):
    %pw = torch.ops.quantized.linear_prepack_fp16.default(W=%w, B=%b)
    %out_0 = torch.ops.quantized.linear_dynamic_fp16.default(X=%i, W_prepack=%pw)
    return (%out_0))";

        SubgraphRewriter rewriter("LinearDynamicFp16UnpackedWeight");
        rewriter.registerRewritePattern(p, p_new);
        rewriter.registerRewritePattern(p_1, p_new);
        return rewriter.run(graph);
      });

  GraphPassRegistry::add_pass(
      "LinearReluDynamicFp16UnpackedWeight", [](Graph* graph) {
        std::string p = R"(
    graph(%i, %w, %b):
    %out_0 = torch.ops.aten.linear.default(input=%i, weight=%w, bias=%b)
    %out_1 = torch.ops.aten.relu.default(self=%out_0)
    return (%out_1))";

        std::string p_1 = R"(
    graph(%i, %w, %b):
    %out_0 = torch.ops.quantized.linear_dynamic_fp16_unpacked_weight.default(X=%i, weight=%w, bias=%b)
    %out_1 = torch.ops.aten.relu.default(self=%out_0)
    return (%out_1))";

        std::string p_new = R"(
    graph(%i, %w, %b):
    %pw = torch.ops.quantized.linear_prepack_fp16.default(W=%w, B=%b)
    %out_0 = torch.ops.quantized.linear_relu_dynamic_fp16.default(X=%i, W_prepack=%pw)
    return (%out_0))";

        SubgraphRewriter rewriter("LinearReluDynamicFp16UnpackedWeight");
        rewriter.registerRewritePattern(p, p_new);
        rewriter.registerRewritePattern(p_1, p_new);
        return rewriter.run(graph);
      });

  GraphPassRegistry::add_pass("CleanUpDeadNodes", [](Graph* graph) {
    return graph->cleanupDeadNodes();
  });

  GraphPassRegistry::add_pass("RemoveDetach", [](Graph* graph) {
    std::vector<Node*> nodesToDestroy;

    for (auto& node : graph->nodes()) {
      if (node.target() == "torch.ops.aten.detach.default") {
        nodesToDestroy.push_back(&node);
        graph->replaceAllUses(node.outputs()[0], node.inputs()[0].value);
      }
    }

    VLOG(1) << "[GraphPasses] Removed " << nodesToDestroy.size()
            << " aten.detach nodes";

    const bool mutated = !nodesToDestroy.empty();

    for (Node* node : nodesToDestroy) {
      node->destroy();
    }

    graph->renumberValues();
    graph->finalize();
    graph->lint();

    return mutated;
  });
}

} // namespace torch::nativert
