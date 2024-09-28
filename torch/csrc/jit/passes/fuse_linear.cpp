#include <torch/csrc/jit/passes/fuse_linear.h>
#include <torch/csrc/jit/passes/graph_rewrite_helper.h>
#include <torch/csrc/jit/passes/quantization/helper.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>

namespace torch::jit {

void FuseLinear(std::shared_ptr<Graph>& graph) {
  std::string addmm_pattern = R"IR(
    graph(%input, %weight_t, %bias, %beta, %alpha):
        %res = aten::addmm(%bias, %input, %weight_t, %beta, %alpha)
        return (%res))IR";
  std::string fused_linear_addmm = R"IR(
    graph(%input, %weight_t, %bias, %beta, %alpha):
        %weight = aten::t(%weight_t)
        %res = aten::linear(%input, %weight, %bias)
        return (%res))IR";

  auto beta_is_one = [](const Match& match,
                        const std::unordered_map<std::string, Value*>& vmap) {
    return is_int_constant(match, vmap, "beta", 1);
  };

  // check %weight_t is produced by `aten::t` to make sure
  // we can transform the pattern to `aten::linear`
  auto weight_transposed =
      [](const Match& match,
         const std::unordered_map<std::string, Value*>& vmap) {
        const auto& match_vmap = match.values_map;
        auto v = match_vmap.at(vmap.at("weight_t"));
        return v->node()->kind() == Symbol::aten("t");
      };

  // replace addmm pattern to linear
  SubgraphRewriter addmm_to_linear;
  std::vector<std::pair<std::string, std::string>> value_mappings(
      {{"weight", "res"}, {"res", "res"}});
  addmm_to_linear.RegisterRewritePattern(
      addmm_pattern, fused_linear_addmm, value_mappings);
  addmm_to_linear.runOnGraph(
      graph, {aten_add_alpha_is_one, beta_is_one, weight_transposed});

  std::string matmul_add_pattern = R"IR(
    graph(%input, %weight_t, %bias, %alpha):
        %output = aten::matmul(%input, %weight_t)
        %res = aten::add_(%output, %bias, %alpha)
        return (%res))IR";
  std::string fused_linear_matmul = R"IR(
    graph(%input, %weight_t, %bias, %alpha):
        %weight = aten::t(%weight_t)
        %res = aten::linear(%input, %weight, %bias)
        return (%res))IR";
  value_mappings = {{"weight", "output"}, {"res", "output"}};
  // replace matmul + add pattern to linear
  SubgraphRewriter matmuladd_to_linear;
  matmuladd_to_linear.RegisterRewritePattern(
      matmul_add_pattern, fused_linear_matmul, value_mappings);
  matmuladd_to_linear.runOnGraph(
      graph, {aten_add_alpha_is_one, weight_transposed});

  std::string matmul_pattern = R"IR(
    graph(%input, %weight_t):
        %output = aten::matmul(%input, %weight_t)
        return (%output))IR";
  std::string fused_linear_bias_none = R"IR(
    graph(%input, %weight_t):
        %weight = aten::t(%weight_t)
        %bias: Tensor? = prim::Constant()
        %res = aten::linear(%input, %weight, %bias)
        return (%res))IR";

  // replace matmul with bias=None pattern to linear
  SubgraphRewriter matmul_to_linear;
  matmul_to_linear.RegisterRewritePattern(
      matmul_pattern, fused_linear_bias_none, value_mappings);
  matmul_to_linear.runOnGraph(graph, weight_transposed);

  // clean up extra transpose for the weight of aten::linear
  std::string linear_weight_extra_transpose = R"IR(
    graph(%input, %weight, %bias):
        %weight_t1 = aten::t(%weight)
        %weight_t2 = aten::t(%weight_t1)
        %res = aten::linear(%input, %weight_t2, %bias)
        return (%res))IR";

  std::string linear_weight_no_transpose = R"IR(
    graph(%input, %weight, %bias):
        %res = aten::linear(%input, %weight, %bias)
        return (%res))IR";

  value_mappings = {{"res", "res"}};
  SubgraphRewriter cleanup;
  cleanup.RegisterRewritePattern(
      linear_weight_extra_transpose,
      linear_weight_no_transpose,
      value_mappings);
  cleanup.runOnGraph(graph);

  SwapFunctionalLinear(graph);
}

void SwapFunctionalLinear(Module& module) {
  for (auto& method : module.get_methods()) {
    std::shared_ptr<Graph> g = method.graph();
    SwapFunctionalLinear(g);
  }
  for (Module m : module.children()) {
    SwapFunctionalLinear(m);
  }
}

void SwapFunctionalLinear(std::shared_ptr<Graph>& graph) {
  std::string functional_linear = R"(
graph(%linear, %input, %weight, %bias):
  %r = prim::CallFunction(%linear, %input, %weight, %bias)
  return (%r) )";
  std::string aten_linear = R"(
graph(%linear, %input, %weight, %bias):
  %r = aten::linear(%input, %weight, %bias)
  return (%r) )";

  auto filter = [](const Match& match,
                   const std::unordered_map<std::string, Value*>& vmap) {
    const auto& match_vmap = match.values_map;
    auto linear = graph_rewrite_helper::getValue("linear", match_vmap, vmap);
    auto func_name = graph_rewrite_helper::getFuncName(linear);
    return func_name == "linear";
  };
  SubgraphRewriter rewriter;
  rewriter.RegisterRewritePattern(functional_linear, aten_linear);
  rewriter.runOnGraph(graph, filter);
}

} // namespace torch::jit
