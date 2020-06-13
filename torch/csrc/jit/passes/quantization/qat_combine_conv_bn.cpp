#include <torch/csrc/jit/passes/quantization/qat_combine_conv_bn.h>

#include <torch/csrc/jit/ir/subgraph_matcher.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/graph_rewrite_helper.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>

#include <stack>

namespace torch {
namespace jit {

namespace {
using graph_rewrite_helper::PatternInfo;

class QATCombineConvBatchNorm2dHelper {
 public:
  /**
   * Find all instances of Conv2d - BatchNorm2d patterns in the graph
   * and record their locations.
   *
   * TODO before land: mark as const?
   */
  void analyze(Module& module);

  /**
   * Combine all Conv2d - BatchNorm2d patterns found in the analyze step.
   */
  void transform(Module& module);

 private:
  // map from graph pointer to vector of matched conv-bn names
  //
  // example:
  //   %x1 = self.conv1(%x0)
  //   %x2 = self.bn1(%x1)
  //   %x3 = self.conv2(%x2)
  //   %x4 = self.bn2(%x3)
  //
  // conv_bn_names_ = {
  //   g: [("conv1", "bn1"), ("conv2", "bn2")]
  // }
  std::unordered_map<std::shared_ptr<Graph>, std::vector<std::tuple<std::string, std::string>>>
      conv_bn_names_;
};

void QATCombineConvBatchNorm2dHelper::analyze(Module& module) {
  // TODO before land: verify what specifically the matching rules are here
  // Dot in the ".Conv2d" and ".BatchNorm2d" is an attempt to
  // prevent matching module's whose name might end with Conv2d
  // But are user defined modules.
  const PatternInfo pattern = PatternInfo::parse_from_str(R"IR(
graph(%self, %x):
    %conv_submodule = match::module[name=".Conv2d"](%self)
    %conv_out = prim::CallMethod[name="forward"](%conv_submodule, %x)
    %bn_submodule = match::module[name=".BatchNorm2d"](%self)
    %bn_out = prim::CallMethod[name="forward"](%bn_submodule, %conv_out)
    return (%bn_out))IR");

  const Graph& pattern_graph = *pattern.pattern_graph;
  const auto& vmap = pattern.vmap;
  Value* pattern_conv_out = vmap.at("conv_out");
  Value* pattern_bn_out = vmap.at("bn_out");
  Value* pattern_conv_submodule = vmap.at("conv_submodule");
  Value* pattern_bn_submodule = vmap.at("bn_submodule");
  Node* pattern_conv = pattern_conv_out->node();
  Node* pattern_bn = pattern_bn_out->node();

  // We will put submodules into this worklist and keep processing items from it
  // one by one. We start by just putting the top module there.
  std::stack<Module> worklist({module});
  while (!worklist.empty()) {
    Module current = worklist.top();
    worklist.pop();

    auto module_name = current.type()->name()->name();
    GRAPH_DEBUG("\nanalyze - while loop - module ", module_name, "\n")

    // Queue submodules for processing
    for (const Module& submodule : current.children()) {
      worklist.push(submodule);
    }

    // Process all method of the current module
    for (auto& method : current.get_methods()) {
      const auto& matches = findPatternMatches(pattern_graph, *method.graph());

      GRAPH_DEBUG("number of Conv2d-BatchNorm2d matches: ", matches.size());
      std::shared_ptr<Graph> g = method.graph();
      if (!conv_bn_names_.count(g)) {
        // This is to make sure we don't visit one graph multiple times
        conv_bn_names_[g] = {};
        for (const Match& match : matches) {
          // TODO (before land): look into reusing this snippet with conv-bn
          // folding pass
          GRAPH_DEBUG("Checking next match...");
          Node* matched_conv = match.nodes_map.at(pattern_conv);
          Node* matched_bn = match.nodes_map.at(pattern_bn);
          Node* matched_conv_submodule =
              match.values_map.at(pattern_conv_submodule)->node();
          Node* matched_bn_submodule =
              match.values_map.at(pattern_bn_submodule)->node();

          TORCH_INTERNAL_ASSERT(
              matched_conv_submodule->kind() == prim::GetAttr);
          TORCH_INTERNAL_ASSERT(matched_bn_submodule->kind() == prim::GetAttr);

          const auto& conv_module_name =
              matched_conv_submodule->s(Symbol::attr("name"));
          const auto& bn_module_name =
              matched_bn_submodule->s(Symbol::attr("name"));
          GRAPH_DEBUG(conv_module_name, bn_module_name);

          conv_bn_names_[g].push_back(
              std::make_tuple(conv_module_name, bn_module_name));
        }
      }
    } // for

  } // while
}

std::pair<std::string, std::string> getBeforeAndAfterPatterns(
    std::string conv_name, std::string bn_name) {

  // A pattern to match a nn.Conv2d -> nn.BatchNorm2d call with
  // exact module names.
  std::string conv_bn = R"(
graph(%self, %input):
  %conv_submodule = prim::GetAttr[name=")" + conv_name + R"("](%self)
  %conv_out = prim::CallMethod[name="forward"](%conv_submodule, %input)
  %bn_submodule = prim::GetAttr[name=")" + bn_name + R"("](%self)
  %bn_out = prim::CallMethod[name="forward"](%bn_submodule, %conv_out)
  return (%bn_out))";

  // A pattern only for testing
  // TODO(before land): remove
  std::string conv = R"(
graph(%self, %input):
  %conv_mod = prim::GetAttr[name=")" + conv_name + R"("](%self)
  %res = prim::CallMethod[name="forward"](%conv_mod, %input)
  return (%res))";

  // The IR corresponding to the forward pass of combined QAT Conv-BN,
  // before observation.
  // TODO(before land) remove or move to gist: base:
  //   https://www.internalfb.com/intern/paste/P133207815/
  //
  // TODO before land: handle the padding if statement (currently only the 'zeros' branch is in the IR)
  //
  // corresponds to this Python code:
  //
  // running_std = torch.sqrt(self.bn.running_var + self.bn.eps)
  // scale_factor = self.bn.weight / running_std
  // scaled_weight_no_fq = self.conv.weight * scale_factor.reshape([-1, 1, 1, 1])
  // conv_res = \
  //     torch.qat_conv2d_and_unscale(input, scaled_weight_no_fq, scale_factor,
  //        self.conv.bias, self.conv.stride, self.conv.padding,
  //        self.conv.dilation, self.conv.groups)
  //  return conv_res
  //
  //  TODO before land: run passes, again, to fix the constants
  //  TODO before land: make sure all the constants are looked up correctly (group,
  //  dilation, etc)
  //
  // Note: currently parsing IR for prim::Constant[value={neg_value}]
  // is not working (likely a bug). So, creating neg_one by subtracting
  // one from zero, and depending on a future pass to optimize it away.
  // Ideally we can remove this before land.
  std::string combined_conv_bn = R"(
graph(%self, %input):
  %one : int = prim::Constant[value=1]()
  %zero : int = prim::Constant[value=0]()
  %neg_one : int = aten::sub(%zero, %one)
  %conv_mod = prim::GetAttr[name=")" + conv_name + R"("](%self)
  %bn_mod = prim::GetAttr[name=")" + bn_name + R"("](%self)
  %bn_eps : float = prim::GetAttr[name="eps"](%bn_mod)
  %bn_running_var : Tensor = prim::GetAttr[name="running_var"](%bn_mod)
  %var_plus_eps : Tensor = aten::add(%bn_running_var, %bn_eps, %one)
  %running_std.1 : Tensor = aten::sqrt(%var_plus_eps)
  %bn_weight : Tensor = prim::GetAttr[name="weight"](%bn_mod)
  %scale_factor.1 : Tensor = aten::div(%bn_weight, %running_std.1)
  %conv_weight : Tensor = prim::GetAttr[name="weight"](%conv_mod)
  %reshape_params : int[] = prim::ListConstruct(%neg_one, %one, %one, %one)
  %scale_factor_reshaped : Tensor = aten::reshape(%scale_factor.1, %reshape_params)
  %scaled_weight_no_fq.1 : Tensor = aten::mul(%conv_weight, %scale_factor_reshaped)
  %conv_bias : Tensor? = prim::GetAttr[name="bias"](%conv_mod)
  %conv_stride : int[] = prim::ListConstruct(%one, %one) # TODO: get from conv
  %conv_padding : int[] = prim::ListConstruct(%zero, %zero) # TODO: get from conv
  %conv_dilation : int[] = prim::ListConstruct(%one, %one) # TODO: get from conv, get groups from conv
  %conv_res.2 : Tensor = aten::qat_conv2d_and_unscale(%input, %scaled_weight_no_fq.1, %scale_factor.1, %conv_bias, %conv_stride, %conv_padding, %conv_dilation, %one)
  return (%conv_res.2))";

  return std::pair<std::string, std::string>(conv_bn, combined_conv_bn);
}

void QATCombineConvBatchNorm2dHelper::transform(Module& module) {

  for (auto graph_to_conv_bn_names : conv_bn_names_) {
    std::shared_ptr<Graph> g = graph_to_conv_bn_names.first;
    for (const auto conv_bn_name : graph_to_conv_bn_names.second) {

      std::string conv_name = std::get<0>(conv_bn_name);
      std::string bn_name = std::get<1>(conv_bn_name);
      auto patterns = getBeforeAndAfterPatterns(conv_name, bn_name);

      // TODO before land: reuse better
      SubgraphRewriter rewriter;
      rewriter.RegisterRewritePattern(patterns.first, patterns.second);

      GRAPH_DUMP("before", g);
      rewriter.runOnGraph(g);
      GRAPH_DUMP("after", g);
    }
  }
}

} // namespace

Module QATCombineConvBatchNorm2d(const Module& module) {
  QATCombineConvBatchNorm2dHelper h;
  Module m = module.clone();
  h.analyze(m);
  h.transform(m);
  return m;
}

} // namespace jit
} // namespace torch
