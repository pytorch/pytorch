#include <torch/csrc/jit/passes/quantization/qat_combine_conv_bn.h>

#include <torch/csrc/jit/ir/subgraph_matcher.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/graph_rewrite_helper.h>

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
   * TODO: mark as const?
   */
  void analyze(Module& module);

  /**
   * Combine all Conv2d - BatchNorm2d patterns found in the analyze step.
   */
  void transform();

 private:
  // map from graph to vector of matched conv-bn names
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
  std::unordered_map<Graph*, std::vector<std::tuple<std::string, std::string>>>
      conv_bn_names_;
};

void QATCombineConvBatchNorm2dHelper::analyze(Module& module) {
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
      // TODO: remove
      if (module_name != "BatchNorm2d" && module_name != "Conv2d") {
        GRAPH_DUMP(
            current.type()->name()->name() + "::" + method.name() +
                "() before Conv2d-BatchNorm2d combining",
            method.graph());
      }
      const auto& matches = findPatternMatches(pattern_graph, *method.graph());

      GRAPH_DEBUG("number of Conv2d-BatchNorm2d matches: ", matches.size());
      Graph* g = method.graph().get();
      if (!conv_bn_names_.count(g)) {
        // This is to make sure we don't visit one graph multiple times
        conv_bn_names_[g] = {};
        for (const Match& match : matches) {
          GRAPH_DEBUG("Checking next match...");
          Node* matched_conv = match.nodes_map.at(pattern_conv);
          GRAPH_DEBUG("matched_conv ", *matched_conv);
          Node* matched_bn = match.nodes_map.at(pattern_bn);
          GRAPH_DEBUG("matched_bn ", *matched_bn);
          Node* matched_conv_submodule =
              match.values_map.at(pattern_conv_submodule)->node();
          GRAPH_DEBUG("matched_conv_submodule ", *matched_conv_submodule);
          Node* matched_bn_submodule =
              match.values_map.at(pattern_bn_submodule)->node();
          GRAPH_DEBUG("matched_bn_submodule ", *matched_bn_submodule);

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

          // TODO: finish
        }
      }
    }

  } // while
}

void QATCombineConvBatchNorm2dHelper::transform() {}

} // namespace

Module QATCombineConvBatchNorm2d(const Module& module) {
  QATCombineConvBatchNorm2dHelper h;
  Module m = module.clone();
  h.analyze(m);
  h.transform();
  return m;
}

} // namespace jit
} // namespace torch
