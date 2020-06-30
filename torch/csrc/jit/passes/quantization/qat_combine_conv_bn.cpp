#include <torch/csrc/jit/passes/quantization/qat_combine_conv_bn.h>

#include <torch/csrc/jit/ir/subgraph_matcher.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/graph_rewrite_helper.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>

#include <stack>
#include <unordered_set>

namespace torch {
namespace jit {

namespace {
using graph_rewrite_helper::PatternInfo;

class QATCombineConvBatchNorm2dHelper {
 public:

  /**
   * Combine adjacent conv-bn modules into QAT conv-bn
   */
  void transform(Module& module);

 private:
  std::unordered_set<std::shared_ptr<Graph>> visited_graphs_;

  /**
   * Rewrites the IR which transforms a single conv-bn pattern
   * into a QAT conv-bn pattern.
   */
  void transformMatch(std::shared_ptr<Graph> g, const Match& match,
      const std::unordered_map<std::string, Value*>& vmap);
};

void QATCombineConvBatchNorm2dHelper::transformMatch(std::shared_ptr<Graph> g,
    const Match& match, const std::unordered_map<std::string, Value*>& vmap) {

  GRAPH_DUMP("before", g);

  Value* pattern_conv_out = vmap.at("conv_out");
  Value* pattern_bn_out = vmap.at("bn_out");
  Value* pattern_conv_submodule = vmap.at("conv_submodule");
  Value* pattern_bn_submodule = vmap.at("bn_submodule");
  Node* pattern_conv = pattern_conv_out->node();
  Node* pattern_bn = pattern_bn_out->node();

  // TODO (before land): look into reusing this snippet with conv-bn
  // folding pass
  GRAPH_DEBUG("Checking next match...");
  Node* matched_conv = match.nodes_map.at(pattern_conv);
  Node* matched_bn = match.nodes_map.at(pattern_bn);
  Node* matched_conv_submodule =
      match.values_map.at(pattern_conv_submodule)->node();
  Node* matched_bn_submodule =
      match.values_map.at(pattern_bn_submodule)->node();
  Node* matched_bn_orig_forward_call = matched_bn_submodule->next();

  // new
  Value* matched_bn_val = match.values_map.at(pattern_bn_submodule);
  Value* matched_conv_val = match.values_map.at(pattern_conv_submodule);

  TORCH_INTERNAL_ASSERT(
      matched_conv_submodule->kind() == prim::GetAttr);
  TORCH_INTERNAL_ASSERT(matched_bn_submodule->kind() == prim::GetAttr);

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
  //  TODO: clean up this docblock
  //
  /*
  std::string combined_conv_bn = R"(
graph(%self, %input):
  %one : int = prim::Constant[value=1]()
  %zero : int = prim::Constant[value=0]()
  %neg_one : int = prim::Constant[value=-1]()
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
  %bn_out = prim::CallMethod[name="forward"](%bn_mod, %conv_res.2)
  return (%bn_out))";
  */

  // insert after the bn node but before the bn forward call node
  g->setInsertPoint(matched_bn_orig_forward_call);

  // -1, 0, 1 constants
  // TODO: add pass to dedup the constants later
  auto zero = g->create(prim::Constant)->i_(attr::value, 0);
  zero->output()->setType(IntType::get())
    ->setDebugName("zero");
  g->insertNode(zero);
  auto one = g->create(prim::Constant)->i_(attr::value, 1);
  one->output()->setType(IntType::get())
    ->setDebugName("one");
  g->insertNode(one);
  auto neg_one = g->create(prim::Constant)->i_(attr::value, -1);
  neg_one->output()->setType(IntType::get())
    ->setDebugName("neg_one");
  g->insertNode(neg_one);

  // TODO: instead of constant, fetch this from BN. However,
  //   it has to be made a buffer, since only buffers and params are
  //   available to do something like
  //   Node* bn_eps = g->createGetAttr(matched_bn_val, "eps");
  auto bn_eps = g->create(prim::Constant)->f_(attr::value, 1.0e-5);
  bn_eps->output()->setType(FloatType::get())
    ->setDebugName("bn_eps");
  g->insertNode(bn_eps);

  Node* bn_running_var = g->createGetAttr(matched_bn_val, "running_var");
  bn_running_var->output()->setDebugName("bn_running_var");
  g->insertNode(bn_running_var);

  // var_plus_eps = bn_running_var + bn_eps
  Node* var_plus_eps = g->insertNode(g->create(aten::add, 1));
  var_plus_eps->addInput(bn_running_var->output());
  var_plus_eps->addInput(bn_eps->output());
  var_plus_eps->addInput(one->output());
  var_plus_eps->output()->setType(TensorType::get())
    ->setDebugName("var_plus_eps");

  // running_std = sqrt(var_plus_eps)
  Node* running_std = g->insertNode(g->create(aten::sqrt));
  running_std->addInput(var_plus_eps->output());
  running_std->output()->setType(TensorType::get())
    ->setDebugName("running_std");

  // bn_weight = bn.weight
  Node* bn_weight = g->createGetAttr(matched_bn_val, "weight");
  bn_weight->output()->setDebugName("bn_weight");
  g->insertNode(bn_weight);

  // scale_factor = bn_weight / running_std
  Node* scale_factor = g->insertNode(g->create(aten::div));
  scale_factor->addInput(bn_weight->output());
  scale_factor->addInput(running_std->output());
  scale_factor->output()->setType(TensorType::get())
    ->setDebugName("scale_factor");

  // conv_weight = conv.weight
  Node* conv_weight = g->createGetAttr(matched_conv_val, "weight");
  conv_weight->output()->setDebugName("conv_weight");
  g->insertNode(conv_weight);

  // reshape_params = [-1, 1, 1, 1]
  Node* reshape_params = g->create(prim::ListConstruct);
  reshape_params->addInput(neg_one->output());
  reshape_params->addInput(one->output());
  reshape_params->addInput(one->output());
  reshape_params->addInput(one->output());
  reshape_params->output()->setType(ListType::create(IntType::get()))
    ->setDebugName("reshape_params");
  g->insertNode(reshape_params);

  // scale_factor_reshaped = scale_factor.reshape(reshape_params)
  Node* scale_factor_reshaped = g->insertNode(g->create(aten::reshape));
  scale_factor_reshaped->addInput(scale_factor->output());
  scale_factor_reshaped->addInput(reshape_params->output());
  scale_factor_reshaped->output()->setType(TensorType::get())
    ->setDebugName("scale_factor_reshaped");

  // scaled_weight_no_fq = conv_weight * scale_factor_reshaped
  Node* scaled_weight_no_fq = g->insertNode(g->create(aten::mul));
  scaled_weight_no_fq->addInput(conv_weight->output());
  scaled_weight_no_fq->addInput(scale_factor_reshaped->output());
  scaled_weight_no_fq->output()->setType(TensorType::get())
    ->setDebugName("scaled_weight_no_fq");

  // conv_bias = conv.bias
  Node* conv_bias = g->createGetAttr(matched_conv_val, "bias");
  conv_bias->output()->setDebugName("conv_bias");
  g->insertNode(conv_bias);

  // conv_stride = [1, 1]. TODO: get from conv instead of hardcode
  Node* conv_stride = g->create(prim::ListConstruct);
  conv_stride->addInput(one->output());
  conv_stride->addInput(one->output());
  conv_stride->output()->setType(ListType::create(IntType::get()))
    ->setDebugName("conv_stride");
  g->insertNode(conv_stride);

  // conv_padding = [0, 0]. TODO: get from conv instead of hardcode
  Node* conv_padding = g->create(prim::ListConstruct);
  conv_padding->addInput(zero->output());
  conv_padding->addInput(zero->output());
  conv_padding->output()->setType(ListType::create(IntType::get()))
    ->setDebugName("conv_padding");
  g->insertNode(conv_padding);

  // conv_dilation = [1, 1]. TODO: get from conv instead of hardcode
  Node* conv_dilation = g->create(prim::ListConstruct);
  conv_dilation->addInput(one->output());
  conv_dilation->addInput(one->output());
  conv_dilation->output()->setType(ListType::create(IntType::get()))
    ->setDebugName("conv_dilation");
  g->insertNode(conv_dilation);

  // conv_res = aten::qat_conv2d_and_unscale(input, scaled_weight_no_fq,
  //   scale_factor, conv_bias, conv_stride, conv_padding, conv_dilation,
  //   one)
  Node* conv_res = g->insertNode(g->create(aten::qat_conv2d_and_unscale));
  Value* input = g->inputs()[1];
  conv_res->addInput(input);
  conv_res->addInput(scaled_weight_no_fq->output());
  conv_res->addInput(scale_factor->output());
  conv_res->addInput(conv_bias->output());
  conv_res->addInput(conv_stride->output());
  conv_res->addInput(conv_padding->output());
  conv_res->addInput(conv_dilation->output());
  conv_res->addInput(one->output());
  conv_res->output()->setDebugName("conv_res");

  // replace bn forward's tensor input with conv_res
  matched_bn_orig_forward_call->replaceInput(1, conv_res->output());

  // TODO: delete the original conv forward

  // TODO: verify numerics match

  GRAPH_DUMP("after", g);

}

void QATCombineConvBatchNorm2dHelper::transform(Module& module) {

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

  // recursively process all the modules
  std::stack<Module> worklist({module});
  while (!worklist.empty()) {
    Module current = worklist.top();
    worklist.pop();

    auto module_name = current.type()->name()->name();
    GRAPH_DEBUG("\nanalyze - while loop - module ", module_name, "\n")

    for (const Module& submodule : current.children()) {
      worklist.push(submodule);
    }

    for (auto& method : current.get_methods()) {
      const auto& matches = findPatternMatches(pattern_graph, *method.graph());

      GRAPH_DEBUG("number of Conv2d-BatchNorm2d matches: ", matches.size());
      std::shared_ptr<Graph> g = method.graph();

      if (visited_graphs_.count(g)) {
        continue;
      }
      visited_graphs_.insert(g);

      for (const Match& match : matches) {
        transformMatch(g, match, pattern.vmap);
      }
    } // for
  } // while
}

} // namespace

Module QATCombineConvBatchNorm2d(const Module& module) {
  QATCombineConvBatchNorm2dHelper h;
  Module m = module.clone();
  h.transform(m);
  return m;
}

} // namespace jit
} // namespace torch
