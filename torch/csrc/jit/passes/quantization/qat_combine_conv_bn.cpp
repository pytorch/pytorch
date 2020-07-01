#include <torch/csrc/jit/passes/quantization/qat_combine_conv_bn.h>

#include <torch/csrc/jit/ir/subgraph_matcher.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/graph_rewrite_helper.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>
#include <torch/csrc/jit/passes/quantization/helper.h>

namespace torch {
namespace jit {

namespace {
using graph_rewrite_helper::PatternInfo;

class QATCombineConvBatchNorm2dHelper {
 public:
  void transform(Module& module);
};

std::pair<std::string, std::string> getBeforeAndAfterPatterns() {

  std::string conv_bn = R"(
graph(%input, %conv, %batchnorm):
  %conv_out = prim::CallMethod[name="forward"](%conv, %input)
  %bn_out = prim::CallMethod[name="forward"](%batchnorm, %conv_out)
  return (%bn_out))";

  // The IR corresponding to the forward pass of combined QAT Conv-BN,
  // before observation.
  //
  // Example: https://gist.github.com/vkuzo/a057c4f7ee2ea94ef9a023580774b138
  //
  // This should be kept in sync with all possible paths through
  // torch/nn/intrinsic/qat/modules/conv_fused.py
  //
  // TODO(future PR): handle the padding if statement (currently only the
  //   'zeros' branch is in the IR)
  // TODO(future PR): add passes to inline and dedup the constants
  // TODO(future PR): make sure all the constants are looked up correctly
  //  (eps, group, dilation, etc).
  std::string combined_conv_bn = R"(
graph(%input, %conv, %batchnorm):
  %one : int = prim::Constant[value=1]()
  %zero : int = prim::Constant[value=0]()
  %neg_one : int = prim::Constant[value=-1]()
  %bn_eps : float = prim::Constant[value=1.0e-5]() # TODO: get from BN
  %bn_running_var : Tensor = prim::GetAttr[name="running_var"](%batchnorm)
  %var_plus_eps : Tensor = aten::add(%bn_running_var, %bn_eps, %one)
  %running_std.1 : Tensor = aten::sqrt(%var_plus_eps)
  %bn_weight : Tensor = prim::GetAttr[name="weight"](%batchnorm)
  %scale_factor.1 : Tensor = aten::div(%bn_weight, %running_std.1)
  %conv_weight : Tensor = prim::GetAttr[name="weight"](%conv)
  %reshape_params : int[] = prim::ListConstruct(%neg_one, %one, %one, %one)
  %scale_factor_reshaped : Tensor = aten::reshape(%scale_factor.1, %reshape_params)
  %scaled_weight_no_fq.1 : Tensor = aten::mul(%conv_weight, %scale_factor_reshaped)
  %conv_bias : Tensor? = prim::GetAttr[name="bias"](%conv)
  %conv_stride : int[] = prim::ListConstruct(%one, %one) # TODO: get from conv
  %conv_padding : int[] = prim::ListConstruct(%zero, %zero) # TODO: get from conv
  %conv_dilation : int[] = prim::ListConstruct(%one, %one) # TODO: get from conv, get groups from conv
  %conv_res.2 : Tensor = aten::qat_conv2d_and_unscale(%input, %scaled_weight_no_fq.1, %scale_factor.1, %conv_bias, %conv_stride, %conv_padding, %conv_dilation, %one)
  %bn_out = prim::CallMethod[name="forward"](%batchnorm, %conv_res.2)
  return (%bn_out))";

  return std::pair<std::string, std::string>(conv_bn, combined_conv_bn);
}

void QATCombineConvBatchNorm2dHelper::transform(Module& module) {
  auto patterns = getBeforeAndAfterPatterns();
  SubgraphRewriter rewriter;
  rewriter.RegisterRewritePattern(patterns.first, patterns.second);

  for (auto& method : module.get_methods()) {
    std::shared_ptr<Graph> g = method.graph();
    GRAPH_DUMP("before", g);
    rewriter.runOnGraph(g, {is_conv2d_module, is_batchnorm2d_module});
    GRAPH_DUMP("after", g);
  }
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
