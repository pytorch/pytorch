
#include <torch/csrc/jit/passes/fuse_relu.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/subgraph_matcher.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>

namespace torch {
namespace jit {

namespace {
void fuseAddReluImpl(std::shared_ptr<Graph>& graph) {
  SubgraphRewriter rewriter;

  std::string add_relu_0 = R"(
    graph(%a, %b, %alpha):
        %add_res = aten::add(%a, %b, %alpha)
        %res = aten::relu(%add_res)
        return (%res))";
  std::string add_relu_fused = R"(
    graph(%a, %b, %alpha):
        %res = aten::add_relu(%a, %b, %alpha)
        return (%res))";
  rewriter.RegisterRewritePattern(add_relu_0, add_relu_fused);

  std::string add_relu_1 = R"(
    graph(%a, %b, %alpha):
        %add_res = aten::add(%a, %b, %alpha)
        %res = aten::relu_(%add_res)
        return (%res))";
  rewriter.RegisterRewritePattern(add_relu_1, add_relu_fused);

  std::string add_inplace_relu_1 = R"(
    graph(%a, %b, %alpha):
        %add_res = aten::add_(%a, %b, %alpha)
        %res = aten::relu_(%add_res)
        return (%res))";
  std::string add_inplace_relu_fused = R"(
    graph(%a, %b, %alpha):
        %res = aten::add_relu_(%a, %b, %alpha)
        return (%res))";
  rewriter.RegisterRewritePattern(add_inplace_relu_1, add_inplace_relu_fused);

  std::string add_out_relu_0 = R"(
    graph(%a, %b, %alpha, %out):
        %add_res = aten::add(%a, %b, %alpha, %out)
        %res = aten::relu(%add_res)
        return (%res))";
  std::string add_out_relu_fused = R"(
    graph(%a, %b, %alpha, %out):
        %res = aten::add_relu(%a, %b, %alpha, %out)
        return (%res))";
  rewriter.RegisterRewritePattern(add_out_relu_0, add_out_relu_fused);

  std::string add_out_relu_1 = R"(
    graph(%a, %b, %alpha, %out):
        %add_res = aten::add(%a, %b, %alpha, %out)
        %res = aten::relu_(%add_res)
        return (%res))";
  rewriter.RegisterRewritePattern(add_out_relu_1, add_out_relu_fused);

  rewriter.runOnGraph(graph);
}
} // namespace

void FuseAddRelu(script::Module& module) {
  auto graph = module.get_method("forward").graph();
  fuseAddReluImpl(graph);
}

void FuseAddRelu(std::shared_ptr<Graph>& graph) {
  fuseAddReluImpl(graph);
}
} // namespace jit
} // namespace torch
