#include <torch/csrc/jit/passes/quantization/fusion_passes.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>

namespace torch {
namespace jit {

namespace {
void fuseQuantizeAddReluImpl(std::shared_ptr<Graph>& graph) {
  SubgraphRewriter fused_add_relu_rewriter;
  std::string quantized_add_relu_pattern = R"(
    graph(%a_quant, %b_quant, %scale, %zero_point):
         %add_out = quantized::add(%a_quant, %b_quant, %scale, %zero_point)
         %r = aten::relu(%add_out)
         return (%r) )";
  std::string fused_add_relu_pattern = R"(
    graph(%a_quant, %b_quant, %scale, %zero_point):
         %r = quantized::add_relu(%a_quant, %b_quant, %scale, %zero_point)
         return (%r) )";
  fused_add_relu_rewriter.RegisterRewritePattern(
      quantized_add_relu_pattern, fused_add_relu_pattern);
  std::string quantized_add_out_relu_pattern = R"(
    graph(%a_quant, %b_quant, %out_quant):
         %add_out = quantized::add_out(%a_quant, %b_quant, %out_quant)
         %r = aten::relu(%add_out)
         return (%r) )";
  std::string fused_add_out_relu_pattern = R"(
    graph(%a_quant, %b_quant, %out_quant):
         %r = quantized::add_relu_out(%a_quant, %b_quant, %out_quant)
         return (%r) )";
  fused_add_relu_rewriter.RegisterRewritePattern(
      quantized_add_out_relu_pattern, fused_add_out_relu_pattern);
  std::string quantized_add_scalar_relu_pattern = R"(
    graph(%a_quant, %b_scalar):
         %add_out = quantized::add_scalar(%a_quant, %b_scalar)
         %r = aten::relu(%add_out)
         return (%r) )";
  std::string fused_add_scalar_relu_pattern = R"(
    graph(%a_quant, %b_scalar):
         %r = quantized::add_scalar_relu(%a_quant, %b_scalar)
         return (%r) )";
  fused_add_relu_rewriter.RegisterRewritePattern(
      quantized_add_scalar_relu_pattern, fused_add_scalar_relu_pattern);
  std::string quantized_add_scalar_out_relu_pattern = R"(
    graph(%a_quant, %b_scalar, %out_quant):
         %add_out = quantized::add_scalar_out(%a_quant, %b_scalar, %out_quant)
         %r = aten::relu(%add_out)
         return (%r) )";
  std::string fused_add_scalar_out_relu_pattern = R"(
    graph(%a_quant, %b_scalar, %out_quant):
         %r = quantized::add_scalar_relu_out(%a_quant, %b_scalar, %out_quant)
         return (%r) )";
  fused_add_relu_rewriter.RegisterRewritePattern(
      quantized_add_scalar_out_relu_pattern, fused_add_scalar_out_relu_pattern);
  fused_add_relu_rewriter.runOnGraph(graph);
}
} // namespace

void FuseQuantizedAddRelu(std::shared_ptr<Graph>& graph) {
  fuseQuantizeAddReluImpl(graph);
}

} // namespace jit
} // namespace torch
