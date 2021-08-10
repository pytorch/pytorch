#include <ATen/Utils.h>

#include <torch/csrc/jit/frontend/code_template.h>
#include <torch/csrc/jit/ir/constants.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/subgraph_matcher.h>
#include <torch/csrc/jit/passes/frozen_conv_add_relu_fusion.h>
#include <torch/csrc/jit/passes/graph_rewrite_helper.h>
#include <torch/csrc/jit/passes/remove_mutation.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>
#ifdef USE_CUDA
#include <ATen/cuda/CUDAConfig.h>
#endif

namespace torch {
namespace jit {

namespace {
void fuseFrozenConvAddReluImpl(std::shared_ptr<Graph>& graph) {
#ifdef USE_CUDA
#if AT_CUDNN_ENABLED()
  SubgraphRewriter rewriter;

  // CUDNN does not support conv1d
  std::array<std::string, 2> conv_operators = {"conv2d", "conv3d"};
  std::array<std::string, 2> add_operators = {"add", "add_"};
  std::array<std::string, 2> relu_operators = {"relu", "relu_"};

  auto conv_relu_rstring = CodeTemplate(R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[], %dilation:int[], %groups:int):
      %x = aten::${conv}(%input, %weight, %bias, %stride, %padding, %dilation, %groups)
      %res = aten::${relu}(%x)
      return (%res))");

  std::string conv_relu_fused = R"(
    graph(%input, %weight, %bias, %stride:int[], %padding:int[], %dilation:int[], %groups:int):
        %res = aten::cudnn_convolution_relu(%input, %weight, %bias, %stride, %padding, %dilation, %groups)
        return (%res))";

  auto conv_add_relu_rstring = CodeTemplate(R"(
    graph(%input, %weight, %bias, %z, %alpha, %stride:int[], %padding:int[], %dilation:int[], %groups:int):
      %x = aten::${conv}(%input, %weight, %bias, %stride, %padding, %dilation, %groups)
      %y = aten::${add}(%x, %z, %alpha)
      %res = aten::${relu}(%y)
      return (%res))");

  std::string conv_add_relu_fused = R"(
    graph(%input, %weight, %bias, %z, %alpha, %stride:int[], %padding:int[], %dilation:int[], %groups:int):
        %res = aten::cudnn_convolution_add_relu(%input, %weight, %z, %alpha, %bias, %stride, %padding, %dilation, %groups)
        return (%res))";

  for (const auto& conv : conv_operators) {
    for (const auto& relu : relu_operators) {
      TemplateEnv env;
      env.s("conv", conv);
      env.s("relu", relu);
      rewriter.RegisterRewritePattern(
          conv_relu_rstring.format(env), conv_relu_fused);
      for (const auto& add : add_operators) {
        env.s("add", add);
        rewriter.RegisterRewritePattern(
            conv_add_relu_rstring.format(env), conv_add_relu_fused);
      }
    }
  }

  auto filter = [](const Match& match,
                   const std::unordered_map<std::string, Value*>& vmap) {
    auto weight = toIValue(match.values_map.at(vmap.at("weight")));
    if (!weight.has_value() || !weight.value().isTensor()) {
      return false;
    }
    const at::Tensor& weight_t = weight.value().toTensor();
    if (!weight_t.device().is_cuda() ||
        !weight_t.is_contiguous()
#if CUDNN_VERSION < 8000
        // Enabling the fusion for half-precision causes perf regression on
        // earlier cuDNN
        || weight_t.dtype() == at::kHalf
#endif
    ) {
      return false;
    }

    // bias is optional
    if (vmap.find("bias") != vmap.end()) {
      auto bias = toIValue(match.values_map.at(vmap.at("bias")));
      if (bias.has_value() && bias.value().isTensor()) {
        const at::Tensor& bias_t = bias.value().toTensor();
        if (bias_t.dtype() != weight_t.dtype() || bias_t.ndimension() != 1 ||
            bias_t.size(0) != weight_t.size(0) || !bias_t.device().is_cuda()) {
          return false;
        }
      }
    }

    // z is optional
    if (vmap.find("z") != vmap.end()) {
      auto z = toIValue(match.values_map.at(vmap.at("z")));
      if (z.has_value() && z.value().isTensor()) {
        const at::Tensor& z_t = z.value().toTensor();
        if (z_t.dtype() != weight_t.dtype() ||
            z_t.size(0) != weight_t.size(0) || !z_t.is_contiguous() ||
            !z_t.device().is_cuda()) {
          return false;
        }
      }
    }
    return true;
  };

  // Convert _convolution and in-place operators for simpler replacement pattern
  // matching
  graph_rewrite_helper::replaceConvolutionWithAtenConv(graph);

  rewriter.runOnGraph(graph, filter);
#endif
#endif
}
} // namespace

void FuseFrozenConvAddRelu(std::shared_ptr<Graph>& graph) {
  fuseFrozenConvAddReluImpl(graph);
}

} // namespace jit
} // namespace torch
