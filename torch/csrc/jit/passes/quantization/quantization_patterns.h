#pragma once

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/subgraph_matcher.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>
#include <string>
#include <unordered_map>

namespace torch {
namespace jit {

struct QuantFusionInfo {
  std::string quantized_op_name;
  std::string pattern;
  std::string replacement;
  std::function<
      bool(const Match&, const std::unordered_map<std::string, Value*>&)>
      filter =
          [](const Match&, const std::unordered_map<std::string, Value*>&) {
            return true;
          };
};

std::vector<QuantFusionInfo> quant_fusion_pattern_and_replacements() {
  // aten::conv2d
  std::string conv2d = R"(
graph(%a_quant, %packed_params, %r_scale, %r_zero_point, %r_dtype, %stride, %padding, %dilation, %groups):
        %a_dequant = aten::dequantize(%a_quant)
        %w_quant : Tensor, %b : Tensor? = quantized::conv2d_unpack(%packed_params)
        %w_dequant = aten::dequantize(%w_quant)
        %r = aten::conv2d(%a_dequant, %w_dequant, %b, %stride, %padding, %dilation, %groups)
        %r_quant = aten::quantize_per_tensor(%r, %r_scale, %r_zero_point, %r_dtype)
        return (%r_quant) )";

  // aten::conv2d - aten::relu
  std::string conv2d_relu = R"(
graph(%a_quant, %packed_params, %r_scale, %r_zero_point, %r_dtype, %stride, %padding, %dilation, %groups):
        %a_dequant = aten::dequantize(%a_quant)
        %w_quant : Tensor, %b : Tensor? = quantized::conv2d_unpack(%packed_params)
        %w_dequant = aten::dequantize(%w_quant)
        %conv_out = aten::conv2d(%a_dequant, %w_dequant, %b, %stride, %padding, %dilation, %groups)
        %r = aten::relu(%conv_out)
        %r_quant = aten::quantize_per_tensor(%r, %r_scale, %r_zero_point, %r_dtype)
        return (%r_quant) )";

  // aten::conv2d - aten::relu_
  std::string conv2d_inplace_relu = R"(
graph(%a_quant, %packed_params, %r_scale, %r_zero_point, %r_dtype, %stride, %padding, %dilation, %groups):
        %a_dequant = aten::dequantize(%a_quant)
        %w_quant : Tensor, %b : Tensor? = quantized::conv2d_unpack(%packed_params)
        %w_dequant = aten::dequantize(%w_quant)
        %conv_out = aten::conv2d(%a_dequant, %w_dequant, %b, %stride, %padding, %dilation, %groups)
        %r = aten::relu_(%conv_out)
        %r_quant = aten::quantize_per_tensor(%r, %r_scale, %r_zero_point, %r_dtype)
        return (%r_quant) )";

  // quantized::conv2d
  std::string quantized_conv2d = R"(
graph(%a_quant, %packed_params, %r_scale, %r_zero_point, %r_dtype, %stride, %padding, %dilation, %groups):
        %r_quant = quantized::conv2d(%a_quant, %packed_params, %r_scale, %r_zero_point)
        return (%r_quant) )";

  // quantized::conv2d_relu
  std::string quantized_conv2d_relu = R"(
graph(%a_quant, %packed_params, %r_scale, %r_zero_point, %r_dtype, %stride, %padding, %dilation, %groups):
        %r_quant = quantized::conv2d_relu(%a_quant, %packed_params, %r_scale, %r_zero_point)
        return (%r_quant) )";

  // aten::conv3d
  std::string conv3d = R"(
graph(%a_quant, %packed_params, %r_scale, %r_zero_point, %r_dtype, %stride, %padding, %dilation, %groups):
        %a_dequant = aten::dequantize(%a_quant)
        %w_quant : Tensor, %b : Tensor? = quantized::conv3d_unpack(%packed_params)
        %w_dequant = aten::dequantize(%w_quant)
        %r = aten::conv3d(%a_dequant, %w_dequant, %b, %stride, %padding, %dilation, %groups)
        %r_quant = aten::quantize_per_tensor(%r, %r_scale, %r_zero_point, %r_dtype)
        return (%r_quant) )";

  // aten::conv3d - aten::relu
  std::string conv3d_relu = R"(
graph(%a_quant, %packed_params, %r_scale, %r_zero_point, %r_dtype, %stride, %padding, %dilation, %groups):
        %a_dequant = aten::dequantize(%a_quant)
        %w_quant : Tensor, %b : Tensor? = quantized::conv3d_unpack(%packed_params)
        %w_dequant = aten::dequantize(%w_quant)
        %conv_out = aten::conv3d(%a_dequant, %w_dequant, %b, %stride, %padding, %dilation, %groups)
        %r = aten::relu(%conv_out)
        %r_quant = aten::quantize_per_tensor(%r, %r_scale, %r_zero_point, %r_dtype)
        return (%r_quant) )";

  // aten::conv3d - aten::relu_
  std::string conv3d_inplace_relu = R"(
graph(%a_quant, %packed_params, %r_scale, %r_zero_point, %r_dtype, %stride, %padding, %dilation, %groups):
        %a_dequant = aten::dequantize(%a_quant)
        %w_quant : Tensor, %b : Tensor? = quantized::conv3d_unpack(%packed_params)
        %w_dequant = aten::dequantize(%w_quant)
        %conv_out = aten::conv3d(%a_dequant, %w_dequant, %b, %stride, %padding, %dilation, %groups)
        %r = aten::relu_(%conv_out)
        %r_quant = aten::quantize_per_tensor(%r, %r_scale, %r_zero_point, %r_dtype)
        return (%r_quant) )";

  // quantized::conv3d
  std::string quantized_conv3d = R"(
graph(%a_quant, %packed_params, %r_scale, %r_zero_point, %r_dtype, %stride, %padding, %dilation, %groups):
        %r_quant = quantized::conv3d(%a_quant, %packed_params, %r_scale, %r_zero_point)
        return (%r_quant) )";

  // quantized::conv3d_relu
  std::string quantized_conv3d_relu = R"(
graph(%a_quant, %packed_params, %r_scale, %r_zero_point, %r_dtype, %stride, %padding, %dilation, %groups):
        %r_quant = quantized::conv3d_relu(%a_quant, %packed_params, %r_scale, %r_zero_point)
        return (%r_quant) )";

  std::string add_relu = R"(
graph(%a_quant, %b_quant, %scale, %zero_point, %dtype):
         %alpha = prim::Constant[value=1]()
         %a_dequant = aten::dequantize(%a_quant)
         %b_dequant = aten::dequantize(%b_quant)
         %r_add = aten::add_(%a_dequant, %b_dequant, %alpha)
         %r_relu = aten::relu(%r_add)
         %r = aten::quantize_per_tensor(%r_relu, %scale, %zero_point, %dtype)
         return (%r) )";

  std::string add_inplace_relu = R"(
graph(%a_quant, %b_quant, %scale, %zero_point, %dtype):
         %alpha = prim::Constant[value=1]()
         %a_dequant = aten::dequantize(%a_quant)
         %b_dequant = aten::dequantize(%b_quant)
         %r_add = aten::add_(%a_dequant, %b_dequant, %alpha)
         %r_relu = aten::relu_(%r_add)
         %r = aten::quantize_per_tensor(%r_relu, %scale, %zero_point, %dtype)
         return (%r) )";

  std::string quantized_add_relu = R"(
graph(%a_quant, %b_quant, %scale, %zero_point, %dtype):
         %r = quantized::add_relu(%a_quant, %b_quant, %scale, %zero_point)
         return (%r) )";

  // aten::linear
  std::string linear = R"(
graph(%packed_params, %a_quant, %r_scale, %r_zero_point, %r_dtype):
        %a_dequant = aten::dequantize(%a_quant)
        %w_quant : Tensor, %b : Tensor? = quantized::linear_unpack(%packed_params)
        %w_dequant = aten::dequantize(%w_quant)
        %r = aten::linear(%a_dequant, %w_dequant, %b)
        %r_quant = aten::quantize_per_tensor(%r, %r_scale, %r_zero_point, %r_dtype)
        return (%r_quant) )";

  // quantized::linear
  std::string quantized_linear = R"(
graph(%packed_params, %a_quant, %r_scale, %r_zero_point, %r_dtype):
        %r = quantized::linear(%a_quant, %packed_params, %r_scale, %r_zero_point)
        return (%r) )";

  std::string cat = R"(
graph(%input_quant, %dim, %r_scale, %r_zero_point, %r_dtype):
        %input_dequant = aten::dequantize(%input_quant)
        %r = aten::cat(%input_dequant, %dim)
        %r_quant = aten::quantize_per_tensor(%r, %r_scale, %r_zero_point, %r_dtype)
        return (%r_quant) )";

  std::string quantized_cat = R"(
graph(%input_quant, %dim, %r_scale, %r_zero_point, %r_dtype):
         %r_quant = quantized::cat(%input_quant, %dim, %r_scale, %r_zero_point)
         return (%r_quant) )";

  // aten::add
  std::string add = R"(
graph(%a_quant, %b_quant, %alpha, %scale, %zero_point, %dtype):
         %a_dequant = aten::dequantize(%a_quant)
         %b_dequant = aten::dequantize(%b_quant)
         %r_add = aten::add(%a_dequant, %b_dequant, %alpha)
         %r = aten::quantize_per_tensor(%r_add, %scale, %zero_point, %dtype)
         return (%r) )";

  // TODO: add %dtype after when https://github.com/pytorch/pytorch/issues/34351
  // is fixed
  // quantized::add
  std::string quantized_add = R"(
graph(%a_quant, %b_quant, %alpha, %scale, %zero_point, %dtype):
         %r = quantized::add(%a_quant, %b_quant, %scale, %zero_point)
         return (%r) )";

  auto add_filter = [](const Match& match,
                       const std::unordered_map<std::string, Value*>& vmap) {
    const auto& match_vmap = match.values_map;
    auto alpha = toIValue(match_vmap.at(vmap.at("alpha")));
    return alpha && alpha->isInt() && alpha->toInt() == 1;
  };

  // aten::add_
  std::string inplace_add = R"(
graph(%a_quant, %b_quant, %alpha, %scale, %zero_point, %dtype):
         %a_dequant = aten::dequantize(%a_quant)
         %b_dequant = aten::dequantize(%b_quant)
         %r_add = aten::add_(%a_dequant, %b_dequant, %alpha)
         %r = aten::quantize_per_tensor(%r_add, %scale, %zero_point, %dtype)
         return (%r) )";

  // quantized::add_scalar
  std::string add_scalar = R"(
graph(%a_quant, %b_scalar, %alpha):
         %a_dequant = aten::dequantize(%a_quant)
         %r = aten::add(%a_dequant, %b_scalar, %alpha)
         return (%r) )";

  std::string quantized_add_scalar = R"(
graph(%a_quant, %b_scalar, %alpha):
         %r = quantized::add_scalar(%a_quant, %b_scalar)
         return (%r) )";

  // filter that checks %alpha is constant 1 and %b_scalar is a scalar
  auto add_scalar_filter =
      [](const Match& match,
         const std::unordered_map<std::string, Value*>& vmap) {
        const auto& match_vmap = match.values_map;
        auto alpha = toIValue(match_vmap.at(vmap.at("alpha")));
        auto b_scalar = match_vmap.at(vmap.at("b_scalar"));
        return alpha && alpha->isInt() && alpha->toInt() == 1 &&
            b_scalar->type()->isSubtypeOf(NumberType::get());
      };

  // quantized::add_scalar_out
  std::string add_scalar_out = R"(
graph(%a_quant, %b_scalar, %alpha):
         %a_dequant = aten::dequantize(%a_quant)
         %r = aten::add_(%a_dequant, %b_scalar, %alpha)
         return (%r) )";

  std::string quantized_add_scalar_out = R"(
graph(%a_quant, %b_scalar, %alpha):
         %r = quantized::add_scalar_out(%a_quant, %b_scalar, %a_quant)
         return (%r) )";

  // quantized::add_scalar_relu
  std::string add_scalar_relu = R"(
graph(%a_quant, %b_scalar, %alpha):
         %a_dequant = aten::dequantize(%a_quant)
         %r_add = aten::add(%a_dequant, %b_scalar, %alpha)
         %r = aten::relu(%r_add)
         return (%r) )";

  std::string quantized_add_scalar_relu = R"(
graph(%a_quant, %b_scalar, %alpha):
         %r = quantized::add_scalar_relu(%a_quant, %b_scalar)
         return (%r) )";

  // quantized::add_scalar_relu_out
  std::string add_scalar_relu_out = R"(
graph(%a_quant, %b_scalar, %alpha):
         %a_dequant = aten::dequantize(%a_quant)
         %r_add = aten::add_(%a_dequant, %b_scalar, %alpha)
         %r = aten::relu(%r_add)
         return (%r) )";

  std::string quantized_add_scalar_relu_out = R"(
graph(%a_quant, %b_scalar, %alpha):
         %r = quantized::add_scalar_relu_out(%a_quant, %b_scalar, %a_quant)
         return (%r) )";

  // quantized::batch_norm
  std::string batch_norm2d = R"(
graph(%a_quant, %weight, %bias, %mean, %var, %training, %eaf, %eps, %7, %scale, %zero_point, %scalar_type):
         %a_dequant = aten::dequantize(%a_quant)
         %r_bn = aten::batch_norm(%a_dequant, %weight, %bias, %mean, %var, %training, %eaf, %eps, %7)
         %r = aten::quantize_per_tensor(%r_bn, %scale, %zero_point, %scalar_type)
         return (%r) )";
  std::string quantized_batch_norm2d = R"(
graph(%a_quant, %weight, %bias, %mean, %var, %training, %eaf, %eps, %7, %scale, %zero_point, %scalar_type):
         %r = quantized::batch_norm2d(%a_quant, %weight, %bias, %mean, %var, %eps, %scale, %zero_point)
         return (%r) )";

  std::string batch_norm2d_relu = R"(
graph(%a_quant, %weight, %bias, %mean, %var, %training, %eaf, %eps, %7, %scale, %zero_point, %scalar_type):
         %a_dequant = aten::dequantize(%a_quant)
         %bn_out = aten::batch_norm(%a_dequant, %weight, %bias, %mean, %var, %training, %eaf, %eps, %7)
         %relu = aten::relu(%bn_out)
         %r = aten::quantize_per_tensor(%relu, %scale, %zero_point, %scalar_type)
         return (%r) )";
  std::string batch_norm2d_inplace_relu = R"(
graph(%a_quant, %weight, %bias, %mean, %var, %training, %eaf, %eps, %7, %scale, %zero_point, %scalar_type):
         %a_dequant = aten::dequantize(%a_quant)
         %bn_out = aten::batch_norm(%a_dequant, %weight, %bias, %mean, %var, %training, %eaf, %eps, %7)
         %relu = aten::relu_(%bn_out)
         %r = aten::quantize_per_tensor(%relu, %scale, %zero_point, %scalar_type)
         return (%r) )";

  std::string quantized_batch_norm2d_relu = R"(
graph(%a_quant, %weight, %bias, %mean, %var, %training, %eaf, %eps, %7, %scale, %zero_point, %scalar_type):
         %r = quantized::batch_norm2d_relu(%a_quant, %weight, %bias, %mean, %var, %eps, %scale, %zero_point)
         return (%r) )";

  // aten::mul
  std::string mul = R"(
graph(%a_quant, %b_quant, %scale, %zero_point, %dtype):
         %a_dequant = aten::dequantize(%a_quant)
         %b_dequant = aten::dequantize(%b_quant)
         %r_mul = aten::mul(%a_dequant, %b_dequant)
         %r = aten::quantize_per_tensor(%r_mul, %scale, %zero_point, %dtype)
         return (%r) )";

  // aten::mul_
  std::string inplace_mul = R"(
graph(%a_quant, %b_quant, %scale, %zero_point, %dtype):
         %a_dequant = aten::dequantize(%a_quant)
         %b_dequant = aten::dequantize(%b_quant)
         %r_mul = aten::mul_(%a_dequant, %b_dequant)
         %r = aten::quantize_per_tensor(%r_mul, %scale, %zero_point, %dtype)
         return (%r) )";

  // quantized::mul
  std::string quantized_mul = R"(
graph(%a_quant, %b_quant, %scale, %zero_point, %dtype):
         %r = quantized::mul(%a_quant, %b_quant, %scale, %zero_point)
         return (%r) )";

  // quantized::mul_scalar
  std::string mul_scalar = R"(
graph(%a_quant, %b_scalar):
         %a_dequant = aten::dequantize(%a_quant)
         %r = aten::mul(%a_dequant, %b_scalar)
         return (%r) )";

  std::string mul_scalar_out = R"(
graph(%a_quant, %b_scalar):
         %a_dequant = aten::dequantize(%a_quant)
         %r = aten::mul_(%a_dequant, %b_scalar)
         return (%r) )";

  std::string quantized_mul_scalar = R"(
graph(%a_quant, %b_scalar):
         %r = quantized::mul_scalar(%a_quant, %b_scalar)
         return (%r) )";

  std::string quantized_mul_scalar_out = R"(
graph(%a_quant, %b_scalar):
         %r = quantized::mul_scalar_out(%a_quant, %b_scalar, %a_quant)
         return (%r) )";

  // filter that checks %b_scalar is a scalar
  auto mul_scalar_filter =
      [](const Match& match,
         const std::unordered_map<std::string, Value*>& vmap) {
        const auto& match_vmap = match.values_map;
        auto b_scalar = match_vmap.at(vmap.at("b_scalar"));
        return b_scalar->type()->isSubtypeOf(NumberType::get());
      };

  // quantized::mul_relu
  std::string mul_relu = R"(
graph(%a_quant, %b_quant, %scale, %zero_point, %dtype):
         %a_dequant = aten::dequantize(%a_quant)
         %b_dequant = aten::dequantize(%b_quant)
         %r_mul = aten::mul(%a_dequant, %b_dequant)
         %r_relu = aten::relu(%r_mul)
         %r = aten::quantize_per_tensor(%r_relu, %scale, %zero_point, %dtype)
         return (%r) )";

  std::string inplace_mul_relu = R"(
graph(%a_quant, %b_quant, %scale, %zero_point, %dtype):
         %a_dequant = aten::dequantize(%a_quant)
         %b_dequant = aten::dequantize(%b_quant)
         %r_mul = aten::mul_(%a_dequant, %b_dequant)
         %r_relu = aten::relu(%r_mul)
         %r = aten::quantize_per_tensor(%r_relu, %scale, %zero_point, %dtype)
         return (%r) )";

  std::string mul_inplace_relu = R"(
graph(%a_quant, %b_quant, %scale, %zero_point, %dtype):
         %a_dequant = aten::dequantize(%a_quant)
         %b_dequant = aten::dequantize(%b_quant)
         %r_mul = aten::mul(%a_dequant, %b_dequant)
         %r_relu = aten::relu_(%r_mul)
         %r = aten::quantize_per_tensor(%r_relu, %scale, %zero_point, %dtype)
         return (%r) )";

  std::string inplace_mul_inplace_relu = R"(
graph(%a_quant, %b_quant, %scale, %zero_point, %dtype):
         %a_dequant = aten::dequantize(%a_quant)
         %b_dequant = aten::dequantize(%b_quant)
         %r_mul = aten::mul_(%a_dequant, %b_dequant)
         %r_relu = aten::relu_(%r_mul)
         %r = aten::quantize_per_tensor(%r_relu, %scale, %zero_point, %dtype)
         return (%r) )";

  std::string quantized_mul_relu = R"(
graph(%a_quant, %b_quant, %scale, %zero_point, %dtype):
         %r = quantized::mul_relu(%a_quant, %b_quant, %scale, %zero_point)
         return (%r) )";

  // quantized::mul_scalar_relu
  std::string mul_scalar_relu = R"(
graph(%a_quant, %b_scalar):
         %a_dequant = aten::dequantize(%a_quant)
         %r_mul = aten::mul(%a_dequant, %b_scalar)
         %r = aten::relu(%r_mul)
         return (%r) )";

  std::string quantized_mul_scalar_relu = R"(
graph(%a_quant, %b_scalar):
         %r = quantized::mul_scalar_relu(%a_quant, %b_scalar)
         return (%r) )";

  // quantized::mul_scalar_relu_out
  std::string mul_scalar_relu_out = R"(
graph(%a_quant, %b_scalar):
         %a_dequant = aten::dequantize(%a_quant)
         %r_mul = aten::mul_(%a_dequant, %b_scalar)
         %r = aten::relu(%r_mul)
         return (%r) )";

  std::string quantized_mul_scalar_relu_out = R"(
graph(%a_quant, %b_scalar):
         %r = quantized::mul_scalar_relu_out(%a_quant, %b_scalar, %a_quant)
         return (%r) )";

  // quantized::hardswish
  std::string hardswish = R"(
graph(%a_quant, %r_scale, %r_zero_point, %r_dtype):
         %a_dequant = aten::dequantize(%a_quant)
         %r = aten::hardswish(%a_dequant)
         %r_quant = aten::quantize_per_tensor(%r, %r_scale, %r_zero_point, %r_dtype)
         return (%r_quant) )";

  std::string quantized_hardswish = R"(
graph(%a_quant, %r_scale, %r_zero_point, %r_dtype):
         %r_quant = quantized::hardswish(%a_quant, %r_scale, %r_zero_point)
         return (%r_quant) )";

  // quantized::layer_norm
  std::string layer_norm = R"(
graph(%a_quant, %normalized_shape, %weight, %bias, %eps, %cudnn_enabled, %output_scale, %output_zero_point, %scalar_type):
         %a_dequant = aten::dequantize(%a_quant)
         %r_ln = aten::layer_norm(%a_dequant, %normalized_shape, %weight, %bias, %eps, %cudnn_enabled)
         %r = aten::quantize_per_tensor(%r_ln, %output_scale, %output_zero_point, %scalar_type)
         return (%r) )";

  std::string quantized_layer_norm = R"(
graph(%a_quant, %normalized_shape, %weight, %bias, %eps, %cudnn_enabled, %output_scale, %output_zero_point, %scalar_type):
         %r = quantized::layer_norm(%a_quant, %normalized_shape, %weight, %bias, %eps, %output_scale, %output_zero_point)
         return (%r) )";

  // ============= General Ops that doesn't require observation =============
  // IR pattern common to all general value ops
  std::string common_general_value_op = R"(
          %r_scale : float = aten::q_scale(%a_quant)
          %r_zero_point : int = aten::q_zero_point(%a_quant)
          %r_dtype : int = prim::dtype(%a_quant)
          %r_quant = aten::quantize_per_tensor(%r, %r_scale, %r_zero_point, %r_dtype)
          return (%r_quant) )";

  // aten::avg_pool1d
  std::string avg_pool1d = R"(
graph(%a_quant, %kernel_size, %stride, %padding, %ceil_mode, %count_include_pad):
          %a_dequant = aten::dequantize(%a_quant)
          %r = aten::avg_pool1d(%a_dequant, %kernel_size, %stride, %padding, %ceil_mode, %count_include_pad)
)" + common_general_value_op;

  std::string aten_avg_pool1d = R"(
graph(%a_quant, %kernel_size, %stride, %padding, %ceil_mode, %count_include_pad):
          %r = aten::avg_pool1d(%a_quant, %kernel_size, %stride, %padding, %ceil_mode, %count_include_pad)
          return (%r) )";

  // aten::avg_pool2d
  std::string avg_pool2d = R"(
graph(%a_quant, %kernel_size, %stride, %padding, %ceil_mode, %count_include_pad, %divisor_override):
          %a_dequant = aten::dequantize(%a_quant)
          %r = aten::avg_pool2d(%a_dequant, %kernel_size, %stride, %padding, %ceil_mode, %count_include_pad, %divisor_override)
)" + common_general_value_op;

  std::string aten_avg_pool2d = R"(
graph(%a_quant, %kernel_size, %stride, %padding, %ceil_mode, %count_include_pad, %divisor_override):
          %r = aten::avg_pool2d(%a_quant, %kernel_size, %stride, %padding, %ceil_mode, %count_include_pad, %divisor_override)
          return (%r) )";

  // aten::avg_pool3d
  std::string avg_pool3d = R"(
graph(%a_quant, %kernel_size, %stride, %padding, %ceil_mode, %count_include_pad, %divisor_override):
          %a_dequant = aten::dequantize(%a_quant)
          %r = aten::avg_pool3d(%a_dequant, %kernel_size, %stride, %padding, %ceil_mode, %count_include_pad, %divisor_override)
)" + common_general_value_op;

  std::string aten_avg_pool3d = R"(
graph(%a_quant, %kernel_size, %stride, %padding, %ceil_mode, %count_include_pad, %divisor_override):
          %r = aten::avg_pool3d(%a_quant, %kernel_size, %stride, %padding, %ceil_mode, %count_include_pad, %divisor_override)
          return (%r) )";

  // aten::adaptive_avg_pool1d
  std::string adaptive_avg_pool1d = R"(
graph(%a_quant, %output_size):
          %a_dequant = aten::dequantize(%a_quant)
          %r = aten::adaptive_avg_pool1d(%a_dequant, %output_size)
)" + common_general_value_op;

  std::string aten_adaptive_avg_pool1d = R"(
graph(%a_quant, %output_size):
          %r = aten::adaptive_avg_pool1d(%a_quant, %output_size)
          return (%r) )";

  // aten::adaptive_avg_pool2d
  std::string adaptive_avg_pool2d = R"(
graph(%a_quant, %output_size):
          %a_dequant = aten::dequantize(%a_quant)
          %r = aten::adaptive_avg_pool2d(%a_dequant, %output_size)
)" + common_general_value_op;

  std::string aten_adaptive_avg_pool2d = R"(
graph(%a_quant, %output_size):
          %r = aten::adaptive_avg_pool2d(%a_quant, %output_size)
          return (%r) )";

  // aten::adaptive_avg_pool3d
  std::string adaptive_avg_pool3d = R"(
graph(%a_quant, %output_size):
          %a_dequant = aten::dequantize(%a_quant)
          %r = aten::adaptive_avg_pool3d(%a_dequant, %output_size)
)" + common_general_value_op;

  std::string aten_adaptive_avg_pool3d = R"(
graph(%a_quant, %output_size):
          %r = aten::adaptive_avg_pool3d(%a_quant, %output_size)
          return (%r) )";

  // aten::mean
  std::string mean = R"(
graph(%a_quant, %dim):
          %a_dequant = aten::dequantize(%a_quant)
          %r = aten::mean(%a_dequant, %dim)
)" + common_general_value_op;

  std::string aten_mean = R"(
graph(%a_quant, %dim):
          %r = aten::mean(%a_quant, %dim)
          return (%r) )";

  // aten::upsample_nearest1d
  std::string upsample_nearest1d = R"(
graph(%a_quant, %size, %scale_factor):
          %a_dequant = aten::dequantize(%a_quant)
          %r = aten::upsample_nearest1d(%a_dequant, %size, %scale_factor)
)" + common_general_value_op;

  std::string aten_upsample_nearest1d = R"(
graph(%a_quant, %size, %scale_factor):
          %r = aten::upsample_nearest1d(%a_quant, %size, %scale_factor)
          return (%r) )";

  // aten::upsample_nearest2d
  std::string upsample_nearest2d = R"(
graph(%a_quant, %size, %scale_h, %scale_w):
          %a_dequant = aten::dequantize(%a_quant)
          %r = aten::upsample_nearest2d(%a_dequant, %size, %scale_h, %scale_w)
)" + common_general_value_op;

  std::string aten_upsample_nearest2d = R"(
graph(%a_quant, %size, %scale_h, %scale_w):
          %r = aten::upsample_nearest2d(%a_quant, %size, %scale_h, %scale_w)
          return (%r) )";

  // aten::upsample_nearest3d
  std::string upsample_nearest3d = R"(
graph(%a_quant, %size, %scale_d, %scale_h, %scale_w):
          %a_dequant = aten::dequantize(%a_quant)
          %r = aten::upsample_nearest3d(%a_dequant, %size, %scale_d, %scale_h, %scale_w)
)" + common_general_value_op;

  std::string aten_upsample_nearest3d = R"(
graph(%a_quant, %size, %scale_d, %scale_h, %scale_w):
          %r = aten::upsample_nearest3d(%a_quant, %size, %scale_d, %scale_h, %scale_w)
          return (%r) )";

  // aten::upsample_linear1d
  std::string upsample_linear1d = R"(
graph(%a_quant, %size, %align_corners, %scales):
          %a_dequant = aten::dequantize(%a_quant)
          %r = aten::upsample_linear1d(%a_dequant, %size, %align_corners, %scales)
)" + common_general_value_op;

  std::string aten_upsample_linear1d = R"(
graph(%a_quant, %size, %align_corners, %scales):
          %r = aten::upsample_linear1d(%a_quant, %size, %align_corners, %scales)
          return (%r) )";

  // aten::upsample_bilinear2d
  std::string upsample_bilinear2d = R"(
graph(%a_quant, %size, %align_corners, %scales_h, %scales_w):
          %a_dequant = aten::dequantize(%a_quant)
          %r = aten::upsample_bilinear2d(%a_dequant, %size, %align_corners, %scales_h, %scales_w)
)" + common_general_value_op;

  std::string aten_upsample_bilinear2d = R"(
graph(%a_quant, %size, %align_corners, %scales_h, %scales_w):
          %r = aten::upsample_bilinear2d(%a_quant, %size, %align_corners, %scales_h, %scales_w)
          return (%r) )";

  // aten::upsample_trilinear3d
  std::string upsample_trilinear3d = R"(
graph(%a_quant, %size, %align_corners, %scales_d, %scales_h, %scales_w):
          %a_dequant = aten::dequantize(%a_quant)
          %r = aten::upsample_trilinear3d(%a_dequant, %size, %align_corners, %scales_h, %scales_w)
)" + common_general_value_op;

  std::string aten_upsample_trilinear3d = R"(
graph(%a_quant, %size, %align_corners, %scales_d, %scales_h, %scales_w):
          %r = aten::upsample_trilinear3d(%a_quant, %size, %align_corners, %scales_d, %scales_h, %scales_w)
          return (%r) )";

  // aten::clamp
  std::string clamp = R"(
graph(%a_quant, %min, %max):
          %a_dequant = aten::dequantize(%a_quant)
          %r = aten::clamp(%a_dequant, %min, %max)
)" + common_general_value_op;

  std::string aten_clamp = R"(
graph(%a_quant, %min, %max):
          %r = aten::clamp(%a_quant, %min, %max)
          return (%r) )";

  // aten::hardtanh
  std::string hardtanh = R"(
graph(%a_quant, %min, %max):
          %a_dequant = aten::dequantize(%a_quant)
          %r = aten::hardtanh(%a_dequant, %min, %max)
)" + common_general_value_op;

  std::string aten_hardtanh = R"(
graph(%a_quant, %min, %max):
          %r = aten::hardtanh(%a_quant, %min, %max)
          return (%r) )";

  // aten::hardtanh_
  std::string hardtanh_ = R"(
graph(%a_quant, %min, %max):
          %a_dequant = aten::dequantize(%a_quant)
          %r = aten::hardtanh_(%a_dequant, %min, %max)
)" + common_general_value_op;

  std::string aten_hardtanh_ = R"(
graph(%a_quant, %min, %max):
          %r = aten::hardtanh_(%a_quant, %min, %max)
          return (%r) )";

  return {
      {"quantized::conv2d", conv2d, quantized_conv2d},
      {"quantized::conv2d_relu", conv2d_relu, quantized_conv2d_relu},
      {"quantized::conv2d_relu", conv2d_inplace_relu, quantized_conv2d_relu},
      {"quantized::conv3d", conv3d, quantized_conv3d},
      {"quantized::conv3d_relu", conv3d_relu, quantized_conv3d_relu},
      {"quantized::conv3d_relu", conv3d_inplace_relu, quantized_conv3d_relu},
      {"quantized::linear", linear, quantized_linear},
      {"quantized::add_relu", add_relu, quantized_add_relu, add_filter},
      {"quantized::add_relu", add_inplace_relu, quantized_add_relu, add_filter},
      {"quantized::add", add, quantized_add, add_filter},
      {"quantized::add", inplace_add, quantized_add, add_filter},
      // note that this must come before quantized::add_scalar
      {"quantized::add_scalar_relu",
       add_scalar_relu,
       quantized_add_scalar_relu,
       add_scalar_filter},
      {"quantized::add_scalar_relu_out",
       add_scalar_relu_out,
       quantized_add_scalar_relu_out,
       add_scalar_filter},
      {"quantized::add_scalar",
       add_scalar,
       quantized_add_scalar,
       add_scalar_filter},
      {"quantized::add_scalar_out",
       add_scalar_out,
       quantized_add_scalar_out,
       add_scalar_filter},
      {"quantized::cat", cat, quantized_cat},
      {"quantized::batch_norm2d", batch_norm2d, quantized_batch_norm2d},
      {"quantized::batch_norm2d_relu",
       batch_norm2d_relu,
       quantized_batch_norm2d_relu},
      {"quantized::batch_norm2d_relu",
       batch_norm2d_inplace_relu,
       quantized_batch_norm2d_relu},
      {"quantized::mul", mul, quantized_mul},
      {"quantized::mul", inplace_mul, quantized_mul},
      {"quantized::mul_scalar_relu",
       mul_scalar_relu,
       quantized_mul_scalar_relu,
       mul_scalar_filter},
      {"quantized::mul_scalar_relu_out",
       mul_scalar_relu_out,
       quantized_mul_scalar_relu_out,
       mul_scalar_filter},
      {"quantized::mul_scalar",
       mul_scalar,
       quantized_mul_scalar,
       mul_scalar_filter},
      {"quantized::mul_scalar",
       mul_scalar_out,
       quantized_mul_scalar_out,
       mul_scalar_filter},
      {"quantized::mul_relu", mul_relu, quantized_mul_relu},
      {"quantized::mul_relu", mul_inplace_relu, quantized_mul_relu},
      {"quantized::mul_relu", inplace_mul_relu, quantized_mul_relu},
      {"quantized::mul_relu", inplace_mul_inplace_relu, quantized_mul_relu},
      {"quantized::hardswish", hardswish, quantized_hardswish},
      {"quantized::layer_norm", layer_norm, quantized_layer_norm},
      {"aten::avg_pool1d", avg_pool1d, aten_avg_pool1d},
      {"aten::avg_pool2d", avg_pool2d, aten_avg_pool2d},
      {"aten::avg_pool3d", avg_pool3d, aten_avg_pool3d},
      {"aten::adaptive_avg_pool1d",
       adaptive_avg_pool1d,
       aten_adaptive_avg_pool1d},
      {"aten::adaptive_avg_pool2d",
       adaptive_avg_pool2d,
       aten_adaptive_avg_pool2d},
      {"aten::adaptive_avg_pool3d",
       adaptive_avg_pool3d,
       aten_adaptive_avg_pool3d},
      {"aten::mean", mean, aten_mean},
      {"aten::upsample_nearest1d", upsample_nearest1d, aten_upsample_nearest1d},
      {"aten::upsample_nearest2d", upsample_nearest2d, aten_upsample_nearest2d},
      {"aten::upsample_nearest3d", upsample_nearest3d, aten_upsample_nearest3d},
      {"aten::upsample_linea1d", upsample_linear1d, aten_upsample_linear1d},
      {"aten::upsample_bilinear2d",
       upsample_bilinear2d,
       aten_upsample_bilinear2d},
      {"aten::upsample_trilinear3d",
       upsample_trilinear3d,
       aten_upsample_trilinear3d},
      {"aten::clamp", clamp, aten_clamp},
      {"aten::hardtanh", hardtanh, aten_hardtanh},
      {"aten::hardtanh_", hardtanh_, aten_hardtanh_},
  };
}

std::vector<QuantFusionInfo> dynamic_quant_fusion_pattern_and_replacements() {
  std::string linear_dynamic = R"(
graph(%packed_params, %a, %reduce_range, %a_dtype):
        %a_scale : float, %a_zero_point : int = aten::_choose_qparams_per_tensor(%a, %reduce_range)
        %a_quant = aten::quantize_per_tensor(%a, %a_scale, %a_zero_point, %a_dtype)
        %a_dequant = aten::dequantize(%a_quant)
        %w_quant : Tensor, %b : Tensor? = quantized::linear_unpack(%packed_params)
        %w_dequant = aten::dequantize(%w_quant)
        %r = aten::linear(%a_dequant, %w_dequant, %b)
        return (%r) )";

  std::string quantized_linear_dynamic = R"(
graph(%packed_params, %a, %reduce_range, %a_dtype):
        %r = quantized::linear_dynamic(%a, %packed_params)
        return (%r) )";
  return {
      {"quantized::linear_dynamic", linear_dynamic, quantized_linear_dynamic},
  };
}

} // namespace jit
} // namespace torch
