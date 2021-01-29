#pragma once

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/subgraph_matcher.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/quantization/helper.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>
#include <string>
#include <unordered_map>

namespace torch {
namespace jit {

struct QuantFusionInfo {
  std::string quantized_op_name;
  std::string pattern;
  std::string replacement;
  std::vector<MatchFilter> filters = {};
};

namespace {
std::string getExtraArgList(std::vector<std::string> extra_args) {
  return std::accumulate(
      extra_args.begin(),
      extra_args.end(),
      std::string(),
      [](std::string acc, const std::string& arg) { return acc + ", " + arg; });
}

// Get the pattern we want to replace the match with
std::string getAtenOpPattern(
    const std::string& graph_header,
    const std::string& op_name,
    const std::vector<std::string>& extra_op_args,
    bool scalar_args = false) {
  std::vector<std::string> _extra_op_args = extra_op_args;
  std::string aten_op_pattern = graph_header;
  if (scalar_args) {
    for (const auto& extra_arg : _extra_op_args) {
      aten_op_pattern += R"(
          )" +
          extra_arg + "_scalar = aten::item(" + extra_arg + ")";
    }

    for (size_t i = 0; i < _extra_op_args.size(); ++i) {
      _extra_op_args[i] = _extra_op_args[i] + "_scalar";
    }
  }
  const auto& extra_op_arg_list = getExtraArgList(_extra_op_args);
  aten_op_pattern += R"(
          %r = )";
  aten_op_pattern += op_name + "(" + "%a_quant" + extra_op_arg_list + ")";
  aten_op_pattern += R"(
          return (%r) )";
  return aten_op_pattern;
}

// generate ops for quantize pattern for a scalar value
std::string getQuantizeForScalar(const std::string& value) {
  // 6 is `torch.float` ScalarType, we are creating a float scalar
  // tensor from a scalar value
  std::string quantize_pattern = R"(
          )" +
      value + "_float_scalar_type : int = prim::Constant[value=6]()";
  quantize_pattern += R"(
          )" +
      value + "_none : None = prim::Constant()";
  quantize_pattern += R"(
          )" +
      value + "_tensor : Tensor = aten::scalar_tensor(" + value + ", " + value +
      "_float_scalar_type";
  for (auto i = 0; i < 3; ++i) {
    quantize_pattern += ", " + value + "_none";
  }
  quantize_pattern += ")";
  quantize_pattern +=
      R"(
          )" +
      value + "_quant = aten::quantize_per_tensor(" + value + "_tensor" +
      getExtraArgList(
          {value + "_scale", value + "_zero_point", value + "_dtype"}) +
      ")";
  return quantize_pattern;
}

std::string getDequantize(const std::string& value) {
  return R"(
          )" +
      value + "_dequant = aten::dequantize(" + value + "_quant)";
}

std::string getItem(const std::string& value) {
  return R"(
          )" +
      value + "_scalar : float = aten::item(" + value + "_dequant)";
}

// Patterns for the ops that inherit parameters from input
std::string getInputTensorQParamOpPattern(
    const std::string& op_name,
    const std::vector<std::string>& extra_op_args) {
  const auto& extra_op_arg_list = getExtraArgList(extra_op_args);
  std::string op_pattern = "graph(%a_quant" + extra_op_arg_list + "):" + R"(
          %a_dequant = aten::dequantize(%a_quant)
          %r = )" +
      op_name + "(" + "%a_dequant" + extra_op_arg_list + ")" + R"(
          %r_scale : float = aten::q_scale(%a_quant)
          %r_zero_point : int = aten::q_zero_point(%a_quant)
          %r_dtype : int = prim::dtype(%a_quant)
          %r_quant = aten::quantize_per_tensor(%r, %r_scale, %r_zero_point, %r_dtype)
          return (%r_quant) )";
  return op_pattern;
}

// QuantFusionInfo for the ops that inherit parameters from input
QuantFusionInfo getInputTensorQParamOpFusionInfo(
    const std::string& op_name,
    const std::vector<std::string>& extra_op_args) {
  std::string op_pattern =
      getInputTensorQParamOpPattern(op_name, extra_op_args);
  const auto& extra_op_arg_list = getExtraArgList(extra_op_args);
  std::string graph_header = "graph(%a_quant" + extra_op_arg_list + "):";
  std::string op_replacement =
      getAtenOpPattern(graph_header, op_name, extra_op_args);

  return {op_name, op_pattern, op_replacement};
}

// quant fusion for ops like `quantized::add_scalar`, `quantized::mul_scalar`
QuantFusionInfo getBinaryOpScalarFusionInfo(
    const std::string& op_name,
    const std::vector<std::string>& extra_op_args,
    const std::string& quantized_op_name,
    const std::vector<std::string>& extra_quantized_op_args,
    const std::vector<MatchFilter>& filters = {}) {
  std::string op_pattern =
      getInputTensorQParamOpPattern(op_name, extra_op_args);

  const auto& extra_op_arg_list = getExtraArgList(extra_op_args);
  std::string graph_header = "graph(%a_quant" + extra_op_arg_list + "):";
  std::string op_replacement = getAtenOpPattern(
      graph_header, quantized_op_name, extra_quantized_op_args);

  return {op_name, op_pattern, op_replacement, filters};
}

QuantFusionInfo getClampOpFusionInfo(
    const std::string& op_name,
    const std::vector<std::string>& extra_op_args) {
  std::vector<std::string> header_args = extra_op_args;
  std::vector<std::string> input_qparams = {"_scale", "_zero_point", "_dtype"};
  for (const auto& arg : extra_op_args) {
    for (const auto& qparam : input_qparams) {
      header_args.push_back(arg + qparam);
    }
  }
  for (const auto& qparam : input_qparams) {
    header_args.push_back("%r" + qparam);
  }
  const auto& extra_header_arg_list = getExtraArgList(header_args);
  std::string graph_header = "graph(%a_quant" + extra_header_arg_list + "):";
  std::string op_pattern = graph_header;
  for (const auto& arg : extra_op_args) {
    op_pattern += getQuantizeForScalar(arg);
    op_pattern += getDequantize(arg);
    op_pattern += getItem(arg);
  }
  op_pattern += getDequantize("%a");
  op_pattern += R"(
          %r = )";
  std::vector<std::string> scalar_extra_args;
  for (const auto& arg : extra_op_args) {
    scalar_extra_args.push_back(arg + "_scalar");
  }
  op_pattern +=
      op_name + "(" + "%a_dequant" + getExtraArgList(scalar_extra_args) + ")";
  // IR pattern common to all ops that inherit qparam from input
  op_pattern += R"(
          %r_quant = aten::quantize_per_tensor(%r, %r_scale, %r_zero_point, %r_dtype)
          return (%r_quant) )";

  std::string aten_op_pattern =
      getAtenOpPattern(graph_header, op_name, extra_op_args);

  return {op_name, op_pattern, aten_op_pattern};
}

// Patterns for the ops that has fixed quantization parameters
QuantFusionInfo getFixedQParamOpFusionInfo(
    const std::string& op_name,
    const std::vector<std::string>& extra_op_args,
    bool is_symmetric) {
  const auto& extra_op_arg_list = getExtraArgList(extra_op_args);
  std::string graph_header = "graph(%a_quant" + extra_op_arg_list + "):";
  std::string op_pattern = graph_header;
  op_pattern += R"(
          %a_dequant = aten::dequantize(%a_quant)
          %r = )";
  op_pattern += op_name + "(" + "%a_dequant" + extra_op_arg_list + ")";
  // IR pattern common to all ops with fixed quantization parameters for
  // asymetric quantization
  std::string asym_fixed_qparam_op_suffix = R"(
          %r_scale : float = prim::Constant[value=0.00390625]()
          %r_zero_point : int = prim::Constant[value=0]()
          %r_dtype : int = prim::Constant[value=13]()
          %r_quant = aten::quantize_per_tensor(%r, %r_scale, %r_zero_point, %r_dtype)
          return (%r_quant) )";

  std::string sym_fixed_qparam_op_suffix = R"(
          %r_scale : float = prim::Constant[value=0.0078125]()
          %r_zero_point : int = prim::Constant[value=128]()
          %r_dtype : int = prim::Constant[value=13]()
          %r_quant = aten::quantize_per_tensor(%r, %r_scale, %r_zero_point, %r_dtype)
          return (%r_quant) )";
  op_pattern +=
      is_symmetric ? sym_fixed_qparam_op_suffix : asym_fixed_qparam_op_suffix;

  std::string aten_op_pattern =
      getAtenOpPattern(graph_header, op_name, extra_op_args);

  return {op_name, op_pattern, aten_op_pattern};
}

// filter that checks %b_scalar is a scalar
bool input_b_is_scalar(
    const Match& match,
    const std::unordered_map<std::string, Value*>& vmap) {
  const auto& match_vmap = match.values_map;
  auto b_scalar = match_vmap.at(vmap.at("b_scalar"));
  return isScalar(b_scalar);
}

// Patterns for ops that require observation for output quantization parameters
// Example:
//
// before fusion:
//
// graph(%a_quant, %r_scale, %r_zero_point, %r_dtype):
//     %a_dequant = aten::dequantize(%a_quant)
//     %r = {op_name}(%a_dequant, {extra_args})
//     %r_quant = aten::quantize_per_tensor(%r, %r_scale, %r_zero_point,
//     %r_dtype) return (%r_quant)
//
// after fusion:
//
// graph(%a_quant, %r_scale, %r_zero_point, %r_dtype):
//     %r_quant = {quantized_op_name}(%a_quant, {extra_args}, %r_scale,
//     %r_zero_point) return (%r_quant)
QuantFusionInfo getObservedQParamOpFusionInfo(
    const std::string& fp_op_name,
    const std::string& q_op_name,
    const std::vector<std::string>& fp_extra_args,
    const std::vector<std::string>& q_extra_args) {
  const auto& fp_extra_arg_list = getExtraArgList(fp_extra_args);
  const auto& q_extra_arg_list = getExtraArgList(q_extra_args);

  std::string op_pattern = "graph(%a_quant" + fp_extra_arg_list +
      ", %r_scale, %r_zero_point, %r_dtype):" + R"(
          %a_dequant = aten::dequantize(%a_quant)
          %r = )" +
      fp_op_name + "(" + "%a_dequant" + fp_extra_arg_list + ")" + R"(
          %r_quant = aten::quantize_per_tensor(%r, %r_scale, %r_zero_point, %r_dtype)
          return (%r_quant) )";

  std::string aten_op_pattern = "graph(%a_quant" + fp_extra_arg_list +
      ", %r_scale, %r_zero_point, %r_dtype):" + R"(
          %r_quant = )" +
      q_op_name + "(%a_quant" + q_extra_arg_list +
      ", %r_scale, %r_zero_point)" + R"(
          return (%r_quant) )";

  return {q_op_name, op_pattern, aten_op_pattern};
}

} // namespace

std::vector<QuantFusionInfo> quant_fusion_pattern_and_replacements() {
  // aten::conv1d
  std::string conv1d = R"(
graph(%a_quant, %packed_params, %r_scale, %r_zero_point, %r_dtype, %stride, %padding, %dilation, %groups):
        %a_dequant = aten::dequantize(%a_quant)
        %w_quant : Tensor, %b : Tensor? = quantized::conv1d_unpack(%packed_params)
        %w_dequant = aten::dequantize(%w_quant)
        %r = aten::conv1d(%a_dequant, %w_dequant, %b, %stride, %padding, %dilation, %groups)
        %r_quant = aten::quantize_per_tensor(%r, %r_scale, %r_zero_point, %r_dtype)
        return (%r_quant) )";

  // aten::conv1d - aten::relu
  std::string conv1d_relu = R"(
graph(%a_quant, %packed_params, %r_scale, %r_zero_point, %r_dtype, %stride, %padding, %dilation, %groups):
        %a_dequant = aten::dequantize(%a_quant)
        %w_quant : Tensor, %b : Tensor? = quantized::conv1d_unpack(%packed_params)
        %w_dequant = aten::dequantize(%w_quant)
        %conv_out = aten::conv1d(%a_dequant, %w_dequant, %b, %stride, %padding, %dilation, %groups)
        %r = aten::relu(%conv_out)
        %r_quant = aten::quantize_per_tensor(%r, %r_scale, %r_zero_point, %r_dtype)
        return (%r_quant) )";

  // aten::conv1d - aten::relu_
  std::string conv1d_inplace_relu = R"(
graph(%a_quant, %packed_params, %r_scale, %r_zero_point, %r_dtype, %stride, %padding, %dilation, %groups):
        %a_dequant = aten::dequantize(%a_quant)
        %w_quant : Tensor, %b : Tensor? = quantized::conv1d_unpack(%packed_params)
        %w_dequant = aten::dequantize(%w_quant)
        %conv_out = aten::conv1d(%a_dequant, %w_dequant, %b, %stride, %padding, %dilation, %groups)
        %r = aten::relu_(%conv_out)
        %r_quant = aten::quantize_per_tensor(%r, %r_scale, %r_zero_point, %r_dtype)
        return (%r_quant) )";

  // quantized::conv1d
  std::string quantized_conv1d = R"(
graph(%a_quant, %packed_params, %r_scale, %r_zero_point, %r_dtype, %stride, %padding, %dilation, %groups):
        %r_quant = quantized::conv1d(%a_quant, %packed_params, %r_scale, %r_zero_point)
        return (%r_quant) )";

  // quantized::conv1d_relu
  std::string quantized_conv1d_relu = R"(
graph(%a_quant, %packed_params, %r_scale, %r_zero_point, %r_dtype, %stride, %padding, %dilation, %groups):
        %r_quant = quantized::conv1d_relu(%a_quant, %packed_params, %r_scale, %r_zero_point)
        return (%r_quant) )";

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

  // aten::conv_transpose1d
  std::string conv_transpose1d = R"(
graph(%a_quant, %packed_params, %r_scale, %r_zero_point, %r_dtype, %stride, %padding, %output_padding, %groups, %dilation):
        %a_dequant = aten::dequantize(%a_quant)
        %w_quant : Tensor, %b : Tensor? = quantized::conv_transpose1d_unpack(%packed_params)
        %w_dequant = aten::dequantize(%w_quant)
        %r = aten::conv_transpose1d(%a_dequant, %w_dequant, %b, %stride, %padding, %output_padding, %groups, %dilation)
        %r_quant = aten::quantize_per_tensor(%r, %r_scale, %r_zero_point, %r_dtype)
        return (%r_quant) )";

  // quantized::conv_transpose1d
  std::string quantized_conv_transpose1d = R"(
graph(%a_quant, %packed_params, %r_scale, %r_zero_point, %r_dtype, %stride, %padding, %output_padding, %groups, %dilation):
        %r_quant = quantized::conv_transpose1d(%a_quant, %packed_params, %r_scale, %r_zero_point)
        return (%r_quant) )";

  // aten::conv_transpose2d
  std::string conv_transpose2d = R"(
graph(%a_quant, %packed_params, %r_scale, %r_zero_point, %r_dtype, %stride, %padding, %output_padding, %groups, %dilation):
        %a_dequant = aten::dequantize(%a_quant)
        %w_quant : Tensor, %b : Tensor? = quantized::conv_transpose2d_unpack(%packed_params)
        %w_dequant = aten::dequantize(%w_quant)
        %r = aten::conv_transpose2d(%a_dequant, %w_dequant, %b, %stride, %padding, %output_padding, %groups, %dilation)
        %r_quant = aten::quantize_per_tensor(%r, %r_scale, %r_zero_point, %r_dtype)
        return (%r_quant) )";

  // quantized::conv_transpose1d
  std::string quantized_conv_transpose2d = R"(
graph(%a_quant, %packed_params, %r_scale, %r_zero_point, %r_dtype, %stride, %padding, %output_padding, %groups, %dilation):
        %r_quant = quantized::conv_transpose2d(%a_quant, %packed_params, %r_scale, %r_zero_point)
        return (%r_quant) )";

  std::string add_relu = R"(
graph(%a_quant, %b_quant, %alpha, %scale, %zero_point, %dtype):
         %a_dequant = aten::dequantize(%a_quant)
         %b_dequant = aten::dequantize(%b_quant)
         %r_add = aten::add(%a_dequant, %b_dequant, %alpha)
         %r_relu = aten::relu(%r_add)
         %r = aten::quantize_per_tensor(%r_relu, %scale, %zero_point, %dtype)
         return (%r) )";

  std::string add_inplace_relu = R"(
graph(%a_quant, %b_quant, %alpha, %scale, %zero_point, %dtype):
         %a_dequant = aten::dequantize(%a_quant)
         %b_dequant = aten::dequantize(%b_quant)
         %r_add = aten::add(%a_dequant, %b_dequant, %alpha)
         %r_relu = aten::relu_(%r_add)
         %r = aten::quantize_per_tensor(%r_relu, %scale, %zero_point, %dtype)
         return (%r) )";

  std::string inplace_add_relu = R"(
graph(%a_quant, %b_quant, %alpha, %scale, %zero_point, %dtype):
         %a_dequant = aten::dequantize(%a_quant)
         %b_dequant = aten::dequantize(%b_quant)
         %r_add = aten::add_(%a_dequant, %b_dequant, %alpha)
         %r_relu = aten::relu(%r_add)
         %r = aten::quantize_per_tensor(%r_relu, %scale, %zero_point, %dtype)
         return (%r) )";

  std::string inplace_add_inplace_relu = R"(
graph(%a_quant, %b_quant, %alpha, %scale, %zero_point, %dtype):
         %a_dequant = aten::dequantize(%a_quant)
         %b_dequant = aten::dequantize(%b_quant)
         %r_add = aten::add_(%a_dequant, %b_dequant, %alpha)
         %r_relu = aten::relu_(%r_add)
         %r = aten::quantize_per_tensor(%r_relu, %scale, %zero_point, %dtype)
         return (%r) )";

  std::string quantized_add_relu = R"(
graph(%a_quant, %b_quant, %alpha, %scale, %zero_point, %dtype):
         %r = quantized::add_relu(%a_quant, %b_quant, %scale, %zero_point)
         return (%r) )";

  // aten::linear
  std::string linear = R"(
graph(%a_quant, %packed_params, %r_scale, %r_zero_point, %r_dtype):
        %a_dequant = aten::dequantize(%a_quant)
        %w_quant : Tensor, %b : Tensor? = quantized::linear_unpack(%packed_params)
        %w_dequant = aten::dequantize(%w_quant)
        %r = aten::linear(%a_dequant, %w_dequant, %b)
        %r_quant = aten::quantize_per_tensor(%r, %r_scale, %r_zero_point, %r_dtype)
        return (%r_quant) )";

  std::string linear_relu = R"(
graph(%a_quant, %packed_params, %r_scale, %r_zero_point, %r_dtype):
        %a_dequant = aten::dequantize(%a_quant)
        %w_quant : Tensor, %b : Tensor? = quantized::linear_unpack(%packed_params)
        %w_dequant = aten::dequantize(%w_quant)
        %linear_out = aten::linear(%a_dequant, %w_dequant, %b)
        %r = aten::relu(%linear_out)
        %r_quant = aten::quantize_per_tensor(%r, %r_scale, %r_zero_point, %r_dtype)
        return (%r_quant) )";

  std::string linear_inplace_relu = R"(
graph(%a_quant, %packed_params, %r_scale, %r_zero_point, %r_dtype):
        %a_dequant = aten::dequantize(%a_quant)
        %w_quant : Tensor, %b : Tensor? = quantized::linear_unpack(%packed_params)
        %w_dequant = aten::dequantize(%w_quant)
        %linear_out = aten::linear(%a_dequant, %w_dequant, %b)
        %r = aten::relu_(%linear_out)
        %r_quant = aten::quantize_per_tensor(%r, %r_scale, %r_zero_point, %r_dtype)
        return (%r_quant) )";

  // quantized::linear
  std::string quantized_linear = R"(
graph(%a_quant, %packed_params, %r_scale, %r_zero_point, %r_dtype):
        %r = quantized::linear(%a_quant, %packed_params, %r_scale, %r_zero_point)
        return (%r) )";

  std::string quantized_linear_relu = R"(
graph(%a_quant, %packed_params, %r_scale, %r_zero_point, %r_dtype):
        %r = quantized::linear_relu(%a_quant, %packed_params, %r_scale, %r_zero_point)
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

  // aten::add_
  std::string inplace_add = R"(
graph(%a_quant, %b_quant, %alpha, %scale, %zero_point, %dtype):
         %a_dequant = aten::dequantize(%a_quant)
         %b_dequant = aten::dequantize(%b_quant)
         %r_add = aten::add_(%a_dequant, %b_dequant, %alpha)
         %r = aten::quantize_per_tensor(%r_add, %scale, %zero_point, %dtype)
         return (%r) )";

  auto add_scalar = getBinaryOpScalarFusionInfo(
      "aten::add",
      {"%b_scalar", "%alpha"},
      "quantized::add_scalar",
      {"%b_scalar"},
      {aten_add_alpha_is_one, input_b_is_scalar});

  auto add_scalar_out = getBinaryOpScalarFusionInfo(
      "aten::add_",
      {"%b_scalar", "%alpha"},
      "quantized::add_scalar_out",
      {"%b_scalar", "%a_quant"},
      {aten_add_alpha_is_one, input_b_is_scalar});

  // quantized::add_scalar_relu -- fusing quantized::add_scalar
  // and aten::relu
  auto quantized_add_scalar_relu_pattern = R"(
graph(%a_quant, %b_scalar):
         %r_add = quantized::add_scalar(%a_quant, %b_scalar)
         %r = aten::relu(%r_add)
         return (%r) )";

  auto quantized_add_scalar_inplace_relu_pattern = R"(
graph(%a_quant, %b_scalar):
         %r_add = quantized::add_scalar(%a_quant, %b_scalar)
         %r = aten::relu_(%r_add)
         return (%r) )";

  auto quantized_add_scalar_relu_replacement = R"(
graph(%a_quant, %b_scalar):
         %r = quantized::add_scalar_relu(%a_quant, %b_scalar)
         return (%r) )";

  // quantized::add_scalar_relu_out -- fusing quantized::add_scalarOut
  // and aten::relu
  auto quantized_add_scalar_relu_out_pattern = R"(
graph(%a_quant, %b_scalar):
         %r_add = quantized::add_scalar_out(%a_quant, %b_scalar, %a_quant)
         %r = aten::relu(%r_add)
         return (%r) )";

  auto quantized_add_scalar_inplace_relu_out_pattern = R"(
graph(%a_quant, %b_scalar):
         %r_add = quantized::add_scalar_out(%a_quant, %b_scalar, %a_quant)
         %r = aten::relu_(%r_add)
         return (%r) )";

  auto quantized_add_scalar_relu_out_replacement = R"(
graph(%a_quant, %b_scalar):
         %r = quantized::add_scalar_relu_out(%a_quant, %b_scalar, %a_quant)
         return (%r) )";

  // quantized::batch_norm
  std::string batch_norm = R"(
graph(%a_quant, %weight, %bias, %mean, %var, %training, %eaf, %eps, %7, %scale, %zero_point, %scalar_type):
         %a_dequant = aten::dequantize(%a_quant)
         %r_bn = aten::batch_norm(%a_dequant, %weight, %bias, %mean, %var, %training, %eaf, %eps, %7)
         %r = aten::quantize_per_tensor(%r_bn, %scale, %zero_point, %scalar_type)
         return (%r) )";
  std::string quantized_batch_norm = R"(
graph(%a_quant, %weight, %bias, %mean, %var, %training, %eaf, %eps, %7, %scale, %zero_point, %scalar_type):
         %r = quantized::batch_norm(%a_quant, %weight, %bias, %mean, %var, %eps, %scale, %zero_point)
         return (%r) )";

  std::string batch_norm_relu = R"(
graph(%a_quant, %weight, %bias, %mean, %var, %training, %eaf, %eps, %7, %scale, %zero_point, %scalar_type):
         %a_dequant = aten::dequantize(%a_quant)
         %bn_out = aten::batch_norm(%a_dequant, %weight, %bias, %mean, %var, %training, %eaf, %eps, %7)
         %relu = aten::relu(%bn_out)
         %r = aten::quantize_per_tensor(%relu, %scale, %zero_point, %scalar_type)
         return (%r) )";
  std::string batch_norm_inplace_relu = R"(
graph(%a_quant, %weight, %bias, %mean, %var, %training, %eaf, %eps, %7, %scale, %zero_point, %scalar_type):
         %a_dequant = aten::dequantize(%a_quant)
         %bn_out = aten::batch_norm(%a_dequant, %weight, %bias, %mean, %var, %training, %eaf, %eps, %7)
         %relu = aten::relu_(%bn_out)
         %r = aten::quantize_per_tensor(%relu, %scale, %zero_point, %scalar_type)
         return (%r) )";

  std::string quantized_batch_norm_relu = R"(
graph(%a_quant, %weight, %bias, %mean, %var, %training, %eaf, %eps, %7, %scale, %zero_point, %scalar_type):
         %r = quantized::batch_norm_relu(%a_quant, %weight, %bias, %mean, %var, %eps, %scale, %zero_point)
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

  auto mul_scalar = getBinaryOpScalarFusionInfo(
      "aten::mul",
      {"%b_scalar"},
      "quantized::mul_scalar",
      {"%b_scalar"},
      {input_b_is_scalar});

  auto mul_scalar_out = getBinaryOpScalarFusionInfo(
      "aten::mul_",
      {"%b_scalar"},
      "quantized::mul_scalar_out",
      {"%b_scalar", "%a_quant"},
      {input_b_is_scalar});

  // quantized::mul_relu
  std::string mul_relu = R"(
graph(%a_quant, %b_quant, %scale, %zero_point, %dtype):
         %a_dequant = aten::dequantize(%a_quant)
         %b_dequant = aten::dequantize(%b_quant)
         %r_mul = aten::mul(%a_dequant, %b_dequant)
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

  std::string inplace_mul_relu = R"(
graph(%a_quant, %b_quant, %scale, %zero_point, %dtype):
         %a_dequant = aten::dequantize(%a_quant)
         %b_dequant = aten::dequantize(%b_quant)
         %r_mul = aten::mul_(%a_dequant, %b_dequant)
         %r_relu = aten::relu(%r_mul)
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

  // quantized::mul_scalar_relu -- fusing quantized::mul_scalar
  // and aten::relu
  auto quantized_mul_scalar_relu_pattern = R"(
graph(%a_quant, %b_scalar):
         %r_mul = quantized::mul_scalar(%a_quant, %b_scalar)
         %r = aten::relu(%r_mul)
         return (%r) )";

  auto quantized_mul_scalar_inplace_relu_pattern = R"(
graph(%a_quant, %b_scalar):
         %r_mul = quantized::mul_scalar(%a_quant, %b_scalar)
         %r = aten::relu_(%r_mul)
         return (%r) )";

  auto quantized_mul_scalar_relu_replacement = R"(
graph(%a_quant, %b_scalar):
         %r = quantized::mul_scalar_relu(%a_quant, %b_scalar)
         return (%r) )";

  // quantized::mul_scalar_relu_out -- fusing quantized::mul_scalarOut
  // and aten::relu
  auto quantized_mul_scalar_relu_out_pattern = R"(
graph(%a_quant, %b_scalar):
         %r_mul = quantized::mul_scalar_out(%a_quant, %b_scalar, %a_quant)
         %r = aten::relu(%r_mul)
         return (%r) )";

  auto quantized_mul_scalar_inplace_relu_out_pattern = R"(
graph(%a_quant, %b_scalar):
         %r_mul = quantized::mul_scalar_out(%a_quant, %b_scalar, %a_quant)
         %r = aten::relu_(%r_mul)
         return (%r) )";

  auto quantized_mul_scalar_relu_out_replacement = R"(
graph(%a_quant, %b_scalar):
         %r = quantized::mul_scalar_relu_out(%a_quant, %b_scalar, %a_quant)
         return (%r) )";

  // quantized::elu
  std::string elu = R"(
graph(%a_quant, %alpha, %scale, %input_scale, %r_scale, %r_zero_point, %r_dtype):
         %a_dequant = aten::dequantize(%a_quant)
         %r = aten::elu(%a_dequant, %alpha, %scale, %input_scale)
         %r_quant = aten::quantize_per_tensor(%r, %r_scale, %r_zero_point, %r_dtype)
         return (%r_quant) )";

  std::string quantized_elu = R"(
graph(%a_quant, %alpha, %scale, %input_scale, %r_scale, %r_zero_point, %r_dtype):
         %r_quant = quantized::elu(%a_quant, %r_scale, %r_zero_point, %alpha, %scale, %input_scale)
         return (%r_quant) )";

  std::string elu_ = R"(
graph(%a_quant, %alpha, %scale, %input_scale, %r_scale, %r_zero_point, %r_dtype):
         %a_dequant = aten::dequantize(%a_quant)
         %r = aten::elu_(%a_dequant, %alpha, %scale, %input_scale)
         %r_quant = aten::quantize_per_tensor(%r, %r_scale, %r_zero_point, %r_dtype)
         return (%r_quant) )";

  // ============= General Ops that inherit quantization paramters from input
  // tensor =============
  auto avg_pool1d = getInputTensorQParamOpFusionInfo(
      "aten::avg_pool1d",
      {"%kernel_size",
       "%stride",
       "%padding",
       "%ceil_mode",
       "%count_include_pad"});

  auto avg_pool2d = getInputTensorQParamOpFusionInfo(
      "aten::avg_pool2d",
      {"%kernel_size",
       "%stride",
       "%padding",
       "%ceil_mode",
       "%count_include_pad",
       "%divisor_override"});

  std::string common_general_value_op = R"(
          %r_scale : float = aten::q_scale(%a_quant)
          %r_zero_point : int = aten::q_zero_point(%a_quant)
          %r_dtype : int = prim::dtype(%a_quant)
          %r_quant = aten::quantize_per_tensor(%r, %r_scale, %r_zero_point, %r_dtype)
          return (%r_quant) )";

  auto avg_pool3d = getInputTensorQParamOpFusionInfo(
      "aten::avg_pool3d",
      {"%kernel_size",
       "%stride",
       "%padding",
       "%ceil_mode",
       "%count_include_pad",
       "%divisor_override"});

  auto adaptive_avg_pool1d = getInputTensorQParamOpFusionInfo(
      "aten::adaptive_avg_pool1d", {"%output_size"});

  auto adaptive_avg_pool2d = getInputTensorQParamOpFusionInfo(
      "aten::adaptive_avg_pool2d", {"%output_size"});

  auto adaptive_avg_pool3d = getInputTensorQParamOpFusionInfo(
      "aten::adaptive_avg_pool3d", {"%output_size"});

  auto mean1 = getInputTensorQParamOpFusionInfo("aten::mean", {"%dim"});

  auto mean2 = getInputTensorQParamOpFusionInfo(
      "aten::mean", {"%dim", "%keepdim", "%out"});

  auto upsample_nearest1d_vec = getInputTensorQParamOpFusionInfo(
      "aten::upsample_nearest1d", {"%output_size", "%scale_factors"});

  auto upsample_nearest2d_vec = getInputTensorQParamOpFusionInfo(
      "aten::upsample_nearest2d", {"%output_size", "%scale_factors"});

  auto upsample_nearest3d_vec = getInputTensorQParamOpFusionInfo(
      "aten::upsample_nearest3d", {"%output_size", "%scale_factors"});

  auto upsample_linear1d_vec = getInputTensorQParamOpFusionInfo(
      "aten::upsample_linear1d",
      {"%output_size", "%align_corners", "%scale_factors"});

  auto upsample_bilinear2d_vec = getInputTensorQParamOpFusionInfo(
      "aten::upsample_bilinear2d",
      {"%output_size", "%align_corners", "%scale_factors"});

  auto upsample_trilinear3d_vec = getInputTensorQParamOpFusionInfo(
      "aten::upsample_trilinear3d",
      {"%output_size", "%align_corners", "%scale_factors"});

  auto upsample_nearest1d = getInputTensorQParamOpFusionInfo(
      "aten::upsample_nearest1d", {"%output_size", "%scales"});

  auto upsample_nearest2d = getInputTensorQParamOpFusionInfo(
      "aten::upsample_nearest2d", {"%output_size", "%scale_h", "%scale_w"});

  auto upsample_nearest3d = getInputTensorQParamOpFusionInfo(
      "aten::upsample_nearest3d",
      {"%output_size", "%scale_d", "%scale_h", "%scale_w"});

  auto upsample_linear1d = getInputTensorQParamOpFusionInfo(
      "aten::upsample_linear1d", {"%output_size", "%align_corners", "%scales"});

  auto upsample_bilinear2d = getInputTensorQParamOpFusionInfo(
      "aten::upsample_bilinear2d",
      {"%output_size", "%align_corners", "%scale_h", "%scale_w"});

  auto upsample_trilinear3d = getInputTensorQParamOpFusionInfo(
      "aten::upsample_trilinear3d",
      {"%output_size", "%align_corners", "%scale_d", "%scale_h", "%scale_w"});

  auto clamp = getClampOpFusionInfo("aten::clamp", {"%min", "%max"});

  auto hardtanh = getClampOpFusionInfo("aten::hardtanh", {"%min", "%max"});

  auto hardtanh_ = getClampOpFusionInfo("aten::hardtanh_", {"%min", "%max"});

  auto leaky_relu =
      getInputTensorQParamOpFusionInfo("aten::leaky_relu", {"%negative_slope"});

  auto leaky_relu_ = getInputTensorQParamOpFusionInfo(
      "aten::leaky_relu_", {"%negative_slope"});

  // Ops with fixed quantization parameters
  auto hardsigmoid = getFixedQParamOpFusionInfo("aten::hardsigmoid", {}, false);

  auto hardsigmoid_ =
      getFixedQParamOpFusionInfo("aten::hardsigmoid_", {}, false);

  auto sigmoid = getFixedQParamOpFusionInfo("aten::sigmoid", {}, false);

  auto sigmoid_ = getFixedQParamOpFusionInfo("aten::sigmoid_", {}, false);

  auto tanh = getFixedQParamOpFusionInfo("aten::tanh", {}, true);

  auto tanh_ = getFixedQParamOpFusionInfo("aten::tanh_", {}, true);

  auto hardswish = getObservedQParamOpFusionInfo(
      "aten::hardswish", "quantized::hardswish", {}, {});

  auto hardswish_ = getObservedQParamOpFusionInfo(
      "aten::hardswish_", "quantized::hardswish", {}, {});

  auto layer_norm = getObservedQParamOpFusionInfo(
      "aten::layer_norm",
      "quantized::layer_norm",
      {"%normalized_shape", "%weight", "%bias", "%eps", "%cudnn_enabled"},
      {"%normalized_shape", "%weight", "%bias", "%eps"});

  auto group_norm = getObservedQParamOpFusionInfo(
      "aten::group_norm",
      "quantized::group_norm",
      {"%num_groups", "%weight", "%bias", "%eps", "%cudnn_enabled"},
      {"%num_groups", "%weight", "%bias", "%eps"});

  auto instance_norm = getObservedQParamOpFusionInfo(
      "aten::instance_norm",
      "quantized::instance_norm",
      {"%weight",
       "%bias",
       "%running_mean",
       "%running_var",
       "%use_input_stats",
       "%momentum",
       "%eps",
       "%cudnn_enabled"},
      {"%weight", "%bias", "%eps"});

  return {
      {"quantized::conv1d", conv1d, quantized_conv1d},
      {"quantized::conv1d_relu", conv1d_relu, quantized_conv1d_relu},
      {"quantized::conv1d_relu", conv1d_inplace_relu, quantized_conv1d_relu},
      {"quantized::conv2d", conv2d, quantized_conv2d},
      {"quantized::conv2d_relu", conv2d_relu, quantized_conv2d_relu},
      {"quantized::conv2d_relu", conv2d_inplace_relu, quantized_conv2d_relu},
      {"quantized::conv3d", conv3d, quantized_conv3d},
      {"quantized::conv3d_relu", conv3d_relu, quantized_conv3d_relu},
      {"quantized::conv3d_relu", conv3d_inplace_relu, quantized_conv3d_relu},
      {"quantized::conv_transpose1d",
       conv_transpose1d,
       quantized_conv_transpose1d},
      {"quantized::conv_transpose2d",
       conv_transpose2d,
       quantized_conv_transpose2d},
      {"quantized::linear", linear, quantized_linear},
      {"quantized::linear_relu", linear_relu, quantized_linear_relu},
      {"quantized::linear_relu", linear_inplace_relu, quantized_linear_relu},
      {"quantized::add_relu",
       add_relu,
       quantized_add_relu,
       {aten_add_alpha_is_one}},
      {"quantized::add_relu",
       add_inplace_relu,
       quantized_add_relu,
       {aten_add_alpha_is_one}},
      {"quantized::add_relu",
       inplace_add_relu,
       quantized_add_relu,
       {aten_add_alpha_is_one}},
      {"quantized::add_relu",
       inplace_add_inplace_relu,
       quantized_add_relu,
       {aten_add_alpha_is_one}},
      add_scalar,
      add_scalar_out,
      // note that these must come after quantized::add_scalar and
      // quantized::add_scalar_out patterns
      {"quantized::add_scalar_relu",
       quantized_add_scalar_relu_pattern,
       quantized_add_scalar_relu_replacement},
      {"quantized::add_scalar_relu",
       quantized_add_scalar_inplace_relu_pattern,
       quantized_add_scalar_relu_replacement},
      {"quantized::add_scalar_relu_out",
       quantized_add_scalar_relu_out_pattern,
       quantized_add_scalar_relu_out_replacement},
      {"quantized::add_scalar_relu_out",
       quantized_add_scalar_inplace_relu_out_pattern,
       quantized_add_scalar_relu_out_replacement},
      {"quantized::add", add, quantized_add, {aten_add_alpha_is_one}},
      {"quantized::add", inplace_add, quantized_add, {aten_add_alpha_is_one}},
      {"quantized::cat", cat, quantized_cat},
      {"quantized::batch_norm", batch_norm, quantized_batch_norm},
      {"quantized::batch_norm_relu",
       batch_norm_relu,
       quantized_batch_norm_relu},
      {"quantized::batch_norm_relu",
       batch_norm_inplace_relu,
       quantized_batch_norm_relu},
      mul_scalar,
      mul_scalar_out,
      // note that these must come after quantized::mul_scalar and
      // quantized::mul_scalar_out patterns
      {"quantized::mul_scalar_relu",
       quantized_mul_scalar_relu_pattern,
       quantized_mul_scalar_relu_replacement},
      {"quantized::mul_scalar_relu",
       quantized_mul_scalar_inplace_relu_pattern,
       quantized_mul_scalar_relu_replacement},
      {"quantized::mul_scalar_relu_out",
       quantized_mul_scalar_relu_out_pattern,
       quantized_mul_scalar_relu_out_replacement},
      {"quantized::mul_scalar_relu_out",
       quantized_mul_scalar_inplace_relu_out_pattern,
       quantized_mul_scalar_relu_out_replacement},
      {"quantized::mul_relu", mul_relu, quantized_mul_relu},
      {"quantized::mul_relu", mul_inplace_relu, quantized_mul_relu},
      {"quantized::mul_relu", inplace_mul_relu, quantized_mul_relu},
      {"quantized::mul_relu", inplace_mul_inplace_relu, quantized_mul_relu},
      {"quantized::mul", mul, quantized_mul},
      {"quantized::mul", inplace_mul, quantized_mul},
      hardswish,
      hardswish_,
      layer_norm,
      group_norm,
      instance_norm,
      {"quantized::elu", elu, quantized_elu},
      {"quantized::elu_", elu_, quantized_elu},
      avg_pool1d,
      avg_pool2d,
      avg_pool3d,
      adaptive_avg_pool1d,
      adaptive_avg_pool2d,
      adaptive_avg_pool3d,
      mean1,
      mean2,
      upsample_nearest1d,
      upsample_nearest2d,
      upsample_nearest3d,
      upsample_linear1d,
      upsample_bilinear2d,
      upsample_trilinear3d,
      upsample_nearest1d_vec,
      upsample_nearest2d_vec,
      upsample_nearest3d_vec,
      upsample_linear1d_vec,
      upsample_bilinear2d_vec,
      upsample_trilinear3d_vec,
      clamp,
      hardtanh,
      hardtanh_,
      leaky_relu,
      leaky_relu_,
      // fixed qparam ops
      hardsigmoid,
      hardsigmoid_,
      sigmoid,
      sigmoid_,
      tanh,
      tanh_,
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
        %r = quantized::linear_dynamic(%a, %packed_params, %reduce_range)
        return (%r) )";

  std::string linear_dynamic_fp16 = R"(
graph(%packed_params, %a):
        %w_unpacked : Tensor, %b : Tensor? = quantized::linear_unpack_fp16(%packed_params)
        %r = aten::linear(%a, %w_unpacked, %b)
        return (%r) )";

  std::string quantized_linear_dynamic_fp16 = R"(
graph(%packed_params, %a):
        %r = quantized::linear_dynamic_fp16(%a, %packed_params)
        return (%r) )";

  return {
      {"quantized::linear_dynamic", linear_dynamic, quantized_linear_dynamic},
      {"quantized::linear_dynamic_fp16",
       linear_dynamic_fp16,
       quantized_linear_dynamic_fp16},
  };
}

std::vector<QuantFusionInfo> linear_prepack_unpack_patterns() {
  std::string linear_with_quant = R"(
graph(%a_dequant, %w_quant, %b):
        %w_dequant = aten::dequantize(%w_quant)
        %r = aten::linear(%a_dequant, %w_dequant, %b)
        return (%r) )";

  std::string linear_with_quant_prepack = R"(
graph(%a_dequant, %w_quant, %b):
        %packed_params = quantized::linear_prepack(%w_quant, %b)
        %w_quant_unpacked : Tensor, %b_unpacked : Tensor? = quantized::linear_unpack(%packed_params)
        %w_dequant = aten::dequantize(%w_quant_unpacked)
        %r = aten::linear(%a_dequant, %w_dequant, %b_unpacked)
        return (%r) )";
  std::string linear_fp16_with_cast = R"(
graph(%w, %a_dq, %b):
        %fp16_tensor = aten::_saturate_weight_to_fp16(%w)
        %r = aten::linear(%a_dq, %fp16_tensor, %b)
        return (%r) )";
  std::string linear_fp16_with_prepack = R"(
graph(%w, %a_dq, %b):
        %packed_params = quantized::linear_prepack_fp16(%w, %b)
        %w_unpacked : Tensor, %b_unpacked : Tensor? = quantized::linear_unpack_fp16(%packed_params)
        %r = aten::linear(%a_dq, %w_unpacked, %b_unpacked)
        return (%r) )";

  return {
      {"linear_prepack_unpack", linear_with_quant, linear_with_quant_prepack},
      {"linear_fp16_prepack_unpack",
       linear_fp16_with_cast,
       linear_fp16_with_prepack},
  };
}

std::vector<QuantFusionInfo> conv_prepack_unpack_patterns() {
  std::string conv1d_with_quant = R"(
graph(%a_dequant, %w_quant, %b, %stride, %padding, %dilation, %groups):
        %w_dequant = aten::dequantize(%w_quant)
        %r = aten::conv1d(%a_dequant, %w_dequant, %b, %stride, %padding, %dilation, %groups)
        return (%r) )";

  std::string conv1d_with_quant_prepack = R"(
graph(%a_dequant, %w_quant, %b, %stride, %padding, %dilation, %groups):
        %packed_params : __torch__.torch.classes.quantized.Conv2dPackedParamsBase = quantized::conv1d_prepack(%w_quant, %b, %stride, %padding, %dilation, %groups)
        %w_quant_unpacked : Tensor, %b_unpacked : Tensor? = quantized::conv1d_unpack(%packed_params)
        %w_dequant = aten::dequantize(%w_quant_unpacked)
        %r = aten::conv1d(%a_dequant, %w_dequant, %b_unpacked, %stride, %padding, %dilation, %groups)
        return (%r) )";

  std::string conv2d_with_quant = R"(
graph(%a_dequant, %w_quant, %b, %stride, %padding, %dilation, %groups):
        %w_dequant = aten::dequantize(%w_quant)
        %r = aten::conv2d(%a_dequant, %w_dequant, %b, %stride, %padding, %dilation, %groups)
        return (%r) )";

  std::string conv2d_with_quant_prepack = R"(
graph(%a_dequant, %w_quant, %b, %stride, %padding, %dilation, %groups):
        %packed_params : __torch__.torch.classes.quantized.Conv2dPackedParamsBase = quantized::conv2d_prepack(%w_quant, %b, %stride, %padding, %dilation, %groups)
        %w_quant_unpacked : Tensor, %b_unpacked : Tensor? = quantized::conv2d_unpack(%packed_params)
        %w_dequant = aten::dequantize(%w_quant_unpacked)
        %r = aten::conv2d(%a_dequant, %w_dequant, %b_unpacked, %stride, %padding, %dilation, %groups)
        return (%r) )";

  std::string conv3d_with_quant = R"(
graph(%a_dequant, %w_quant, %b, %stride, %padding, %dilation, %groups):
        %w_dequant = aten::dequantize(%w_quant)
        %r = aten::conv3d(%a_dequant, %w_dequant, %b, %stride, %padding, %dilation, %groups)
        return (%r) )";

  std::string conv3d_with_quant_prepack = R"(
graph(%a_dequant, %w_quant, %b, %stride, %padding, %dilation, %groups):
        %packed_params : __torch__.torch.classes.quantized.Conv3dPackedParamsBase = quantized::conv3d_prepack(%w_quant, %b, %stride, %padding, %dilation, %groups)
        %w_quant_unpacked : Tensor, %b_unpacked : Tensor? = quantized::conv3d_unpack(%packed_params)
        %w_dequant = aten::dequantize(%w_quant_unpacked)
        %r = aten::conv3d(%a_dequant, %w_dequant, %b_unpacked, %stride, %padding, %dilation, %groups)
        return (%r) )";

  std::string conv_transpose1d_with_quant = R"(
graph(%a_dequant, %w_quant, %b, %stride, %padding, %output_padding, %groups, %dilation):
        %w_dequant = aten::dequantize(%w_quant)
        %r = aten::conv_transpose1d(%a_dequant, %w_dequant, %b, %stride, %padding, %output_padding, %groups, %dilation)
        return (%r) )";

  std::string conv_transpose1d_with_quant_prepack = R"(
graph(%a_dequant, %w_quant, %b, %stride, %padding, %output_padding, %groups, %dilation):
        %packed_params : __torch__.torch.classes.quantized.Conv2dPackedParamsBase = quantized::conv_transpose1d_prepack(%w_quant, %b, %stride, %padding, %output_padding, %dilation, %groups)
        %w_quant_unpacked : Tensor, %b_unpacked : Tensor? = quantized::conv_transpose1d_unpack(%packed_params)
        %w_dequant = aten::dequantize(%w_quant_unpacked)
        %r = aten::conv_transpose1d(%a_dequant, %w_dequant, %b_unpacked, %stride, %padding, %output_padding, %groups, %dilation)
        return (%r) )";

  std::string conv_transpose2d_with_quant = R"(
graph(%a_dequant, %w_quant, %b, %stride, %padding, %output_padding, %groups, %dilation):
        %w_dequant = aten::dequantize(%w_quant)
        %r = aten::conv_transpose2d(%a_dequant, %w_dequant, %b, %stride, %padding, %output_padding, %groups, %dilation)
        return (%r) )";

  std::string conv_transpose2d_with_quant_prepack = R"(
graph(%a_dequant, %w_quant, %b, %stride, %padding, %output_padding, %groups, %dilation):
        %packed_params : __torch__.torch.classes.quantized.Conv2dPackedParamsBase = quantized::conv_transpose2d_prepack(%w_quant, %b, %stride, %padding, %output_padding, %dilation, %groups)
        %w_quant_unpacked : Tensor, %b_unpacked : Tensor? = quantized::conv_transpose2d_unpack(%packed_params)
        %w_dequant = aten::dequantize(%w_quant_unpacked)
        %r = aten::conv_transpose2d(%a_dequant, %w_dequant, %b_unpacked, %stride, %padding, %output_padding, %groups, %dilation)
        return (%r) )";

  return {
      {"conv1d_prepack_unpack", conv1d_with_quant, conv1d_with_quant_prepack},
      {"conv2d_prepack_unpack", conv2d_with_quant, conv2d_with_quant_prepack},
      {"conv3d_prepack_unpack", conv3d_with_quant, conv3d_with_quant_prepack},
      {"conv_transpose1d_prepack_unpack",
       conv_transpose1d_with_quant,
       conv_transpose1d_with_quant_prepack},
      {"conv_transpose2d_prepack_unpack",
       conv_transpose2d_with_quant,
       conv_transpose2d_with_quant_prepack}};
}

} // namespace jit
} // namespace torch
