#pragma once

namespace torch {
namespace jit {

static std::string conv2d = R"(
graph(%a_quant, %w_quant, %b, %r_scale, %r_zero_point, %r_dtype, %stride, %padding, %dilation, %groups):
        %a_dequant = aten::dequantize(%a_quant)
        %w_dequant = aten::dequantize(%w_quant)
        %r = aten::conv2d(%a_dequant, %w_dequant, %b, %stride, %padding, %dilation, %groups)
        %r_quant = aten::quantize_per_tensor(%r, %r_scale, %r_zero_point, %r_dtype)
        return (%r_quant))";

static std::string quantized_conv2d = R"(
graph(%a_quant, %w_quant, %b, %r_scale, %r_zero_point, %r_dtype, %stride, %padding, %dilation, %groups):
        %packed_params = quantized::conv_prepack(%w_quant, %b, %stride, %padding, %dilation, %groups)
        %r_quant = quantized::conv2d(%a_quant, %packed_params, %stride, %padding, %dilation, %groups, %r_scale, %r_zero_point)
        %0 : int = prim::Constant[value=0]()
        %1 : int = prim::Constant[value=1]()
        %2 : int = prim::Constant[value=2]()
        %3 : int = prim::Constant[value=3]()
        %out_param : int[] = prim::ListConstruct(%0, %3, %1, %2)
        %r_perm = aten::permute(%r_quant, %out_param)
        return (%r_perm))";

static std::string addmm = R"(
graph(%packed_params_module, %a_quant, %r_scale, %r_zero_point, %r_dtype, %4):
        %a_dequant = aten::dequantize(%a_quant)
        %zero : int = prim::Constant[value=0]()
        %one : int = prim::Constant[value=1]()
        %weight_bias : (Tensor, Tensor?) = prim::CallMethod[name="_weight_bias"](%packed_params_module)
        %w_quant : Tensor = prim::TupleIndex(%weight_bias, %zero)
        %b : Tensor = prim::TupleIndex(%weight_bias, %one)
        %w_dequant = aten::dequantize(%w_quant)
        %w_dequant_t = aten::t(%w_dequant)
        %r = aten::addmm(%b, %a_dequant, %w_dequant_t, %4, %4)
        %r_quant = aten::quantize_per_tensor(%r, %r_scale, %r_zero_point, %r_dtype)
        return (%r_quant))";

static std::string matmul_with_bias = R"(
graph(%packed_params_module, %a_quant, %r_scale, %r_zero_point, %r_dtype, %4):
        %a_dequant = aten::dequantize(%a_quant)
        %zero : int = prim::Constant[value=0]()
        %one : int = prim::Constant[value=1]()
        %weight_bias : (Tensor, Tensor?) = prim::CallMethod[name="_weight_bias"](%packed_params_module)
        %w_quant : Tensor = prim::TupleIndex(%weight_bias, %zero)
        %b : Tensor = prim::TupleIndex(%weight_bias, %one)
        %w_dequant = aten::dequantize(%w_quant)
        %w_dequant_t = aten::t(%w_dequant)
        %output = aten::matmul(%a_dequant, %w_dequant_t)
        %r = aten::add_(%output, %b, %4)
        %r_quant = aten::quantize_per_tensor(%r, %r_scale, %r_zero_point, %r_dtype)
        return (%r_quant))";

static std::string quantized_linear = R"(
graph(%packed_params_module, %a_quant, %r_scale, %r_zero_point, %r_dtype, %4):
        %packed_params = prim::GetAttr[name="_packed_params"](%packed_params_module)
        %r = quantized::linear(%a_quant, %packed_params, %r_scale, %r_zero_point)
        return (%r))";

static std::string matmul_no_bias = R"(
graph(%packed_params_module, %a_quant, %r_scale, %r_zero_point, %r_dtype):
        %a_dequant = aten::dequantize(%a_quant)
        %zero : int = prim::Constant[value=0]()
        %one : int = prim::Constant[value=1]()
        %weight_bias : (Tensor, Tensor?) = prim::CallMethod[name="_weight_bias"](%packed_params_module)
        %w_quant = prim::TupleIndex(%weight_bias, %zero)
        %w_dequant = aten::dequantize(%w_quant)
        %w_dequant_t = aten::t(%w_dequant)
        %r = aten::matmul(%a_dequant, %w_dequant_t)
        %r_quant = aten::quantize_per_tensor(%r, %r_scale, %r_zero_point, %r_dtype)
        return (%r_quant))";

static std::string quantized_linear_no_bias = R"(
graph(%packed_params_module, %a_quant, %r_scale, %r_zero_point, %r_dtype):
        %packed_params = prim::GetAttr[name="_packed_params"](%packed_params_module)
        %r = quantized::linear(%a_quant, %packed_params, %r_scale, %r_zero_point)
        return (%r))";

static std::string linear_with_quant = R"(
graph(%self, %a_dequant, %w_scale, %w_zero_point, %w_dtype, %r_scale, %r_zero_point, %r_dtype, %4):
        %w = prim::GetAttr[name="weight"](%self)
        %b = prim::GetAttr[name="bias"](%self)
        %w_quant = aten::quantize_per_tensor(%w, %w_scale, %w_zero_point, %w_dtype)
        %w_dequant = aten::dequantize(%w_quant)
        %linear = prim::Constant[name="linear"]()
        %r = prim::CallFunction(%linear, %a_dequant, %w_dequant, %b)
        %r_quant = aten::quantize_per_tensor(%r, %r_scale, %r_zero_point, %r_dtype)
        return (%r_quant))";

}} // torch::jit
