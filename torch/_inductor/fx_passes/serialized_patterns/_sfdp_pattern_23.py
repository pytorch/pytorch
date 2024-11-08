# mypy: ignore-errors

# noqa: F401, E501
# This is an auto-generated file. Please do not modify it by hand.
# To re-generate, run:
# cd ~/pytorch && python torchgen/fuse/gen_patterns.py

import torch
import torch._inductor

aten = torch.ops.aten
prims = torch.ops.prims
quantized_decomposed = torch.ops.quantized_decomposed

from torch._inductor.pattern_matcher import (
   Arg,
   CallFunction,
   CallFunctionVarArgs,
   CallMethod,
   CallMethodVarArgs,
   CallModule,
   CallModuleVarArgs,
   ExclusiveKeywordArg,
   Ignored,
   KeywordArg,
   ListOf,
   MultiOutputPattern,
   PatternExpr,
   RepeatedExpr,
   _TargetArgsExpr,
   _TargetExpr,
   _TargetExprVarArgs,
)
permute_default = CallFunction(aten.permute.default, KeywordArg('query'), Ignored())
quantized_decomposed_dequantize_per_tensor_default = CallFunction(quantized_decomposed.dequantize_per_tensor.default, permute_default, KeywordArg('q_scale'), KeywordArg('q_zp'), Ignored(), Ignored(), Ignored())
convert_element_type_default = CallFunction(prims.convert_element_type.default, quantized_decomposed_dequantize_per_tensor_default, Ignored())
expand_default = CallFunction(aten.expand.default, convert_element_type_default, Ignored())
clone_default = CallFunction(aten.clone.default, expand_default, memory_format=torch.contiguous_format)
view_default = CallFunction(aten.view.default, clone_default, Ignored())
permute_default_1 = CallFunction(aten.permute.default, KeywordArg('key'), Ignored())
permute_default_2 = CallFunction(aten.permute.default, permute_default_1, Ignored())
quantized_decomposed_dequantize_per_tensor_default_1 = CallFunction(quantized_decomposed.dequantize_per_tensor.default, permute_default_2, KeywordArg('k_scale'), KeywordArg('k_zp'), Ignored(), Ignored(), Ignored())
convert_element_type_default_1 = CallFunction(prims.convert_element_type.default, quantized_decomposed_dequantize_per_tensor_default_1, Ignored())
expand_default_1 = CallFunction(aten.expand.default, convert_element_type_default_1, Ignored())
clone_default_1 = CallFunction(aten.clone.default, expand_default_1, memory_format=torch.contiguous_format)
view_default_1 = CallFunction(aten.view.default, clone_default_1, Ignored())
bmm_default = CallFunction(aten.bmm.default, view_default, view_default_1)
view_default_2 = CallFunction(aten.view.default, bmm_default, Ignored())
div_Tensor = CallFunction(aten.div.Tensor, view_default_2, KeywordArg('inv_scale'))
convert_element_type_default_2 = CallFunction(prims.convert_element_type.default, div_Tensor, Ignored(), _users=2)
amax_default = CallFunction(aten.amax.default, convert_element_type_default_2, Ignored(), True)
sub_Tensor = CallFunction(aten.sub.Tensor, convert_element_type_default_2, amax_default)
exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)
div_Tensor_1 = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList)
convert_element_type_default_3 = CallFunction(prims.convert_element_type.default, div_Tensor_1, Ignored())
quantized_decomposed_quantize_per_tensor_default = CallFunction(quantized_decomposed.quantize_per_tensor.default, convert_element_type_default_3, KeywordArg('a_scale'), KeywordArg('a_zp'), Ignored(), Ignored(), Ignored())
quantized_decomposed_dequantize_per_tensor_default_2 = CallFunction(quantized_decomposed.dequantize_per_tensor.default, quantized_decomposed_quantize_per_tensor_default, KeywordArg('a_scale'), KeywordArg('a_zp'), Ignored(), Ignored(), Ignored())
convert_element_type_default_4 = CallFunction(prims.convert_element_type.default, quantized_decomposed_dequantize_per_tensor_default_2, Ignored())
expand_default_2 = CallFunction(aten.expand.default, convert_element_type_default_4, Ignored())
view_default_3 = CallFunction(aten.view.default, expand_default_2, Ignored())
permute_default_3 = CallFunction(aten.permute.default, KeywordArg('value'), Ignored())
quantized_decomposed_dequantize_per_tensor_default_3 = CallFunction(quantized_decomposed.dequantize_per_tensor.default, permute_default_3, KeywordArg('v_scale'), KeywordArg('v_zp'), Ignored(), Ignored(), Ignored())
convert_element_type_default_5 = CallFunction(prims.convert_element_type.default, quantized_decomposed_dequantize_per_tensor_default_3, Ignored())
expand_default_3 = CallFunction(aten.expand.default, convert_element_type_default_5, Ignored())
clone_default_2 = CallFunction(aten.clone.default, expand_default_3, memory_format=torch.contiguous_format)
view_default_4 = CallFunction(aten.view.default, clone_default_2, Ignored())
bmm_default_1 = CallFunction(aten.bmm.default, view_default_3, view_default_4)
view_default_5 = CallFunction(aten.view.default, bmm_default_1, Ignored())
permute_default_4 = CallFunction(aten.permute.default, view_default_5, Ignored())
clone_default_3 = CallFunction(aten.clone.default, permute_default_4, memory_format=torch.contiguous_format)
_sfdp_pattern_23_u8_inference = CallFunction(quantized_decomposed.quantize_per_tensor.default, clone_default_3, KeywordArg('o_scale'), KeywordArg('o_zp'), Ignored(), Ignored(), Ignored(), _users=0)


permute_default = CallFunction(aten.permute.default, KeywordArg('query'), Ignored())
quantized_decomposed_dequantize_per_tensor_default = CallFunction(quantized_decomposed.dequantize_per_tensor.default, permute_default, KeywordArg('q_scale'), KeywordArg('q_zp'), Ignored(), Ignored(), Ignored())
convert_element_type_default = CallFunction(prims.convert_element_type.default, quantized_decomposed_dequantize_per_tensor_default, Ignored())
expand_default = CallFunction(aten.expand.default, convert_element_type_default, Ignored())
view_default = CallFunction(aten.view.default, expand_default, Ignored())
permute_default_1 = CallFunction(aten.permute.default, KeywordArg('key'), Ignored())
permute_default_2 = CallFunction(aten.permute.default, permute_default_1, Ignored())
quantized_decomposed_dequantize_per_tensor_default_1 = CallFunction(quantized_decomposed.dequantize_per_tensor.default, permute_default_2, KeywordArg('k_scale'), KeywordArg('k_zp'), Ignored(), Ignored(), Ignored())
convert_element_type_default_1 = CallFunction(prims.convert_element_type.default, quantized_decomposed_dequantize_per_tensor_default_1, Ignored())
expand_default_1 = CallFunction(aten.expand.default, convert_element_type_default_1, Ignored())
view_default_1 = CallFunction(aten.view.default, expand_default_1, Ignored())
bmm_default = CallFunction(aten.bmm.default, view_default, view_default_1)
view_default_2 = CallFunction(aten.view.default, bmm_default, Ignored())
div_Tensor = CallFunction(aten.div.Tensor, view_default_2, KeywordArg('inv_scale'))
convert_element_type_default_2 = CallFunction(prims.convert_element_type.default, div_Tensor, Ignored(), _users=2)
amax_default = CallFunction(aten.amax.default, convert_element_type_default_2, Ignored(), True)
sub_Tensor = CallFunction(aten.sub.Tensor, convert_element_type_default_2, amax_default)
exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)
div_Tensor_1 = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList)
convert_element_type_default_3 = CallFunction(prims.convert_element_type.default, div_Tensor_1, Ignored())
quantized_decomposed_quantize_per_tensor_default = CallFunction(quantized_decomposed.quantize_per_tensor.default, convert_element_type_default_3, KeywordArg('a_scale'), KeywordArg('a_zp'), Ignored(), Ignored(), Ignored())
quantized_decomposed_dequantize_per_tensor_default_2 = CallFunction(quantized_decomposed.dequantize_per_tensor.default, quantized_decomposed_quantize_per_tensor_default, KeywordArg('a_scale'), KeywordArg('a_zp'), Ignored(), Ignored(), Ignored())
convert_element_type_default_4 = CallFunction(prims.convert_element_type.default, quantized_decomposed_dequantize_per_tensor_default_2, Ignored())
expand_default_2 = CallFunction(aten.expand.default, convert_element_type_default_4, Ignored())
view_default_3 = CallFunction(aten.view.default, expand_default_2, Ignored())
permute_default_3 = CallFunction(aten.permute.default, KeywordArg('value'), Ignored())
quantized_decomposed_dequantize_per_tensor_default_3 = CallFunction(quantized_decomposed.dequantize_per_tensor.default, permute_default_3, KeywordArg('v_scale'), KeywordArg('v_zp'), Ignored(), Ignored(), Ignored())
convert_element_type_default_5 = CallFunction(prims.convert_element_type.default, quantized_decomposed_dequantize_per_tensor_default_3, Ignored())
expand_default_3 = CallFunction(aten.expand.default, convert_element_type_default_5, Ignored())
view_default_4 = CallFunction(aten.view.default, expand_default_3, Ignored())
bmm_default_1 = CallFunction(aten.bmm.default, view_default_3, view_default_4)
view_default_5 = CallFunction(aten.view.default, bmm_default_1, Ignored())
permute_default_4 = CallFunction(aten.permute.default, view_default_5, Ignored())
clone_default = CallFunction(aten.clone.default, permute_default_4, memory_format=torch.contiguous_format)
_sfdp_pattern_23_u8_bs1_inference = CallFunction(quantized_decomposed.quantize_per_tensor.default, clone_default, KeywordArg('o_scale'), KeywordArg('o_zp'), Ignored(), Ignored(), Ignored(), _users=0)
