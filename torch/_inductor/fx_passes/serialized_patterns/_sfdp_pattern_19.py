# mypy: ignore-errors

# noqa: F401, E501
# This is an auto-generated file. Please do not modify it by hand.
# To re-generate, run:
# cd ~/pytorch && python torchgen/fuse/gen_patterns.py

import torch
import torch._inductor

aten = torch.ops.aten
prims = torch.ops.prims

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
convert_element_type_default = CallFunction(prims.convert_element_type.default, permute_default, Ignored())
sub_Tensor = CallFunction(aten.sub.Tensor, convert_element_type_default, KeywordArg('q_zp'))
mul_Tensor = CallFunction(aten.mul.Tensor, sub_Tensor, KeywordArg('q_scale'))
expand_default = CallFunction(aten.expand.default, mul_Tensor, Ignored())
clone_default = CallFunction(aten.clone.default, expand_default, memory_format=torch.contiguous_format)
view_default = CallFunction(aten.view.default, clone_default, Ignored())
permute_default_1 = CallFunction(aten.permute.default, KeywordArg('key'), Ignored())
permute_default_2 = CallFunction(aten.permute.default, permute_default_1, Ignored())
convert_element_type_default_1 = CallFunction(prims.convert_element_type.default, permute_default_2, Ignored())
sub_Tensor_1 = CallFunction(aten.sub.Tensor, convert_element_type_default_1, KeywordArg('k_zp'))
mul_Tensor_1 = CallFunction(aten.mul.Tensor, sub_Tensor_1, KeywordArg('k_scale'))
expand_default_1 = CallFunction(aten.expand.default, mul_Tensor_1, Ignored())
clone_default_1 = CallFunction(aten.clone.default, expand_default_1, memory_format=torch.contiguous_format)
view_default_1 = CallFunction(aten.view.default, clone_default_1, Ignored())
bmm_default = CallFunction(aten.bmm.default, view_default, view_default_1)
view_default_2 = CallFunction(aten.view.default, bmm_default, Ignored())
div_Tensor = CallFunction(aten.div.Tensor, view_default_2, KeywordArg('inv_scale'))
add_Tensor = CallFunction(aten.add.Tensor, div_Tensor, KeywordArg('attn_mask'), _users=2)
amax_default = CallFunction(aten.amax.default, add_Tensor, Ignored(), True)
sub_Tensor_2 = CallFunction(aten.sub.Tensor, add_Tensor, amax_default)
exp_default = CallFunction(aten.exp.default, sub_Tensor_2, _users=2)
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)
div_Tensor_1 = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList)
clone_default_2 = CallFunction(aten.clone.default, div_Tensor_1)
mul_Tensor_2 = CallFunction(aten.mul.Tensor, clone_default_2, KeywordArg('a_inv_scale'))
round_default = CallFunction(aten.round.default, mul_Tensor_2)
add_Tensor_1 = CallFunction(aten.add.Tensor, round_default, KeywordArg('a_zp'))
clamp_min_default = CallFunction(aten.clamp_min.default, add_Tensor_1, Ignored())
clamp_max_default = CallFunction(aten.clamp_max.default, clamp_min_default, Ignored())
convert_element_type_default_2 = CallFunction(prims.convert_element_type.default, clamp_max_default, Ignored())
convert_element_type_default_3 = CallFunction(prims.convert_element_type.default, convert_element_type_default_2, Ignored())
sub_Tensor_3 = CallFunction(aten.sub.Tensor, convert_element_type_default_3, KeywordArg('a_zp'))
mul_Tensor_3 = CallFunction(aten.mul.Tensor, sub_Tensor_3, KeywordArg('a_scale'))
expand_default_2 = CallFunction(aten.expand.default, mul_Tensor_3, Ignored())
view_default_3 = CallFunction(aten.view.default, expand_default_2, Ignored())
permute_default_3 = CallFunction(aten.permute.default, KeywordArg('value'), Ignored())
convert_element_type_default_4 = CallFunction(prims.convert_element_type.default, permute_default_3, Ignored())
sub_Tensor_4 = CallFunction(aten.sub.Tensor, convert_element_type_default_4, KeywordArg('v_zp'))
mul_Tensor_4 = CallFunction(aten.mul.Tensor, sub_Tensor_4, KeywordArg('v_scale'))
expand_default_3 = CallFunction(aten.expand.default, mul_Tensor_4, Ignored())
clone_default_3 = CallFunction(aten.clone.default, expand_default_3, memory_format=torch.contiguous_format)
view_default_4 = CallFunction(aten.view.default, clone_default_3, Ignored())
bmm_default_1 = CallFunction(aten.bmm.default, view_default_3, view_default_4)
view_default_5 = CallFunction(aten.view.default, bmm_default_1, Ignored())
permute_default_4 = CallFunction(aten.permute.default, view_default_5, Ignored())
clone_default_4 = CallFunction(aten.clone.default, permute_default_4, memory_format=torch.contiguous_format)
mul_Tensor_5 = CallFunction(aten.mul.Tensor, clone_default_4, KeywordArg('o_inv_scale'))
round_default_1 = CallFunction(aten.round.default, mul_Tensor_5)
add_Tensor_2 = CallFunction(aten.add.Tensor, round_default_1, KeywordArg('o_zp'))
clamp_min_default_1 = CallFunction(aten.clamp_min.default, add_Tensor_2, Ignored())
clamp_max_default_1 = CallFunction(aten.clamp_max.default, clamp_min_default_1, Ignored())
_sfdp_pattern_19_u8_inference = CallFunction(prims.convert_element_type.default, clamp_max_default_1, Ignored(), _users=0)
