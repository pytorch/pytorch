# noqa: F401, E501
# This is an auto-generated file. Please do not modify it by hand.
# To re-generate, run:
# cd ~/pytorch && python
# torchgen/fuse_attention_patterns/gen_attention_patterns.py

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
expand_default = CallFunction(aten.expand.default, KeywordArg('query'), Ignored())
view_default = CallFunction(aten.view.default, expand_default, Ignored())
permute_default = CallFunction(aten.permute.default, KeywordArg('key'), Ignored())
expand_default_1 = CallFunction(aten.expand.default, permute_default, Ignored())
view_default_1 = CallFunction(aten.view.default, expand_default_1, Ignored())
bmm_default = CallFunction(aten.bmm.default, view_default, view_default_1)
view_default_2 = CallFunction(aten.view.default, bmm_default, Ignored())
div_Tensor = CallFunction(aten.div.Tensor, view_default_2, KeywordArg('inv_scale_factor'), _users=2)
amax_default = CallFunction(aten.amax.default, div_Tensor, Ignored(), True)
sub_Tensor = CallFunction(aten.sub.Tensor, div_Tensor, amax_default)
exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)
div_Tensor_1 = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList)
expand_default_2 = CallFunction(aten.expand.default, div_Tensor_1, Ignored())
view_default_3 = CallFunction(aten.view.default, expand_default_2, Ignored())
expand_default_3 = CallFunction(aten.expand.default, KeywordArg('value'), Ignored())
view_default_4 = CallFunction(aten.view.default, expand_default_3, Ignored())
bmm_default_1 = CallFunction(aten.bmm.default, view_default_3, view_default_4)
_sfdp_pattern_3_inference = CallFunction(aten.view.default, bmm_default_1, Ignored())


expand_default = CallFunction(aten.expand.default, KeywordArg('query'), Ignored())
view_default = CallFunction(aten.view.default, expand_default, Ignored())
permute_default = CallFunction(aten.permute.default, KeywordArg('key'), Ignored())
expand_default_1 = CallFunction(aten.expand.default, permute_default, Ignored())
view_default_1 = CallFunction(aten.view.default, expand_default_1, Ignored())
bmm_default = CallFunction(aten.bmm.default, view_default, view_default_1)
view_default_2 = CallFunction(aten.view.default, bmm_default, Ignored())
mul_Tensor = CallFunction(aten.mul.Tensor, view_default_2, KeywordArg('scale_factor'), _users=2)
amax_default = CallFunction(aten.amax.default, mul_Tensor, Ignored(), True)
sub_Tensor = CallFunction(aten.sub.Tensor, mul_Tensor, amax_default)
exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)
div_Tensor = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList)
expand_default_2 = CallFunction(aten.expand.default, div_Tensor, Ignored())
view_default_3 = CallFunction(aten.view.default, expand_default_2, Ignored())
expand_default_3 = CallFunction(aten.expand.default, KeywordArg('value'), Ignored())
view_default_4 = CallFunction(aten.view.default, expand_default_3, Ignored())
bmm_default_1 = CallFunction(aten.bmm.default, view_default_3, view_default_4)
_sfdp_pattern_4_inference = CallFunction(aten.view.default, bmm_default_1, Ignored())


expand_default = CallFunction(aten.expand.default, KeywordArg('query'), Ignored())
view_default = CallFunction(aten.view.default, expand_default, Ignored())
permute_default = CallFunction(aten.permute.default, KeywordArg('key'), Ignored())
expand_default_1 = CallFunction(aten.expand.default, permute_default, Ignored())
view_default_1 = CallFunction(aten.view.default, expand_default_1, Ignored())
bmm_default = CallFunction(aten.bmm.default, view_default, view_default_1)
view_default_2 = CallFunction(aten.view.default, bmm_default, Ignored())
div_Tensor = CallFunction(aten.div.Tensor, view_default_2, Ignored())
add_Tensor = CallFunction(aten.add.Tensor, div_Tensor, KeywordArg('attn_mask'), _users=2)
amax_default = CallFunction(aten.amax.default, add_Tensor, Ignored(), True)
sub_Tensor = CallFunction(aten.sub.Tensor, add_Tensor, amax_default)
exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)
div_Tensor_1 = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList)
expand_default_2 = CallFunction(aten.expand.default, div_Tensor_1, Ignored())
view_default_3 = CallFunction(aten.view.default, expand_default_2, Ignored())
expand_default_3 = CallFunction(aten.expand.default, KeywordArg('value'), Ignored())
view_default_4 = CallFunction(aten.view.default, expand_default_3, Ignored())
bmm_default_1 = CallFunction(aten.bmm.default, view_default_3, view_default_4)
_sfdp_pattern_6_inference = CallFunction(aten.view.default, bmm_default_1, Ignored())


permute_default = CallFunction(aten.permute.default, KeywordArg('query'), Ignored())
expand_default = CallFunction(aten.expand.default, permute_default, Ignored())
clone_default = CallFunction(aten.clone.default, expand_default, memory_format=torch.contiguous_format)
view_default = CallFunction(aten.view.default, clone_default, Ignored())
permute_default_1 = CallFunction(aten.permute.default, KeywordArg('key'), Ignored())
permute_default_2 = CallFunction(aten.permute.default, permute_default_1, Ignored())
expand_default_1 = CallFunction(aten.expand.default, permute_default_2, Ignored())
clone_default_1 = CallFunction(aten.clone.default, expand_default_1, memory_format=torch.contiguous_format)
view_default_1 = CallFunction(aten.view.default, clone_default_1, Ignored())
bmm_default = CallFunction(aten.bmm.default, view_default, view_default_1)
view_default_2 = CallFunction(aten.view.default, bmm_default, Ignored())
div_Tensor = CallFunction(aten.div.Tensor, view_default_2, Ignored())
convert_element_type_default = CallFunction(prims.convert_element_type.default, div_Tensor, Ignored(), _users=2)
amax_default = CallFunction(aten.amax.default, convert_element_type_default, Ignored(), True)
sub_Tensor = CallFunction(aten.sub.Tensor, convert_element_type_default, amax_default)
exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)
div_Tensor_1 = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList)
convert_element_type_default_1 = CallFunction(prims.convert_element_type.default, div_Tensor_1, Ignored())
expand_default_2 = CallFunction(aten.expand.default, convert_element_type_default_1, Ignored())
view_default_3 = CallFunction(aten.view.default, expand_default_2, Ignored())
permute_default_3 = CallFunction(aten.permute.default, KeywordArg('value'), Ignored())
expand_default_3 = CallFunction(aten.expand.default, permute_default_3, Ignored())
clone_default_2 = CallFunction(aten.clone.default, expand_default_3, memory_format=torch.contiguous_format)
view_default_4 = CallFunction(aten.view.default, clone_default_2, Ignored())
bmm_default_1 = CallFunction(aten.bmm.default, view_default_3, view_default_4)
_sfdp_pattern_7_inference = CallFunction(aten.view.default, bmm_default_1, Ignored())


permute_default = CallFunction(aten.permute.default, KeywordArg('query'), Ignored())
div_Tensor = CallFunction(aten.div.Tensor, permute_default, Ignored())
expand_default = CallFunction(aten.expand.default, div_Tensor, Ignored())
clone_default = CallFunction(aten.clone.default, expand_default, memory_format=torch.contiguous_format)
view_default = CallFunction(aten.view.default, clone_default, Ignored())
permute_default_1 = CallFunction(aten.permute.default, KeywordArg('key'), Ignored())
permute_default_2 = CallFunction(aten.permute.default, permute_default_1, Ignored())
expand_default_1 = CallFunction(aten.expand.default, permute_default_2, Ignored())
clone_default_1 = CallFunction(aten.clone.default, expand_default_1, memory_format=torch.contiguous_format)
view_default_1 = CallFunction(aten.view.default, clone_default_1, Ignored())
bmm_default = CallFunction(aten.bmm.default, view_default, view_default_1)
view_default_2 = CallFunction(aten.view.default, bmm_default, Ignored())
convert_element_type_default = CallFunction(prims.convert_element_type.default, view_default_2, Ignored(), _users=2)
amax_default = CallFunction(aten.amax.default, convert_element_type_default, Ignored(), True)
sub_Tensor = CallFunction(aten.sub.Tensor, convert_element_type_default, amax_default)
exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)
div_Tensor_1 = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList)
convert_element_type_default_1 = CallFunction(prims.convert_element_type.default, div_Tensor_1, Ignored())
expand_default_2 = CallFunction(aten.expand.default, convert_element_type_default_1, Ignored())
view_default_3 = CallFunction(aten.view.default, expand_default_2, Ignored())
permute_default_3 = CallFunction(aten.permute.default, KeywordArg('value'), Ignored())
expand_default_3 = CallFunction(aten.expand.default, permute_default_3, Ignored())
clone_default_2 = CallFunction(aten.clone.default, expand_default_3, memory_format=torch.contiguous_format)
view_default_4 = CallFunction(aten.view.default, clone_default_2, Ignored())
bmm_default_1 = CallFunction(aten.bmm.default, view_default_3, view_default_4)
_sfdp_pattern_9_inference = CallFunction(aten.view.default, bmm_default_1, Ignored())


permute_default = CallFunction(aten.permute.default, KeywordArg('query'), Ignored())
expand_default = CallFunction(aten.expand.default, permute_default, Ignored())
clone_default = CallFunction(aten.clone.default, expand_default, memory_format=torch.contiguous_format)
view_default = CallFunction(aten.view.default, clone_default, Ignored())
permute_default_1 = CallFunction(aten.permute.default, KeywordArg('key'), Ignored())
permute_default_2 = CallFunction(aten.permute.default, permute_default_1, Ignored())
expand_default_1 = CallFunction(aten.expand.default, permute_default_2, Ignored())
clone_default_1 = CallFunction(aten.clone.default, expand_default_1, memory_format=torch.contiguous_format)
view_default_1 = CallFunction(aten.view.default, clone_default_1, Ignored())
bmm_default = CallFunction(aten.bmm.default, view_default, view_default_1)
view_default_2 = CallFunction(aten.view.default, bmm_default, Ignored())
div_Tensor = CallFunction(aten.div.Tensor, view_default_2, KeywordArg('inv_scale_factor'), _users=2)
amax_default = CallFunction(aten.amax.default, div_Tensor, Ignored(), True)
sub_Tensor = CallFunction(aten.sub.Tensor, div_Tensor, amax_default)
exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)
div_Tensor_1 = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList)
expand_default_2 = CallFunction(aten.expand.default, div_Tensor_1, Ignored())
view_default_3 = CallFunction(aten.view.default, expand_default_2, Ignored())
permute_default_3 = CallFunction(aten.permute.default, KeywordArg('value'), Ignored())
expand_default_3 = CallFunction(aten.expand.default, permute_default_3, Ignored())
clone_default_2 = CallFunction(aten.clone.default, expand_default_3, memory_format=torch.contiguous_format)
view_default_4 = CallFunction(aten.view.default, clone_default_2, Ignored())
bmm_default_1 = CallFunction(aten.bmm.default, view_default_3, view_default_4)
_sfdp_pattern_12_inference = CallFunction(aten.view.default, bmm_default_1, Ignored())
