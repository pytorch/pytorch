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
permute_default = CallFunction(aten.permute.default, KeywordArg('query'), Ignored())
div_Tensor = CallFunction(aten.div.Tensor, permute_default, Ignored())
expand_default = CallFunction(aten.expand.default, div_Tensor, Ignored())
clone_default = CallFunction(aten.clone.default, expand_default, memory_format=torch.contiguous_format)
view_default = CallFunction(aten.view.default, clone_default, Ignored(), _users=2)
permute_default_1 = CallFunction(aten.permute.default, KeywordArg('key'), Ignored())
permute_default_2 = CallFunction(aten.permute.default, permute_default_1, Ignored())
expand_default_1 = CallFunction(aten.expand.default, permute_default_2, Ignored())
clone_default_1 = CallFunction(aten.clone.default, expand_default_1, memory_format=torch.contiguous_format)
view_default_1 = CallFunction(aten.view.default, clone_default_1, Ignored(), _users=2)
bmm_default = CallFunction(aten.bmm.default, view_default, view_default_1)
view_default_2 = CallFunction(aten.view.default, bmm_default, Ignored())
convert_element_type_default = CallFunction(prims.convert_element_type.default, view_default_2, Ignored(), _users=2)
amax_default = CallFunction(aten.amax.default, convert_element_type_default, Ignored(), True)
sub_Tensor = CallFunction(aten.sub.Tensor, convert_element_type_default, amax_default)
exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)
div_Tensor_1 = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList, _users=3)
convert_element_type_default_1 = CallFunction(prims.convert_element_type.default, div_Tensor_1, Ignored())
expand_default_2 = CallFunction(aten.expand.default, convert_element_type_default_1, Ignored())
view_default_3 = CallFunction(aten.view.default, expand_default_2, Ignored(), _users=2)
permute_default_3 = CallFunction(aten.permute.default, KeywordArg('value'), Ignored())
expand_default_3 = CallFunction(aten.expand.default, permute_default_3, Ignored())
clone_default_2 = CallFunction(aten.clone.default, expand_default_3, memory_format=torch.contiguous_format)
view_default_4 = CallFunction(aten.view.default, clone_default_2, Ignored(), _users=2)
bmm_default_1 = CallFunction(aten.bmm.default, view_default_3, view_default_4)
view_default_5 = CallFunction(aten.view.default, bmm_default_1, Ignored())
view_default_6 = CallFunction(aten.view.default, KeywordArg('tangents_1'), Ignored(), _users=2)
permute_default_4 = CallFunction(aten.permute.default, view_default_4, Ignored())
bmm_default_2 = CallFunction(aten.bmm.default, view_default_6, permute_default_4)
view_default_7 = CallFunction(aten.view.default, bmm_default_2, Ignored())
convert_element_type_default_2 = CallFunction(prims.convert_element_type.default, view_default_7, Ignored())
mul_Tensor = CallFunction(aten.mul.Tensor, convert_element_type_default_2, div_Tensor_1, _users=2)
sum_dim_IntList_1 = CallFunction(aten.sum.dim_IntList, mul_Tensor, Ignored(), True)
mul_Tensor_1 = CallFunction(aten.mul.Tensor, div_Tensor_1, sum_dim_IntList_1)
sub_Tensor_1 = CallFunction(aten.sub.Tensor, mul_Tensor, mul_Tensor_1)
convert_element_type_default_3 = CallFunction(prims.convert_element_type.default, sub_Tensor_1, Ignored())
view_default_8 = CallFunction(aten.view.default, convert_element_type_default_3, Ignored(), _users=2)
permute_default_5 = CallFunction(aten.permute.default, view_default_1, Ignored())
bmm_default_3 = CallFunction(aten.bmm.default, view_default_8, permute_default_5)
view_default_9 = CallFunction(aten.view.default, bmm_default_3, Ignored())
div_Tensor_2 = CallFunction(aten.div.Tensor, view_default_9, Ignored())
permute_default_6 = CallFunction(aten.permute.default, div_Tensor_2, Ignored())
permute_default_7 = CallFunction(aten.permute.default, view_default, Ignored())
bmm_default_4 = CallFunction(aten.bmm.default, permute_default_7, view_default_8)
view_default_10 = CallFunction(aten.view.default, bmm_default_4, Ignored())
permute_default_8 = CallFunction(aten.permute.default, view_default_10, Ignored())
permute_default_9 = CallFunction(aten.permute.default, permute_default_8, Ignored())
permute_default_10 = CallFunction(aten.permute.default, view_default_3, Ignored())
bmm_default_5 = CallFunction(aten.bmm.default, permute_default_10, view_default_6)
view_default_11 = CallFunction(aten.view.default, bmm_default_5, Ignored())
permute_default_11 = CallFunction(aten.permute.default, view_default_11, Ignored())
_sfdp_pattern_10_training = MultiOutputPattern([view_default_5,
  permute_default_6,
  permute_default_9,
  permute_default_11
])


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
_sfdp_pattern_10_inference = CallFunction(aten.view.default, bmm_default_1, Ignored())
