# mypy: ignore-errors

# noqa: F401, E501
# This is an auto-generated file. Please do not modify it by hand.
# To re-generate, run:
# cd ~/pytorch && python torchgen/fuse/gen_patterns.py

import torch
import torch._inductor
import operator

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
view_default = CallFunction(aten.view.default, KeywordArg('query'), Ignored(), _users=2)
view_default_1 = CallFunction(aten.view.default, KeywordArg('key'), Ignored())
permute_default = CallFunction(aten.permute.default, view_default_1, Ignored(), _users=2)
bmm_default = CallFunction(aten.bmm.default, view_default, permute_default)
view_default_2 = CallFunction(aten.view.default, bmm_default, Ignored())
add_Tensor = CallFunction(aten.add.Tensor, view_default_2, KeywordArg('attention_mask'))
view_default_3 = CallFunction(aten.view.default, add_Tensor, Ignored(), _users=2)
amax_default = CallFunction(aten.amax.default, view_default_3, Ignored(), True)
sub_Tensor = CallFunction(aten.sub.Tensor, view_default_3, amax_default)
exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)
div_Tensor = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList, _users=4)
view_default_4 = CallFunction(aten.view.default, KeywordArg('value'), Ignored(), _users=2)
bmm_default_1 = CallFunction(aten.bmm.default, div_Tensor, view_default_4)
view_default_5 = CallFunction(aten.view.default, bmm_default_1, Ignored())
neg_default = CallFunction(aten.neg.default, div_Tensor)
view_default_6 = CallFunction(aten.view.default, KeywordArg('tangents_1'), Ignored(), _users=2)
permute_default_1 = CallFunction(aten.permute.default, view_default_4, Ignored())
bmm_default_2 = CallFunction(aten.bmm.default, view_default_6, permute_default_1)
mul_Tensor = CallFunction(aten.mul.Tensor, bmm_default_2, div_Tensor, _users=2)
sum_dim_IntList_1 = CallFunction(aten.sum.dim_IntList, mul_Tensor, Ignored(), True)
fma_default = CallFunction(prims.fma.default, neg_default, sum_dim_IntList_1, mul_Tensor, _users=2)
permute_default_2 = CallFunction(aten.permute.default, permute_default, Ignored())
bmm_default_3 = CallFunction(aten.bmm.default, fma_default, permute_default_2)
view_default_7 = CallFunction(aten.view.default, bmm_default_3, Ignored())
permute_default_3 = CallFunction(aten.permute.default, view_default, Ignored())
bmm_default_4 = CallFunction(aten.bmm.default, permute_default_3, fma_default)
permute_default_4 = CallFunction(aten.permute.default, bmm_default_4, Ignored())
view_default_8 = CallFunction(aten.view.default, permute_default_4, Ignored())
permute_default_5 = CallFunction(aten.permute.default, div_Tensor, Ignored())
bmm_default_5 = CallFunction(aten.bmm.default, permute_default_5, view_default_6)
view_default_9 = CallFunction(aten.view.default, bmm_default_5, Ignored())
_sfdp_pattern_24_training = MultiOutputPattern([view_default_5,
  view_default_7,
  view_default_8,
  view_default_9,
  None
])


view_default = CallFunction(aten.view.default, KeywordArg('query'), Ignored())
view_default_1 = CallFunction(aten.view.default, KeywordArg('key'), Ignored())
permute_default = CallFunction(aten.permute.default, view_default_1, Ignored())
bmm_default = CallFunction(aten.bmm.default, view_default, permute_default)
view_default_2 = CallFunction(aten.view.default, bmm_default, Ignored())
add_Tensor = CallFunction(aten.add.Tensor, view_default_2, KeywordArg('attention_mask'))
view_default_3 = CallFunction(aten.view.default, add_Tensor, Ignored(), _users=2)
amax_default = CallFunction(aten.amax.default, view_default_3, Ignored(), True)
sub_Tensor = CallFunction(aten.sub.Tensor, view_default_3, amax_default)
exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)
div_Tensor = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList)
view_default_4 = CallFunction(aten.view.default, KeywordArg('value'), Ignored())
bmm_default_1 = CallFunction(aten.bmm.default, div_Tensor, view_default_4)
_sfdp_pattern_24_inference = CallFunction(aten.view.default, bmm_default_1, Ignored(), _users=0)


view_default = CallFunction(aten.view.default, KeywordArg('query'), Ignored(), _users=2)
view_default_1 = CallFunction(aten.view.default, KeywordArg('key'), Ignored())
permute_default = CallFunction(aten.permute.default, view_default_1, Ignored(), _users=2)
bmm_default = CallFunction(aten.bmm.default, view_default, permute_default)
view_default_2 = CallFunction(aten.view.default, bmm_default, Ignored())
add_Tensor = CallFunction(aten.add.Tensor, view_default_2, KeywordArg('attention_mask'))
view_default_3 = CallFunction(aten.view.default, add_Tensor, Ignored(), _users=2)
amax_default = CallFunction(aten.amax.default, view_default_3, Ignored(), True)
sub_Tensor = CallFunction(aten.sub.Tensor, view_default_3, amax_default)
exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)
div_Tensor = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList, _users=3)
convert_element_type_default = CallFunction(prims.convert_element_type.default, div_Tensor, Ignored(), _users=2)
view_default_4 = CallFunction(aten.view.default, KeywordArg('value'), Ignored(), _users=2)
bmm_default_1 = CallFunction(aten.bmm.default, convert_element_type_default, view_default_4)
view_default_5 = CallFunction(aten.view.default, bmm_default_1, Ignored())
neg_default = CallFunction(aten.neg.default, div_Tensor)
view_default_6 = CallFunction(aten.view.default, KeywordArg('tangents_1'), Ignored(), _users=2)
permute_default_1 = CallFunction(aten.permute.default, view_default_4, Ignored())
bmm_default_2 = CallFunction(aten.bmm.default, view_default_6, permute_default_1)
convert_element_type_default_1 = CallFunction(prims.convert_element_type.default, bmm_default_2, Ignored())
mul_Tensor = CallFunction(aten.mul.Tensor, convert_element_type_default_1, div_Tensor, _users=2)
sum_dim_IntList_1 = CallFunction(aten.sum.dim_IntList, mul_Tensor, Ignored(), True)
fma_default = CallFunction(prims.fma.default, neg_default, sum_dim_IntList_1, mul_Tensor)
view_default_7 = CallFunction(aten.view.default, fma_default, Ignored())
convert_element_type_default_2 = CallFunction(prims.convert_element_type.default, view_default_7, Ignored())
view_default_8 = CallFunction(aten.view.default, convert_element_type_default_2, Ignored(), _users=2)
permute_default_2 = CallFunction(aten.permute.default, permute_default, Ignored())
bmm_default_3 = CallFunction(aten.bmm.default, view_default_8, permute_default_2)
view_default_9 = CallFunction(aten.view.default, bmm_default_3, Ignored())
permute_default_3 = CallFunction(aten.permute.default, view_default, Ignored())
bmm_default_4 = CallFunction(aten.bmm.default, permute_default_3, view_default_8)
permute_default_4 = CallFunction(aten.permute.default, bmm_default_4, Ignored())
view_default_10 = CallFunction(aten.view.default, permute_default_4, Ignored())
permute_default_5 = CallFunction(aten.permute.default, convert_element_type_default, Ignored())
bmm_default_5 = CallFunction(aten.bmm.default, permute_default_5, view_default_6)
view_default_11 = CallFunction(aten.view.default, bmm_default_5, Ignored())
_sfdp_pattern_24_half_training = MultiOutputPattern([view_default_5,
  view_default_9,
  view_default_10,
  view_default_11,
  None
])


view_default = CallFunction(aten.view.default, KeywordArg('query'), Ignored())
view_default_1 = CallFunction(aten.view.default, KeywordArg('key'), Ignored())
permute_default = CallFunction(aten.permute.default, view_default_1, Ignored())
bmm_default = CallFunction(aten.bmm.default, view_default, permute_default)
view_default_2 = CallFunction(aten.view.default, bmm_default, Ignored())
add_Tensor = CallFunction(aten.add.Tensor, view_default_2, KeywordArg('attention_mask'))
view_default_3 = CallFunction(aten.view.default, add_Tensor, Ignored(), _users=2)
amax_default = CallFunction(aten.amax.default, view_default_3, Ignored(), True)
sub_Tensor = CallFunction(aten.sub.Tensor, view_default_3, amax_default)
exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)
div_Tensor = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList)
convert_element_type_default = CallFunction(prims.convert_element_type.default, div_Tensor, Ignored())
view_default_4 = CallFunction(aten.view.default, KeywordArg('value'), Ignored())
bmm_default_1 = CallFunction(aten.bmm.default, convert_element_type_default, view_default_4)
_sfdp_pattern_24_half_inference = CallFunction(aten.view.default, bmm_default_1, Ignored(), _users=0)
