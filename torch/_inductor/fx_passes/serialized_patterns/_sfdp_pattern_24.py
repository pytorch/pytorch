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
repeat_default = CallFunction(aten.repeat.default, KeywordArg('attn_mask'), Ignored(), _users=2)
full_default = CallFunction(aten.full.default, [], Ignored(), dtype=Ignored(), device=Ignored(), pin_memory=False)
permute_default = CallFunction(aten.permute.default, KeywordArg('query'), Ignored())
clone_default = CallFunction(aten.clone.default, permute_default, memory_format=torch.contiguous_format)
view_default = CallFunction(aten.view.default, clone_default, Ignored(), _users=2)
permute_default_1 = CallFunction(aten.permute.default, KeywordArg('key'), Ignored())
clone_default_1 = CallFunction(aten.clone.default, permute_default_1, memory_format=torch.contiguous_format)
view_default_1 = CallFunction(aten.view.default, clone_default_1, Ignored())
permute_default_2 = CallFunction(aten.permute.default, view_default_1, Ignored(), _users=2)
bmm_default = CallFunction(aten.bmm.default, view_default, permute_default_2)
div_Tensor = CallFunction(aten.div.Tensor, bmm_default, KeywordArg('inv_scale'))
where_self = CallFunction(aten.where.self, repeat_default, full_default, div_Tensor, _users=2)
amax_default = CallFunction(aten.amax.default, where_self, Ignored(), True)
sub_Tensor = CallFunction(aten.sub.Tensor, where_self, amax_default)
exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)
div_Tensor_1 = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList, _users=4)
permute_default_3 = CallFunction(aten.permute.default, KeywordArg('value'), Ignored())
clone_default_2 = CallFunction(aten.clone.default, permute_default_3, memory_format=torch.contiguous_format)
view_default_2 = CallFunction(aten.view.default, clone_default_2, Ignored(), _users=2)
bmm_default_1 = CallFunction(aten.bmm.default, div_Tensor_1, view_default_2)
scalar_tensor_default = CallFunction(aten.scalar_tensor.default, Ignored(), dtype=Ignored(), layout=torch.strided, device=Ignored())
neg_default = CallFunction(aten.neg.default, div_Tensor_1)
permute_default_4 = CallFunction(aten.permute.default, view_default_2, Ignored())
bmm_default_2 = CallFunction(aten.bmm.default, KeywordArg('tangents_1'), permute_default_4)
mul_Tensor = CallFunction(aten.mul.Tensor, bmm_default_2, div_Tensor_1, _users=2)
sum_dim_IntList_1 = CallFunction(aten.sum.dim_IntList, mul_Tensor, Ignored(), True)
fma_default = CallFunction(prims.fma.default, neg_default, sum_dim_IntList_1, mul_Tensor)
where_self_1 = CallFunction(aten.where.self, repeat_default, scalar_tensor_default, fma_default)
div_Tensor_2 = CallFunction(aten.div.Tensor, where_self_1, KeywordArg('inv_scale'), _users=2)
permute_default_5 = CallFunction(aten.permute.default, permute_default_2, Ignored())
bmm_default_3 = CallFunction(aten.bmm.default, div_Tensor_2, permute_default_5)
view_default_3 = CallFunction(aten.view.default, bmm_default_3, Ignored())
permute_default_6 = CallFunction(aten.permute.default, view_default_3, Ignored())
permute_default_7 = CallFunction(aten.permute.default, view_default, Ignored())
bmm_default_4 = CallFunction(aten.bmm.default, permute_default_7, div_Tensor_2)
permute_default_8 = CallFunction(aten.permute.default, bmm_default_4, Ignored())
view_default_4 = CallFunction(aten.view.default, permute_default_8, Ignored())
permute_default_9 = CallFunction(aten.permute.default, view_default_4, Ignored())
permute_default_10 = CallFunction(aten.permute.default, div_Tensor_1, Ignored())
bmm_default_5 = CallFunction(aten.bmm.default, permute_default_10, KeywordArg('tangents_1'))
view_default_5 = CallFunction(aten.view.default, bmm_default_5, Ignored())
permute_default_11 = CallFunction(aten.permute.default, view_default_5, Ignored())
_sfdp_pattern_24_training = MultiOutputPattern([bmm_default_1,
  permute_default_6,
  permute_default_9,
  permute_default_11,
  None,
  None
])


repeat_default = CallFunction(aten.repeat.default, KeywordArg('attn_mask'), Ignored())
full_default = CallFunction(aten.full.default, [], Ignored(), dtype=Ignored(), device=Ignored(), pin_memory=False)
permute_default = CallFunction(aten.permute.default, KeywordArg('query'), Ignored())
clone_default = CallFunction(aten.clone.default, permute_default, memory_format=torch.contiguous_format)
view_default = CallFunction(aten.view.default, clone_default, Ignored())
permute_default_1 = CallFunction(aten.permute.default, KeywordArg('key'), Ignored())
clone_default_1 = CallFunction(aten.clone.default, permute_default_1, memory_format=torch.contiguous_format)
view_default_1 = CallFunction(aten.view.default, clone_default_1, Ignored())
permute_default_2 = CallFunction(aten.permute.default, view_default_1, Ignored())
bmm_default = CallFunction(aten.bmm.default, view_default, permute_default_2)
div_Tensor = CallFunction(aten.div.Tensor, bmm_default, KeywordArg('inv_scale'))
where_self = CallFunction(aten.where.self, repeat_default, full_default, div_Tensor, _users=2)
amax_default = CallFunction(aten.amax.default, where_self, Ignored(), True)
sub_Tensor = CallFunction(aten.sub.Tensor, where_self, amax_default)
exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)
div_Tensor_1 = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList)
permute_default_3 = CallFunction(aten.permute.default, KeywordArg('value'), Ignored())
clone_default_2 = CallFunction(aten.clone.default, permute_default_3, memory_format=torch.contiguous_format)
view_default_2 = CallFunction(aten.view.default, clone_default_2, Ignored())
_sfdp_pattern_24_inference = CallFunction(aten.bmm.default, div_Tensor_1, view_default_2, _users=0)


repeat_default = CallFunction(aten.repeat.default, KeywordArg('attn_mask'), Ignored(), _users=2)
full_default = CallFunction(aten.full.default, [], Ignored(), dtype=Ignored(), device=Ignored(), pin_memory=False)
permute_default = CallFunction(aten.permute.default, KeywordArg('query'), Ignored())
clone_default = CallFunction(aten.clone.default, permute_default, memory_format=torch.contiguous_format)
view_default = CallFunction(aten.view.default, clone_default, Ignored(), _users=2)
permute_default_1 = CallFunction(aten.permute.default, KeywordArg('key'), Ignored())
clone_default_1 = CallFunction(aten.clone.default, permute_default_1, memory_format=torch.contiguous_format)
view_default_1 = CallFunction(aten.view.default, clone_default_1, Ignored())
permute_default_2 = CallFunction(aten.permute.default, view_default_1, Ignored(), _users=2)
bmm_default = CallFunction(aten.bmm.default, view_default, permute_default_2)
div_Tensor = CallFunction(aten.div.Tensor, bmm_default, KeywordArg('inv_scale'))
where_self = CallFunction(aten.where.self, repeat_default, full_default, div_Tensor)
convert_element_type_default = CallFunction(prims.convert_element_type.default, where_self, Ignored(), _users=2)
amax_default = CallFunction(aten.amax.default, convert_element_type_default, Ignored(), True)
sub_Tensor = CallFunction(aten.sub.Tensor, convert_element_type_default, amax_default)
exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)
div_Tensor_1 = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList)
convert_element_type_default_1 = CallFunction(prims.convert_element_type.default, div_Tensor_1, Ignored(), _users=3)
permute_default_3 = CallFunction(aten.permute.default, KeywordArg('value'), Ignored())
clone_default_2 = CallFunction(aten.clone.default, permute_default_3, memory_format=torch.contiguous_format)
view_default_2 = CallFunction(aten.view.default, clone_default_2, Ignored(), _users=2)
bmm_default_1 = CallFunction(aten.bmm.default, convert_element_type_default_1, view_default_2)
scalar_tensor_default = CallFunction(aten.scalar_tensor.default, Ignored(), dtype=Ignored(), layout=torch.strided, device=Ignored())
convert_element_type_default_2 = CallFunction(prims.convert_element_type.default, convert_element_type_default_1, Ignored(), _users=2)
neg_default = CallFunction(aten.neg.default, convert_element_type_default_2)
permute_default_4 = CallFunction(aten.permute.default, view_default_2, Ignored())
bmm_default_2 = CallFunction(aten.bmm.default, KeywordArg('tangents_1'), permute_default_4)
convert_element_type_default_3 = CallFunction(prims.convert_element_type.default, bmm_default_2, Ignored())
mul_Tensor = CallFunction(aten.mul.Tensor, convert_element_type_default_3, convert_element_type_default_2, _users=2)
sum_dim_IntList_1 = CallFunction(aten.sum.dim_IntList, mul_Tensor, Ignored(), True)
fma_default = CallFunction(prims.fma.default, neg_default, sum_dim_IntList_1, mul_Tensor)
convert_element_type_default_4 = CallFunction(prims.convert_element_type.default, fma_default, Ignored())
where_self_1 = CallFunction(aten.where.self, repeat_default, scalar_tensor_default, convert_element_type_default_4)
div_Tensor_2 = CallFunction(aten.div.Tensor, where_self_1, KeywordArg('inv_scale'), _users=2)
permute_default_5 = CallFunction(aten.permute.default, permute_default_2, Ignored())
bmm_default_3 = CallFunction(aten.bmm.default, div_Tensor_2, permute_default_5)
view_default_3 = CallFunction(aten.view.default, bmm_default_3, Ignored())
permute_default_6 = CallFunction(aten.permute.default, view_default_3, Ignored())
permute_default_7 = CallFunction(aten.permute.default, view_default, Ignored())
bmm_default_4 = CallFunction(aten.bmm.default, permute_default_7, div_Tensor_2)
permute_default_8 = CallFunction(aten.permute.default, bmm_default_4, Ignored())
view_default_4 = CallFunction(aten.view.default, permute_default_8, Ignored())
permute_default_9 = CallFunction(aten.permute.default, view_default_4, Ignored())
permute_default_10 = CallFunction(aten.permute.default, convert_element_type_default_1, Ignored())
bmm_default_5 = CallFunction(aten.bmm.default, permute_default_10, KeywordArg('tangents_1'))
view_default_5 = CallFunction(aten.view.default, bmm_default_5, Ignored())
permute_default_11 = CallFunction(aten.permute.default, view_default_5, Ignored())
_sfdp_pattern_24_half_training = MultiOutputPattern([bmm_default_1,
  permute_default_6,
  permute_default_9,
  permute_default_11,
  None,
  None
])


repeat_default = CallFunction(aten.repeat.default, KeywordArg('attn_mask'), Ignored())
full_default = CallFunction(aten.full.default, [], Ignored(), dtype=Ignored(), device=Ignored(), pin_memory=False)
permute_default = CallFunction(aten.permute.default, KeywordArg('query'), Ignored())
clone_default = CallFunction(aten.clone.default, permute_default, memory_format=torch.contiguous_format)
view_default = CallFunction(aten.view.default, clone_default, Ignored())
permute_default_1 = CallFunction(aten.permute.default, KeywordArg('key'), Ignored())
clone_default_1 = CallFunction(aten.clone.default, permute_default_1, memory_format=torch.contiguous_format)
view_default_1 = CallFunction(aten.view.default, clone_default_1, Ignored())
permute_default_2 = CallFunction(aten.permute.default, view_default_1, Ignored())
bmm_default = CallFunction(aten.bmm.default, view_default, permute_default_2)
div_Tensor = CallFunction(aten.div.Tensor, bmm_default, KeywordArg('inv_scale'))
where_self = CallFunction(aten.where.self, repeat_default, full_default, div_Tensor)
convert_element_type_default = CallFunction(prims.convert_element_type.default, where_self, Ignored(), _users=2)
amax_default = CallFunction(aten.amax.default, convert_element_type_default, Ignored(), True)
sub_Tensor = CallFunction(aten.sub.Tensor, convert_element_type_default, amax_default)
exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)
div_Tensor_1 = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList)
convert_element_type_default_1 = CallFunction(prims.convert_element_type.default, div_Tensor_1, Ignored())
permute_default_3 = CallFunction(aten.permute.default, KeywordArg('value'), Ignored())
clone_default_2 = CallFunction(aten.clone.default, permute_default_3, memory_format=torch.contiguous_format)
view_default_2 = CallFunction(aten.view.default, clone_default_2, Ignored())
_sfdp_pattern_24_half_inference = CallFunction(aten.bmm.default, convert_element_type_default_1, view_default_2, _users=0)
