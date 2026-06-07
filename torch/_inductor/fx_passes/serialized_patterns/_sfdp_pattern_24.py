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
