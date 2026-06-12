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
permute_default = CallFunction(aten.permute.default, KeywordArg('key'), Ignored())
bmm_default = CallFunction(aten.bmm.default, KeywordArg('query'), permute_default, _users=2)
amax_default = CallFunction(aten.amax.default, bmm_default, Ignored(), True)
sub_Tensor = CallFunction(aten.sub.Tensor, bmm_default, amax_default)
exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)
div_Tensor = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList)
_sfdp_pattern_13_inference = CallFunction(aten.bmm.default, div_Tensor, KeywordArg('value'), _users=0)


permute_default = CallFunction(aten.permute.default, KeywordArg('key'), Ignored())
bmm_default = CallFunction(aten.bmm.default, KeywordArg('query'), permute_default)
convert_element_type_default = CallFunction(prims.convert_element_type.default, bmm_default, Ignored(), _users=2)
amax_default = CallFunction(aten.amax.default, convert_element_type_default, Ignored(), True)
sub_Tensor = CallFunction(aten.sub.Tensor, convert_element_type_default, amax_default)
exp_default = CallFunction(aten.exp.default, sub_Tensor, _users=2)
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, exp_default, Ignored(), True)
div_Tensor = CallFunction(aten.div.Tensor, exp_default, sum_dim_IntList)
convert_element_type_default_1 = CallFunction(prims.convert_element_type.default, div_Tensor, Ignored())
_sfdp_pattern_13_half_inference = CallFunction(aten.bmm.default, convert_element_type_default_1, KeywordArg('value'), _users=0)
