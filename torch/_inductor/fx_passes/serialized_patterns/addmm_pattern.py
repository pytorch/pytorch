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
addmm_default = CallFunction(aten.addmm.default, KeywordArg('input'), KeywordArg('mat1'), KeywordArg('mat2'), beta=KeywordArg('beta'), alpha=KeywordArg('alpha'))
mul_Scalar = CallFunction(aten.mul.Scalar, KeywordArg('tangents_1'), KeywordArg('beta'))
sum_dim_IntList = CallFunction(aten.sum.dim_IntList, mul_Scalar, Ignored(), True)
view_default = CallFunction(aten.view.default, sum_dim_IntList, Ignored())
permute_default = CallFunction(aten.permute.default, KeywordArg('mat2'), Ignored())
mm_default = CallFunction(aten.mm.default, KeywordArg('tangents_1'), permute_default)
mul_Scalar_1 = CallFunction(aten.mul.Scalar, mm_default, KeywordArg('alpha'))
permute_default_1 = CallFunction(aten.permute.default, KeywordArg('mat1'), Ignored())
mm_default_1 = CallFunction(aten.mm.default, permute_default_1, KeywordArg('tangents_1'))
mul_Scalar_2 = CallFunction(aten.mul.Scalar, mm_default_1, KeywordArg('alpha'))
addmm_pattern_training = MultiOutputPattern([addmm_default,
  view_default,
  mul_Scalar_1,
  mul_Scalar_2,
  None,
  None
])


addmm_pattern_inference = CallFunction(aten.addmm.default, KeywordArg('input'), KeywordArg('mat1'), KeywordArg('mat2'), beta=KeywordArg('beta'), alpha=KeywordArg('alpha'), _users=0)
