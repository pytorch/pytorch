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
mm_default = CallFunction(aten.mm.default, KeywordArg('mat1'), KeywordArg('mat2'))
permute_default = CallFunction(aten.permute.default, KeywordArg('mat2'), Ignored())
mm_default_1 = CallFunction(aten.mm.default, KeywordArg('tangents_1'), permute_default)
permute_default_1 = CallFunction(aten.permute.default, KeywordArg('mat1'), Ignored())
mm_default_2 = CallFunction(aten.mm.default, permute_default_1, KeywordArg('tangents_1'))
mm_pattern_training = MultiOutputPattern([mm_default,
  mm_default_1,
  mm_default_2
])


mm_pattern_inference = CallFunction(aten.mm.default, KeywordArg('mat1'), KeywordArg('mat2'), _users=0)
