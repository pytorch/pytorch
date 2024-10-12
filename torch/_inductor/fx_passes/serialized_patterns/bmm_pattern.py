

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
bmm_default = CallFunction(aten.bmm.default, KeywordArg('mat1'), KeywordArg('mat2'))
permute_default = CallFunction(aten.permute.default, KeywordArg('mat2'), Ignored())
bmm_default_1 = CallFunction(aten.bmm.default, KeywordArg('tangents_1'), permute_default)
permute_default_1 = CallFunction(aten.permute.default, KeywordArg('mat1'), Ignored())
bmm_default_2 = CallFunction(aten.bmm.default, permute_default_1, KeywordArg('tangents_1'))
bmm_pattern_training = MultiOutputPattern([bmm_default,
  bmm_default_1,
  bmm_default_2
])


bmm_pattern_inference = CallFunction(aten.bmm.default, KeywordArg('mat1'), KeywordArg('mat2'), _users=0)
