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
addmm_default = CallFunction(aten.addmm.default, KeywordArg('inp'), KeywordArg('m1'), KeywordArg('m2'), beta=KeywordArg('beta'), alpha=KeywordArg('alpha'), _users=4)
mul_Tensor = CallFunction(aten.mul.Tensor, addmm_default, Ignored())
mul_Tensor_1 = CallFunction(aten.mul.Tensor, addmm_default, addmm_default)
mul_Tensor_2 = CallFunction(aten.mul.Tensor, mul_Tensor_1, addmm_default)
mul_Tensor_3 = CallFunction(aten.mul.Tensor, mul_Tensor_2, Ignored())
add_Tensor = CallFunction(aten.add.Tensor, addmm_default, mul_Tensor_3)
mul_Tensor_4 = CallFunction(aten.mul.Tensor, add_Tensor, Ignored())
tanh_default = CallFunction(aten.tanh.default, mul_Tensor_4)
add_Tensor_1 = CallFunction(aten.add.Tensor, tanh_default, Ignored())
addmm_gelu_pattern = CallFunction(aten.mul.Tensor, mul_Tensor, add_Tensor_1, _users=0)
