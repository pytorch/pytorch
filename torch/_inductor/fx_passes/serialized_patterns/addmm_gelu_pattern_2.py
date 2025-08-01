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
add_Tensor = CallFunction(aten.add.Tensor, KeywordArg('input'), mm_default, _users=4)
mul_Tensor = CallFunction(aten.mul.Tensor, add_Tensor, Ignored())
mul_Tensor_1 = CallFunction(aten.mul.Tensor, add_Tensor, add_Tensor)
mul_Tensor_2 = CallFunction(aten.mul.Tensor, mul_Tensor_1, add_Tensor)
mul_Tensor_3 = CallFunction(aten.mul.Tensor, mul_Tensor_2, Ignored())
add_Tensor_1 = CallFunction(aten.add.Tensor, add_Tensor, mul_Tensor_3)
mul_Tensor_4 = CallFunction(aten.mul.Tensor, add_Tensor_1, Ignored())
tanh_default = CallFunction(aten.tanh.default, mul_Tensor_4)
add_Tensor_2 = CallFunction(aten.add.Tensor, tanh_default, Ignored())
addmm_gelu_pattern_2_fp32 = CallFunction(aten.mul.Tensor, mul_Tensor, add_Tensor_2, _users=0)


mm_default = CallFunction(aten.mm.default, KeywordArg('mat1'), KeywordArg('mat2'))
add_Tensor = CallFunction(aten.add.Tensor, KeywordArg('input'), mm_default)
convert_element_type_default = CallFunction(prims.convert_element_type.default, add_Tensor, Ignored(), _users=4)
mul_Tensor = CallFunction(aten.mul.Tensor, convert_element_type_default, Ignored())
mul_Tensor_1 = CallFunction(aten.mul.Tensor, convert_element_type_default, convert_element_type_default)
mul_Tensor_2 = CallFunction(aten.mul.Tensor, mul_Tensor_1, convert_element_type_default)
mul_Tensor_3 = CallFunction(aten.mul.Tensor, mul_Tensor_2, Ignored())
add_Tensor_1 = CallFunction(aten.add.Tensor, convert_element_type_default, mul_Tensor_3)
mul_Tensor_4 = CallFunction(aten.mul.Tensor, add_Tensor_1, Ignored())
tanh_default = CallFunction(aten.tanh.default, mul_Tensor_4)
add_Tensor_2 = CallFunction(aten.add.Tensor, tanh_default, Ignored())
mul_Tensor_5 = CallFunction(aten.mul.Tensor, mul_Tensor, add_Tensor_2)
addmm_gelu_pattern_2_bf16 = CallFunction(prims.convert_element_type.default, mul_Tensor_5, Ignored(), _users=0)
