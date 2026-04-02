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
bmm_default = CallFunction(aten.bmm.default, KeywordArg('hidden_3d'), KeywordArg('gate_up_proj'))
split_Tensor = CallFunction(aten.split.Tensor, bmm_default, Ignored(), Ignored(), _users=2)
operator_getitem = CallFunction(operator.getitem, split_Tensor, 1)
operator_getitem_1 = CallFunction(operator.getitem, split_Tensor, 0, _users=2)
neg_default = CallFunction(aten.neg.default, operator_getitem_1)
exp_default = CallFunction(aten.exp.default, neg_default)
add_Tensor = CallFunction(aten.add.Tensor, exp_default, Ignored())
div_Tensor = CallFunction(aten.div.Tensor, operator_getitem_1, add_Tensor)
mul_Tensor = CallFunction(aten.mul.Tensor, operator_getitem, div_Tensor)
_moe_expert_pattern_1_inference = CallFunction(aten.bmm.default, mul_Tensor, KeywordArg('down_proj'), _users=0)


bmm_default = CallFunction(aten.bmm.default, KeywordArg('hidden_3d'), KeywordArg('gate_up_proj'))
split_Tensor = CallFunction(aten.split.Tensor, bmm_default, Ignored(), Ignored(), _users=2)
operator_getitem = CallFunction(operator.getitem, split_Tensor, 1)
operator_getitem_1 = CallFunction(operator.getitem, split_Tensor, 0)
convert_element_type_default = CallFunction(prims.convert_element_type.default, operator_getitem_1, Ignored(), _users=2)
neg_default = CallFunction(aten.neg.default, convert_element_type_default)
exp_default = CallFunction(aten.exp.default, neg_default)
add_Tensor = CallFunction(aten.add.Tensor, exp_default, Ignored())
div_Tensor = CallFunction(aten.div.Tensor, convert_element_type_default, add_Tensor)
convert_element_type_default_1 = CallFunction(prims.convert_element_type.default, div_Tensor, Ignored())
mul_Tensor = CallFunction(aten.mul.Tensor, operator_getitem, convert_element_type_default_1)
_moe_expert_pattern_1_half_inference = CallFunction(aten.bmm.default, mul_Tensor, KeywordArg('down_proj'), _users=0)
