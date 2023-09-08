# noqa: F401, E501
# This is an auto-generated file. Please do not modify it by hand.
# To re-generate, run:
# cd ~/pytorch && python
# torchgen/fuse_attention_patterns/gen_attention_patterns.py

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
tmp_0 = CallFunction(aten.view.default,
  CallFunction(aten.expand.default,
    KeywordArg('query'),
    Ignored()
  ),
  Ignored(),
  _users=2
)
tmp_1 = CallFunction(aten.view.default,
  CallFunction(aten.expand.default,
    CallFunction(aten.permute.default,
      KeywordArg('key'),
      Ignored()
    ),
    Ignored()
  ),
  Ignored(),
  _users=2
)
tmp_2 = CallFunction(aten.add.Tensor,
  CallFunction(aten.div.Tensor,
    CallFunction(aten.view.default,
      CallFunction(aten.bmm.default,
        tmp_0,
        tmp_1
      ),
      Ignored()
    ),
    Ignored()
  ),
  KeywordArg('attn_mask'),
  _users=2
)
tmp_3 = CallFunction(aten.exp.default,
  CallFunction(aten.sub.Tensor,
    tmp_2,
    CallFunction(aten.amax.default,
      tmp_2,
      Ignored(),
      True
    )
  ),
  _users=2
)
tmp_4 = CallFunction(aten.div.Tensor,
  tmp_3,
  CallFunction(aten.sum.dim_IntList,
    tmp_3,
    Ignored(),
    True
  ),
  _users=3
)
tmp_5 = CallFunction(aten.view.default,
  CallFunction(aten.expand.default,
    tmp_4,
    Ignored()
  ),
  Ignored(),
  _users=2
)
tmp_6 = CallFunction(aten.view.default,
  CallFunction(aten.expand.default,
    KeywordArg('value'),
    Ignored()
  ),
  Ignored(),
  _users=2
)
tmp_7 = CallFunction(aten.view.default,
  KeywordArg('tangents_1'),
  Ignored(),
  _users=2
)
tmp_8 = CallFunction(aten.mul.Tensor,
  CallFunction(aten.view.default,
    CallFunction(aten.bmm.default,
      tmp_7,
      CallFunction(aten.permute.default,
        tmp_6,
        Ignored()
      )
    ),
    Ignored()
  ),
  tmp_4,
  _users=2
)
tmp_9 = CallFunction(aten.view.default,
  CallFunction(aten.div.Tensor,
    CallFunction(aten.sub.Tensor,
      tmp_8,
      CallFunction(aten.mul.Tensor,
        tmp_4,
        CallFunction(aten.sum.dim_IntList,
          tmp_8,
          Ignored(),
          True
        )
      )
    ),
    Ignored()
  ),
  Ignored(),
  _users=2
)
_sfdp_pattern_5_training = MultiOutputPattern([CallFunction(aten.view.default,
    CallFunction(aten.bmm.default,
      tmp_5,
      tmp_6
    ),
    Ignored()
  ),
  CallFunction(aten.view.default,
    CallFunction(aten.bmm.default,
      tmp_9,
      CallFunction(aten.permute.default,
        tmp_1,
        Ignored()
      )
    ),
    Ignored()
  ),
  CallFunction(aten.permute.default,
    CallFunction(aten.view.default,
      CallFunction(aten.bmm.default,
        CallFunction(aten.permute.default,
          tmp_0,
          Ignored()
        ),
        tmp_9
      ),
      Ignored()
    ),
    Ignored()
  ),
  CallFunction(aten.view.default,
    CallFunction(aten.bmm.default,
      CallFunction(aten.permute.default,
        tmp_5,
        Ignored()
      ),
      tmp_7
    ),
    Ignored()
  ),
  None
])
