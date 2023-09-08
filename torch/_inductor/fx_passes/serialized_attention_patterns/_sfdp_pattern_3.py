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
tmp_0 = CallFunction(aten.gt.Scalar,
  CallFunction(aten.rand.default,
    Ignored(),
    dtype=Ignored(),
    device=Ignored(),
    pin_memory=False
  ),
  KeywordArg('dropout_p'),
  _users=2
)
tmp_1 = CallFunction(aten.view.default,
  CallFunction(aten.expand.default,
    KeywordArg('query'),
    Ignored()
  ),
  Ignored(),
  _users=2
)
tmp_2 = CallFunction(aten.view.default,
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
tmp_3 = CallFunction(aten.div.Tensor,
  CallFunction(aten.view.default,
    CallFunction(aten.bmm.default,
      tmp_1,
      tmp_2
    ),
    Ignored()
  ),
  KeywordArg('inv_scale_factor'),
  _users=2
)
tmp_4 = CallFunction(aten.exp.default,
  CallFunction(aten.sub.Tensor,
    tmp_3,
    CallFunction(aten.amax.default,
      tmp_3,
      Ignored(),
      True
    )
  ),
  _users=2
)
tmp_5 = CallFunction(aten.div.Tensor,
  tmp_4,
  CallFunction(aten.sum.dim_IntList,
    tmp_4,
    Ignored(),
    True
  ),
  _users=3
)
tmp_6 = CallFunction(aten.view.default,
  CallFunction(aten.expand.default,
    CallFunction(aten.mul.Tensor,
      CallFunction(aten.mul.Tensor,
        tmp_0,
        tmp_5
      ),
      Ignored()
    ),
    Ignored()
  ),
  Ignored(),
  _users=2
)
tmp_7 = CallFunction(aten.view.default,
  CallFunction(aten.expand.default,
    KeywordArg('value'),
    Ignored()
  ),
  Ignored(),
  _users=2
)
tmp_8 = CallFunction(aten.view.default,
  KeywordArg('tangents_1'),
  Ignored(),
  _users=2
)
tmp_9 = CallFunction(aten.mul.Tensor,
  CallFunction(aten.clone.default,
    CallFunction(aten.mul.Tensor,
      CallFunction(aten.view.default,
        CallFunction(aten.bmm.default,
          tmp_8,
          CallFunction(aten.permute.default,
            tmp_7,
            Ignored()
          )
        ),
        Ignored()
      ),
      CallFunction(aten.mul.Tensor,
        CallFunction(prims.convert_element_type.default,
          tmp_0,
          Ignored()
        ),
        Ignored()
      )
    ),
    memory_format=torch.contiguous_format
  ),
  tmp_5,
  _users=2
)
tmp_10 = CallFunction(aten.view.default,
  CallFunction(aten.div.Tensor,
    CallFunction(aten.sub.Tensor,
      tmp_9,
      CallFunction(aten.mul.Tensor,
        tmp_5,
        CallFunction(aten.sum.dim_IntList,
          tmp_9,
          Ignored(),
          True
        )
      )
    ),
    KeywordArg('inv_scale_factor')
  ),
  Ignored(),
  _users=2
)
_sfdp_pattern_3_training = MultiOutputPattern([CallFunction(aten.view.default,
    CallFunction(aten.bmm.default,
      tmp_6,
      tmp_7
    ),
    Ignored()
  ),
  CallFunction(aten.view.default,
    CallFunction(aten.bmm.default,
      tmp_10,
      CallFunction(aten.permute.default,
        tmp_2,
        Ignored()
      )
    ),
    Ignored()
  ),
  CallFunction(aten.permute.default,
    CallFunction(aten.view.default,
      CallFunction(aten.bmm.default,
        CallFunction(aten.permute.default,
          tmp_1,
          Ignored()
        ),
        tmp_10
      ),
      Ignored()
    ),
    Ignored()
  ),
  CallFunction(aten.view.default,
    CallFunction(aten.bmm.default,
      CallFunction(aten.permute.default,
        tmp_6,
        Ignored()
      ),
      tmp_8
    ),
    Ignored()
  ),
  None,
  None
])
