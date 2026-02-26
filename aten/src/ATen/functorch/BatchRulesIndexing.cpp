// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <ATen/Operators.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/functorch/BatchRulesHelper.h>

namespace at::functorch {

#define OP_DECOMPOSE(op)  m.impl(#op, static_cast<decltype(&ATEN_FN(op))>(native::op));
#define OP_DECOMPOSE2(op, overload)  m.impl(#op"."#overload, static_cast<decltype(&ATEN_FN2(op, overload))>(native::op));

TORCH_LIBRARY_IMPL(aten, FuncTorchBatched, m) {
  OP_DECOMPOSE2(_unsafe_index, Tensor);
  OP_DECOMPOSE(_unsafe_masked_index);
  OP_DECOMPOSE(_unsafe_index_put);
  OP_DECOMPOSE(_unsafe_masked_index_put_accumulate);
}

}
