
// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <functorch/csrc/BatchRulesHelper.h>
#include <ATen/Operators.h>
#include <functorch/csrc/PlumbingHelper.h>
#include <functorch/csrc/BatchedFallback.h>
#include <ATen/core/dispatch/Dispatcher.h>

namespace at { namespace functorch {

TORCH_LIBRARY_IMPL(aten, FT_BATCHED_KEY, m) {
//   m.impl("index_add_", torch::CppFunction::makeFromBoxedFunction<&batchedTensorForLoopFallback>());
}

}}
