// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <ATen/ATen.h>
#include <functorch/csrc/BatchRulesHelper.h>
#include <functorch/csrc/BatchedFallback.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/util/Metaprogramming.h>


namespace at { namespace functorch {

void unsupportedDynamicOp(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
    TORCH_CHECK(false, "vmap: We do not support batching operators that can output dynamic shape. ",
        "Attempted to vmap over ", op.schema().operator_name());
}
#define UNSUPPORTED_DYNAMIC(op) \
    m.impl(#op, torch::CppFunction::makeFromBoxedFunction<&unsupportedDynamicOp>());

TORCH_LIBRARY_IMPL(aten, FT_BATCHED_KEY, m) {
    UNSUPPORTED_DYNAMIC(nonzero);
    UNSUPPORTED_DYNAMIC(unique);
}

}}
