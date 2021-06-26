// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <functorch/csrc/BatchRulesHelper.h>

namespace at { namespace functorch {


#define SINGLE_ARG(...) __VA_ARGS__
#define FACTORY_TYPES optional<ScalarType>, optional<Layout>, optional<Device>, optional<bool>, optional<MemoryFormat>
#define FACTORY_BATCH_RULE(op) SINGLE_ARG(basic_unary_batch_rule<decltype(&ATEN_FN(op)), &at::op, FACTORY_TYPES>)

TORCH_LIBRARY_IMPL(aten, FT_BATCHED_KEY, m) {
  VMAP_SUPPORT("ones_like", FACTORY_BATCH_RULE(ones_like));
  VMAP_SUPPORT("zeros_like", FACTORY_BATCH_RULE(zeros_like));
  VMAP_SUPPORT("empty_like", FACTORY_BATCH_RULE(empty_like));
  VMAP_SUPPORT("randn_like", FACTORY_BATCH_RULE(randn_like));
  VMAP_SUPPORT("rand_like", FACTORY_BATCH_RULE(rand_like));
  VMAP_SUPPORT("full_like", SINGLE_ARG(basic_unary_batch_rule<decltype(&ATEN_FN(full_like)), &at::full_like, const Scalar&, FACTORY_TYPES>));
  // Not sure how to add the ones with irregular args to the mix cleanly (i.e. randint takes an extra int parameter)
}
}}

