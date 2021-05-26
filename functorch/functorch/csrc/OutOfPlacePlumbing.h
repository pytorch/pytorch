// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once


namespace at { namespace functorch {

template <typename batch_rule_t, typename Result, typename... Args>
Result lowerToNextLayer(batch_rule_t batch_rule, Args... args);

}}
