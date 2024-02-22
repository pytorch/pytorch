/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/native/vulkan/graph/Functions.h>
#include <ATen/native/vulkan/graph/OperatorRegistry.h>

namespace at {
namespace native {
namespace vulkan {

bool hasOpsFn(const std::string& name) {
  return OperatorRegistry::getInstance().hasOpsFn(name);
}

OpFunction& getOpsFn(const std::string& name) {
  return OperatorRegistry::getInstance().getOpsFn(name);
}

OperatorRegistry& OperatorRegistry::getInstance() {
  static OperatorRegistry instance;
  return instance;
}

bool OperatorRegistry::hasOpsFn(const std::string& name) {
  return OperatorRegistry::kTable.count(name) > 0;
}

OpFunction& OperatorRegistry::getOpsFn(const std::string& name) {
  return OperatorRegistry::kTable.find(name)->second;
}

// @lint-ignore-every CLANGTIDY modernize-avoid-bind
// clang-format off
#define OPERATOR_ENTRY(name, function) \
  { #name, std::bind(&at::native::vulkan::function, std::placeholders::_1, std::placeholders::_2) }
// clang-format on

const OperatorRegistry::OpTable OperatorRegistry::kTable = {
    OPERATOR_ENTRY(aten.add.Tensor, add),
    OPERATOR_ENTRY(aten.sub.Tensor, sub),
    OPERATOR_ENTRY(aten.mul.Tensor, mul),
    OPERATOR_ENTRY(aten.div.Tensor, div),
    OPERATOR_ENTRY(aten.div.Tensor_mode, floor_div),
    OPERATOR_ENTRY(aten.pow.Tensor_Tensor, pow),
};

} // namespace vulkan
} // namespace native
} // namespace at
