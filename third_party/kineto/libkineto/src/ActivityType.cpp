/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "ActivityType.h"

#include <fmt/format.h>

namespace libkineto {

static constexpr bool matchingOrder(int idx = 0) {
  return _activityTypeNames[idx].type == ActivityType::ENUM_COUNT ||
      ((idx == static_cast<int>(_activityTypeNames[idx].type)) &&
       matchingOrder(idx + 1));
}
static_assert(matchingOrder(), "ActivityTypeName map is out of order");

ActivityType toActivityType(const std::string& str) {
  for (int i = 0; i < activityTypeCount; i++) {
    if (str == _activityTypeNames[i].name) {
      return _activityTypeNames[i].type;
    }
  }
  throw std::invalid_argument(fmt::format("Invalid activity type: {}", str));
}

std::array<ActivityType, activityTypeCount> activityTypes() {
  std::array<ActivityType, activityTypeCount> res;
  for (int i = 0; i < activityTypeCount; i++) {
    res[i] = _activityTypeNames[i].type;
  }
  return res;
}

std::array<ActivityType, defaultActivityTypeCount> defaultActivityTypes() {
  std::array<ActivityType, defaultActivityTypeCount> res;
  for (int i = 0; i < defaultActivityTypeCount; i++) {
    res[i] = _activityTypeNames[i].type;
  }
  return res;
}

} // namespace libkineto
