/*
 * Copyright (C) Intel Corporation
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "ActivityType.h"

#include <bitset>
#include <concepts>
#include <cstddef>
#include <set>

namespace KINETO_NAMESPACE {

// A set of activity types held as a bitset: built once, queried per record.
// Sized from the enum, so activity types can be added without outgrowing it,
// and a value outside the enum throws instead of shifting out of range.
class ActivityTypeMask {
 public:
  explicit ActivityTypeMask(const std::set<ActivityType>& types) {
    for (const auto type : types) {
      bits_.set(asIndex(type));
    }
  }

  bool contains(ActivityType type) const {
    return bits_.test(asIndex(type));
  }

  // Calls visit() for each activity type in the set, in ascending order.
  void forEach(std::invocable<ActivityType> auto&& visit) const {
    for (size_t type = 0; type < bits_.size(); ++type) {
      if (bits_.test(type)) {
        visit(static_cast<ActivityType>(type));
      }
    }
  }

 private:
  static constexpr size_t asIndex(ActivityType type) {
    return static_cast<size_t>(type);
  }

  std::bitset<libkineto::activityTypeCount> bits_;
};

} // namespace KINETO_NAMESPACE
