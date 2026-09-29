/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace libkineto {

struct XpuMetricInfo {
  std::string name;
  std::string description;
  std::string unit;
  std::string metricType;
  std::string valueType;
};

struct XpuMetricGroupInfo {
  std::string name;
  std::string description;
  bool scopeCompatible{false};
  std::vector<XpuMetricInfo> metrics;
};

std::vector<XpuMetricGroupInfo> xpuptiAvailableMetrics(
    const std::array<uint8_t, 16>& deviceUuid);

} // namespace libkineto
