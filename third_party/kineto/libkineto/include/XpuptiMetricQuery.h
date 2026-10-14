/*
 * Copyright (C) Intel Corporation
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace libkineto {

constexpr std::size_t kXpuDeviceUuidSize = 16;
using XpuDeviceUuid = std::array<uint8_t, kXpuDeviceUuidSize>;

// A metric group's name and the names of its metrics.
using XpuMetricGroupNames = std::pair<std::string, std::vector<std::string>>;

std::vector<XpuMetricGroupNames> xpuptiAvailableMetrics(
    const XpuDeviceUuid& deviceUuid);

} // namespace libkineto
