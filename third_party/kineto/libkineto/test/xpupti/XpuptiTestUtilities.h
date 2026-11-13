/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "include/output_base.h"

#include <functional>

namespace KN = KINETO_NAMESPACE;

bool IsEnvVerbose();

std::pair<
    std::unique_ptr<KN::IActivityProfilerSession>,
    std::unique_ptr<KN::CpuTraceBuffer>>
RunProfilerTest(
    const std::vector<std::string_view>& metrics,
    const std::set<KN::ActivityType>& activities,
    const KN::Config& cfg,
    unsigned repeatCount,
    std::vector<std::string_view>&& expectedActivities,
    std::vector<std::string_view>&& expectedTypes,
    int64_t userCorrelationId = 0,
    const KN::ITraceActivity* linkedCpuActivity = nullptr,
    std::function<const KN::ITraceActivity*(int32_t)> linkedActivityCallback =
        nullptr);
