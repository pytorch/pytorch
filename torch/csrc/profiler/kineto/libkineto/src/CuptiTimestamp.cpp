/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "CuptiTimestamp.h"

#include <chrono>

#include "DeviceUtil.h"
#include "Logger.h"

namespace KINETO_NAMESPACE {
namespace {

bool timestampSourceReady{false};

} // namespace

bool& use_cupti_tsc() {
  static bool use_cupti_tsc = true;
  return use_cupti_tsc;
}

void configureCuptiTimestampSource([[maybe_unused]] bool useTsc) {
  bool configured{true};
#ifdef _WIN32
  configured =
      CUPTI_CALL(cuptiActivityRegisterTimestampCallback([]() -> uint64_t {
        auto system = std::chrono::time_point_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now());
        return system.time_since_epoch().count();
      })) == CUPTI_SUCCESS;
#else
#if CUDA_VERSION >= 11060
  use_cupti_tsc() = useTsc;
  if (use_cupti_tsc()) {
    configured =
        CUPTI_CALL(cuptiActivityRegisterTimestampCallback([]() -> uint64_t {
          return getApproximateTime();
        })) == CUPTI_SUCCESS;
  }
#endif // CUDA_VERSION >= 11060
#endif // _WIN32
  timestampSourceReady = configured;
}

bool isCuptiTimestampSourceReady() {
  return timestampSourceReady;
}

} // namespace KINETO_NAMESPACE
