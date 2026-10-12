/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cupti.h>

#include <cstdint>

#include "ApproximateClock.h"

namespace KINETO_NAMESPACE {

// Configure the timestamp source shared by CUPTI Activity and PM Sampling.
void configureCuptiTimestampSource(bool useTsc);
bool isCuptiTimestampSourceReady();

// This function allows us to activate/deactivate TSC CUPTI callbacks
// via a killswitch.
bool& use_cupti_tsc();

// If we are running on Windows or are on a CUDA version < 11.6,
// we use the default system clock so no conversion needed.
inline int64_t convertCuptiTimestamp(uint64_t timestamp) {
#if defined(_WIN32) || CUDA_VERSION < 11060
  return timestamp;
#else
  if (use_cupti_tsc()) {
    return get_time_converter()(timestamp);
  } else {
    return timestamp;
  }
#endif
}

inline int64_t convertCuptiDuration(uint64_t start, uint64_t end) {
#if defined(_WIN32) || CUDA_VERSION < 11060
  return end - start;
#else
  if (use_cupti_tsc()) {
    return get_time_converter()(end) - get_time_converter()(start);
  } else {
    return end - start;
  }
#endif
}

} // namespace KINETO_NAMESPACE
