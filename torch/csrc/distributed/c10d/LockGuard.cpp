// Copyright (c) Meta Platforms, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <torch/csrc/distributed/c10d/LockGuard.hpp>

#include <torch/csrc/distributed/c10d/logging.h>

namespace c10d::detail {

void lockWithLogging(
    std::unique_lock<std::timed_mutex>& lock,
    std::chrono::milliseconds log_interval,
    const char* desc,
    const char* file,
    int line) {
  while (!lock.try_lock_for(log_interval)) {
    C10D_WARNING(
        "{}:{} {}: waiting for lock for {}ms",
        file,
        line,
        desc,
        log_interval.count());
  }
}

} // namespace c10d::detail
