// Copyright (c) Meta Platforms, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <chrono>
#include <mutex>

#include <c10/macros/Export.h>

namespace c10d::detail {

// logWithLogging is a wrapper around std::unique_lock<std::timed_mutex>
// that automatically logs if the lock cannot be acquired within a given
// timeout.
TORCH_API void lockWithLogging(
    std::unique_lock<std::timed_mutex>& lock,
    std::chrono::milliseconds log_interval,
    const char* desc,
    const char* file,
    int line);

} // namespace c10d::detail

// TODO: use std::source_location() when we can use C++20
#define C10D_LOCK_GUARD(name, mutex)                               \
  std::unique_lock<std::timed_mutex> name{mutex, std::defer_lock}; \
  ::c10d::detail::lockWithLogging(                                 \
      name, std::chrono::seconds(30), #mutex, __FILE__, __LINE__)
