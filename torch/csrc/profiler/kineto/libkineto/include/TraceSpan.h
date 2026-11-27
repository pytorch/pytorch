/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <atomic>
#include <string>
#include <thread>

namespace libkineto {

struct TraceSpan {
  TraceSpan() = delete;
  TraceSpan(int64_t startTime, int64_t endTime, std::string name)
      : startTime(startTime), endTime(endTime), name(std::move(name)) {}
  TraceSpan(int opCount, int it, std::string name, std::string prefix)
      : opCount(opCount),
        iteration(it),
        name(std::move(name)),
        prefix(std::move(prefix)) {}

  // FIXME: change to duration?
  int64_t startTime{0};
  int64_t endTime{0};
  int opCount{0};
  int iteration{-1};
  // Name is used to identify timeline
  std::string name;
  // Prefix used to distinguish trace spans on the same timeline
  std::string prefix;
};

} // namespace libkineto
