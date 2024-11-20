// Copyright (c) Meta Platforms, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <c10/util/env.h>
#include <torch/csrc/distributed/c10d/debug.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <string>

#include <torch/csrc/distributed/c10d/exception.h>
#include <torch/csrc/distributed/c10d/logging.h>

namespace c10d {
namespace detail {
namespace {

DebugLevel loadDebugLevelFromEnvironment() {
  auto env_value = c10::utils::get_env("TORCH_DISTRIBUTED_DEBUG");

  if (!env_value.has_value()) {
    return DebugLevel::Off;
  }

  DebugLevel level{};

  std::string level_str = std::move(env_value.value());

  std::transform(
      level_str.begin(),
      level_str.end(),
      level_str.begin(),
      [](unsigned char c) { return toupper(c); });

  if (level_str == "OFF") {
    level = DebugLevel::Off;
  } else if (level_str == "INFO") {
    level = DebugLevel::Info;
  } else if (level_str == "DETAIL") {
    level = DebugLevel::Detail;
  } else {
    throw std::invalid_argument(
        "The value of TORCH_DISTRIBUTED_DEBUG must be OFF, INFO, or DETAIL.");
  }

  C10D_INFO("The debug level is set to {}.", level_str);

  return level;
}

} // namespace
} // namespace detail

namespace {

DebugLevel g_debug_level = DebugLevel::Off;

} // namespace

void setDebugLevel(DebugLevel level) {
  g_debug_level = level;
}

void setDebugLevelFromEnvironment() {
  g_debug_level = detail::loadDebugLevelFromEnvironment();
}

DebugLevel debug_level() noexcept {
  return g_debug_level;
}

} // namespace c10d
