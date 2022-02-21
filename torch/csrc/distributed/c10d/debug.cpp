// Copyright (c) Meta Platforms, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "debug.h"

#include <cstdlib>
#include <string>

#include "logging.h"
#include "exception.h"

namespace c10d {
namespace detail {
namespace {

DebugLevel loadDebugLevelFromEnvironment() {
  char* env_value = std::getenv("TORCH_DISTRIBUTED_DEBUG");

  if (env_value == nullptr) {
    return DebugLevel::Off;
  }

  DebugLevel level{};

  std::string level_str{env_value};

  if (level_str == "OFF" || level_str == "off") {
    level = DebugLevel::Off;
  } else if (level_str == "INFO" || level_str == "info") {
    level = DebugLevel::Info;
  } else if (level_str == "DETAIL" || level_str == "detail") {
    level = DebugLevel::Detail;
  } else {
    throw C10dError{"The value of TORCH_DISTRIBUTED_DEBUG must be OFF, INFO, or DETAIL."};
  }

  C10D_INFO("The debug level is set to {}.", level_str);

  return level;
}

} // namespace
} // namespace detail

namespace {

DebugLevel g_debug_level = DebugLevel::Off;

} // namespace

void setDebugLevel(at::optional<DebugLevel> opt_level, bool force) {
  static bool is_set = false;

  if (is_set && !force) {
    C10D_WARNING("The debug level is already set.");

    return;
  }

  if (opt_level) {
    g_debug_level = opt_level.value();
  } else {
    g_debug_level = detail::loadDebugLevelFromEnvironment();
  }

  is_set = true;
}

DebugLevel debug_level() noexcept {
  return g_debug_level;
}

} // namespace c10d
