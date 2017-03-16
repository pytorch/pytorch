/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#pragma once

#include <exception>

#include "gloo/common/string.h"

namespace gloo {

// A base class for all gloo runtime errors
class Exception : public std::runtime_error {
 public:
  Exception() = default;
  explicit Exception(const std::string& msg) : std::runtime_error(msg) {}
};

#define GLOO_THROW(...)                                    \
  throw ::gloo::Exception(::gloo::MakeString(__VA_ARGS__))

} // namespace gloo
