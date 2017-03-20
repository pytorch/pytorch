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
struct Exception : public std::runtime_error {
  Exception() = default;
  explicit Exception(const std::string& msg) : std::runtime_error(msg) {}
};

#define GLOO_THROW(...) \
  throw ::gloo::Exception(::gloo::MakeString(__VA_ARGS__))


// Thrown for invalid operations on gloo APIs
struct InvalidOperationException : public ::gloo::Exception {
  InvalidOperationException() = default;
  explicit InvalidOperationException(const std::string& msg)
      : ::gloo::Exception(msg) {}
};

#define GLOO_THROW_INVALID_OPERATION_EXCEPTION(...) \
  throw ::gloo::InvalidOperationException(::gloo::MakeString(__VA_ARGS__))


// Thrown for unrecoverable IO errors
struct IoException : public ::gloo::Exception {
  IoException() = default;
  explicit IoException(const std::string& msg) : ::gloo::Exception(msg) {}
};

#define GLOO_THROW_IO_EXCEPTION(...) \
  throw ::gloo::IoException(::gloo::MakeString(__VA_ARGS__))

} // namespace gloo
