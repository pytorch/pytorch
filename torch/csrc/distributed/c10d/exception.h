// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <stdexcept>

#include <c10/macros/Macros.h>

namespace c10d {

class TORCH_API C10dError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;

  C10dError(const C10dError&) = default;

  C10dError& operator=(const C10dError&) = default;

  C10dError(C10dError&&) = default;

  C10dError& operator=(C10dError&&) = default;

  ~C10dError() override;
};

class TORCH_API TimeoutException : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;

  TimeoutException(const TimeoutException&) = default;

  TimeoutException& operator=(const TimeoutException&) = default;

  TimeoutException(TimeoutException&&) = default;

  TimeoutException& operator=(TimeoutException&&) = default;

  ~TimeoutException() override;
};

} // namespace c10d
