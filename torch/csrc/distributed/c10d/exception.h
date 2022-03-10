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

class TORCH_API TimeoutError : public C10dError {
 public:
  using C10dError::C10dError;

  TimeoutError(const TimeoutError&) = default;

  TimeoutError& operator=(const TimeoutError&) = default;

  TimeoutError(TimeoutError&&) = default;

  TimeoutError& operator=(TimeoutError&&) = default;

  ~TimeoutError() override;
};

} // namespace c10d
