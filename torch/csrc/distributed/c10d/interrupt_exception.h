// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <exception>

#include <c10/macros/Macros.h>

namespace c10d {

class TORCH_API InterruptException : public std::exception {
 public:
  InterruptException() = default;

  InterruptException(const InterruptException&) = default;

  InterruptException& operator=(const InterruptException&) = default;

  InterruptException(InterruptException&&) = default;

  InterruptException& operator=(InterruptException&&) = default;

  ~InterruptException() override;
};

} // namespace c10d
