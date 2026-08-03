// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstdint>

#include <c10/macros/Export.h>

namespace c10d {

class TORCH_API NCCLCommProvider {
 public:
  virtual ~NCCLCommProvider() = default;
  virtual int64_t getCommPtr() = 0;
};

} // namespace c10d
