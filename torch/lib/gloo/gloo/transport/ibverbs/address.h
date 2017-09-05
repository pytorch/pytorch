/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#pragma once

#include <string>

#include <infiniband/verbs.h>

#include "gloo/transport/address.h"

namespace gloo {
namespace transport {
namespace ibverbs {

// Forward declaration
class Pair;

class Address : public ::gloo::transport::Address {
 public:
  Address();
  explicit Address(const std::vector<char>&);
  virtual ~Address() {}

  virtual std::vector<char> bytes() const override;
  virtual std::string str() const override;

 protected:
  struct {
    uint32_t lid;
    uint32_t qpn;
    uint32_t psn;
    union ibv_gid ibv_gid;
  } addr_;

  // Pair can access addr_ directly
  friend class Pair;
};

} // namespace ibverbs
} // namespace transport
} // namespace gloo
