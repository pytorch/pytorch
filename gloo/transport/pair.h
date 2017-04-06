/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#pragma once

#include <memory>

#include "gloo/transport/address.h"
#include "gloo/transport/buffer.h"

namespace gloo {
namespace transport {

class Pair {
 public:
  virtual ~Pair() = 0;

  virtual const Address& address() const = 0;

  virtual void connect(const std::vector<char>& bytes) = 0;

  virtual void setSync(bool enable, bool busyPoll) = 0;

  virtual std::unique_ptr<Buffer>
  createSendBuffer(int slot, void* ptr, size_t size) = 0;

  virtual std::unique_ptr<Buffer>
  createRecvBuffer(int slot, void* ptr, size_t size) = 0;
};

} // namespace transport
} // namespace gloo
