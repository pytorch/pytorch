/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#pragma once

#include <chrono>
#include <memory>

#include "gloo/transport/pair.h"

namespace gloo {
namespace transport {

// Forward declarations
class Pair;
class Buffer;

class Device {
 public:
  virtual ~Device() = 0;

  virtual std::string str() const = 0;

  virtual const std::string& getPCIBusID() const = 0;

  virtual int getInterfaceSpeed() const { return 0; }

  virtual std::chrono::milliseconds getTimeout() const = 0;

  virtual void setTimeout(const std::chrono::milliseconds& timeout) = 0;

  virtual std::unique_ptr<Pair> createPair() = 0;
};

} // namespace transport
} // namespace gloo
