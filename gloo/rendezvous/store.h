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
#include <string>
#include <vector>

#include "gloo/common/logging.h"

namespace gloo {
namespace rendezvous {

class Store {
 public:
  static constexpr std::chrono::milliseconds kDefaultTimeout =
      std::chrono::seconds(30);

  virtual ~Store();

  virtual void set(const std::string& key, const std::vector<char>& data) = 0;

  virtual std::vector<char> get(const std::string& key) = 0;

  virtual void wait(
      const std::vector<std::string>& keys) = 0;

  virtual void wait(
      const std::vector<std::string>& keys,
      const std::chrono::milliseconds& /*timeout*/) {
    // Base implementation ignores the timeout for backward compatibility.
    // Derived Store implementations should override this function.
    wait(keys);
  }

};

} // namespace rendezvous
} // namespace gloo
