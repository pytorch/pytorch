/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#pragma once

#include "store.h"

#include <memory>

namespace gloo {
namespace rendezvous {

class PrefixStore : public Store {
 public:
  PrefixStore(const std::string& prefix, Store& store);

  virtual ~PrefixStore() {}

  virtual void set(const std::string& key, const std::vector<char>& data)
      override;

  virtual std::vector<char> get(const std::string& key) override;

  virtual void wait(const std::vector<std::string>& keys) override {
    wait(keys, Store::kDefaultTimeout);
  }

  virtual void wait(
      const std::vector<std::string>& keys,
      const std::chrono::milliseconds& timeout) override;

 protected:
  const std::string prefix_;
  Store& store_;

  std::string joinKey(const std::string& key);
};

} // namespace rendezvous
} // namespace gloo
