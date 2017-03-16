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
#include <vector>

#ifdef HIREDIS_NESTED_INCLUDE
#include <hiredis/hiredis.h>
#else
#include <hiredis.h>
#endif

#include "gloo/rendezvous/store.h"

namespace gloo {
namespace rendezvous {

class RedisStore : public Store {
 public:
  explicit RedisStore(const std::string& host, int port = 6379);
  virtual ~RedisStore();

  virtual void set(const std::string& key, const std::vector<char>& data)
      override;

  virtual std::vector<char> get(const std::string& key) override;

  bool check(const std::vector<std::string>& keys);

  virtual void wait(const std::vector<std::string>& keys) override;

 protected:
  redisContext* redis_;
};

} // namespace rendezvous
} // namespace gloo
