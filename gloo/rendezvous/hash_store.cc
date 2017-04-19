/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "gloo/rendezvous/hash_store.h"

#include "gloo/common/error.h"
#include "gloo/common/logging.h"

namespace gloo {
namespace rendezvous {

void HashStore::set(const std::string& key, const std::vector<char>& data) {
  std::unique_lock<std::mutex> lock(m_);
  GLOO_ENFORCE(map_.find(key) == map_.end(), "Key '", key, "' already set");
  map_[key] = data;
  cv_.notify_all();
}

std::vector<char> HashStore::get(const std::string& key) {
  std::unique_lock<std::mutex> lock(m_);
  auto it = map_.find(key);
  if (it == map_.end()) {
    return std::vector<char>();
  }

  return it->second;
}

void HashStore::wait(
    const std::vector<std::string>& keys,
    const std::chrono::milliseconds& timeout) {
  const auto end = std::chrono::steady_clock::now() + timeout;
  auto pred = [&](){
    auto done = true;
    for (const auto& key : keys) {
      if (map_.find(key) == map_.end()) {
        done = false;
        break;
      }
    }
    return done;
  };

  std::unique_lock<std::mutex> lock(m_);
  if (timeout == kNoTimeout) {
    cv_.wait(lock, pred);
  } else {
    if (!cv_.wait_until(lock, end, pred)) {
      GLOO_THROW_IO_EXCEPTION(GLOO_ERROR_MSG(
          "Wait timeout for key(s): ", ::gloo::MakeString(keys)));
    }
  }
}

} // namespace rendezvous
} // namespace gloo
