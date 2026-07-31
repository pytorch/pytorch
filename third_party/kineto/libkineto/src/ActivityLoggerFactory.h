/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <fmt/format.h>
#include <algorithm>
#include <cctype>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <utility>

namespace KINETO_NAMESPACE {

class ActivityLogger;

class ActivityLoggerFactory {
 public:
  using FactoryFunc =
      std::function<std::unique_ptr<ActivityLogger>(const std::string& url)>;

  // Add a logger factory for a protocol prefix. Returns true if a factory was
  // already registered for the protocol and is being overwritten.
  bool addProtocol(const std::string& protocol, FactoryFunc f) {
    const std::string key = tolower(protocol);
    std::lock_guard<std::mutex> guard(mutex_);
    const bool overwritten = factories_.contains(key);
    factories_[key] = std::move(f);
    return overwritten;
  }

  // Create a logger, invoking the factory for the protocol specified in url
  std::unique_ptr<ActivityLogger> makeLogger(const std::string& url) const {
    const std::string protocol = extractProtocol(url);
    FactoryFunc factory;
    {
      std::lock_guard<std::mutex> guard(mutex_);
      auto it = factories_.find(tolower(protocol));
      if (it != factories_.end()) {
        factory = it->second;
      }
    }
    // Invoke the factory outside the lock so a user-supplied callback cannot
    // deadlock by re-entering the registry while mutex_ is held.
    if (factory) {
      return factory(stripProtocol(url));
    }
    throw std::invalid_argument(fmt::format(
        "No logger registered for the {} protocol prefix", protocol));
  }

 private:
  static std::string tolower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
      return std::tolower(c);
    });
    return s;
  }

  static std::string extractProtocol(const std::string& url) {
    return url.substr(0, url.find("://"));
  }

  static std::string stripProtocol(const std::string& url) {
    size_t pos = url.find("://");
    return pos == url.npos ? url : url.substr(pos + 3);
  }

  std::map<std::string, FactoryFunc> factories_;
  // registerLoggerFactory is public API and can run concurrently with trace
  // saves that call makeLogger; serialize all access to factories_.
  mutable std::mutex mutex_;
};

} // namespace KINETO_NAMESPACE
