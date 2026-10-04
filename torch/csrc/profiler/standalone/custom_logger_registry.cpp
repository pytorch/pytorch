/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifdef USE_KINETO

#include <vector>

#include <c10/util/Exception.h>
#include <torch/csrc/profiler/standalone/custom_logger_registry.h>

#include <libkineto.h>

namespace torch::profiler::impl {

CustomLoggerRegistry& CustomLoggerRegistry::instance() {
  static CustomLoggerRegistry registry;
  return registry;
}

void CustomLoggerRegistry::registerLogger(
    const std::string& protocol,
    CustomLoggerFactory factory) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (loggers_.contains(protocol)) {
    TORCH_WARN(
        "Custom logger for protocol '",
        protocol,
        "' already registered, overwriting");
  }

  loggers_[protocol] = std::move(factory);

  if (kineto_initialized_ && !registered_with_kineto_) {
    registerWithKineto();
  } else if (kineto_initialized_ && registered_with_kineto_) {
    libkineto::registerLoggerFactory(protocol, loggers_[protocol]);
  }
}

bool CustomLoggerRegistry::hasLogger(const std::string& protocol) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return loggers_.contains(protocol);
}

size_t CustomLoggerRegistry::numLoggers() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return loggers_.size();
}

bool CustomLoggerRegistry::isRegisteredWithKineto() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return registered_with_kineto_;
}

void CustomLoggerRegistry::registerWithKineto() {
  if (loggers_.empty() || registered_with_kineto_) {
    return;
  }

  std::vector<std::string> failed;
  for (const auto& [protocol, factory] : loggers_) {
    try {
      libkineto::registerLoggerFactory(protocol, factory);
    } catch (const std::exception& e) {
      TORCH_WARN(
          "Failed to register logger for protocol '",
          protocol,
          "' with Kineto: ",
          e.what());
      failed.push_back(protocol);
    }
  }
  for (const auto& protocol : failed) {
    loggers_.erase(protocol);
  }

  registered_with_kineto_ = true;
}

void CustomLoggerRegistry::onKinetoInit() {
  std::lock_guard<std::mutex> lock(mutex_);
  kineto_initialized_ = true;

  if (!loggers_.empty() && !registered_with_kineto_) {
    registerWithKineto();
  }
}

} // namespace torch::profiler::impl

#endif // USE_KINETO
