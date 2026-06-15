/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifdef USE_KINETO

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

  if (loggers_.count(protocol)) {
    TORCH_WARN(
        "Custom logger for protocol '", protocol,
        "' already registered, overwriting");
  }

  loggers_[protocol] = std::move(factory);

  // If Kineto was already initialized, register immediately
  if (kineto_initialized_ && !registered_with_kineto_) {
    registerWithKineto();
  } else if (kineto_initialized_ && registered_with_kineto_) {
    // Kineto already initialized and we already registered others,
    // register this new one immediately
    libkineto::registerLoggerFactory(protocol, loggers_[protocol]);
  }
}

bool CustomLoggerRegistry::hasLogger(const std::string& protocol) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return loggers_.count(protocol) > 0;
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
  // Note: Caller must hold mutex_
  if (loggers_.empty() || registered_with_kineto_) {
    return;
  }

  // Register all logger factories with Kineto
  for (const auto& [protocol, factory] : loggers_) {
    libkineto::registerLoggerFactory(protocol, factory);
  }

  registered_with_kineto_ = true;
}

void CustomLoggerRegistry::onKinetoInit() {
  std::lock_guard<std::mutex> lock(mutex_);
  kineto_initialized_ = true;

  // If loggers were registered before Kineto init, register them now
  if (!loggers_.empty() && !registered_with_kineto_) {
    registerWithKineto();
  }
}

} // namespace torch::profiler::impl

#endif // USE_KINETO
