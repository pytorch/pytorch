/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#ifdef USE_KINETO

#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include <torch/csrc/Export.h>

#include <output_base.h>

namespace torch::profiler::impl {

using CustomLoggerFactory =
    std::function<std::unique_ptr<libkineto::ActivityLogger>(
        const std::string&)>;

// Registry for custom ActivityLogger factories that are forwarded to Kineto.
class TORCH_API CustomLoggerRegistry {
 public:
  static CustomLoggerRegistry& instance();

  void registerLogger(const std::string& protocol, CustomLoggerFactory factory);
  bool hasLogger(const std::string& protocol) const;
  size_t numLoggers() const;
  bool isRegisteredWithKineto() const;
  void onKinetoInit();

 private:
  CustomLoggerRegistry() = default;

  void registerWithKineto(); // caller must hold mutex_

  mutable std::mutex mutex_;
  std::unordered_map<std::string, CustomLoggerFactory> loggers_;
  bool registered_with_kineto_ = false;
  bool kineto_initialized_ = false;
};

template <typename LoggerClass>
struct RegisterCustomLogger {
  RegisterCustomLogger(const std::string& protocol) {
    CustomLoggerRegistry::instance().registerLogger(
        protocol,
        [](const std::string& url)
            -> std::unique_ptr<libkineto::ActivityLogger> {
          return std::make_unique<LoggerClass>(url);
        });
  }
};

// Usage: REGISTER_CUSTOM_LOGGER("perfetto", PerfettoLogger)
#define REGISTER_CUSTOM_LOGGER(protocol, LoggerClass)               \
  static ::torch::profiler::impl::RegisterCustomLogger<LoggerClass> \
      custom_logger_register_##LoggerClass(protocol)

} // namespace torch::profiler::impl

#endif // USE_KINETO
