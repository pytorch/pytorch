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

// Factory function type that creates an ActivityLogger instance
// The string parameter is the URL/filename passed to the logger
using CustomLoggerFactory =
    std::function<std::unique_ptr<libkineto::ActivityLogger>(const std::string&)>;

// Registry for custom ActivityLogger factories.
//
// This registry allows users to register custom ActivityLogger implementations
// with Kineto, enabling custom trace output formats (e.g., Perfetto, Chrome
// Trace with extensions, custom binary formats) without modifying Kineto code.
//
// Usage:
//   1. Implement libkineto::ActivityLogger with custom output format
//   2. Use REGISTER_CUSTOM_LOGGER macro to register with a protocol name
//   3. PyTorch forwards the factory to Kineto during initialization
//   4. Use the logger by specifying the protocol in profiler config
//
// Example:
//   class PerfettoLogger : public libkineto::ActivityLogger {
//    public:
//     explicit PerfettoLogger(const std::string& filename);
//     void handleActivity(const libkineto::ITraceActivity& activity) override;
//     void finalizeTrace(...) override;
//   };
//
//   REGISTER_CUSTOM_LOGGER("perfetto", PerfettoLogger)
//
//   Then in Python:
//     with torch.profiler.profile(
//       activities=[...],
//       on_trace_ready=torch.profiler.tensorboard_trace_handler(
//         "perfetto://output.perfetto"
//       )
//     )
//
class TORCH_API CustomLoggerRegistry {
 public:
  static CustomLoggerRegistry& instance();

  // Register a factory function for creating a custom logger.
  // The protocol string identifies this logger type (e.g., "perfetto", "custom").
  // This should be called during static initialization.
  void registerLogger(const std::string& protocol, CustomLoggerFactory factory);

  // Check if a logger has been registered for the given protocol.
  bool hasLogger(const std::string& protocol) const;

  // Get the number of registered loggers.
  size_t numLoggers() const;

  // Check if the loggers have been registered with Kineto.
  // Useful for testing to verify the registration logic.
  bool isRegisteredWithKineto() const;

  // Mark that Kineto has been initialized.
  // If loggers were registered before Kineto init, they will be forwarded.
  void onKinetoInit();

 private:
  CustomLoggerRegistry() = default;

  // Register all loggers with Kineto's activity logger registry.
  // Caller must hold mutex_.
  void registerWithKineto();

  mutable std::mutex mutex_;
  std::unordered_map<std::string, CustomLoggerFactory> loggers_;
  bool registered_with_kineto_ = false;
  bool kineto_initialized_ = false;
};

// Helper struct for static registration via macro.
template <typename LoggerClass>
struct RegisterCustomLogger {
  RegisterCustomLogger(const std::string& protocol) {
    CustomLoggerRegistry::instance().registerLogger(
        protocol,
        [](const std::string& url) -> std::unique_ptr<libkineto::ActivityLogger> {
          return std::make_unique<LoggerClass>(url);
        });
  }
};

// Macro for registering a custom activity logger.
// The logger class must implement libkineto::ActivityLogger.
//
// Usage:
//   REGISTER_CUSTOM_LOGGER("perfetto", PerfettoLogger)
//
// The protocol string will be used to identify this logger when configuring
// the profiler output.
#define REGISTER_CUSTOM_LOGGER(protocol, LoggerClass)                \
  static ::torch::profiler::impl::RegisterCustomLogger<LoggerClass> \
      custom_logger_register_##LoggerClass(protocol)

} // namespace torch::profiler::impl

#endif // USE_KINETO
