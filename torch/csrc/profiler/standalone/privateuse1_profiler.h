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
#include <type_traits>

#include <torch/csrc/Export.h>

#include <IActivityProfiler.h>

namespace torch::profiler::impl {

// Factory function type that creates an IActivityProfiler instance
using PrivateUse1ProfilerFactory =
    std::function<std::unique_ptr<libkineto::IActivityProfiler>()>;

// Registry for PrivateUse1 activity profiler factories.
//
// This registry allows custom accelerator backends to register their
// IActivityProfiler implementation with Kineto, enabling full profiling
// integration without modifying Kineto code.
//
// Usage:
//   1. Backend implements libkineto::IActivityProfiler
//   2. Backend uses REGISTER_PRIVATEUSE1_PROFILER macro to register
//   3. PyTorch forwards the factory to Kineto during initialization
//
// Example:
//   class MyAcceleratorProfiler : public libkineto::IActivityProfiler {
//     const std::string& name() const override { return name_; }
//     const std::set<libkineto::ActivityType>& availableActivities() const
//     override; std::unique_ptr<libkineto::IActivityProfilerSession>
//     configure(...) override;
//   private:
//     std::string name_{"my_accelerator"};
//   };
//
//   REGISTER_PRIVATEUSE1_PROFILER(MyAcceleratorProfiler)
//
class TORCH_API PrivateUse1ProfilerRegistry {
 public:
  static PrivateUse1ProfilerRegistry& instance();

  // Register a factory function for creating the PrivateUse1 profiler.
  // This should be called during static initialization.
  void registerFactory(PrivateUse1ProfilerFactory factory);

  // Check if a factory has been registered.
  bool hasFactory() const;

  // Check if the factory has been registered with Kineto.
  // Useful for testing to verify the registration logic.
  bool isRegisteredWithKineto() const;

  // Register the factory with Kineto's activity profiler.
  // This is called internally when Kineto is ready.
  // Safe to call multiple times - will only register once.
  void registerWithKineto();

  // Mark that Kineto has been initialized.
  // If a factory was registered before Kineto init, it will be forwarded.
  void onKinetoInit();

 private:
  PrivateUse1ProfilerRegistry() = default;

  mutable std::mutex mutex_;
  PrivateUse1ProfilerFactory factory_;
  bool registered_with_kineto_ = false;
  bool kineto_initialized_ = false;
};

// Helper struct for static registration via macro.
// Enforces at compile-time that ProfilerClass inherits from
// libkineto::IActivityProfiler.
template <typename ProfilerClass>
struct RegisterPrivateUse1Profiler {
  static_assert(
      std::is_base_of_v<libkineto::IActivityProfiler, ProfilerClass>,
      "ProfilerClass must inherit from libkineto::IActivityProfiler. "
      "Please ensure your profiler class implements the IActivityProfiler interface.");

  RegisterPrivateUse1Profiler() {
    PrivateUse1ProfilerRegistry::instance().registerFactory(
        []() -> std::unique_ptr<libkineto::IActivityProfiler> {
          return std::make_unique<ProfilerClass>();
        });
  }
};

// Macro for registering a PrivateUse1 activity profiler.
// The profiler class must implement libkineto::IActivityProfiler.
//
// Usage:
//   REGISTER_PRIVATEUSE1_PROFILER(MyAcceleratorProfiler)
#define REGISTER_PRIVATEUSE1_PROFILER(ProfilerClass)                         \
  static ::torch::profiler::impl::RegisterPrivateUse1Profiler<ProfilerClass> \
      privateuse1_profiler_register_##ProfilerClass

} // namespace torch::profiler::impl

#endif // USE_KINETO
