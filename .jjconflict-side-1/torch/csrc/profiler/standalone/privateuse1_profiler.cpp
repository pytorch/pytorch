/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifdef USE_KINETO

#include <c10/util/Exception.h>
#include <torch/csrc/profiler/standalone/privateuse1_profiler.h>

#include <libkineto.h>

namespace torch::profiler::impl {

PrivateUse1ProfilerRegistry& PrivateUse1ProfilerRegistry::instance() {
  static PrivateUse1ProfilerRegistry registry;
  return registry;
}

void PrivateUse1ProfilerRegistry::registerFactory(
    PrivateUse1ProfilerFactory factory) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (factory_) {
    TORCH_WARN("PrivateUse1 profiler factory already registered, overwriting");
  }

  factory_ = std::move(factory);

  // If Kineto was already initialized, register immediately
  if (kineto_initialized_ && !registered_with_kineto_) {
    registerWithKineto();
  }
}

bool PrivateUse1ProfilerRegistry::hasFactory() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return factory_ != nullptr;
}

bool PrivateUse1ProfilerRegistry::isRegisteredWithKineto() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return registered_with_kineto_;
}

void PrivateUse1ProfilerRegistry::registerWithKineto() {
  // Note: Caller must hold mutex_
  if (!factory_ || registered_with_kineto_) {
    return;
  }

  // Register the factory with Kineto's activity profiler
  // Kineto will call the factory to create the profiler instance
  libkineto::api().registerProfilerFactory(factory_);
  registered_with_kineto_ = true;
}

void PrivateUse1ProfilerRegistry::onKinetoInit() {
  std::lock_guard<std::mutex> lock(mutex_);
  kineto_initialized_ = true;

  // If a factory was registered before Kineto init, register it now
  if (factory_ && !registered_with_kineto_) {
    registerWithKineto();
  }
}

} // namespace torch::profiler::impl

#endif // USE_KINETO
