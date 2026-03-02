/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <torch/csrc/profiler/standalone/privateuse1_profiler.h>

#ifdef USE_KINETO
#include <libkineto.h>
#endif

namespace torch::profiler::impl {

#ifdef USE_KINETO

PrivateUse1ProfilerRegistry& PrivateUse1ProfilerRegistry::instance() {
  static PrivateUse1ProfilerRegistry registry;
  return registry;
}

void PrivateUse1ProfilerRegistry::registerFactory(
    PrivateUse1ProfilerFactory factory) {
  std::lock_guard<std::mutex> lock(mutex_);
  factory_ = std::move(factory);

  // If Kineto was already initialized, forward immediately
  if (kineto_initialized_ && !forwarded_to_kineto_) {
    forwardToKineto();
  }
}

bool PrivateUse1ProfilerRegistry::hasFactory() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return factory_ != nullptr;
}

void PrivateUse1ProfilerRegistry::forwardToKineto() {
  // Note: Caller must hold mutex_
  if (!factory_ || forwarded_to_kineto_) {
    return;
  }

  // Forward the factory to Kineto's activity profiler
  // Kineto will call the factory to create the profiler instance
  libkineto::api().registerProfilerFactory(factory_);
  forwarded_to_kineto_ = true;
}

void PrivateUse1ProfilerRegistry::onKinetoInit() {
  std::lock_guard<std::mutex> lock(mutex_);
  kineto_initialized_ = true;

  // If a factory was registered before Kineto init, forward it now
  if (factory_ && !forwarded_to_kineto_) {
    forwardToKineto();
  }
}

#endif // USE_KINETO

} // namespace torch::profiler::impl
