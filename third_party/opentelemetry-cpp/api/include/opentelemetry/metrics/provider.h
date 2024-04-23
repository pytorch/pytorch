// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <mutex>

#include "opentelemetry/common/macros.h"
#include "opentelemetry/common/spin_lock_mutex.h"
#include "opentelemetry/metrics/noop.h"
#include "opentelemetry/nostd/shared_ptr.h"
#include "opentelemetry/version.h"

OPENTELEMETRY_BEGIN_NAMESPACE
namespace metrics
{

class MeterProvider;

/**
 * Stores the singleton global MeterProvider.
 */
class Provider
{
public:
  /**
   * Returns the singleton MeterProvider.
   *
   * By default, a no-op MeterProvider is returned. This will never return a
   * nullptr MeterProvider.
   */
  static nostd::shared_ptr<MeterProvider> GetMeterProvider() noexcept
  {
    std::lock_guard<common::SpinLockMutex> guard(GetLock());
    return nostd::shared_ptr<MeterProvider>(GetProvider());
  }

  /**
   * Changes the singleton MeterProvider.
   */
  static void SetMeterProvider(nostd::shared_ptr<MeterProvider> tp) noexcept
  {
    std::lock_guard<common::SpinLockMutex> guard(GetLock());
    GetProvider() = tp;
  }

private:
  OPENTELEMETRY_API_SINGLETON static nostd::shared_ptr<MeterProvider> &GetProvider() noexcept
  {
    static nostd::shared_ptr<MeterProvider> provider(new NoopMeterProvider);
    return provider;
  }

  OPENTELEMETRY_API_SINGLETON static common::SpinLockMutex &GetLock() noexcept
  {
    static common::SpinLockMutex lock;
    return lock;
  }
};

}  // namespace metrics
OPENTELEMETRY_END_NAMESPACE
