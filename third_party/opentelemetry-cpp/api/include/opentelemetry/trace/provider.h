// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <mutex>

#include "opentelemetry/common/macros.h"
#include "opentelemetry/common/spin_lock_mutex.h"
#include "opentelemetry/nostd/shared_ptr.h"
#include "opentelemetry/trace/noop.h"
#include "opentelemetry/version.h"

OPENTELEMETRY_BEGIN_NAMESPACE
namespace trace
{

class TracerProvider;

/**
 * Stores the singleton global TracerProvider.
 */
class OPENTELEMETRY_EXPORT Provider
{
public:
  /**
   * Returns the singleton TracerProvider.
   *
   * By default, a no-op TracerProvider is returned. This will never return a
   * nullptr TracerProvider.
   */
  static nostd::shared_ptr<TracerProvider> GetTracerProvider() noexcept
  {
    std::lock_guard<common::SpinLockMutex> guard(GetLock());
    return nostd::shared_ptr<TracerProvider>(GetProvider());
  }

  /**
   * Changes the singleton TracerProvider.
   */
  static void SetTracerProvider(nostd::shared_ptr<TracerProvider> tp) noexcept
  {
    std::lock_guard<common::SpinLockMutex> guard(GetLock());
    GetProvider() = tp;
  }

private:
  OPENTELEMETRY_API_SINGLETON static nostd::shared_ptr<TracerProvider> &GetProvider() noexcept
  {
    static nostd::shared_ptr<TracerProvider> provider(new NoopTracerProvider);
    return provider;
  }

  OPENTELEMETRY_API_SINGLETON static common::SpinLockMutex &GetLock() noexcept
  {
    static common::SpinLockMutex lock;
    return lock;
  }
};

}  // namespace trace
OPENTELEMETRY_END_NAMESPACE
