// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <mutex>

#include "opentelemetry/common/macros.h"
#include "opentelemetry/common/spin_lock_mutex.h"
#include "opentelemetry/logs/noop.h"
#include "opentelemetry/nostd/shared_ptr.h"
#include "opentelemetry/version.h"

OPENTELEMETRY_BEGIN_NAMESPACE
namespace logs
{

class EventLoggerProvider;
class LoggerProvider;

/**
 * Stores the singleton global LoggerProvider.
 */
class OPENTELEMETRY_EXPORT Provider
{
public:
  /**
   * Returns the singleton LoggerProvider.
   *
   * By default, a no-op LoggerProvider is returned. This will never return a
   * nullptr LoggerProvider.
   */
  static nostd::shared_ptr<LoggerProvider> GetLoggerProvider() noexcept
  {
    std::lock_guard<common::SpinLockMutex> guard(GetLock());
    return nostd::shared_ptr<LoggerProvider>(GetProvider());
  }

  /**
   * Changes the singleton LoggerProvider.
   */
  static void SetLoggerProvider(nostd::shared_ptr<LoggerProvider> tp) noexcept
  {
    std::lock_guard<common::SpinLockMutex> guard(GetLock());
    GetProvider() = tp;
  }

  /**
   * Returns the singleton EventLoggerProvider.
   *
   * By default, a no-op EventLoggerProvider is returned. This will never return a
   * nullptr EventLoggerProvider.
   */
  static nostd::shared_ptr<EventLoggerProvider> GetEventLoggerProvider() noexcept
  {
    std::lock_guard<common::SpinLockMutex> guard(GetLock());
    return nostd::shared_ptr<EventLoggerProvider>(GetEventProvider());
  }

  /**
   * Changes the singleton EventLoggerProvider.
   */
  static void SetEventLoggerProvider(nostd::shared_ptr<EventLoggerProvider> tp) noexcept
  {
    std::lock_guard<common::SpinLockMutex> guard(GetLock());
    GetEventProvider() = tp;
  }

private:
  OPENTELEMETRY_API_SINGLETON static nostd::shared_ptr<LoggerProvider> &GetProvider() noexcept
  {
    static nostd::shared_ptr<LoggerProvider> provider(new NoopLoggerProvider);
    return provider;
  }

  OPENTELEMETRY_API_SINGLETON static nostd::shared_ptr<EventLoggerProvider>
      &GetEventProvider() noexcept
  {
    static nostd::shared_ptr<EventLoggerProvider> provider(new NoopEventLoggerProvider);
    return provider;
  }

  OPENTELEMETRY_API_SINGLETON static common::SpinLockMutex &GetLock() noexcept
  {
    static common::SpinLockMutex lock;
    return lock;
  }
};

}  // namespace logs
OPENTELEMETRY_END_NAMESPACE
