// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <mutex>

#include "opentelemetry/context/propagation/noop_propagator.h"

#include "opentelemetry/common/macros.h"
#include "opentelemetry/common/spin_lock_mutex.h"
#include "opentelemetry/nostd/shared_ptr.h"

#include "opentelemetry/version.h"

OPENTELEMETRY_BEGIN_NAMESPACE
namespace context
{
namespace propagation
{

class TextMapPropagator;

/* Stores the singleton TextMapPropagator */

class OPENTELEMETRY_EXPORT GlobalTextMapPropagator
{
public:
  static nostd::shared_ptr<TextMapPropagator> GetGlobalPropagator() noexcept
  {
    std::lock_guard<common::SpinLockMutex> guard(GetLock());
    return nostd::shared_ptr<TextMapPropagator>(GetPropagator());
  }

  static void SetGlobalPropagator(nostd::shared_ptr<TextMapPropagator> prop) noexcept
  {
    std::lock_guard<common::SpinLockMutex> guard(GetLock());
    GetPropagator() = prop;
  }

private:
  OPENTELEMETRY_API_SINGLETON static nostd::shared_ptr<TextMapPropagator> &GetPropagator() noexcept
  {
    static nostd::shared_ptr<TextMapPropagator> propagator(new NoOpPropagator());
    return propagator;
  }

  OPENTELEMETRY_API_SINGLETON static common::SpinLockMutex &GetLock() noexcept
  {
    static common::SpinLockMutex lock;
    return lock;
  }
};

}  // namespace propagation
}  // namespace context
OPENTELEMETRY_END_NAMESPACE
