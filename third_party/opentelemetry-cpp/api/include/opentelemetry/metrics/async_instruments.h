// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "opentelemetry/metrics/observer_result.h"
#include "opentelemetry/version.h"

OPENTELEMETRY_BEGIN_NAMESPACE
namespace metrics
{

using ObservableCallbackPtr = void (*)(ObserverResult, void *);

class ObservableInstrument
{
public:
  ObservableInstrument()          = default;
  virtual ~ObservableInstrument() = default;

  /**
   * Sets up a function that will be called whenever a metric collection is initiated.
   */
  virtual void AddCallback(ObservableCallbackPtr, void *state) noexcept = 0;

  /**
   * Remove a function that was configured to be called whenever a metric collection is initiated.
   */
  virtual void RemoveCallback(ObservableCallbackPtr, void *state) noexcept = 0;
};

}  // namespace metrics
OPENTELEMETRY_END_NAMESPACE
