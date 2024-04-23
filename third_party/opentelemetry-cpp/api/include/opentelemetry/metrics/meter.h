// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "opentelemetry/nostd/shared_ptr.h"
#include "opentelemetry/nostd/string_view.h"
#include "opentelemetry/nostd/unique_ptr.h"
#include "opentelemetry/version.h"

OPENTELEMETRY_BEGIN_NAMESPACE
namespace metrics
{

template <typename T>
class Counter;

template <typename T>
class Histogram;

template <typename T>
class UpDownCounter;

class ObservableInstrument;

/**
 * Handles instrument creation and provides a facility for batch recording.
 *
 * This class provides methods to create new metric instruments, record a
 * batch of values to a specified set of instruments, and collect
 * measurements from all instruments.
 *
 */
class Meter
{
public:
  virtual ~Meter() = default;

  /**
   * Creates a Counter with the passed characteristics and returns a unique_ptr to that Counter.
   *
   * @param name the name of the new Counter.
   * @param description a brief description of what the Counter is used for.
   * @param unit the unit of metric values following https://unitsofmeasure.org/ucum.html.
   * @return a shared pointer to the created Counter.
   */

  virtual nostd::unique_ptr<Counter<uint64_t>> CreateUInt64Counter(
      nostd::string_view name,
      nostd::string_view description = "",
      nostd::string_view unit        = "") noexcept = 0;

  virtual nostd::unique_ptr<Counter<double>> CreateDoubleCounter(
      nostd::string_view name,
      nostd::string_view description = "",
      nostd::string_view unit        = "") noexcept = 0;

  /**
   * Creates a Asynchronous (Observable) counter with the passed characteristics and returns a
   * shared_ptr to that Observable Counter
   *
   * @param name the name of the new Observable Counter.
   * @param description a brief description of what the Observable Counter is used for.
   * @param unit the unit of metric values following https://unitsofmeasure.org/ucum.html.
   */
  virtual nostd::shared_ptr<ObservableInstrument> CreateInt64ObservableCounter(
      nostd::string_view name,
      nostd::string_view description = "",
      nostd::string_view unit        = "") noexcept = 0;

  virtual nostd::shared_ptr<ObservableInstrument> CreateDoubleObservableCounter(
      nostd::string_view name,
      nostd::string_view description = "",
      nostd::string_view unit        = "") noexcept = 0;

  /**
   * Creates a Histogram with the passed characteristics and returns a unique_ptr to that Histogram.
   *
   * @param name the name of the new Histogram.
   * @param description a brief description of what the Histogram is used for.
   * @param unit the unit of metric values following https://unitsofmeasure.org/ucum.html.
   * @return a shared pointer to the created Histogram.
   */
  virtual nostd::unique_ptr<Histogram<uint64_t>> CreateUInt64Histogram(
      nostd::string_view name,
      nostd::string_view description = "",
      nostd::string_view unit        = "") noexcept = 0;

  virtual nostd::unique_ptr<Histogram<double>> CreateDoubleHistogram(
      nostd::string_view name,
      nostd::string_view description = "",
      nostd::string_view unit        = "") noexcept = 0;

  /**
   * Creates a Asynchronouse (Observable) Gauge with the passed characteristics and returns a
   * shared_ptr to that Observable Gauge
   *
   * @param name the name of the new Observable Gauge.
   * @param description a brief description of what the Observable Gauge is used for.
   * @param unit the unit of metric values following https://unitsofmeasure.org/ucum.html.
   */
  virtual nostd::shared_ptr<ObservableInstrument> CreateInt64ObservableGauge(
      nostd::string_view name,
      nostd::string_view description = "",
      nostd::string_view unit        = "") noexcept = 0;

  virtual nostd::shared_ptr<ObservableInstrument> CreateDoubleObservableGauge(
      nostd::string_view name,
      nostd::string_view description = "",
      nostd::string_view unit        = "") noexcept = 0;

  /**
   * Creates an UpDownCounter with the passed characteristics and returns a unique_ptr to that
   * UpDownCounter.
   *
   * @param name the name of the new UpDownCounter.
   * @param description a brief description of what the UpDownCounter is used for.
   * @param unit the unit of metric values following https://unitsofmeasure.org/ucum.html.
   * @return a shared pointer to the created UpDownCounter.
   */
  virtual nostd::unique_ptr<UpDownCounter<int64_t>> CreateInt64UpDownCounter(
      nostd::string_view name,
      nostd::string_view description = "",
      nostd::string_view unit        = "") noexcept = 0;

  virtual nostd::unique_ptr<UpDownCounter<double>> CreateDoubleUpDownCounter(
      nostd::string_view name,
      nostd::string_view description = "",
      nostd::string_view unit        = "") noexcept = 0;

  /**
   * Creates a Asynchronouse (Observable) UpDownCounter with the passed characteristics and returns
   * a shared_ptr to that Observable UpDownCounter
   *
   * @param name the name of the new Observable UpDownCounter.
   * @param description a brief description of what the Observable UpDownCounter is used for.
   * @param unit the unit of metric values following https://unitsofmeasure.org/ucum.html.
   */
  virtual nostd::shared_ptr<ObservableInstrument> CreateInt64ObservableUpDownCounter(
      nostd::string_view name,
      nostd::string_view description = "",
      nostd::string_view unit        = "") noexcept = 0;

  virtual nostd::shared_ptr<ObservableInstrument> CreateDoubleObservableUpDownCounter(
      nostd::string_view name,
      nostd::string_view description = "",
      nostd::string_view unit        = "") noexcept = 0;
};
}  // namespace metrics
OPENTELEMETRY_END_NAMESPACE
