/**
 * This file is adapted from PyTorch/XLA
 * https://github.com/pytorch/xla/blob/master/third_party/xla_client/metrics.h
 */

#pragma once

#include <atomic>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <c10/macros/Export.h>

namespace torch {
namespace lazy {

struct TORCH_API Sample {
  Sample() = default;
  Sample(int64_t timestamp_ns, double value)
      : timestamp_ns(timestamp_ns), value(value) {}

  int64_t timestamp_ns = 0;
  double value = 0;
};

using MetricReprFn = std::function<std::string(double)>;

// Class used to collect time-stamped numeric samples. The samples are stored in
// a circular buffer whose size can be configured at constructor time.
class TORCH_API MetricData {
 public:
  // Creates a new MetricData object with the internal circular buffer storing
  // max_samples samples. The repr_fn argument allow to specify a function which
  // pretty-prints a sample value.
  MetricData(MetricReprFn repr_fn, size_t max_samples);

  // Returns the total values of all the samples being posted to this metric.
  double Accumulator() const;

  size_t TotalSamples() const;

  void AddSample(int64_t timestamp_ns, double value);

  // Returns a vector with all the current samples, from the oldest to the
  // newer. If accumulator is not nullptr, it will receive the current value of
  // the metrics' accumulator (the sum of all posted values). If total_samples
  // is not nullptr, it will receive the count of the posted values.
  std::vector<Sample> Samples(double* accumulator, size_t* total_samples) const;

  std::string Repr(double value) const {
    return repr_fn_(value);
  }

 private:
  mutable std::mutex lock_;
  MetricReprFn repr_fn_;
  size_t count_ = 0;
  std::vector<Sample> samples_;
  double accumulator_ = 0.0;
};

// Counters are a very lightweight form of metrics which do not need to track
// sample time.
class TORCH_API CounterData {
 public:
  CounterData() : value_(0) {}

  void AddValue(int64_t value) {
    value_ += value;
  }

  int64_t Value() const {
    return value_;
  }

  void Reset() {
    value_ = 0;
  }

 private:
  std::atomic<int64_t> value_;
};

class TORCH_API MetricsArena {
 public:
  static MetricsArena* Get();

  void Reset();

  // Registers a new metric in the global arena.
  void RegisterMetric(
      const std::string& name,
      MetricReprFn repr_fn,
      size_t max_samples,
      std::shared_ptr<MetricData>* data);

  void RegisterCounter(
      const std::string& name,
      std::shared_ptr<CounterData>* data);

  void ForEachMetric(
      const std::function<void(const std::string&, MetricData*)>& metric_func);

  void ForEachCounter(
      const std::function<void(const std::string&, CounterData*)>&
          counter_func);

  std::vector<std::string> GetMetricNames();

  MetricData* GetMetric(const std::string& name);

  std::vector<std::string> GetCounterNames();

  CounterData* GetCounter(const std::string& name);

 private:
  std::mutex lock_;
  std::map<std::string, std::shared_ptr<MetricData>> metrics_;
  std::map<std::string, std::shared_ptr<CounterData>> counters_;
};

// Emits the value in a to_string() conversion.
TORCH_API std::string MetricFnValue(double value);
// Emits the value in a humanized bytes representation.
TORCH_API std::string MetricFnBytes(double value);
// Emits the value in a humanized time representation. The value is expressed in
// nanoseconds EPOCH time.
TORCH_API std::string MetricFnTime(double value);

// The typical use of a Metric is one in which it gets created either in a
// global scope context:
//   static Metric* metric = new Metric("RpcCount");
// Or within a function scope:
//   void MyFunction(...) {
//     static Metric* metric = new Metric("RpcCount");
//     ...
//     metric->AddSample(ts_nanos, some_value);
//   }
class TORCH_API Metric {
 public:
  explicit Metric(
      std::string name,
      MetricReprFn repr_fn = MetricFnValue,
      size_t max_samples = 0);

  const std::string& Name() const {
    return name_;
  }

  double Accumulator() const;

  void AddSample(int64_t timestamp_ns, double value);

  void AddSample(double value);

  std::vector<Sample> Samples(double* accumulator, size_t* total_samples) const;

  std::string Repr(double value) const;

 private:
  MetricData* GetData() const;

  std::string name_;
  MetricReprFn repr_fn_;
  size_t max_samples_;
  mutable std::shared_ptr<MetricData> data_ptr_;
  mutable std::atomic<MetricData*> data_;
};

// A Counter is a lightweight form of metric which tracks an integer value which
// can increase or decrease.
// A typical use is as:
//   static Counter* counter = new Counter("MyCounter");
//   ...
//   counter->AddValue(+1);
class TORCH_API Counter {
 public:
  explicit Counter(std::string name);

  void AddValue(int64_t value) {
    GetData()->AddValue(value);
  }

  int64_t Value() const {
    return GetData()->Value();
  }

 private:
  CounterData* GetData() const;

  std::string name_;
  mutable std::shared_ptr<CounterData> data_ptr_;
  mutable std::atomic<CounterData*> data_;
};

#define TORCH_LAZY_COUNTER(name, value)        \
  do {                                         \
    static ::torch::lazy::Counter* __counter = \
        new ::torch::lazy::Counter(name);      \
    __counter->AddValue(value);                \
  } while (0)

#define TORCH_LAZY_FN_COUNTER(ns) TORCH_LAZY_COUNTER(c10::str(ns, __func__), 1)

#define TORCH_LAZY_VALUE_METRIC(name, value)                         \
  do {                                                               \
    static ::torch::lazy::Metric* __metric =                         \
        new ::torch::lazy::Metric(name, torch::lazy::MetricFnValue); \
    __metric->AddSample(value);                                      \
  } while (0)

// Creates a report with the current metrics statistics.
TORCH_API std::string CreateMetricReport();

// Returns the currently registered metric names. Note that the list can grow
// since metrics are usually function intialized (they are static function
// variables).
TORCH_API std::vector<std::string> GetMetricNames();

// Retrieves the metric data of a given metric, or nullptr if such metric does
// not exist.
TORCH_API MetricData* GetMetric(const std::string& name);

// Returns the currently registered counter names. Note that the list can grow
// since counters are usually function intialized (they are static function
// variables).
TORCH_API std::vector<std::string> GetCounterNames();

// Retrieves the counter data of a given counter, or nullptr if such counter
// does not exist.
TORCH_API CounterData* GetCounter(const std::string& name);

// Retrieves the current EPOCH time in nanoseconds.
TORCH_API int64_t NowNs();

// Scope based utility class TORCH_API to measure the time the code takes within
// a given C++ scope.
class TORCH_API TimedSection {
 public:
  explicit TimedSection(Metric* metric) : metric_(metric), start_(NowNs()) {}

  ~TimedSection() {
    int64_t now = NowNs();
    metric_->AddSample(now, now - start_);
  }

  double Elapsed() const {
    return 1e-9 * static_cast<double>(NowNs() - start_);
  }

 private:
  Metric* metric_;
  int64_t start_;
};

#define TORCH_LAZY_TIMED(name)                                  \
  static torch::lazy::Metric* timed_metric =                    \
      new torch::lazy::Metric(name, torch::lazy::MetricFnTime); \
  torch::lazy::TimedSection timed_section(timed_metric)

} // namespace lazy
} // namespace torch
