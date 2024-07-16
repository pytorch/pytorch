#include <torch/csrc/lazy/core/metrics.h>

#include <c10/util/Logging.h>
#include <c10/util/irange.h>
#include <torch/csrc/lazy/backend/backend_interface.h>
#include <torch/csrc/lazy/core/config.h>
#include <torch/csrc/lazy/core/helpers.h>
#include <torch/csrc/lazy/core/util.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <sstream>

namespace torch {
namespace lazy {
namespace {

const std::vector<double>* ReadEnvPercentiles() {
  std::vector<std::string> percentiles_list =
      StrSplit(FLAGS_torch_lazy_metrics_percentiles, ':');
  std::unique_ptr<std::vector<double>> metrics_percentiles =
      std::make_unique<std::vector<double>>();
  for (auto& pct_str : percentiles_list) {
    double pct = std::stod(pct_str);
    TORCH_CHECK(pct > 0.0 && pct < 1.0, "Invalid percentile: ", pct);
    metrics_percentiles->push_back(pct);
  }
  std::sort(metrics_percentiles->begin(), metrics_percentiles->end());
  return metrics_percentiles.release();
}

const std::vector<double>& GetPercentiles() {
  static const std::vector<double>* metrics_percentiles = ReadEnvPercentiles();
  return *metrics_percentiles;
}

void EmitMetricInfo(
    const std::string& name,
    MetricData* data,
    std::stringstream* ss) {
  double accumulator = 0.0;
  size_t total_samples = 0;
  std::vector<Sample> samples = data->Samples(&accumulator, &total_samples);
  (*ss) << "Metric: " << name << std::endl;
  (*ss) << "  TotalSamples: " << total_samples << std::endl;
  (*ss) << "  Accumulator: " << data->Repr(accumulator) << std::endl;
  if (!samples.empty()) {
    double total = 0.0;
    for (auto& sample : samples) {
      total += sample.value;
    }
    int64_t delta_time =
        samples.back().timestamp_ns - samples.front().timestamp_ns;
    if (delta_time > 0) {
      double value_sec = 1e6 * (total / (delta_time / 1000.0));
      (*ss) << "  ValueRate: " << data->Repr(value_sec) << " / second"
            << std::endl;
      double count_sec =
          1e6 * (static_cast<double>(samples.size()) / (delta_time / 1000.0));
      (*ss) << "  Rate: " << count_sec << " / second" << std::endl;
    }
  }

  const std::vector<double>& metrics_percentiles = GetPercentiles();
  std::sort(
      samples.begin(), samples.end(), [](const Sample& s1, const Sample& s2) {
        return s1.value < s2.value;
      });
  (*ss) << "  Percentiles: ";
  for (const auto i : c10::irange(metrics_percentiles.size())) {
    size_t index = metrics_percentiles[i] * samples.size();
    if (i > 0) {
      (*ss) << "; ";
    }
    (*ss) << (metrics_percentiles[i] * 100.0)
          << "%=" << data->Repr(samples[index].value);
  }
  (*ss) << std::endl;
}

void EmitCounterInfo(
    const std::string& name,
    CounterData* data,
    std::stringstream* ss) {
  (*ss) << "Counter: " << name << std::endl;
  (*ss) << "  Value: " << data->Value() << std::endl;
}

template <typename T, typename G>
const typename T::mapped_type& MapInsert(
    T* cont,
    const typename T::key_type& key,
    const G& gen) {
  auto it = cont->find(key);
  if (it == cont->end()) {
    it = cont->emplace(key, gen()).first;
  }
  return it->second;
}

} // namespace

MetricsArena* MetricsArena::Get() {
  static MetricsArena* arena = new MetricsArena();
  return arena;
}

void MetricsArena::ResetCounters() {
  for (auto& pair : counters_) {
    if (pair.second) {
      pair.second->Reset();
    }
  }
}

void MetricsArena::ResetMetrics() {
  for (auto& pair : metrics_) {
    if (pair.second) {
      pair.second->Reset();
    }
  }
}

void MetricsArena::RegisterMetric(
    const std::string& name,
    MetricReprFn repr_fn,
    size_t max_samples,
    std::shared_ptr<MetricData>* data) {
  std::lock_guard<std::mutex> lock(lock_);
  if (*data == nullptr) {
    *data = MapInsert(&metrics_, name, [&]() {
      return std::make_shared<MetricData>(std::move(repr_fn), max_samples);
    });
  }
}

void MetricsArena::RegisterCounter(
    const std::string& name,
    std::shared_ptr<CounterData>* data) {
  std::lock_guard<std::mutex> lock(lock_);
  if (*data == nullptr) {
    *data = MapInsert(
        &counters_, name, []() { return std::make_shared<CounterData>(); });
  }
}

void MetricsArena::ForEachMetric(
    const std::function<void(const std::string&, MetricData*)>& metric_func) {
  std::lock_guard<std::mutex> lock(lock_);
  for (auto& name_data : metrics_) {
    if (!name_data.second->IsValid()) {
      continue;
    }
    metric_func(name_data.first, name_data.second.get());
  }
}

void MetricsArena::ForEachCounter(
    const std::function<void(const std::string&, CounterData*)>& counter_func) {
  std::lock_guard<std::mutex> lock(lock_);
  for (auto& name_data : counters_) {
    if (!name_data.second->IsValid())
      continue;
    counter_func(name_data.first, name_data.second.get());
  }
}

std::vector<std::string> MetricsArena::GetMetricNames() {
  std::vector<std::string> names;
  ForEachMetric([&names](const std::string& name, MetricData* data) {
    names.push_back(name);
  });
  return names;
}

MetricData* MetricsArena::GetMetric(const std::string& name) {
  std::lock_guard<std::mutex> lock(lock_);
  auto it = metrics_.find(name);
  if (it == metrics_.end()) {
    return nullptr;
  }
  return it->second->IsValid() ? it->second.get() : nullptr;
}

std::vector<std::string> MetricsArena::GetCounterNames() {
  std::vector<std::string> names;
  ForEachCounter([&names](const std::string& name, CounterData* data) {
    names.push_back(name);
  });
  return names;
}

CounterData* MetricsArena::GetCounter(const std::string& name) {
  std::lock_guard<std::mutex> lock(lock_);
  auto it = counters_.find(name);
  if (it == counters_.end()) {
    return nullptr;
  }
  return it->second->IsValid() ? it->second.get() : nullptr;
}

MetricData::MetricData(MetricReprFn repr_fn, size_t max_samples)
    : repr_fn_(std::move(repr_fn)), samples_(max_samples) {}

void MetricData::AddSample(int64_t timestamp_ns, double value) {
  std::lock_guard<std::mutex> lock(lock_);
  size_t position = count_ % samples_.size();
  ++count_;
  accumulator_ += value;
  samples_[position] = Sample(timestamp_ns, value);
}

double MetricData::Accumulator() const {
  std::lock_guard<std::mutex> lock(lock_);
  return accumulator_;
}

size_t MetricData::TotalSamples() const {
  std::lock_guard<std::mutex> lock(lock_);
  return count_;
}

std::vector<Sample> MetricData::Samples(
    double* accumulator,
    size_t* total_samples) const {
  std::lock_guard<std::mutex> lock(lock_);
  std::vector<Sample> samples;
  if (count_ <= samples_.size()) {
    samples.insert(samples.end(), samples_.begin(), samples_.begin() + count_);
  } else {
    size_t position = count_ % samples_.size();
    samples.insert(samples.end(), samples_.begin() + position, samples_.end());
    samples.insert(
        samples.end(), samples_.begin(), samples_.begin() + position);
  }
  if (accumulator != nullptr) {
    *accumulator = accumulator_;
  }
  if (total_samples != nullptr) {
    *total_samples = count_;
  }
  return samples;
}

void MetricData::Reset() {
  std::lock_guard<std::mutex> lock(lock_);
  count_ = 0;
  // Don't clear. samples_ are init with placeholders.
  samples_ = std::vector<Sample>(samples_.size());
  accumulator_ = 0.0;
}

Metric::Metric(std::string name, MetricReprFn repr_fn, size_t max_samples)
    : name_(std::move(name)),
      repr_fn_(std::move(repr_fn)),
      max_samples_(
          max_samples != 0 ? max_samples : FLAGS_torch_lazy_metrics_samples),
      data_(nullptr) {}

double Metric::Accumulator() const {
  return GetData()->Accumulator();
}

void Metric::AddSample(int64_t timestamp_ns, double value) {
  GetData()->AddSample(timestamp_ns, value);
}

void Metric::AddSample(double value) {
  GetData()->AddSample(NowNs(), value);
}

std::vector<Sample> Metric::Samples(double* accumulator, size_t* total_samples)
    const {
  return GetData()->Samples(accumulator, total_samples);
}

std::string Metric::Repr(double value) const {
  return GetData()->Repr(value);
}

MetricData* Metric::GetData() const {
  MetricData* data = data_.load();
  if (C10_UNLIKELY(data == nullptr)) {
    // The RegisterMetric() API is a synchronization point, and even if multiple
    // threads enters it, the data will be created only once.
    MetricsArena* arena = MetricsArena::Get();
    arena->RegisterMetric(name_, repr_fn_, max_samples_, &data_ptr_);
    // Even if multiple threads will enter this IF statement, they will all
    // fetch the same value, and hence store the same value below.
    data = data_ptr_.get();
    data_.store(data);
  }
  return data;
}

Counter::Counter(std::string name) : name_(std::move(name)), data_(nullptr) {}

CounterData* Counter::GetData() const {
  CounterData* data = data_.load();
  if (C10_UNLIKELY(data == nullptr)) {
    // The RegisterCounter() API is a synchronization point, and even if
    // multiple threads enters it, the data will be created only once.
    MetricsArena* arena = MetricsArena::Get();
    arena->RegisterCounter(name_, &data_ptr_);
    // Even if multiple threads will enter this IF statement, they will all
    // fetch the same value, and hence store the same value below.
    data = data_ptr_.get();
    data_.store(data);
  }
  return data;
}

std::string MetricFnValue(double value) {
  std::stringstream ss;
  ss.precision(2);
  ss << std::fixed << value;
  return ss.str();
}

std::string MetricFnBytes(double value) {
  static const std::array<const char*, 6> kSizeSuffixes{
      "B", "KB", "MB", "GB", "TB", "PB"};
  unsigned sfix = 0;
  for (; (sfix + 1) < kSizeSuffixes.size() && value >= 1024.0; ++sfix) {
    value /= 1024.0;
  }
  std::stringstream ss;
  ss.precision(2);
  ss << std::fixed << value << kSizeSuffixes[sfix];
  return ss.str();
}

std::string MetricFnTime(double value) {
  struct TimePart {
    const char* suffix;
    double scaler;
    int width;
    int precision;
    char fill;
  };
  static const std::array<TimePart, 6> time_parts{
      {{"d", 86400.0 * 1e9, 2, 0, '0'},
       {"h", 3600.0 * 1e9, 2, 0, '0'},
       {"m", 60.0 * 1e9, 2, 0, '0'},
       {"s", 1e9, 2, 0, '0'},
       {"ms", 1e6, 3, 0, '0'},
       {"us", 1e3, 7, 3, '0'}}};
  int count = 0;
  std::stringstream ss;
  for (const auto i : c10::irange(time_parts.size())) {
    const TimePart& part = time_parts[i];
    double ctime = value / part.scaler;
    if (ctime >= 1.0 || count > 0 || i + 1 == time_parts.size()) {
      ss.precision(part.precision);
      ss.width(part.width);
      ss.fill(part.fill);
      ss << std::fixed << ctime << part.suffix;
      value -= std::floor(ctime) * part.scaler;
      ++count;
    }
  }
  return ss.str();
}

std::string CreateMetricReport() {
  MetricsArena* arena = MetricsArena::Get();
  std::stringstream ss;
  arena->ForEachMetric([&ss](const std::string& name, MetricData* data) {
    EmitMetricInfo(name, data, &ss);
  });
  arena->ForEachCounter([&ss](const std::string& name, CounterData* data) {
    EmitCounterInfo(name, data, &ss);
  });

  // Append the backend metrics report
  ss << getBackend()->CreateMetricReport();
  return ss.str();
}

std::string CreateMetricReport(
    const std::vector<std::string>& counter_names,
    const std::vector<std::string>& metric_names) {
  MetricsArena* arena = MetricsArena::Get();
  std::stringstream ss;
  std::set<std::string> metric_name_set(
      metric_names.begin(), metric_names.end());
  arena->ForEachMetric(
      [&ss, &metric_name_set](const std::string& name, MetricData* data) {
        if (metric_name_set.find(name) != metric_name_set.end()) {
          EmitMetricInfo(name, data, &ss);
        }
      });
  std::set<std::string> counter_name_set(
      counter_names.begin(), counter_names.end());
  arena->ForEachCounter(
      [&ss, &counter_name_set](const std::string& name, CounterData* data) {
        if (counter_name_set.find(name) != counter_name_set.end()) {
          EmitCounterInfo(name, data, &ss);
        }
      });

  static std::string fall_back_counter_prefix = "aten::";
  arena->ForEachCounter([&ss](const std::string& name, CounterData* data) {
    if (name.rfind(fall_back_counter_prefix, 0) == 0) {
      // it might emit duplicated counter if user also specified exact aten
      // counter in the `counter_names` but it should be very rare.
      EmitCounterInfo(name, data, &ss);
    }
  });
  return ss.str();
}

std::vector<std::string> GetMetricNames() {
  return MetricsArena::Get()->GetMetricNames();
}

MetricData* GetMetric(const std::string& name) {
  return MetricsArena::Get()->GetMetric(name);
}

std::vector<std::string> GetCounterNames() {
  return MetricsArena::Get()->GetCounterNames();
}

CounterData* GetCounter(const std::string& name) {
  return MetricsArena::Get()->GetCounter(name);
}

int64_t NowNs() {
  auto now = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             now.time_since_epoch())
      .count();
}

} // namespace lazy
} // namespace torch
