#include "metrics_snapshot.h"

#include <regex>

#include "lazy_tensors/computation_client/ltc_logging.h"
#include "lazy_tensors/computation_client/util.h"

namespace torch_lazy_tensors {
namespace cpp_test {

MetricsSnapshot::MetricsSnapshot() {
  for (auto& name : lazy_tensors::metrics::GetMetricNames()) {
    lazy_tensors::metrics::MetricData* metric =
        lazy_tensors::metrics::GetMetric(name);
    MetricSamples msamples;
    msamples.samples =
        metric->Samples(&msamples.accumulator, &msamples.total_samples);
    metrics_map_.emplace(name, std::move(msamples));
  }
  for (auto& name : lazy_tensors::metrics::GetCounterNames()) {
    lazy_tensors::metrics::CounterData* counter =
        lazy_tensors::metrics::GetCounter(name);
    counters_map_.emplace(name, counter->Value());
  }
}

std::vector<MetricsSnapshot::ChangedCounter> MetricsSnapshot::CounterChanged(
    const std::string& counter_regex, const MetricsSnapshot& after,
    const std::unordered_set<std::string>* ignore_set) const {
  std::vector<ChangedCounter> changed;
  std::regex cregex(counter_regex);
  for (auto& name_counter : after.counters_map_) {
    std::smatch match;
    if ((ignore_set == nullptr || ignore_set->count(name_counter.first) == 0) &&
        std::regex_match(name_counter.first, match, cregex)) {
      lazy_tensors::int64 start_value =
          lazy_tensors::util::FindOr(counters_map_, name_counter.first, 0);
      if (name_counter.second != start_value) {
        changed.push_back(
            {name_counter.first, start_value, name_counter.second});
      }
    }
  }
  return changed;
}

std::string MetricsSnapshot::DumpDifferences(
    const MetricsSnapshot& after,
    const std::unordered_set<std::string>* ignore_set) const {
  std::stringstream ss;
  for (auto& name_counter : after.counters_map_) {
    if (ignore_set == nullptr || ignore_set->count(name_counter.first) == 0) {
      lazy_tensors::int64 start_value =
          lazy_tensors::util::FindOr(counters_map_, name_counter.first, 0);
      if (name_counter.second != start_value) {
        ss << "Counter '" << name_counter.first << "' changed from "
           << start_value << " to " << name_counter.second << "\n";
      }
    }
  }
  MetricSamples no_samples;
  for (auto& name_metrics : after.metrics_map_) {
    if (ignore_set == nullptr || ignore_set->count(name_metrics.first) == 0) {
      auto it = metrics_map_.find(name_metrics.first);
      if (it == metrics_map_.end() ||
          name_metrics.second.total_samples != it->second.total_samples) {
        const MetricSamples* start_samples =
            (it != metrics_map_.end()) ? &it->second : &no_samples;
        DumpMetricDifference(name_metrics.first, *start_samples,
                             name_metrics.second, &ss);
      }
    }
  }
  return ss.str();
}

void MetricsSnapshot::DumpMetricDifference(const std::string& name,
                                           const MetricSamples& before,
                                           const MetricSamples& after,
                                           std::stringstream* ss) {
  // Dump only the sample count difference for now.
  *ss << "Metric '" << name << "' collected extra "
      << (after.total_samples - before.total_samples) << " samples\n";
}

}  // namespace cpp_test
}  // namespace torch_lazy_tensors
