#include <torch/csrc/monitor/counters.h>

#include <unordered_set>

namespace torch::monitor {

const char* aggregationName(Aggregation agg) {
  switch (agg) {
    case Aggregation::NONE:
      return "none";
    case Aggregation::VALUE:
      return "value";
    case Aggregation::MEAN:
      return "mean";
    case Aggregation::COUNT:
      return "count";
    case Aggregation::SUM:
      return "sum";
    case Aggregation::MAX:
      return "max";
    case Aggregation::MIN:
      return "min";
    default:
      throw std::runtime_error(
          "unknown aggregation: " + std::to_string(static_cast<int>(agg)));
  }
}

namespace {
struct Stats {
  std::mutex mu;

  std::unordered_set<Stat<double>*> doubles;
  std::unordered_set<Stat<int64_t>*> int64s;
};

Stats& stats() {
  static Stats stats;
  return stats;
}
} // namespace

namespace detail {
void registerStat(Stat<double>* stat) {
  std::lock_guard<std::mutex> guard(stats().mu);

  stats().doubles.insert(stat);
}
void registerStat(Stat<int64_t>* stat) {
  std::lock_guard<std::mutex> guard(stats().mu);

  stats().int64s.insert(stat);
}
void unregisterStat(Stat<double>* stat) {
  std::lock_guard<std::mutex> guard(stats().mu);

  stats().doubles.erase(stat);
}
void unregisterStat(Stat<int64_t>* stat) {
  std::lock_guard<std::mutex> guard(stats().mu);

  stats().int64s.erase(stat);
}
} // namespace detail

} // namespace torch::monitor
