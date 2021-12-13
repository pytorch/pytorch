#include <torch/csrc/monitor/counters.h>

#include <sstream>
#include <unordered_set>

namespace torch {
namespace monitor {

const char* aggregationName(Aggregation agg) {
  switch (agg) {
    case NONE:
      return "none";
    case VALUE:
      return "value";
    case COUNT:
      return "count";
    case SUM:
      return "sum";
    case MAX:
      return "max";
    case MIN:
      return "min";
    default:
      throw std::runtime_error("unknown aggregation: " + std::to_string(agg));
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

template <typename T>
void closeAndGetStat(Stat<T>* s, std::unordered_map<std::string, T>& m) {
  s->closeWindow();
  auto out = s->get();
  for (auto& kv : out) {
    std::stringstream key;
    key << s->name();
    key << ".";
    key << aggregationName(kv.first);
    m[key.str()] = kv.second;
  }
}

std::pair<
    std::unordered_map<std::string, double>,
    std::unordered_map<std::string, int64_t>>
closeAndGetStats() noexcept {
  std::pair<
      std::unordered_map<std::string, double>,
      std::unordered_map<std::string, int64_t>>
      out;

  std::lock_guard<std::mutex> guard(stats().mu);

  for (auto* s : stats().doubles) {
    closeAndGetStat(s, out.first);
  }
  for (auto* s : stats().int64s) {
    closeAndGetStat(s, out.second);
  }

  return out;
}

} // namespace monitor
} // namespace torch
