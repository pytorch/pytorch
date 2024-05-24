#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include "caffe2/core/logging.h"
#include "c10/util/static_tracepoint.h"

namespace caffe2 {

class TORCH_API StatValue {
  std::atomic<int64_t> v_{0};

 public:
  int64_t increment(int64_t inc) {
    return v_ += inc;
  }

  int64_t reset(int64_t value = 0) {
    return v_.exchange(value);
  }

  int64_t get() const {
    return v_.load();
  }
};

struct TORCH_API ExportedStatValue {
  std::string key;
  int64_t value;
  std::chrono::time_point<std::chrono::high_resolution_clock> ts;
};

/**
 * @brief Holds names and values of counters exported from a StatRegistry.
 */
using ExportedStatList = std::vector<ExportedStatValue>;
using ExportedStatMap = std::unordered_map<std::string, int64_t>;

TORCH_API ExportedStatMap toMap(const ExportedStatList& stats);

/**
 * @brief Holds a map of atomic counters keyed by name.
 *
 * The StatRegistry singleton, accessed through StatRegistry::get(), holds
 * counters registered through the macro CAFFE_EXPORTED_STAT. Example of usage:
 *
 * struct MyCaffeClass {
 *   MyCaffeClass(const std::string& instanceName): stats_(instanceName) {}
 *   void run(int numRuns) {
 *     try {
 *       CAFFE_EVENT(stats_, num_runs, numRuns);
 *       tryRun(numRuns);
 *       CAFFE_EVENT(stats_, num_successes);
 *     } catch (std::exception& e) {
 *       CAFFE_EVENT(stats_, num_failures, 1, "arg_to_usdt", e.what());
 *     }
 *     CAFFE_EVENT(stats_, usdt_only, 1, "arg_to_usdt");
 *   }
 *  private:
 *   struct MyStats {
 *     CAFFE_STAT_CTOR(MyStats);
 *     CAFFE_EXPORTED_STAT(num_runs);
 *     CAFFE_EXPORTED_STAT(num_successes);
 *     CAFFE_EXPORTED_STAT(num_failures);
 *     CAFFE_STAT(usdt_only);
 *   } stats_;
 * };
 *
 * int main() {
 *   MyCaffeClass a("first");
 *   MyCaffeClass b("second");
 *   for (const auto i : c10::irange(10)) {
 *     a.run(10);
 *     b.run(5);
 *   }
 *   ExportedStatList finalStats;
 *   StatRegistry::get().publish(finalStats);
 * }
 *
 * For every new instance of MyCaffeClass, a new counter is created with
 * the instance name as prefix. Everytime run() is called, the corresponding
 * counter will be incremented by the given value, or 1 if value not provided.
 *
 * Counter values can then be exported into an ExportedStatList. In the
 * example above, considering "tryRun" never throws, `finalStats` will be
 * populated as follows:
 *
 *   first/num_runs       100
 *   first/num_successes   10
 *   first/num_failures     0
 *   second/num_runs       50
 *   second/num_successes  10
 *   second/num_failures    0
 *
 * The event usdt_only is not present in ExportedStatList because it is declared
 * as CAFFE_STAT, which does not create a counter.
 *
 * Additionally, for each call to CAFFE_EVENT, a USDT probe is generated.
 * The probe will be set up with the following arguments:
 *   - Probe name: field name (e.g. "num_runs")
 *   - Arg #0: instance name (e.g. "first", "second")
 *   - Arg #1: For CAFFE_EXPORTED_STAT, value of the updated counter
 *             For CAFFE_STAT, -1 since no counter is available
 *   - Args ...: Arguments passed to CAFFE_EVENT, including update value
 *             when provided.
 *
 * It is also possible to create additional StatRegistry instances beyond
 * the singleton. These instances are not automatically populated with
 * CAFFE_EVENT. Instead, they can be populated from an ExportedStatList
 * structure by calling StatRegistry::update().
 *
 */
class TORCH_API StatRegistry {
  std::mutex mutex_;
  std::unordered_map<std::string, std::unique_ptr<StatValue>> stats_;

 public:
  /**
   * Retrieve the singleton StatRegistry, which gets populated
   * through the CAFFE_EVENT macro.
   */
  static StatRegistry& get();

  /**
   * Add a new counter with given name. If a counter for this name already
   * exists, returns a pointer to it.
   */
  StatValue* add(const std::string& name);

  /**
   * Populate an ExportedStatList with current counter values.
   * If `reset` is true, resets all counters to zero. It is guaranteed that no
   * count is lost.
   */
  void publish(ExportedStatList& exported, bool reset = false);

  ExportedStatList publish(bool reset = false) {
    ExportedStatList stats;
    publish(stats, reset);
    return stats;
  }

  /**
   * Update values of counters contained in the given ExportedStatList to
   * the values provided, creating counters that don't exist.
   */
  void update(const ExportedStatList& data);

  ~StatRegistry();
};

struct TORCH_API Stat {
  std::string groupName;
  std::string name;
  Stat(const std::string& gn, const std::string& n) : groupName(gn), name(n) {}

  template <typename... Unused>
  int64_t increment(Unused...) {
    return -1;
  }
};

class TORCH_API ExportedStat : public Stat {
  StatValue* value_;

 public:
  ExportedStat(const std::string& gn, const std::string& n)
      : Stat(gn, n), value_(StatRegistry::get().add(gn + "/" + n)) {}

  int64_t increment(int64_t value = 1) {
    return value_->increment(value);
  }

  template <typename T, typename Unused1, typename... Unused>
  int64_t increment(T value, Unused1, Unused...) {
    return increment(value);
  }
};

class TORCH_API AvgExportedStat : public ExportedStat {
 private:
  ExportedStat count_;

 public:
  AvgExportedStat(const std::string& gn, const std::string& n)
      : ExportedStat(gn, n + "/sum"), count_(gn, n + "/count") {}

  int64_t increment(int64_t value = 1) {
    count_.increment();
    return ExportedStat::increment(value);
  }

  template <typename T, typename Unused1, typename... Unused>
  int64_t increment(T value, Unused1, Unused...) {
    return increment(value);
  }
};

class TORCH_API StdDevExportedStat : public ExportedStat {
  // Uses an offset (first_) to remove issue of cancellation
  // Variance is then (sumsqoffset_ - (sumoffset_^2) / count_) / (count_ - 1)
 private:
  ExportedStat count_;
  ExportedStat sumsqoffset_;
  ExportedStat sumoffset_;
  std::atomic<int64_t> first_{std::numeric_limits<int64_t>::min()};
  int64_t const_min_{std::numeric_limits<int64_t>::min()};

 public:
  StdDevExportedStat(const std::string& gn, const std::string& n)
      : ExportedStat(gn, n + "/sum"),
        count_(gn, n + "/count"),
        sumsqoffset_(gn, n + "/sumsqoffset"),
        sumoffset_(gn, n + "/sumoffset") {}

  int64_t increment(int64_t value = 1) {
    first_.compare_exchange_strong(const_min_, value);
    int64_t offset_value = first_.load();
    int64_t orig_value = value;
    value -= offset_value;
    count_.increment();
    sumsqoffset_.increment(value * value);
    sumoffset_.increment(value);
    return ExportedStat::increment(orig_value);
  }

  template <typename T, typename Unused1, typename... Unused>
  int64_t increment(T value, Unused1, Unused...) {
    return increment(value);
  }
};

class TORCH_API DetailedExportedStat : public ExportedStat {
 private:
  std::vector<ExportedStat> details_;

 public:
  DetailedExportedStat(const std::string& gn, const std::string& n)
      : ExportedStat(gn, n) {}

  void setDetails(const std::vector<std::string>& detailNames) {
    details_.clear();
    for (const auto& detailName : detailNames) {
      details_.emplace_back(groupName, name + "/" + detailName);
    }
  }

  template <typename T, typename... Unused>
  int64_t increment(T value, size_t detailIndex, Unused...) {
    if (detailIndex < details_.size()) {
      details_[detailIndex].increment(value);
    }
    return ExportedStat::increment(value);
  }
};

class TORCH_API StaticStat : public Stat {
 private:
  StatValue* value_;

 public:
  StaticStat(const std::string& groupName, const std::string& name)
      : Stat(groupName, name),
        value_(StatRegistry::get().add(groupName + "/" + name)) {}

  int64_t increment(int64_t value = 1) {
    return value_->reset(value);
  }

  template <typename T, typename Unused1, typename... Unused>
  int64_t increment(T value, Unused1, Unused...) {
    return increment(value);
  }
};

namespace detail {

template <class T>
struct _ScopeGuard {
  T f_;
  std::chrono::high_resolution_clock::time_point start_;

  explicit _ScopeGuard(T f)
      : f_(f), start_(std::chrono::high_resolution_clock::now()) {}
  ~_ScopeGuard() {
    using namespace std::chrono;
    auto duration = high_resolution_clock::now() - start_;
    int64_t nanos = duration_cast<nanoseconds>(duration).count();
    f_(nanos);
  }

  // Using implicit cast to bool so that it can be used in an 'if' condition
  // within CAFFE_DURATION macro below.
  /* implicit */ operator bool() {
    return true;
  }
};

template <class T>
_ScopeGuard<T> ScopeGuard(T f) {
  return _ScopeGuard<T>(f);
}
} // namespace detail

#define CAFFE_STAT_CTOR(ClassName)                 \
  ClassName(std::string name) : groupName(name) {} \
  std::string groupName

#define CAFFE_EXPORTED_STAT(name) \
  ExportedStat name {             \
    groupName, #name              \
  }

#define CAFFE_AVG_EXPORTED_STAT(name) \
  AvgExportedStat name {              \
    groupName, #name                  \
  }

#define CAFFE_STDDEV_EXPORTED_STAT(name) \
  StdDevExportedStat name {              \
    groupName, #name                     \
  }

#define CAFFE_DETAILED_EXPORTED_STAT(name) \
  DetailedExportedStat name {              \
    groupName, #name                       \
  }

#define CAFFE_STAT(name) \
  Stat name {            \
    groupName, #name     \
  }

#define CAFFE_STATIC_STAT(name) \
  StaticStat name {             \
    groupName, #name            \
  }

#define CAFFE_EVENT(stats, field, ...)                              \
  {                                                                 \
    auto __caffe_event_value_ = stats.field.increment(__VA_ARGS__); \
    TORCH_SDT(                                                      \
        field,                                                      \
        stats.field.groupName.c_str(),                              \
        __caffe_event_value_,                                       \
        ##__VA_ARGS__);                                             \
    (void)__caffe_event_value_;                                     \
  }

#define CAFFE_DURATION(stats, field, ...)                        \
  if (auto g = ::caffe2::detail::ScopeGuard([&](int64_t nanos) { \
        CAFFE_EVENT(stats, field, nanos, ##__VA_ARGS__);         \
      }))
} // namespace caffe2
