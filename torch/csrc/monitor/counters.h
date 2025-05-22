#pragma once

#include <bitset>
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <vector>

#include <c10/macros/Macros.h>

#include <torch/csrc/monitor/events.h>

namespace torch::monitor {

constexpr int NUM_AGGREGATIONS = 7;

// Aggregation is the list of possible aggregations for Stats.
// These use bitwise flags so they can be efficiently stored.
enum class C10_API_ENUM Aggregation {
  // NONE means no aggregations are set.
  NONE = 0,
  // VALUE exports the most recently set value.
  VALUE = 1,
  // MEAN computes the mean of the set values within the window. Zero if no
  // values.
  MEAN = 2,
  // COUNT tracks the number of times a value is set within the window.
  COUNT = 3,
  // SUM computes the sum of the values set within the window.
  SUM = 4,
  // MIN computes the minimum of the values set within the window. Zero if no
  // values.
  MAX = 5,
  // MAX computes the maximum of the values set within the window. Zero if no
  // values.
  MIN = 6,
};

struct TORCH_API AggregationHash{template <typename T> std::size_t operator()(
    T t) const {return static_cast<std::size_t>(t);
} // namespace torch::monitor
}
;

// aggregationName returns the human readable name corresponding to the
// aggregation.
TORCH_API const char* aggregationName(Aggregation agg);

template <typename T>
class Stat;

namespace {
template <typename T>
inline std::bitset<NUM_AGGREGATIONS> merge(T& list) {
  std::bitset<NUM_AGGREGATIONS> a;
  for (Aggregation b : list) {
    a.set(static_cast<int>(b));
  }
  return a;
}
} // namespace

namespace detail {
void TORCH_API registerStat(Stat<double>* stat);
void TORCH_API registerStat(Stat<int64_t>* stat);
void TORCH_API unregisterStat(Stat<double>* stat);
void TORCH_API unregisterStat(Stat<int64_t>* stat);
} // namespace detail

// Stat is used to compute summary statistics in a performant way over fixed
// intervals. Stat logs the statistics as an Event once every `windowSize`
// duration. When the window closes the stats are logged via the event handlers
// as a `torch.monitor.Stat` event.
//
// `windowSize` should be set to something relatively high to avoid a huge
// number of events being logged. Ex: 60s. Stat uses millisecond precision.
//
// If maxSamples is set, the stat will cap the number of samples per window by
// discarding `add` calls once `maxSamples` adds have occurred. If it's not set,
// all `add` calls during the window will be included.
// This is an optional field to make aggregations more directly comparable
// across windows when the number of samples might vary.
//
// Stats support double and int64_t data types depending on what needs to be
// logged and needs to be templatized with one of them.
//
// When the Stat is destructed it will log any remaining data even if the window
// hasn't elapsed.
template <typename T>
class Stat {
 private:
  struct Values {
    T value{0};
    T sum{0};
    T min{0};
    T max{0};
    int64_t count{0};
  };

 public:
  Stat(
      std::string name,
      std::initializer_list<Aggregation> aggregations,
      std::chrono::milliseconds windowSize,
      int64_t maxSamples = std::numeric_limits<int64_t>::max())
      : name_(std::move(name)),
        aggregations_(merge(aggregations)),
        windowSize_(windowSize),
        maxSamples_(maxSamples) {
    detail::registerStat(this);
  }

  Stat(
      std::string name,
      std::vector<Aggregation> aggregations,
      std::chrono::milliseconds windowSize,
      int64_t maxSamples = std::numeric_limits<int64_t>::max())
      : name_(std::move(name)),
        aggregations_(merge(aggregations)),
        windowSize_(windowSize),
        maxSamples_(maxSamples) {
    detail::registerStat(this);
  }
  Stat(const Stat&) = delete;
  Stat(Stat&&) = delete;
  Stat& operator=(const Stat&) = delete;
  Stat& operator=(Stat&&) = delete;

  virtual ~Stat() {
    {
      // on destruction log if there's unlogged data
      std::lock_guard<std::mutex> guard(mu_);
      logLocked();
    }
    detail::unregisterStat(this);
  }

  // add adds the value v to the current window.
  void add(T v) {
    std::lock_guard<std::mutex> guard(mu_);
    maybeLogLocked();

    if (alreadyLogged()) {
      return;
    }

    if (aggregations_.test(static_cast<int>(Aggregation::VALUE))) {
      current_.value = v;
    }
    if (aggregations_.test(static_cast<int>(Aggregation::MEAN)) ||
        aggregations_.test(static_cast<int>(Aggregation::SUM))) {
      current_.sum += v;
    }

    if (aggregations_.test(static_cast<int>(Aggregation::MAX))) {
      if (current_.max < v || current_.count == 0) {
        current_.max = v;
      }
    }
    if (aggregations_.test(static_cast<int>(Aggregation::MIN))) {
      if (current_.min > v || current_.count == 0) {
        current_.min = v;
      }
    }

    current_.count += 1;
    maybeLogLocked();
  }

  const std::string& name() const noexcept {
    return name_;
  }

  // count returns the number of items in the current open window.
  int64_t count() noexcept {
    std::lock_guard<std::mutex> guard(mu_);

    return current_.count;
  }

  std::unordered_map<Aggregation, T, AggregationHash> get() noexcept {
    std::lock_guard<std::mutex> guard(mu_);
    return getLocked();
  }

 protected:
  virtual uint64_t currentWindowId() const {
    std::chrono::milliseconds now =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch());

    // always returns a currentWindowId of at least 1 to avoid 0 window issues
    return (now / windowSize_) + 1;
  }

 private:
  bool alreadyLogged() {
    return lastLoggedWindowId_ == currentWindowId();
  }

  void maybeLogLocked() {
    auto windowId = currentWindowId();
    bool shouldLog = windowId_ != windowId || current_.count >= maxSamples_;
    if (shouldLog && !alreadyLogged()) {
      logLocked();
      lastLoggedWindowId_ = windowId_;
      windowId_ = windowId;
    }
  }

  void logLocked() {
    prev_ = current_;
    current_ = Values();

    // don't log event if there's no data
    if (prev_.count == 0) {
      return;
    }

    Event e;
    e.name = "torch.monitor.Stat";
    e.timestamp = std::chrono::system_clock::now();

    auto stats = getLocked();
    e.data.reserve(stats.size());
    for (auto& kv : stats) {
      std::stringstream key;
      key << name_;
      key << ".";
      key << aggregationName(kv.first);
      e.data[key.str()] = kv.second;
    }

    logEvent(e);
  }

  std::unordered_map<Aggregation, T, AggregationHash> getLocked()
      const noexcept {
    std::unordered_map<Aggregation, T, AggregationHash> out;
    out.reserve(aggregations_.count());

    if (aggregations_.test(static_cast<int>(Aggregation::VALUE))) {
      out.emplace(Aggregation::VALUE, prev_.value);
    }
    if (aggregations_.test(static_cast<int>(Aggregation::MEAN))) {
      if (prev_.count == 0) {
        out.emplace(Aggregation::MEAN, 0);
      } else {
        out.emplace(Aggregation::MEAN, prev_.sum / prev_.count);
      }
    }
    if (aggregations_.test(static_cast<int>(Aggregation::COUNT))) {
      out.emplace(Aggregation::COUNT, prev_.count);
    }
    if (aggregations_.test(static_cast<int>(Aggregation::SUM))) {
      out.emplace(Aggregation::SUM, prev_.sum);
    }
    if (aggregations_.test(static_cast<int>(Aggregation::MAX))) {
      out.emplace(Aggregation::MAX, prev_.max);
    }
    if (aggregations_.test(static_cast<int>(Aggregation::MIN))) {
      out.emplace(Aggregation::MIN, prev_.min);
    }

    return out;
  }

  const std::string name_;
  const std::bitset<NUM_AGGREGATIONS> aggregations_;

  std::mutex mu_;
  Values current_;
  Values prev_;

  uint64_t windowId_{0};
  uint64_t lastLoggedWindowId_{0};
  const std::chrono::milliseconds windowSize_;
  const int64_t maxSamples_;
};
} // namespace torch::monitor
