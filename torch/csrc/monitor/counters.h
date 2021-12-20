#pragma once

#include <bitset>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace torch {
namespace monitor {

constexpr int NUM_AGGREGATIONS = 7;

// Aggregation is the list of possible aggregations for Stats.
// These use bitwise flags so they can be efficiently stored.
enum Aggregation {
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

const char* aggregationName(Aggregation agg);

template <typename T>
class Stat;

namespace {
inline std::bitset<NUM_AGGREGATIONS> merge(
    std::initializer_list<Aggregation>& list) {
  std::bitset<NUM_AGGREGATIONS> a;
  for (Aggregation b : list) {
    a.set(b);
  }
  return a;
}
} // namespace

namespace detail {
void registerStat(Stat<double>* stat);
void registerStat(Stat<int64_t>* stat);
void unregisterStat(Stat<double>* stat);
void unregisterStat(Stat<int64_t>* stat);
} // namespace detail

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
      int64_t windowSize = -1)
      : name_(std::move(name)),
        aggregations_(merge(aggregations)),
        windowSize_(windowSize) {
    detail::registerStat(this);
  }

  ~Stat() {
    detail::unregisterStat(this);
  }

  // add adds the value v to the current window.
  void add(T v) noexcept {
    std::lock_guard<std::mutex> guard(mu_);

    if (aggregations_.test(VALUE)) {
      current_.value = v;
    }
    if (aggregations_.test(MEAN) || aggregations_.test(SUM)) {
      current_.sum += v;
    }

    if (aggregations_.test(MAX)) {
      if (current_.max < v || current_.count == 0) {
        current_.max = v;
      }
    }
    if (aggregations_.test(MIN)) {
      if (current_.min > v || current_.count == 0) {
        current_.min = v;
      }
    }

    current_.count += 1;
    if (windowSize_ > 0 && current_.count >= windowSize_) {
      saveCurrentLocked();
    }
  }

  const std::string& name() const noexcept {
    return name_;
  }

  int64_t windowSize() const noexcept {
    return windowSize_;
  }

  // count returns the number of items in the current open window.
  int64_t count() noexcept {
    std::lock_guard<std::mutex> guard(mu_);

    return current_.count;
  }

  // closeWindow finalizes the collected stats window so they can be accessed
  // via get().
  // If the Stat has a windowSize specified this doesn't do anything since the
  // window is automatically closed when enough samples have been logged.
  void closeWindow() noexcept {
    if (windowSize_ <= 0) {
      std::lock_guard<std::mutex> guard(mu_);

      saveCurrentLocked();
    }
  }

  std::vector<std::pair<Aggregation, T>> get() noexcept {
    std::vector<std::pair<Aggregation, T>> out;
    out.reserve(aggregations_.count());

    std::lock_guard<std::mutex> guard(mu_);

    if (aggregations_.test(VALUE)) {
      out.emplace_back(VALUE, prev_.value);
    }
    if (aggregations_.test(MEAN)) {
      if (prev_.count == 0) {
        out.emplace_back(MEAN, 0);
      } else {
        out.emplace_back(MEAN, prev_.sum / prev_.count);
      }
    }
    if (aggregations_.test(COUNT)) {
      out.emplace_back(COUNT, prev_.count);
    }
    if (aggregations_.test(SUM)) {
      out.emplace_back(SUM, prev_.sum);
    }
    if (aggregations_.test(MAX)) {
      out.emplace_back(MAX, prev_.max);
    }
    if (aggregations_.test(MIN)) {
      out.emplace_back(MIN, prev_.min);
    }

    return out;
  }

 private:
  void saveCurrentLocked() {
    prev_ = current_;
    current_ = Values();
  }

  const std::string name_;
  const std::bitset<NUM_AGGREGATIONS> aggregations_;
  const int64_t windowSize_;

  std::mutex mu_;
  Values current_;
  Values prev_;
};

std::pair<
    std::unordered_map<std::string, double>,
    std::unordered_map<std::string, int64_t>>
closeAndGetStats() noexcept;
} // namespace monitor
} // namespace torch
