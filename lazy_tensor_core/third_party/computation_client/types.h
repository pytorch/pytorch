#ifndef COMPUTATION_CLIENT_TYPES_H_
#define COMPUTATION_CLIENT_TYPES_H_

#include <c10/util/Optional.h>

#include <cmath>
#include <vector>

namespace lazy_tensors {

struct Percentile {
  enum class UnitOfMeaure {
    kNumber,
    kTime,
    kBytes,
  };
  struct Point {
    double percentile = 0.0;
    double value = 0.0;
  };

  UnitOfMeaure unit_of_measure = UnitOfMeaure::kNumber;
  uint64_t start_nstime = 0;
  uint64_t end_nstime = 0;
  double min_value = NAN;
  double max_value = NAN;
  double mean = NAN;
  double stddev = NAN;
  size_t num_samples = 0;
  size_t total_samples = 0;
  double accumulator = NAN;
  std::vector<Point> points;
};

struct Metric {
  c10::optional<Percentile> percentile;
  c10::optional<int64_t> int64_value;
};

}  // namespace lazy_tensors

#endif  // COMPUTATION_CLIENT_TYPES_H_
