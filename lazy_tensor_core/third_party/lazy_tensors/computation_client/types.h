#ifndef COMPUTATION_CLIENT_TYPES_H_
#define COMPUTATION_CLIENT_TYPES_H_

#include <c10/util/Optional.h>

#include <cmath>
#include <vector>

#include "lazy_tensors/int128.h"
#include "lazy_tensors/types.h"

namespace lazy_tensors {

using hash_t = uint128;

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
  uint64 start_nstime = 0;
  uint64 end_nstime = 0;
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
  c10::optional<int64> int64_value;
};

}  // namespace lazy_tensors

#endif  // COMPUTATION_CLIENT_TYPES_H_
