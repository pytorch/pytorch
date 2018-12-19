#pragma once
#include "output_formatter.h"

namespace caffe2 {
namespace emulator {

const uint64_t MS_IN_SECOND = 1000;

/*
 * Print the output of the emulator run to stdout.
 */
class StdOutputFormatter : public OutputFormatter {
 private:
  template <typename T>
  static float get_mean(const std::vector<T>& values) {
    float sum = std::accumulate(values.begin(), values.end(), 0.0);
    return sum / values.size();
  }

  template <typename T>
  static float get_stdev(const std::vector<T>& values) {
    auto mean = get_mean(values);
    double sq_sum =
        std::inner_product(values.begin(), values.end(), values.begin(), 0.0);
    return std::sqrt(sq_sum / values.size() - mean * mean);
  }

 public:
  std::string format(
      const std::vector<float>& durations_ms,
      uint64_t threads,
      uint64_t iterations) override {
    auto mean = get_mean(durations_ms);
    auto throughput = iterations / (mean / MS_IN_SECOND);
    return std::string("\n\n====================================\n") +
        "Predictor benchmark finished with " + c10::to_string(threads) +
        " threads.\nThroughput:\t\t" + c10::to_string(throughput) +
        " iterations/s\nVariation:\t\t" +
        c10::to_string(get_stdev(durations_ms) * 100 / mean) +
        "%\n====================================";
  }
};

} // namespace emulator
} // namespace caffe2
