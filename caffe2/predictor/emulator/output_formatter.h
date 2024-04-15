#pragma once
#include "caffe2/core/logging.h"

namespace caffe2 {
namespace emulator {

/*
 * An interface that formats the output of the emulator runs.
 */
class OutputFormatter {
 public:
  virtual std::string format(
      const std::vector<float>& durations_ms,
      uint64_t threads,
      uint64_t iterations) = 0;

  virtual ~OutputFormatter() noexcept {}
};

} // namespace emulator
} // namespace caffe2
