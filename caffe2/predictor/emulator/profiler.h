#pragma once
#include "caffe2/core/logging.h"
#include "caffe2/core/timer.h"

namespace caffe2 {
namespace emulator {

/*
 * An interface to profile the metrics of a @runnable.
 * It should return execution walltime in milliseconds.
 */
class Profiler {
 public:
  virtual float profile(std::function<void()> runnable) = 0;

  virtual ~Profiler() noexcept {}
};

} // namespace emulator
} // namespace caffe2
