#pragma once
#include "profiler.h"

namespace caffe2 {
namespace emulator {

/*
 * An profiler that measures the walltime of a @runnable
 */
class TimeProfiler : public Profiler {
 public:
  float profile(std::function<void()> runnable) override {
    Timer timer;
    runnable();
    return timer.MilliSeconds();
  }
};

} // namespace emulator
} // namespace caffe2
