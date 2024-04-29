#ifndef CAFFE2_CORE_TIMER_H_
#define CAFFE2_CORE_TIMER_H_

#include <chrono>

#include "caffe2/core/common.h"

namespace caffe2 {

/**
 * @brief A simple timer object for measuring time.
 *
 * This is a minimal class around a std::chrono::high_resolution_clock that
 * serves as a utility class for testing code.
 */
class Timer {
 public:
  typedef std::chrono::high_resolution_clock clock;
  typedef std::chrono::nanoseconds ns;
  Timer() { Start(); }
  /**
   * @brief Starts a timer.
   */
  inline void Start() { start_time_ = clock::now(); }
  inline float NanoSeconds() {
    return static_cast<float>(
        std::chrono::duration_cast<ns>(clock::now() - start_time_).count());
  }
  /**
   * @brief Returns the elapsed time in milliseconds.
   */
  inline float MilliSeconds() { return NanoSeconds() / 1000000.f; }
  /**
   * @brief Returns the elapsed time in microseconds.
   */
  inline float MicroSeconds() { return NanoSeconds() / 1000.f; }
  /**
   * @brief Returns the elapsed time in seconds.
   */
  inline float Seconds() { return NanoSeconds() / 1000000000.f; }

 protected:
  std::chrono::time_point<clock> start_time_;
  C10_DISABLE_COPY_AND_ASSIGN(Timer);
};
}

#endif  // CAFFE2_CORE_TIMER_H_
