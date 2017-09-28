/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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
    return std::chrono::duration_cast<ns>(clock::now() - start_time_).count();
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
  DISABLE_COPY_AND_ASSIGN(Timer);
};
}

#endif  // CAFFE2_CORE_TIMER_H_
