#pragma once
#include "caffe2/core/logging.h"

namespace caffe2 {
namespace emulator {

/*
 * A net emulator. In short, it can run nets with given @iterations.
 */
class Emulator {
 public:
  virtual void init() = 0;

  virtual void run(const uint64_t iterations) = 0;

  virtual ~Emulator() noexcept {}
};

} // namespace emulator
} // namespace caffe2
