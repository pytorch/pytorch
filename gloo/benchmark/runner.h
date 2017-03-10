/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#pragma once

#include <functional>
#include <memory>

#include "gloo/barrier.h"
#include "gloo/benchmark/benchmark.h"
#include "gloo/benchmark/options.h"
#include "gloo/benchmark/timer.h"
#include "gloo/broadcast.h"
#include "gloo/transport/device.h"

namespace gloo {
namespace benchmark {

class Runner {
 public:
  using BenchmarkFn = std::function<
   std::unique_ptr<Benchmark>(
     std::shared_ptr<::gloo::Context>&)>;

  explicit Runner(const options& options);

  void run(BenchmarkFn& fn);
  void run(BenchmarkFn& fn, int n);

 protected:
  long broadcast(long value);

  std::shared_ptr<Context> newContext();

  void printHeader();
  void printDistribution(int elements);

  options options_;
  int prefixCounter_ = 0;

  std::shared_ptr<transport::Device> device_;

  long broadcastValue_;
  std::unique_ptr<Broadcast<long> > broadcast_;
  std::unique_ptr<Barrier> barrier_;

  Distribution samples_;
};

} // namespace benchmark
} // namespace gloo
