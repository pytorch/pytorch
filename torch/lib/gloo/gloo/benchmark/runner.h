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

#include "gloo/algorithm.h"
#include "gloo/barrier.h"
#include "gloo/benchmark/benchmark.h"
#include "gloo/benchmark/options.h"
#include "gloo/benchmark/timer.h"
#include "gloo/rendezvous/context.h"
#include "gloo/transport/device.h"

namespace gloo {
namespace benchmark {

class Runner {
 public:
  template <typename T>
  using BenchmarkFn = std::function<std::unique_ptr<Benchmark<T>>(
      std::shared_ptr<::gloo::Context>&)>;

  explicit Runner(const options& options);
  ~Runner();

  template <typename T>
  void run(BenchmarkFn<T>& fn);

  template <typename T>
  void run(BenchmarkFn<T>& fn, int n);

 protected:
  long broadcast(long value);

  std::shared_ptr<Context> newContext();

  void printHeader();
  void printDistribution(int elements, int elemSize);

  options options_;
  std::shared_ptr<transport::Device> device_;
  std::shared_ptr<rendezvous::ContextFactory> contextFactory_;

  long broadcastValue_;
  std::unique_ptr<Algorithm> broadcast_;
  std::unique_ptr<Barrier> barrier_;

  Distribution samples_;
};

} // namespace benchmark
} // namespace gloo
