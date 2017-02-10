/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#pragma once

#include <memory>
#include <vector>

#include "gloo/algorithm.h"
#include "gloo/benchmark/options.h"
#include "gloo/context.h"

namespace gloo {
namespace benchmark {

class Benchmark {
 public:
  Benchmark(
    std::shared_ptr<::gloo::Context>& context,
    struct options& options)
      : context_(context),
        options_(options) {}

  virtual ~Benchmark() {}

  virtual void initialize(int elements) = 0;

  virtual void run() {
    algorithm_->run();
  }

  virtual bool verify() = 0;

 protected:
  virtual float* allocate(int elements) {
    data_.resize(elements);
    for (int i = 0; i < data_.size(); i++) {
      data_[i] = context_->rank_;
    }
    return data_.data();
  }

  std::shared_ptr<::gloo::Context> context_;
  struct options options_;
  std::unique_ptr<::gloo::Algorithm> algorithm_;
  std::vector<float> data_;
};

} // namespace benchmark
} // namespace gloo
