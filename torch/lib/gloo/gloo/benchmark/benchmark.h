/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#pragma once

#include <stdlib.h>

#include <memory>
#include <vector>

#include "gloo/algorithm.h"
#include "gloo/benchmark/options.h"
#include "gloo/context.h"
#include "gloo/common/common.h"

namespace gloo {
namespace benchmark {

template <typename T>
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

  virtual void verify() {}

  const options& getOptions() const {
    return options_;
  }
 protected:
  virtual std::vector<T*> allocate(int inputs, int elements) {
    std::vector<T*> ptrs;

    // Stride between successive values in any input.
    const auto stride = context_->size * inputs;
    for (int i = 0; i < inputs; i++) {
      std::vector<T, aligned_allocator<T, kBufferAlignment>> memory(elements);

      // Value at memory[0]. Different for every input at every node.
      // This means all values across all inputs and all nodes are
      // different and we can accurately detect correctness errors.
      auto value = (context_->rank * inputs) + i;
      for (int j = 0; j < elements; j++) {
        memory[j] = (j * stride) + value;
      }
      ptrs.push_back(memory.data());
      inputs_.push_back(std::move(memory));
    }
    return ptrs;
  }

  std::shared_ptr<::gloo::Context> context_;
  struct options options_;
  std::unique_ptr<::gloo::Algorithm> algorithm_;
  std::vector<std::vector<T, aligned_allocator<T, kBufferAlignment>>>
      inputs_;
};

} // namespace benchmark
} // namespace gloo
