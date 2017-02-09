/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#pragma once

#include "gloo/algorithm.h"

namespace gloo {

template <typename T>
class Allreduce : public Algorithm {
 public:
  using ReduceFunction = void(T*, const T*, size_t n);

  Allreduce(const std::shared_ptr<Context>& context, ReduceFunction fn)
      : Algorithm(context), fn_(fn) {
    if (fn_ == nullptr) {
      // Default to addition
      fn_ = [](T* dst, const T* src, size_t n) {
        for (int i = 0; i < n; i++) {
          dst[i] += src[i];
        }
      };
    }
  }

  virtual ~Allreduce(){};

 protected:
  ReduceFunction* fn_;
};

} // namespace gloo
