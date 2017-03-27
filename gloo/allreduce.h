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
  Allreduce(
    const std::shared_ptr<Context>& context,
    const ReductionFunction<T>* fn)
      : Algorithm(context), fn_(fn) {
    if (fn_ == nullptr) {
      fn_= ReductionFunction<T>::sum;
    }
  }

  virtual ~Allreduce() {};

 protected:
  const ReductionFunction<T>* fn_;
};

} // namespace gloo
