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

#include "gloo/context.h"
#include "gloo/math.h"

namespace gloo {

extern const size_t kOnDeviceThreshold;

class Algorithm {
 public:
  explicit Algorithm(const std::shared_ptr<Context>&);
  virtual ~Algorithm() = 0;

  virtual void run() = 0;

 protected:
  std::shared_ptr<Context> context_;

  const int contextRank_;
  const int contextSize_;

  std::unique_ptr<transport::Pair>& getPair(int i);

  // Helpers for ring algorithms
  std::unique_ptr<transport::Pair>& getLeftPair();
  std::unique_ptr<transport::Pair>& getRightPair();
};

// Type of reduction function.
//
// If the reduction type is one of the built-ins, algorithm
// implementations may use accelerated versions if available.
//
// For example, if a ReductionFunction with ReductionType equal
// SUM is passed to CUDA aware Allreduce, it knows it can
// use a NCCL implementation instead of the specified function.
//
enum ReductionType {
  SUM = 1,
  PRODUCT = 2,
  MAX = 3,
  MIN = 4,

  // Use larger number so we have plenty of room to add built-ins
  CUSTOM = 1000,
};

template <typename T>
class ReductionFunction {
 public:
  using Function = void(T*, const T*, size_t n);

  static const ReductionFunction<T>* sum;
  static const ReductionFunction<T>* product;
  static const ReductionFunction<T>* min;
  static const ReductionFunction<T>* max;

  ReductionFunction(ReductionType type, Function* fn)
      : type_(type), fn_(fn) {}

  ReductionType type() const {
    return type_;
  }

  void call(T* x, const T* y, size_t n) const {
    fn_(x, y, n);
  }

 protected:
  ReductionType type_;
  Function* fn_;
};

template <typename T>
const ReductionFunction<T>* ReductionFunction<T>::sum =
  new ReductionFunction<T>(SUM, &::gloo::sum<T>);
template <typename T>
const ReductionFunction<T>* ReductionFunction<T>::product =
  new ReductionFunction<T>(PRODUCT, &::gloo::product<T>);
template <typename T>
const ReductionFunction<T>* ReductionFunction<T>::min =
  new ReductionFunction<T>(MIN, &::gloo::min<T>);
template <typename T>
const ReductionFunction<T>* ReductionFunction<T>::max =
  new ReductionFunction<T>(MAX, &::gloo::max<T>);

// Local operation.
// If an algorithm uses multiple local pointers, local operations
// can be used for local reduction, broadcast, gathering, etc.
template <typename T>
class LocalOp {
 public:
  virtual ~LocalOp() {}
  virtual void runAsync() = 0;
  virtual void wait() = 0;

  // Synchronous run is equal to asynchronous run and wait.
  inline void run() {
    runAsync();
    wait();
  }
};

} // namespace gloo
