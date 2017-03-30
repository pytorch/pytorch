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
  static const ReductionFunction<T>* sum;
  static const ReductionFunction<T>* product;
  static const ReductionFunction<T>* min;
  static const ReductionFunction<T>* max;

  virtual ~ReductionFunction() {}
  virtual ReductionType type() const = 0;
  virtual void call(T*, const T*, size_t n) const = 0;
};

template <typename T>
class BuiltInReductionFunction : public ReductionFunction<T> {
  using Fn = void(T*, const T*, size_t n);

 public:
  BuiltInReductionFunction(ReductionType type, Fn* fn)
      : type_(type), fn_(fn) {}

  virtual ReductionType type() const override {
    return type_;
  }

  virtual void call(T* x, const T* y, size_t n) const override {
    fn_(x, y, n);
  }

 protected:
  ReductionType type_;
  Fn* fn_;
};

template <typename T>
const ReductionFunction<T>* ReductionFunction<T>::sum =
  new BuiltInReductionFunction<T>(SUM, &::gloo::sum<T>);
template <typename T>
const ReductionFunction<T>* ReductionFunction<T>::product =
  new BuiltInReductionFunction<T>(PRODUCT, &::gloo::product<T>);
template <typename T>
const ReductionFunction<T>* ReductionFunction<T>::min =
  new BuiltInReductionFunction<T>(MIN, &::gloo::min<T>);
template <typename T>
const ReductionFunction<T>* ReductionFunction<T>::max =
  new BuiltInReductionFunction<T>(MAX, &::gloo::max<T>);

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
