/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#pragma once

#include <Eigen/Core>

namespace gloo {
namespace benchmark {

template <typename T>
using EigenVectorMap = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>>;

template <typename T>
using ConstEigenVectorMap =
  Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>;

template <typename T>
void sum(T* x, const T* y, size_t n) {
  EigenVectorMap<T>(x, n) =
    ConstEigenVectorMap<T>(x, n) + ConstEigenVectorMap<T>(y, n);
};

} // namespace benchmark
} // namespace gloo
