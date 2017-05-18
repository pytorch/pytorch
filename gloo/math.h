/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#pragma once

#include <algorithm>

#ifdef GLOO_USE_EIGEN
#include <Eigen/Core>
#endif

#include "gloo/types.h"

namespace gloo {

#ifdef GLOO_USE_EIGEN

template <typename T>
using EigenVectorArrayMap =
  Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1> >;
template <typename T>
using ConstEigenVectorArrayMap =
  Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, 1> >;

template <typename T>
void sum(T* x, const T* y, size_t n) {
  EigenVectorArrayMap<T>(x, n) =
    ConstEigenVectorArrayMap<T>(x, n) + ConstEigenVectorArrayMap<T>(y, n);
};

template <typename T>
void product(T* x, const T* y, size_t n) {
  EigenVectorArrayMap<T>(x, n) =
    ConstEigenVectorArrayMap<T>(x, n) * ConstEigenVectorArrayMap<T>(y, n);
};

template <typename T>
void min(T* x, const T* y, size_t n) {
  EigenVectorArrayMap<T>(x, n) =
    ConstEigenVectorArrayMap<T>(x, n).min(ConstEigenVectorArrayMap<T>(y, n));
};

template <typename T>
void max(T* x, const T* y, size_t n) {
  EigenVectorArrayMap<T>(x, n) =
    ConstEigenVectorArrayMap<T>(x, n).max(ConstEigenVectorArrayMap<T>(y, n));
};

#else

template <typename T>
void sum(T* x, const T* y, size_t n) {
  for (auto i = 0; i < n; i++) {
    x[i] = x[i] + y[i];
  }
}

template <typename T>
void product(T* x, const T* y, size_t n) {
  for (auto i = 0; i < n; i++) {
    x[i] = x[i] * y[i];
  }
}

template <typename T>
void max(T* x, const T* y, size_t n) {
  for (auto i = 0; i < n; i++) {
    x[i] = std::max(x[i], y[i]);
  }
}

template <typename T>
void min(T* x, const T* y, size_t n) {
  for (auto i = 0; i < n; i++) {
    x[i] = std::min(x[i], y[i]);
  }
}
#endif

} // namespace gloo
