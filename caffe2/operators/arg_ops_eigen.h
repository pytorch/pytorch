/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef CAFFE2_OPERATORS_ARG_OPS_EIGEN_H_
#define CAFFE2_OPERATORS_ARG_OPS_EIGEN_H_

#include "caffe2/core/context.h"
#include "caffe2/core/types.h"

#include "Eigen/Core"

#if EIGEN_VERSION_AT_LEAST(3, 3, 0)

#include "unsupported/Eigen/CXX11/Tensor"

namespace caffe2 {
namespace arg_ops_eigen {

template <typename T>
using EigenTensorMap1D = Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>>;

template <typename T>
using EigenTensorMap2D = Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>>;

template <typename T>
using EigenTensorMap3D = Eigen::TensorMap<Eigen::Tensor<T, 3, Eigen::RowMajor>>;

template <class Device, typename T>
void ComputeArgMaxEigen(
    const Device& device,
    const T* X,
    const TIndex prev_size,
    const TIndex next_size,
    const TIndex n,
    TIndex* Y) {
  if (next_size == 1) {
    EigenTensorMap1D<TIndex>(Y, prev_size).device(device) =
        EigenTensorMap2D<T>(const_cast<T*>(X), prev_size, n)
            .argmax(1)
            .template cast<TIndex>();
  } else if (prev_size == 1) {
    EigenTensorMap1D<TIndex>(Y, next_size).device(device) =
        EigenTensorMap2D<T>(const_cast<T*>(X), n, next_size)
            .argmax(0)
            .template cast<TIndex>();
  } else {
    EigenTensorMap2D<TIndex>(Y, prev_size, next_size).device(device) =
        EigenTensorMap3D<T>(const_cast<T*>(X), prev_size, n, next_size)
            .argmax(1)
            .template cast<TIndex>();
  }
}

template <class Device, typename T>
void ComputeArgMinEigen(
    const Device& device,
    const T* X,
    const TIndex prev_size,
    const TIndex next_size,
    const TIndex n,
    TIndex* Y) {
  if (next_size == 1) {
    EigenTensorMap1D<TIndex>(Y, prev_size).device(device) =
        EigenTensorMap2D<T>(const_cast<T*>(X), prev_size, n)
            .argmin(1)
            .template cast<TIndex>();
  } else if (prev_size == 1) {
    EigenTensorMap1D<TIndex>(Y, next_size).device(device) =
        EigenTensorMap2D<T>(const_cast<T*>(X), n, next_size)
            .argmin(0)
            .template cast<TIndex>();
  } else {
    EigenTensorMap2D<TIndex>(Y, prev_size, next_size).device(device) =
        EigenTensorMap3D<T>(const_cast<T*>(X), prev_size, n, next_size)
            .argmin(1)
            .template cast<TIndex>();
  }
}

} // namespace arg_ops_eigen
} // namespace caffe2

#endif // EIGEN_VERSION_AT_LEAST(3, 3, 0)

#endif // CAFFE2_OPERATORS_ARG_OPS_EIGEN_H_
