/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// work around CK assuming only a single FP8 interpretation at a time
#if defined(__gfx942__) && __HIP_DEVICE_COMPILE__
#define CK_USE_FNUZ_FP8 1
#undef CK_USE_OCP_FP8
#elif __HIP_DEVICE_COMPILE__
#undef CK_USE_FNUZ_FP8
#define CK_USE_OCP_FP8 1
#endif

#include <ATen/ATen.h>
#include <ck/ck.hpp>
#include <ck/tensor_operation/gpu/device/tensor_layout.hpp>
#include <ck/utility/data_type.hpp>

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

namespace at::native {

template <typename T>
struct CkMathType {
  using dtype = T;
};

template <>
struct CkMathType<at::BFloat16> {
  using dtype = ck::bhalf_t;
};

template <>
struct CkMathType<at::Half> {
  using dtype = ck::half_t;
};

template <bool A, bool B>
struct CkTensorLayout {
  // default goes to row-wise for now
  using a_layout = Row;
  using b_layout = Row;
};

// True denotes transpose is necessary. Default is Col, so return Row
template <>
struct CkTensorLayout<true, true> {
  using a_layout = Col;
  using b_layout = Col;
};

template <>
struct CkTensorLayout<true, false> {
  using a_layout = Row;
  using b_layout = Col;
};

template <>
struct CkTensorLayout<false, true> {
  using a_layout = Col;
  using b_layout = Row;
};

template <>
struct CkTensorLayout<false, false> {
  using a_layout = Row;
  using b_layout = Row;
};

} // namespace at::native
