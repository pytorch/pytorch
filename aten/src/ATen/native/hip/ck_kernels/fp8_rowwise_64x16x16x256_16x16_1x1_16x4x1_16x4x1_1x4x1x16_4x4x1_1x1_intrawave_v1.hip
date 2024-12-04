/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ATen/native/hip/ck_kernels/fp8_rowwise_common.h>

namespace at::cuda::detail {

at::Tensor
fp8_rowwise_64x16x16x256_16x16_1x1_16x4x1_16x4x1_1x4x1x16_4x4x1_1x1_intrawave_v1(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y) {
  // Secret kernel that seems good with small M but large N and K.
  using DeviceGemmInstance = DeviceGemmHelper<
      64,
      16,
      16,
      256,
      16,
      16,
      1,
      1,
      S<16, 4, 1>,
      S<16, 4, 1>,
      S<1, 16, 1, 4>,
      S<4, 4, 1>,
      1,
      1,
      ck::BlockGemmPipelineScheduler::Intrawave,
      ck::BlockGemmPipelineVersion::v1,
      ck::tensor_operation::device::GemmSpecialization::Default>;
  // Run kernel instance.
  return f8f8bf16_rowwise_impl<DeviceGemmInstance>(XQ, WQ, x_scale, w_scale, Y);
}

} // namespace at::cuda::detail
