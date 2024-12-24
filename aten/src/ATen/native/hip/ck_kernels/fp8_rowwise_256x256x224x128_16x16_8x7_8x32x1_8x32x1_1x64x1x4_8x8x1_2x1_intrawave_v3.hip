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
fp8_rowwise_256x256x224x128_16x16_8x7_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y) {
  // Check if this input needs to be padded.
  int M = size_to_dim_(XQ.dim() - 1, XQ.sizes());
  int N = WQ.size(0);
  int K = WQ.size(1);
  bool pad = (M % 256 != 0) || (N % 224 != 0) || (K % 128 != 0);

  // This kernel seems optimal in the most purely compute bound tasks.
  if (pad) {
    using DeviceGemmInstance = DeviceGemmHelper<
        256,
        256,
        224,
        128,
        16,
        16,
        8,
        7,
        S<8, 32, 1>,
        S<8, 32, 1>,
        S<1, 64, 1, 4>,
        S<8, 8, 1>,
        2,
        1,
        ck::BlockGemmPipelineScheduler::Intrawave,
        ck::BlockGemmPipelineVersion::v3>;
    // Run kernel instance.
    return f8f8bf16_rowwise_impl<DeviceGemmInstance>(
        XQ, WQ, x_scale, w_scale, Y);
  } else {
    using DeviceGemmInstance = DeviceGemmHelper<
        256,
        256,
        224,
        128,
        16,
        16,
        8,
        7,
        S<8, 32, 1>,
        S<8, 32, 1>,
        S<1, 64, 1, 4>,
        S<8, 8, 1>,
        2,
        1,
        ck::BlockGemmPipelineScheduler::Intrawave,
        ck::BlockGemmPipelineVersion::v3,
        ck::tensor_operation::device::GemmSpecialization::Default>;
    // Run kernel instance.
    return f8f8bf16_rowwise_impl<DeviceGemmInstance>(
        XQ, WQ, x_scale, w_scale, Y);
  }
}

} // namespace at::cuda::detail
