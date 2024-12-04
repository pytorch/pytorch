/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdlib>

#include <ATen/ATen.h>

namespace at::cuda::detail {

// Tiny tile kernel for small shapes.
at::Tensor
fp8_rowwise_64x16x16x128_16x16_1x1_8x8x1_8x8x1_1x16x1x4_4x4x1_1x1_interwave_v2(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

// Another variant of tiny kernel.
at::Tensor
fp8_rowwise_64x16x16x64_16x16_1x1_4x16x1_4x16x1_1x16x1x4_4x4x1_1x1_interwave_v2(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

at::Tensor
fp8_rowwise_64x16x16x512_16x16_1x1_8x8x1_8x8x1_1x16x1x4_4x4x1_1x1_interwave_v2(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

at::Tensor
fp8_rowwise_64x16x16x512_16x16_1x1_32x2x1_32x2x1_1x16x1x4_4x4x1_1x1_interwave_v2(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

// Alternate tiny kernel that seems to do well when M and K are all small.
at::Tensor
fp8_rowwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v2(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

// Special kernel that works well for small M but large N and K.
at::Tensor
fp8_rowwise_64x16x16x256_16x16_1x1_16x4x1_16x4x1_1x4x1x16_4x4x1_1x1_intrawave_v1(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

// Kernel that seems to work well for small M and N.
at::Tensor
fp8_rowwise_128x128x16x128_16x16_4x1_8x16x1_8x16x1_1x16x1x8_8x8x1_1x1_interwave_v2(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

// Variant that uses interwave scheduling for small batches.
at::Tensor
fp8_rowwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_interwave_v2(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

at::Tensor
fp8_rowwise_128x16x32x512_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_interwave_v2(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

// Alternate tiny kernel that seems to do well when M and N are all small.
at::Tensor
fp8_rowwise_128x32x16x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_2x2x1_1x1_interwave_v2(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

// Alternate split k tiny kernel that seems to do well when M and N are small.
at::Tensor
fp8_rowwise_128x32x16x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_2x2x1_1x1_interwave_v2_16(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

// Small kernel for small shapes with a little beef to them.
at::Tensor
fp8_rowwise_128x32x64x128_32x32_1x1_8x16x1_8x16x1_1x16x1x8_8x8x1_1x1_intrawave_v2(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

// Small kernel for small shapes with a little beef to them.
at::Tensor
fp8_rowwise_128x32x64x128_32x32_1x1_8x16x1_8x16x1_1x16x1x8_8x8x1_1x1_interwave_v2(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

// Kernel that works well on squarish mid sized tensors.
at::Tensor
fp8_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_interwave_v1(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

// V4 Kernel that seems to do well for small-medium shapes.
at::Tensor
fp8_rowwise_256x128x128x64_32x32_2x2_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

// V4 Kernel that seems to do well for square medium shapes.
at::Tensor
fp8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

// V3 Kernel that works well for medium batch sizes.
at::Tensor
fp8_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

// V5 Kernel that works well for medium batch sizes.
at::Tensor
fp8_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v5(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

// Kernel that is well optimized for many medium to large shapes.
at::Tensor
fp8_rowwise_256x224x256x128_16x16_7x8_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

// Kernel that seems optimal for highly compute bound problems.
at::Tensor
fp8_rowwise_256x256x224x128_16x16_8x7_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

// Kernel for medium batch sizes.
at::Tensor
fp8_rowwise_256x64x64x128_32x32_1x1_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

// Kernel for larger batch sizes.
at::Tensor
fp8_rowwise_256x128x64x128_32x32_2x1_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

// Larger batch size with small tensors kernel.
at::Tensor
fp8_rowwise_128x64x32x128_32x32_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v2(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

// Another varient of larger batch size support.
at::Tensor
fp8_rowwise_128x32x128x128_32x32_1x2_8x16x1_8x16x1_1x16x1x8_8x8x1_1x1_interwave_v2(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

// Kernel that seems optimal for highly compute bound problems.
at::Tensor
fp8_rowwise_256x256x256x128_16x16_8x8_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

// Decent mid-size kernel.
at::Tensor
fp8_rowwise_256x256x256x64_16x16_8x8_4x64x1_4x64x1_1x32x1x8_8x8x1_1x2_intrawave_v3(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

// Kernel for small but not too small batch sizes.
at::Tensor
fp8_rowwise_128x128x32x128_32x32_2x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v2(
    at::Tensor XQ,
    at::Tensor WQ,
    at::Tensor x_scale,
    at::Tensor w_scale,
    at::Tensor Y);

} // namespace at::cuda::detail
