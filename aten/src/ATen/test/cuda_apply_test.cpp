#include <gtest/gtest.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/CUDAContext.h>
#ifdef USE_ROCM
#include <ATen/ATen.h>
#include <ATen/native/cuda/GridSampler.h>
#include <limits>
#endif
#define ASSERT_EQ_CUDA(X, Y) \
  {                          \
    bool _isEQ = X == Y;     \
    ASSERT_TRUE(_isEQ);      \
  }
/*
   Tests related to tensor indexing and applying operations.
*/
#ifndef _WIN32

// CATCH_TEST_CASE("2D Contiguous", "Collapses a 2D contiguous tensor to 1D
// contiguous") {
TEST(ApplyTest, Contiguous2D) {
  if (!at::cuda::is_available()) return;
  int sizes[] = {4, 4};
  int strides[] = {4, 1};
  ::at::cuda::detail::TensorInfo<void, int> ti{nullptr, 2, sizes, strides};
  ti.collapseDims();
  ASSERT_EQ_CUDA(ti.dims, 1);
  ASSERT_EQ_CUDA(ti.sizes[0], (4 * 4));
}

// CATCH_TEST_CASE("3D Contiguous", "Collapses a 3D contiguous tensor to a 1D
// contiguous") {
TEST(ApplyTest, Contiguous3D) {
  if (!at::cuda::is_available()) return;
  int sizes[] = {6, 3, 7};
  int strides[] = {3 * 7, 7, 1};
  ::at::cuda::detail::TensorInfo<void, int> ti{nullptr, 3, sizes, strides};
  ti.collapseDims();
  ASSERT_EQ_CUDA(ti.dims, 1);
  ASSERT_EQ_CUDA(ti.sizes[0], (6 * 3 * 7));
}
// CATCH_TEST_CASE("3D Partial Collapse", "Collapses a 3D noncontiguous tensor
// to a 2D tensor") {
TEST(ApplyTest, PartialCollapse3D) {
  if (!at::cuda::is_available()) return;
  int sizes[] = {4, 3, 2};
  int strides[] = {3 * 3, 3, 1};
  ::at::cuda::detail::TensorInfo<void, int> ti{nullptr, 3, sizes, strides};
  ti.collapseDims();
  ASSERT_EQ_CUDA(ti.dims, 2);
  ASSERT_EQ_CUDA(ti.sizes[0], (4 * 3));
  ASSERT_EQ_CUDA(ti.sizes[1], 2);
}

// Collapses a 2D skip contiguous tensor to a 1D skip contiguous tensor
TEST(ApplyTest, StridedCollapse2D) {
  if (!at::cuda::is_available()) return;
  int sizes[] = {3, 2};
  int strides[] = {2 * 2, 2};
  ::at::cuda::detail::TensorInfo<void, int> ti{nullptr, 2, sizes, strides};
  ti.collapseDims();
  ASSERT_EQ_CUDA(ti.dims, 1);
  ASSERT_EQ_CUDA(ti.sizes[0], (3 * 2));
  ASSERT_EQ_CUDA(ti.strides[0], 2);
}

// Collapses a 4D tensor to a 2D tensor
TEST(ApplyTest, PartialStridedCollapse4D) {
  if (!at::cuda::is_available()) return;
  int sizes[] = {3, 6, 5, 2};
  int strides[] = {6 * 22, 22, 2 * 2, 2};
  ::at::cuda::detail::TensorInfo<void, int> ti{nullptr, 4, sizes, strides};
  ti.collapseDims();
  ASSERT_EQ_CUDA(ti.dims, 2);
  ASSERT_EQ_CUDA(ti.sizes[0], (3 * 6));
  ASSERT_EQ_CUDA(ti.strides[0], 22);
  ASSERT_EQ_CUDA(ti.sizes[1], (5 * 2));
  ASSERT_EQ_CUDA(ti.strides[1], 2);
}

// Collapses a 5D tensor to a 1D tensor
TEST(ApplyTest, CollapsesZerosAndOnes) {
  if (!at::cuda::is_available()) return;
  int sizes[] = {1, 10, 1, 5, 4};
  int strides[] = {4, 0, 16, 0, 1};
  ::at::cuda::detail::TensorInfo<void, int> ti{nullptr, 5, sizes, strides};
  ti.collapseDims();
  ASSERT_EQ_CUDA(ti.dims, 2);
  ASSERT_EQ_CUDA(ti.sizes[0], (10 * 5));
  ASSERT_EQ_CUDA(ti.strides[0], 0);
  ASSERT_EQ_CUDA(ti.sizes[1], 4);
  ASSERT_EQ_CUDA(ti.strides[1], 1);
}

// Collapses a 3D tensor to a point tensor
TEST(ApplyTest, CollapseToPointTensor) {
  if (!at::cuda::is_available()) return;
  int sizes[] = {1, 1, 1};
  int strides[] = {17, 12, 3};
  ::at::cuda::detail::TensorInfo<void, int> ti{nullptr, 3, sizes, strides};
  ASSERT_EQ_CUDA(ti.collapseDims(), 0);
  ASSERT_EQ_CUDA(ti.dims, 1);
  ASSERT_EQ_CUDA(ti.sizes[0], 1);
  ASSERT_EQ_CUDA(ti.strides[0], 1);
}

// Collapses a 4D tensor to a 3D tensor
TEST(ApplyTest, ExcludingInContiguous4D) {
  if (!at::cuda::is_available()) return;
  int sizes[] = {3, 6, 5, 2};
  int strides[] = {6 * 22, 22, 2 * 2, 2};
  ::at::cuda::detail::TensorInfo<void, int> ti{nullptr, 4, sizes, strides};
  ASSERT_EQ_CUDA(ti.collapseDims(1), 1);
  ASSERT_EQ_CUDA(ti.dims, 3);
  ASSERT_EQ_CUDA(ti.sizes[0], 3);
  ASSERT_EQ_CUDA(ti.strides[0], (6 * 22));
  ASSERT_EQ_CUDA(ti.sizes[1], 6);
  ASSERT_EQ_CUDA(ti.strides[1], 22);
  ASSERT_EQ_CUDA(ti.sizes[2], (5 * 2));
  ASSERT_EQ_CUDA(ti.strides[2], 2);
}

// Collapses a 4D tensor to a 3D tensor
TEST(ApplyTest, RovingExclusion) {
  if (!at::cuda::is_available()) return;
  int sizes[] = {3, 6, 5, 2};
  int strides[] = {6 * 22, 22, 2 * 2, 2};
  ::at::cuda::detail::TensorInfo<void, int> ti{nullptr, 4, sizes, strides};
  ASSERT_EQ_CUDA(ti.collapseDims(2), 1);
  ASSERT_EQ_CUDA(ti.dims, 3);
  ASSERT_EQ_CUDA(ti.sizes[0], (3 * 6));
  ASSERT_EQ_CUDA(ti.strides[0], 22);
  ASSERT_EQ_CUDA(ti.sizes[1], 5);
  ASSERT_EQ_CUDA(ti.strides[1], 4);
  ASSERT_EQ_CUDA(ti.sizes[2], 2);
  ASSERT_EQ_CUDA(ti.strides[2], 2);
}

// Attempts to exclude a nonexisting dimension
TEST(ApplyTest, InvalidExclusion) {
  if (!at::cuda::is_available()) return;
  int sizes[] = {1, 1, 1};
  int strides[] = {17, 12, 3};
  ::at::cuda::detail::TensorInfo<void, int> ti{nullptr, 3, sizes, strides};
  ASSERT_ANY_THROW(ti.collapseDims(5));
}
#ifdef USE_ROCM

// The ROCm grid_sampler_2d backward channel-parallel kernel advances each
// lane's channel counter by the lane count, so 32-bit indexing also needs
// C - 1 + lanes to fit. Meta tensors keep this host-only: no GPU or large
// allocation is needed.
TEST(ApplyTest, GridSampler2dBackwardChannelLaneIndexWidth) {
  using at::native::canUse32BitIndexMath;
  using at::native::rocm_grid_sampler_2d_backward_channel_lane_use_32bit_index;
  const auto meta = at::TensorOptions().device(at::kMeta).dtype(at::kBFloat16);
  const auto grid = at::empty({1, 1, 1, 2}, meta);
  const int64_t int32_max = std::numeric_limits<int32_t>::max();
  ASSERT_EQ(at::native::rocm_grid_sampler_2d_backward_lanes(int32_max), 256);
  // (C, expected selection); the large C use scalar-expanded tensors.
  const std::pair<int64_t, bool> cases[] = {
      {257, true},
      {int32_max - 255, true},
      {int32_max - 254, false},
      {(int64_t{1} << 31) - 128, false},
  };
  for (const auto& [C, use_32bit] : cases) {
    const auto input = at::empty({1, 1, 1, 1}, meta).expand({1, C, 1, 1});
    const auto grad_output = at::empty({1, 1, 1, 1}, meta).expand({1, C, 1, 1});
    // The tensor-offset checks alone accept every case.
    ASSERT_TRUE(canUse32BitIndexMath(input));
    ASSERT_TRUE(canUse32BitIndexMath(grid));
    ASSERT_TRUE(canUse32BitIndexMath(grad_output));
    EXPECT_EQ(
        rocm_grid_sampler_2d_backward_channel_lane_use_32bit_index(
            input, grid, grad_output),
        use_32bit)
        << "C=" << C;
  }
  // A tensor offset beyond the 32-bit range still selects 64-bit indexing.
  const auto strided =
      at::empty_strided({1, 4, 1, 1}, {4, int64_t{1} << 31, 1, 1}, meta);
  ASSERT_FALSE(canUse32BitIndexMath(strided));
  EXPECT_FALSE(rocm_grid_sampler_2d_backward_channel_lane_use_32bit_index(
      strided, grid, at::empty({1, 4, 1, 1}, meta)));
}
#endif
#endif
