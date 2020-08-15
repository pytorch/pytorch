#include <gtest/gtest.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/cuda/CUDAContext.h>
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
#endif
