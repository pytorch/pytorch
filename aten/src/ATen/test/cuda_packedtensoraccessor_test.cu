#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/core/TensorAccessor.h>
#include <ATen/cuda/CUDAContext.h>

#include <assert.h>

__global__ void test_tensor_packed_accessor_kernel(
    at::PackedTensorAccessor64<float, 1, at::RestrictPtrTraits> resa,
    at::PackedTensorAccessor64<float, 2, at::RestrictPtrTraits> t1a,
    at::PackedTensorAccessor64<float, 1, at::RestrictPtrTraits> t2a) {
  for (int64_t i = 0; i < resa.size(0); i++) {
    float val = 0.0f;
    for (int64_t j = 0; j < t1a.size(1); j++) {
      val += t1a[i][j] * t2a[j];
    }
    resa[i] = val;
  }
}

// test GenericPackedTensorAccessor and Tensor.generic_packed_accessor
TEST(PackedtensoraccessorTest, PackedtensoraccessorTestCUDA) {
  if (!at::cuda::is_available()) return;
  at::manual_seed(123);

  at::Tensor t1 = at::rand({4, 4}, at::CUDA(at::kFloat));
  at::Tensor t2 = at::rand({4}, at::CUDA(at::kFloat));
  at::Tensor res = at::empty({4}, at::CUDA(at::kFloat));

  auto t1a = t1.packed_accessor64<float, 2, at::RestrictPtrTraits>();
  auto t2a = t2.packed_accessor64<float, 1, at::RestrictPtrTraits>();
  auto resa = res.packed_accessor64<float, 1, at::RestrictPtrTraits>();

  auto stream = at::cuda::getCurrentCUDAStream();

  test_tensor_packed_accessor_kernel<<<1, 1, 0, stream>>>(resa, t1a, t2a);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  ASSERT_TRUE(cudaSuccess == cudaDeviceSynchronize());

  auto expected = mv(t1, t2);

  ASSERT_TRUE(res.allclose(expected));
}
