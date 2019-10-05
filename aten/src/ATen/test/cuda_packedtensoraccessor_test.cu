#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/core/TensorAccessor.h>
#include <ATen/cuda/CUDAContext.h>

#include <assert.h>

using namespace at;

__global__ void test_tensor_packed_accessor_kernel(
    PackedTensorAccessor64<float, 1, RestrictPtrTraits> resa,
    PackedTensorAccessor64<float, 2, RestrictPtrTraits> t1a,
    PackedTensorAccessor64<float, 1, RestrictPtrTraits> t2a) {
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
  manual_seed(123);

  Tensor t1 = rand({4, 4}, CUDA(kFloat));
  Tensor t2 = rand({4}, CUDA(kFloat));
  Tensor res = empty({4}, CUDA(kFloat));

  auto t1a = t1.packed_accessor64<float, 2, RestrictPtrTraits>();
  auto t2a = t2.packed_accessor64<float, 1, RestrictPtrTraits>();
  auto resa = res.packed_accessor64<float, 1, RestrictPtrTraits>();

  auto stream = at::cuda::getCurrentCUDAStream();

  test_tensor_packed_accessor_kernel<<<1, 1, 0, stream>>>(resa, t1a, t2a);
  cudaError_t err = cudaDeviceSynchronize();
  bool isEQ = err == cudaSuccess;
  ASSERT_TRUE(isEQ);

  auto expected = mv(t1, t2);

  ASSERT_TRUE(res.allclose(expected));
}
