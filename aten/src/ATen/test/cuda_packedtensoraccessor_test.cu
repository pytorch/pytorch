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
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  ASSERT_TRUE(cudaSuccess == cudaDeviceSynchronize());

  auto expected = mv(t1, t2);

  ASSERT_TRUE(res.allclose(expected));
}

// Test the metadata-based approach for RestrictPtrTraits workaround
__global__ void test_metadata_accessor_kernel(
    float* __restrict__ res_data,
    const float* __restrict__ t1_data,
    const float* __restrict__ t2_data,
    PackedTensorAccessorMetadata<1, int64_t> res_meta,
    PackedTensorAccessorMetadata<2, int64_t> t1_meta,
    PackedTensorAccessorMetadata<1, int64_t> t2_meta) {
  for (int64_t i = 0; i < res_meta.size(0); i++) {
    float val = 0.0f;
    for (int64_t j = 0; j < t1_meta.size(1); j++) {
      val += packed_accessor_get(t1_data, t1_meta, i, j) *
             packed_accessor_get(t2_data, t2_meta, j);
    }
    packed_accessor_get(res_data, res_meta, i) = val;
  }
}

// Test using packed_accessor_offset directly
__global__ void test_offset_accessor_kernel(
    float* __restrict__ res_data,
    const float* __restrict__ t1_data,
    const float* __restrict__ t2_data,
    PackedTensorAccessorMetadata<1, int64_t> res_meta,
    PackedTensorAccessorMetadata<2, int64_t> t1_meta,
    PackedTensorAccessorMetadata<1, int64_t> t2_meta) {
  for (int64_t i = 0; i < res_meta.size(0); i++) {
    float val = 0.0f;
    for (int64_t j = 0; j < t1_meta.size(1); j++) {
      int64_t t1_offset = packed_accessor_offset(t1_meta, i, j);
      int64_t t2_offset = packed_accessor_offset(t2_meta, j);
      val += t1_data[t1_offset] * t2_data[t2_offset];
    }
    int64_t res_offset = packed_accessor_offset(res_meta, i);
    res_data[res_offset] = val;
  }
}

// Verify metadata + __restrict__ pointer approach computes correct matrix-vector product
TEST(PackedtensoraccessorTest, MetadataAccessorTestCUDA) {
  if (!at::cuda::is_available()) return;
  manual_seed(456);

  Tensor t1 = rand({4, 4}, CUDA(kFloat));
  Tensor t2 = rand({4}, CUDA(kFloat));
  Tensor res = empty({4}, CUDA(kFloat));

  auto t1a = t1.packed_accessor64<float, 2, RestrictPtrTraits>();
  auto t2a = t2.packed_accessor64<float, 1, RestrictPtrTraits>();
  auto resa = res.packed_accessor64<float, 1, RestrictPtrTraits>();

  auto t1_meta = make_packed_accessor_metadata(t1a);
  auto t2_meta = make_packed_accessor_metadata(t2a);
  auto res_meta = make_packed_accessor_metadata(resa);

  auto stream = at::cuda::getCurrentCUDAStream();

  test_metadata_accessor_kernel<<<1, 1, 0, stream>>>(
      res.data_ptr<float>(),
      t1.data_ptr<float>(),
      t2.data_ptr<float>(),
      res_meta, t1_meta, t2_meta);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  ASSERT_TRUE(cudaSuccess == cudaDeviceSynchronize());

  auto expected = mv(t1, t2);
  ASSERT_TRUE(res.allclose(expected));

  res.zero_();
  test_offset_accessor_kernel<<<1, 1, 0, stream>>>(
      res.data_ptr<float>(),
      t1.data_ptr<float>(),
      t2.data_ptr<float>(),
      res_meta, t1_meta, t2_meta);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  ASSERT_TRUE(cudaSuccess == cudaDeviceSynchronize());

  ASSERT_TRUE(res.allclose(expected));
}

// Test metadata extraction preserves correct values
TEST(PackedtensoraccessorTest, MetadataValuesTestCUDA) {
  if (!at::cuda::is_available()) return;

  Tensor t = rand({8, 6, 4}, CUDA(kFloat));
  Tensor t_strided = t.transpose(0, 1);

  auto acc = t_strided.packed_accessor64<float, 3, RestrictPtrTraits>();
  auto meta = make_packed_accessor_metadata(acc);

  ASSERT_EQ(meta.size(0), acc.size(0));
  ASSERT_EQ(meta.size(1), acc.size(1));
  ASSERT_EQ(meta.size(2), acc.size(2));

  ASSERT_EQ(meta.stride(0), acc.stride(0));
  ASSERT_EQ(meta.stride(1), acc.stride(1));
  ASSERT_EQ(meta.stride(2), acc.stride(2));
}
