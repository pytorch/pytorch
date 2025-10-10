#include <cuda.h>
#include <cuda_runtime.h>

#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/DLConvertor.h>
#include <ATen/cuda/CUDAConfig.h>
#include <ATen/cuda/CUDAContext.h>

using namespace at;

TEST(TestDlconvertor, TestDlconvertorCUDA) {
  manual_seed(123);

  Tensor a = rand({3, 4}, at::kCUDA);
  DLManagedTensor* dlMTensor = toDLPack(a);

  Tensor b = fromDLPack(dlMTensor);

  ASSERT_TRUE(a.equal(b));
}

TEST(TestDlconvertor, TestDlconvertorNoStridesCUDA) {
  manual_seed(123);

  Tensor a = rand({3, 4}, at::kCUDA);
  DLManagedTensor* dlMTensor = toDLPack(a);
  dlMTensor->dl_tensor.strides = nullptr;

  Tensor b = fromDLPack(dlMTensor);

  ASSERT_TRUE(a.equal(b));
}

TEST(TestDlconvertor, TestDlconvertorCUDAHIP) {
  if (!at::cuda::is_available())
    return;
  manual_seed(123);

  Tensor a = rand({3, 4}, at::kCUDA);
  DLManagedTensor* dlMTensor = toDLPack(a);

#if AT_ROCM_ENABLED()
  ASSERT_TRUE(dlMTensor->dl_tensor.device.device_type == DLDeviceType::kDLROCM);
#else
  ASSERT_TRUE(dlMTensor->dl_tensor.device.device_type == DLDeviceType::kDLCUDA);
#endif

  Tensor b = fromDLPack(dlMTensor);

  ASSERT_TRUE(a.equal(b));
}

TEST(TestDlconvertorVersioned, TestDlconvertorCUDA) {
  manual_seed(123);

  Tensor a = rand({3, 4}, at::kCUDA);
  DLManagedTensorVersioned* dlMTensor = toDLPackVersioned(a);

  Tensor b = fromDLPackVersioned(dlMTensor);

  ASSERT_TRUE(a.equal(b));
}

TEST(TestDlconvertorVersioned, TestDlconvertorNoStridesCUDA) {
  manual_seed(123);

  Tensor a = rand({3, 4}, at::kCUDA);
  DLManagedTensorVersioned* dlMTensor = toDLPackVersioned(a);
  dlMTensor->dl_tensor.strides = nullptr;

  Tensor b = fromDLPackVersioned(dlMTensor);

  ASSERT_TRUE(a.equal(b));
}

TEST(TestDlconvertorVersioned, TestDlconvertorCUDAHIP) {
  if (!at::cuda::is_available())
    return;
  manual_seed(123);

  Tensor a = rand({3, 4}, at::kCUDA);
  DLManagedTensorVersioned* dlMTensor = toDLPackVersioned(a);

#if AT_ROCM_ENABLED()
  ASSERT_TRUE(dlMTensor->dl_tensor.device.device_type == DLDeviceType::kDLROCM);
#else
  ASSERT_TRUE(dlMTensor->dl_tensor.device.device_type == DLDeviceType::kDLCUDA);
#endif

  Tensor b = fromDLPackVersioned(dlMTensor);

  ASSERT_TRUE(a.equal(b));
}
