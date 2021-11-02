#include <cuda.h>
#include <cuda_runtime.h>

#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/DLConvertor.h>
#include <ATen/cuda/CUDAConfig.h>
#include <ATen/cuda/CUDAContext.h>

#include <string.h>
#include <iostream>
#include <sstream>

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
  ASSERT_TRUE(dlMTensor->dl_tensor.device.device_type == DLDeviceType::kDLGPU);
#endif

  Tensor b = fromDLPack(dlMTensor);

  ASSERT_TRUE(a.equal(b));
}
