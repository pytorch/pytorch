#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/DLConvertor.h>

#include <iostream>
#include <string.h>
#include <sstream>

using namespace at;

// internal to DLConvertor.cpp
struct ATenDLMTensor {
  Tensor handle;
  DLManagedTensor tensor;
};

DLManagedTensor* toDLPackNoDeleter(const Tensor& src) {
  ATenDLMTensor* atDLMTensor(new ATenDLMTensor);
  atDLMTensor->handle = src;
  atDLMTensor->tensor.manager_ctx = atDLMTensor;
  atDLMTensor->tensor.dl_tensor.data = src.data_ptr();
  int64_t device_id = 0;
  if (src.is_cuda()) {
    device_id = src.get_device();
  }
  atDLMTensor->tensor.dl_tensor.ctx = getDLContext(src, device_id);
  atDLMTensor->tensor.dl_tensor.ndim = src.dim();
  atDLMTensor->tensor.dl_tensor.dtype = getDLDataType(src);
  atDLMTensor->tensor.dl_tensor.shape =
      const_cast<int64_t*>(src.sizes().data());
  atDLMTensor->tensor.dl_tensor.strides =
      const_cast<int64_t*>(src.strides().data());
  atDLMTensor->tensor.dl_tensor.byte_offset = 0;
  return &(atDLMTensor->tensor);
}

TEST(TestDlconvertor, TestDlconvertor) {
  manual_seed(123);

  Tensor a = rand({3, 4});
  DLManagedTensor* dlMTensor = toDLPack(a);

  Tensor b = fromDLPack(dlMTensor);

  ASSERT_TRUE(a.equal(b));
}

TEST(TestDlconvertor, TestDlconvertorNoStrides) {
  manual_seed(123);

  Tensor a = rand({3, 4});
  DLManagedTensor* dlMTensor = toDLPack(a);
  dlMTensor->dl_tensor.strides = nullptr;

  Tensor b = fromDLPack(dlMTensor);

  ASSERT_TRUE(a.equal(b));
}

TEST(TestDlconvertor, TestDlconvertorNullDeleter) {
  manual_seed(123);

  Tensor a = rand({3, 4});

  // no deleter necessary since dlManagedTensor is initialized on the stack
  DLManagedTensor dlMTensor;
  dlMTensor.deleter = nullptr;
  dlMTensor.dl_tensor.data = a.data_ptr();
  dlMTensor.dl_tensor.ctx = getDLContext(a, 0);
  dlMTensor.dl_tensor.ndim = a.dim();
  dlMTensor.dl_tensor.dtype = getDLDataType(a);
  dlMTensor.dl_tensor.shape = const_cast<int64_t*>(a.sizes().data());
  dlMTensor.dl_tensor.strides = const_cast<int64_t*>(a.strides().data());
  dlMTensor.dl_tensor.byte_offset = 0;

  Tensor b = fromDLPack(&dlMTensor);

  ASSERT_TRUE(a.equal(b));
}
