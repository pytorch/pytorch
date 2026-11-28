#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/DLConvertor.h>

using namespace at;

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

TEST(TestDlconvertorUnversioned, TestDlconvertor) {
  manual_seed(123);

  Tensor a = rand({3, 4});
  DLManagedTensorVersioned* dlMTensor = toDLPackVersioned(a);

  Tensor b = fromDLPackVersioned(dlMTensor);

  ASSERT_TRUE(a.equal(b));
}

TEST(TestDlconvertorUnversioned, TestDlconvertorNoStrides) {
  manual_seed(123);

  Tensor a = rand({3, 4});
  DLManagedTensorVersioned* dlMTensor = toDLPackVersioned(a);
  dlMTensor->dl_tensor.strides = nullptr;

  Tensor b = fromDLPackVersioned(dlMTensor);

  ASSERT_TRUE(a.equal(b));
}
