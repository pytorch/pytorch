#include <iostream>

#include "caffe2/core/blob.h"
#include "caffe2/core/common.h"
#include "caffe2/core/context.h"
#include "caffe2/proto/caffe2.pb.h"
#include "gtest/gtest.h"

namespace caffe2 {

using namespace internal;  // NOLINT

class Foo {};
class Bar {};

TEST(BlobTest, TypeId) {
  TypeId int_id = GetTypeId<int>();
  TypeId float_id = GetTypeId<float>();
  TypeId foo_id = GetTypeId<Foo>();
  TypeId bar_id = GetTypeId<Bar>();
  EXPECT_NE(int_id, float_id);
  EXPECT_NE(float_id, foo_id);
  EXPECT_NE(foo_id, bar_id);
  EXPECT_TRUE(IsTypeId<int>(int_id));
  EXPECT_TRUE(IsTypeId<float>(float_id));
  EXPECT_TRUE(IsTypeId<Foo>(foo_id));
  EXPECT_TRUE(IsTypeId<Bar>(bar_id));
  EXPECT_FALSE(IsTypeId<int>(float_id));
  EXPECT_FALSE(IsTypeId<int>(foo_id));
  EXPECT_FALSE(IsTypeId<Foo>(int_id));
  EXPECT_FALSE(IsTypeId<Foo>(bar_id));
}

TEST(BlobTest, Blob) {
  Blob blob;

  int* int_unused UNUSED_VARIABLE = blob.GetMutable<int>();
  EXPECT_TRUE(blob.IsType<int>());
  EXPECT_FALSE(blob.IsType<Foo>());

  Foo* foo_unused UNUSED_VARIABLE = blob.GetMutable<Foo>();
  EXPECT_TRUE(blob.IsType<Foo>());
  EXPECT_FALSE(blob.IsType<int>());
}

TEST(BlobDeathTest, BlobUninitialized) {
  Blob blob;
  ASSERT_DEATH(blob.Get<int>(), ".*wrong type for the Blob instance.*");
}

TEST(BlobDeathTest, BlobWrongType) {
  Blob blob;
  Foo* foo_unused UNUSED_VARIABLE = blob.GetMutable<Foo>();
  EXPECT_TRUE(blob.IsType<Foo>());
  EXPECT_FALSE(blob.IsType<int>());
  // When not null, we should only call with the right type.
  EXPECT_NE(&blob.Get<Foo>(), nullptr);
  ASSERT_DEATH(blob.Get<int>(), ".*wrong type for the Blob instance.*");
}

template <typename dtype> class TensorCPUTest : public ::testing::Test {};
template <typename dtype> class TensorCPUDeathTest : public ::testing::Test {};
typedef ::testing::Types<char, int, float> TensorTypes;
TYPED_TEST_CASE(TensorCPUTest, TensorTypes);
TYPED_TEST_CASE(TensorCPUDeathTest, TensorTypes);

TYPED_TEST(TensorCPUTest, TensorInitializedEmpty) {
  Tensor<TypeParam, CPUContext> tensor;
  EXPECT_EQ(tensor.ndim(), 0);
  vector<int> dims(3);
  dims[0] = 2;
  dims[1] = 3;
  dims[2] = 5;
  tensor.Reshape(dims);
  EXPECT_EQ(tensor.ndim(), 3);
  EXPECT_EQ(tensor.dim(0), 2);
  EXPECT_EQ(tensor.dim(1), 3);
  EXPECT_EQ(tensor.dim(2), 5);
  EXPECT_EQ(tensor.size(), 2 * 3 * 5);
  EXPECT_TRUE(tensor.mutable_data() != nullptr);
  EXPECT_TRUE(tensor.data() != nullptr);
}

TYPED_TEST(TensorCPUTest, TensorInitializedNonEmpty) {
  vector<int> dims(3);
  dims[0] = 2;
  dims[1] = 3;
  dims[2] = 5;
  Tensor<TypeParam, CPUContext> tensor(dims);
  EXPECT_EQ(tensor.ndim(), 3);
  EXPECT_EQ(tensor.dim(0), 2);
  EXPECT_EQ(tensor.dim(1), 3);
  EXPECT_EQ(tensor.dim(2), 5);
  EXPECT_TRUE(tensor.mutable_data() != nullptr);
  EXPECT_TRUE(tensor.data() != nullptr);
  dims[0] = 7;
  dims[1] = 11;
  dims[2] = 13;
  dims.push_back(17);
  tensor.Reshape(dims);
  EXPECT_EQ(tensor.ndim(), 4);
  EXPECT_EQ(tensor.dim(0), 7);
  EXPECT_EQ(tensor.dim(1), 11);
  EXPECT_EQ(tensor.dim(2), 13);
  EXPECT_EQ(tensor.dim(3), 17);
  EXPECT_TRUE(tensor.mutable_data() != nullptr);
  EXPECT_TRUE(tensor.data() != nullptr);
}

TYPED_TEST(TensorCPUTest, TensorShareData) {
  vector<int> dims(3);
  dims[0] = 2;
  dims[1] = 3;
  dims[2] = 5;
  Tensor<TypeParam, CPUContext> tensor(dims);
  Tensor<TypeParam, CPUContext> other_tensor(dims);
  other_tensor.ShareData(tensor);
  EXPECT_TRUE(tensor.mutable_data() != nullptr);
  EXPECT_TRUE(tensor.data() != nullptr);
  EXPECT_TRUE(other_tensor.data() != nullptr);
  EXPECT_EQ(tensor.data(), other_tensor.data());
  // Set one value, check the other
  for (int i = 0; i < tensor.size(); ++i) {
    tensor.mutable_data()[i] = i;
    EXPECT_EQ(other_tensor.data()[i], i);
  }
}

TYPED_TEST(TensorCPUTest, TensorShareDataCanUseDifferentShapes) {
  vector<int> dims(3);
  dims[0] = 2;
  dims[1] = 3;
  dims[2] = 5;
  vector<int> alternate_dims(1);
  alternate_dims[0] = 2 * 3 * 5;
  Tensor<TypeParam, CPUContext> tensor(dims);
  Tensor<TypeParam, CPUContext> other_tensor(alternate_dims);
  other_tensor.ShareData(tensor);
  EXPECT_EQ(other_tensor.ndim(), 1);
  EXPECT_EQ(other_tensor.dim(0), alternate_dims[0]);
  EXPECT_TRUE(tensor.mutable_data() != nullptr);
  EXPECT_TRUE(tensor.data() != nullptr);
  EXPECT_TRUE(other_tensor.data() != nullptr);
  EXPECT_EQ(tensor.data(), other_tensor.data());
  // Set one value, check the other
  for (int i = 0; i < tensor.size(); ++i) {
    tensor.mutable_data()[i] = i;
    EXPECT_EQ(other_tensor.data()[i], i);
  }
}

TYPED_TEST(TensorCPUDeathTest, ShareDataCannotInitializeDataFromSharedTensor) {
  vector<int> dims(3);
  dims[0] = 2;
  dims[1] = 3;
  dims[2] = 5;
  Tensor<TypeParam, CPUContext> tensor(dims);
  Tensor<TypeParam, CPUContext> other_tensor(dims);
  other_tensor.ShareData(tensor);
  ASSERT_DEATH(other_tensor.mutable_data(), "");
}

TYPED_TEST(TensorCPUDeathTest, CannotDoReshapewithAlias) {
  vector<int> dims(3);
  dims[0] = 2;
  dims[1] = 3;
  dims[2] = 5;
  Tensor<TypeParam, CPUContext> tensor(dims);
  Tensor<TypeParam, CPUContext> other_tensor(dims);
  other_tensor.ShareData(tensor);
  dims[0] = 7;
  tensor.Reshape(dims);
  EXPECT_TRUE(tensor.mutable_data() != nullptr);
  ASSERT_DEATH(other_tensor.data(), ".*Source data size has changed..*");
}

TYPED_TEST(TensorCPUDeathTest, CannotAccessDataWhenEmpty) {
  Tensor<TypeParam, CPUContext> tensor;
  EXPECT_EQ(tensor.ndim(), 0);
  ASSERT_DEATH(tensor.data(), ".*Check failed: 'data_' Must be non NULL.*");
}


}  // namespace caffe2


