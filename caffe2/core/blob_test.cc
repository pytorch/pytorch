#include <iostream>

#include "caffe2/core/blob.h"
#include "caffe2/core/common.h"
#include "caffe2/core/context.h"
#include "caffe2/core/tensor.h"
#include "caffe2/proto/caffe2.pb.h"
#include "gtest/gtest.h"

namespace caffe2 {
namespace {

class Foo {};
class Bar {};

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

TEST(TensorNonTypedTest, TensorChangeType) {
  vector<int> dims(3);
  dims[0] = 2;
  dims[1] = 3;
  dims[2] = 5;
  TensorCPU tensor(dims);
  EXPECT_TRUE(tensor.mutable_data<int>() != nullptr);
  EXPECT_TRUE(tensor.data<int>() != nullptr);
  EXPECT_TRUE(tensor.meta().Match<int>());

  EXPECT_TRUE(tensor.mutable_data<float>() != nullptr);
  EXPECT_TRUE(tensor.data<float>() != nullptr);
  EXPECT_TRUE(tensor.meta().Match<float>());
}

template <typename T> class TensorCPUTest : public ::testing::Test {};
template <typename T> class TensorCPUDeathTest : public ::testing::Test {};
typedef ::testing::Types<char, int, float> TensorTypes;
TYPED_TEST_CASE(TensorCPUTest, TensorTypes);
TYPED_TEST_CASE(TensorCPUDeathTest, TensorTypes);

TYPED_TEST(TensorCPUTest, TensorInitializedEmpty) {
  TensorCPU tensor;
  EXPECT_EQ(tensor.ndim(), 0);
  vector<int> dims(3);
  dims[0] = 2;
  dims[1] = 3;
  dims[2] = 5;
  tensor.Reshape(dims);
  EXPECT_EQ(tensor.ndim(), 3);
  EXPECT_EQ(tensor.dim32(0), 2);
  EXPECT_EQ(tensor.dim32(1), 3);
  EXPECT_EQ(tensor.dim32(2), 5);
  EXPECT_EQ(tensor.size(), 2 * 3 * 5);
  EXPECT_TRUE(tensor.mutable_data<TypeParam>() != nullptr);
  EXPECT_TRUE(tensor.data<TypeParam>() != nullptr);
}

TYPED_TEST(TensorCPUTest, TensorInitializedNonEmpty) {
  vector<int> dims(3);
  dims[0] = 2;
  dims[1] = 3;
  dims[2] = 5;
  TensorCPU tensor(dims);
  EXPECT_EQ(tensor.ndim(), 3);
  EXPECT_EQ(tensor.dim32(0), 2);
  EXPECT_EQ(tensor.dim32(1), 3);
  EXPECT_EQ(tensor.dim32(2), 5);
  EXPECT_TRUE(tensor.mutable_data<TypeParam>() != nullptr);
  EXPECT_TRUE(tensor.data<TypeParam>() != nullptr);
  dims[0] = 7;
  dims[1] = 11;
  dims[2] = 13;
  dims.push_back(17);
  tensor.Reshape(dims);
  EXPECT_EQ(tensor.ndim(), 4);
  EXPECT_EQ(tensor.dim32(0), 7);
  EXPECT_EQ(tensor.dim32(1), 11);
  EXPECT_EQ(tensor.dim32(2), 13);
  EXPECT_EQ(tensor.dim32(3), 17);
  EXPECT_TRUE(tensor.mutable_data<TypeParam>() != nullptr);
  EXPECT_TRUE(tensor.data<TypeParam>() != nullptr);
}

TYPED_TEST(TensorCPUTest, TensorShareData) {
  vector<int> dims(3);
  dims[0] = 2;
  dims[1] = 3;
  dims[2] = 5;
  TensorCPU tensor(dims);
  TensorCPU other_tensor(dims);
  EXPECT_TRUE(tensor.mutable_data<TypeParam>() != nullptr);
  other_tensor.ShareData(tensor);
  EXPECT_TRUE(tensor.data<TypeParam>() != nullptr);
  EXPECT_TRUE(other_tensor.data<TypeParam>() != nullptr);
  EXPECT_EQ(tensor.data<TypeParam>(), other_tensor.data<TypeParam>());
  // Set one value, check the other
  for (int i = 0; i < tensor.size(); ++i) {
    tensor.mutable_data<TypeParam>()[i] = i;
    EXPECT_EQ(other_tensor.data<TypeParam>()[i], i);
  }
}


TYPED_TEST(TensorCPUTest, TensorShareDataRawPointer) {
  vector<int> dims(3);
  dims[0] = 2;
  dims[1] = 3;
  dims[2] = 5;
  std::unique_ptr<TypeParam[]> raw_buffer(new TypeParam[2*3*5]);
  TensorCPU tensor(dims);
  tensor.ShareExternalPointer(raw_buffer.get());
  EXPECT_EQ(tensor.mutable_data<TypeParam>(), raw_buffer.get());
  EXPECT_EQ(tensor.data<TypeParam>(), raw_buffer.get());
  // Set one value, check the other
  for (int i = 0; i < tensor.size(); ++i) {
    raw_buffer.get()[i] = i;
    EXPECT_EQ(tensor.data<TypeParam>()[i], i);
  }
}

TYPED_TEST(TensorCPUDeathTest, CannotShareDataWhenShapeNotSet) {
  std::unique_ptr<TypeParam[]> raw_buffer(new TypeParam[10]);
  TensorCPU tensor;
  EXPECT_DEATH(tensor.ShareExternalPointer(raw_buffer.get()), "");
}

TYPED_TEST(TensorCPUTest, TensorShareDataCanUseDifferentShapes) {
  vector<int> dims(3);
  dims[0] = 2;
  dims[1] = 3;
  dims[2] = 5;
  vector<int> alternate_dims(1);
  alternate_dims[0] = 2 * 3 * 5;
  TensorCPU tensor(dims);
  TensorCPU other_tensor(alternate_dims);
  EXPECT_TRUE(tensor.mutable_data<TypeParam>() != nullptr);
  other_tensor.ShareData(tensor);
  EXPECT_EQ(other_tensor.ndim(), 1);
  EXPECT_EQ(other_tensor.dim32(0), alternate_dims[0]);
  EXPECT_TRUE(tensor.data<TypeParam>() != nullptr);
  EXPECT_TRUE(other_tensor.data<TypeParam>() != nullptr);
  EXPECT_EQ(tensor.data<TypeParam>(), other_tensor.data<TypeParam>());
  // Set one value, check the other
  for (int i = 0; i < tensor.size(); ++i) {
    tensor.mutable_data<TypeParam>()[i] = i;
    EXPECT_EQ(other_tensor.data<TypeParam>()[i], i);
  }
}


TYPED_TEST(TensorCPUTest, NoLongerSharesAfterReshape) {
  vector<int> dims(3);
  dims[0] = 2;
  dims[1] = 3;
  dims[2] = 5;
  TensorCPU tensor(dims);
  TensorCPU other_tensor(dims);
  EXPECT_TRUE(tensor.mutable_data<TypeParam>() != nullptr);
  other_tensor.ShareData(tensor);
  EXPECT_EQ(tensor.data<TypeParam>(), other_tensor.data<TypeParam>());
  auto* old_pointer = other_tensor.data<TypeParam>();

  dims[0] = 7;
  tensor.Reshape(dims);
  EXPECT_EQ(old_pointer, other_tensor.data<TypeParam>());
  EXPECT_NE(old_pointer, tensor.mutable_data<TypeParam>());
}


TYPED_TEST(TensorCPUTest, NoKeepOnShrinkAsDefaultCase) {
  FLAGS_caffe2_keep_on_shrink = false;
  vector<int> dims{2, 3, 5};
  TensorCPU tensor(dims);
  TypeParam* ptr = tensor.mutable_data<TypeParam>();
  EXPECT_TRUE(ptr != nullptr);
  tensor.Reshape(vector<int>{3, 4, 6});
  TypeParam* larger_ptr = tensor.mutable_data<TypeParam>();
  EXPECT_TRUE(larger_ptr != nullptr);
  EXPECT_NE(ptr, larger_ptr);
  tensor.Reshape(vector<int>{1, 2, 4});
  TypeParam* smaller_ptr = tensor.mutable_data<TypeParam>();
  EXPECT_TRUE(smaller_ptr != nullptr);
  EXPECT_NE(larger_ptr, smaller_ptr);
}

TYPED_TEST(TensorCPUTest, KeepOnShrink) {
  FLAGS_caffe2_keep_on_shrink = true;
  vector<int> dims{2, 3, 5};
  TensorCPU tensor(dims);
  TypeParam* ptr = tensor.mutable_data<TypeParam>();
  EXPECT_TRUE(ptr != nullptr);
  // Expanding - will reallocate
  tensor.Reshape(vector<int>{3, 4, 6});
  TypeParam* larger_ptr = tensor.mutable_data<TypeParam>();
  EXPECT_TRUE(larger_ptr != nullptr);
  EXPECT_NE(ptr, larger_ptr);
  // Shrinking - will not reallocate
  tensor.Reshape(vector<int>{1, 2, 4});
  TypeParam* smaller_ptr = tensor.mutable_data<TypeParam>();
  EXPECT_TRUE(smaller_ptr != nullptr);
  EXPECT_EQ(larger_ptr, smaller_ptr);
  // Expanding but still under capacity - will not reallocate
  tensor.Reshape(vector<int>{2, 3, 5});
  TypeParam* new_ptr = tensor.mutable_data<TypeParam>();
  EXPECT_TRUE(new_ptr != nullptr);
  EXPECT_EQ(larger_ptr, new_ptr);
}

TYPED_TEST(TensorCPUDeathTest, CannotAccessDataWhenEmpty) {
  TensorCPU tensor;
  EXPECT_EQ(tensor.ndim(), 0);
  ASSERT_DEATH(tensor.data<TypeParam>(), "");
}


TEST(TensorTest, TensorNonFundamentalType) {
  TensorCPU tensor(vector<int>{2, 3, 4});
  EXPECT_TRUE(tensor.mutable_data<std::string>() != nullptr);
  const std::string* ptr = tensor.data<std::string>();
  for (int i = 0; i < tensor.size(); ++i) {
    EXPECT_TRUE(ptr[i] == "");
  }
}

TEST(TensorTest, TensorNonFundamentalTypeCopy) {
  TensorCPU tensor(vector<int>{2, 3, 4});
  std::string* ptr = tensor.mutable_data<std::string>();
  EXPECT_TRUE(ptr != nullptr);
  for (int i = 0; i < tensor.size(); ++i) {
    EXPECT_TRUE(ptr[i] == "");
    ptr[i] = "filled";
  }
  TensorCPU dst_tensor(tensor);
  const std::string* dst_ptr = dst_tensor.data<std::string>();
  for (int i = 0; i < dst_tensor.size(); ++i) {
    EXPECT_TRUE(dst_ptr[i] == "filled");
  }
}

TEST(TensorTest, Tensor64BitDimension) {
  // Initialize a large tensor.
  TIndex large_number =
      static_cast<int64_t>(std::numeric_limits<int>::max()) + 1;
  TensorCPU tensor(vector<TIndex>{large_number});
  EXPECT_EQ(tensor.ndim(), 1);
  EXPECT_EQ(tensor.dim(0), large_number);
  EXPECT_EQ(tensor.size(), large_number);
  EXPECT_TRUE(tensor.mutable_data<char>() != nullptr);
  EXPECT_EQ(tensor.nbytes(), large_number * sizeof(char));
  EXPECT_EQ(tensor.itemsize(), sizeof(char));
  // Try to go even larger, but this time we will not do mutable_data because we
  // do not have a large enough memory.
  tensor.Reshape(vector<TIndex>{large_number, 100});
  EXPECT_EQ(tensor.ndim(), 2);
  EXPECT_EQ(tensor.dim(0), large_number);
  EXPECT_EQ(tensor.dim(1), 100);
  EXPECT_EQ(tensor.size(), large_number * 100);
}

TEST(TensorDeathTest, CannotCastDownLargeDims) {
  TIndex large_number =
      static_cast<int64_t>(std::numeric_limits<int>::max()) + 1;
  TensorCPU tensor(vector<TIndex>{large_number});
  EXPECT_EQ(tensor.ndim(), 1);
  EXPECT_EQ(tensor.dim(0), large_number);
  ASSERT_DEATH(tensor.dim32(0), "");
}

TEST(TensorTest, TensorSerialization) {
  Blob blob;
  TensorCPU* tensor = blob.GetMutable<TensorCPU>();
  tensor->Reshape(2, 3);
  for (int i = 0; i < 6; ++i) {
    tensor->mutable_data<float>()[i] = i;
  }
  string serialized = blob.Serialize("test");
  TensorProto proto;
  CAFFE_CHECK(proto.ParseFromString(serialized));
  EXPECT_EQ(proto.name(), "test");
  EXPECT_EQ(proto.data_type(), TensorProto::FLOAT);
  EXPECT_EQ(proto.float_data_size(), 6);
  for (int i = 0; i < 6; ++i) {
    EXPECT_EQ(proto.float_data(i), i);
  }
}

}  // namespace
}  // namespace caffe2


