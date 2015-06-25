#include <iostream>  // NOLINT

#include "caffe2/core/blob.h"
#include "caffe2/core/common_gpu.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/proto/caffe2.pb.h"
#include "gtest/gtest.h"

namespace caffe2 {

template <typename dtype> class TensorGPUTest : public ::testing::Test {};
template <typename dtype> class TensorGPUDeathTest : public ::testing::Test {};
typedef ::testing::Types<char, int, float> TensorTypes;
TYPED_TEST_CASE(TensorGPUTest, TensorTypes);
TYPED_TEST_CASE(TensorGPUDeathTest, TensorTypes);

TYPED_TEST(TensorGPUTest, TensorInitializedEmpty) {
  Tensor<TypeParam, CUDAContext> tensor;
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
  EXPECT_TRUE(tensor.mutable_data() != nullptr);
  EXPECT_TRUE(tensor.data() != nullptr);
}

TYPED_TEST(TensorGPUTest, TensorInitializedNonEmpty) {
  vector<int> dims(3);
  dims[0] = 2;
  dims[1] = 3;
  dims[2] = 5;
  Tensor<TypeParam, CUDAContext> tensor(dims);
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

TYPED_TEST(TensorGPUTest, TensorShareData) {
  vector<int> dims(3);
  dims[0] = 2;
  dims[1] = 3;
  dims[2] = 5;
  Tensor<TypeParam, CUDAContext> tensor(dims);
  Tensor<TypeParam, CUDAContext> other_tensor(dims);
  other_tensor.ShareData(tensor);
  EXPECT_TRUE(tensor.mutable_data() != nullptr);
  EXPECT_TRUE(tensor.data() != nullptr);
  EXPECT_TRUE(other_tensor.data() != nullptr);
  EXPECT_EQ(tensor.data(), other_tensor.data());
}

TYPED_TEST(TensorGPUDeathTest, ShareDataCannotInitializeDataFromSharedTensor) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  vector<int> dims(3);
  dims[0] = 2;
  dims[1] = 3;
  dims[2] = 5;
  Tensor<TypeParam, CUDAContext> tensor(dims);
  Tensor<TypeParam, CUDAContext> other_tensor(dims);
  other_tensor.ShareData(tensor);
  ASSERT_DEATH(other_tensor.mutable_data(), "");
}

TYPED_TEST(TensorGPUDeathTest, CannotDoReshapewithAlias) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  vector<int> dims(3);
  dims[0] = 2;
  dims[1] = 3;
  dims[2] = 5;
  Tensor<TypeParam, CUDAContext> tensor(dims);
  Tensor<TypeParam, CUDAContext> other_tensor(dims);
  other_tensor.ShareData(tensor);
  dims[0] = 7;
  tensor.Reshape(dims);
  EXPECT_TRUE(tensor.mutable_data() != nullptr);
  ASSERT_DEATH(other_tensor.data(), "Source data size has changed.");
}

TYPED_TEST(TensorGPUDeathTest, CannotAccessDataWhenEmpty) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  Tensor<TypeParam, CUDAContext> tensor;
  EXPECT_EQ(tensor.ndim(), 0);
  ASSERT_DEATH(tensor.data(), "Check failed: 'data_' Must be non NULL");
}

}  // namespace caffe2


