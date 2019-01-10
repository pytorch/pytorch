#include <gtest/gtest.h>

#include <ATen/ATen.h>

#include <cmath>

#define REQUIRE_TENSOR_OPTIONS(device_, index_, type_, layout_)            \
  ASSERT_TRUE(                                                             \
      tensor.device().type() == at::Device((device_), (index_)).type());   \
  ASSERT_TRUE(                                                             \
      tensor.device().index() == at::Device((device_), (index_)).index()); \
  ASSERT_EQ(tensor.dtype(), (type_));                                      \
  ASSERT_TRUE(tensor.layout() == (layout_))

TEST(TensorTest, AllocatesTensorOnTheCorrectDevice_MultiCUDA) {
  auto tensor = at::tensor({1, 2, 3}, at::device({at::kCUDA, 1}));
  ASSERT_EQ(tensor.device().type(), at::Device::Type::CUDA);
  ASSERT_EQ(tensor.device().index(), 1);
}

TEST(TensorTest, ToDevice_MultiCUDA) {
  auto tensor = at::empty({3, 4});
  REQUIRE_TENSOR_OPTIONS(at::kCPU, -1, at::kFloat, at::kStrided);

  tensor = tensor.to({at::kCUDA, 1});
  REQUIRE_TENSOR_OPTIONS(at::kCUDA, 1, at::kFloat, at::kStrided);

  tensor = tensor.to({at::kCUDA, 0});
  REQUIRE_TENSOR_OPTIONS(at::kCUDA, 0, at::kFloat, at::kStrided);

  tensor = tensor.to({at::kCUDA, 1});
  REQUIRE_TENSOR_OPTIONS(at::kCUDA, 1, at::kFloat, at::kStrided);

  tensor = tensor.to(at::Device(at::kCPU));
  REQUIRE_TENSOR_OPTIONS(at::kCPU, -1, at::kFloat, at::kStrided);

  tensor = tensor.to(at::kCUDA);
  REQUIRE_TENSOR_OPTIONS(at::kCUDA, 0, at::kFloat, at::kStrided);

  tensor = tensor.to(at::TensorOptions({at::kCUDA, 1}));
  REQUIRE_TENSOR_OPTIONS(at::kCUDA, 1, at::kFloat, at::kStrided);

  tensor = tensor.to(at::TensorOptions({at::kCUDA, 0}));
  REQUIRE_TENSOR_OPTIONS(at::kCUDA, 0, at::kFloat, at::kStrided);

  tensor = tensor.to(at::TensorOptions(at::kDouble));
  REQUIRE_TENSOR_OPTIONS(at::kCUDA, 0, at::kDouble, at::kStrided);

  tensor = tensor.to(at::TensorOptions({at::kCUDA, 1}));
  REQUIRE_TENSOR_OPTIONS(at::kCUDA, 1, at::kDouble, at::kStrided);

  tensor = tensor.to(at::TensorOptions(at::kInt));
  REQUIRE_TENSOR_OPTIONS(at::kCUDA, 1, at::kInt, at::kStrided);

  tensor = tensor.to(at::TensorOptions(at::Device(at::kCPU)));
  REQUIRE_TENSOR_OPTIONS(at::kCPU, -1, at::kInt, at::kStrided);

  tensor = tensor.to(at::TensorOptions(at::kCUDA));
  REQUIRE_TENSOR_OPTIONS(at::kCUDA, 0, at::kInt, at::kStrided);
}

TEST(TensorTest, ToTensorAndTensorAttributes_MultiCUDA) {
  auto tensor = at::empty({3, 4});
  REQUIRE_TENSOR_OPTIONS(at::kCPU, -1, at::kFloat, at::kStrided);

  auto other = at::empty({3, 4}, at::kFloat);
  tensor = tensor.to(other);
  REQUIRE_TENSOR_OPTIONS(at::kCPU, -1, at::kFloat, at::kStrided);

  other = at::empty({3, 4}, at::TensorOptions(at::kCUDA).dtype(at::kDouble));
  tensor = tensor.to(other.dtype());
  REQUIRE_TENSOR_OPTIONS(at::kCPU, -1, at::kDouble, at::kStrided);
  tensor = tensor.to(other.device());
  REQUIRE_TENSOR_OPTIONS(at::kCUDA, 0, at::kDouble, at::kStrided);

  other = at::empty({3, 4}, at::TensorOptions({at::kCUDA, 1}).dtype(at::kLong));
  tensor = tensor.to(other.device(), other.dtype());
  REQUIRE_TENSOR_OPTIONS(at::kCUDA, 1, at::kLong, at::kStrided);

  other = at::empty({3, 4}, at::kFloat);
  tensor = tensor.to(other.options());
  REQUIRE_TENSOR_OPTIONS(at::kCPU, -1, at::kFloat, at::kStrided);
}


TEST(TensorTest, ToDoesNotCopyWhenOptionsAreAllTheSame_CUDA) {
  auto tensor = at::empty({3, 4}, at::TensorOptions(at::kFloat).device(at::Device("cuda")));
  auto hopefully_not_copy = tensor.to(tensor.options());
  ASSERT_EQ(hopefully_not_copy.data<float>(), tensor.data<float>());
  hopefully_not_copy = tensor.to(at::kFloat);
  ASSERT_EQ(hopefully_not_copy.data<float>(), tensor.data<float>());
  hopefully_not_copy = tensor.to("cuda");
  ASSERT_EQ(hopefully_not_copy.data<float>(), tensor.data<float>());
  hopefully_not_copy = tensor.to(at::TensorOptions("cuda"));
  ASSERT_EQ(hopefully_not_copy.data<float>(), tensor.data<float>());
  hopefully_not_copy = tensor.to(at::TensorOptions(at::kFloat));
  ASSERT_EQ(hopefully_not_copy.data<float>(), tensor.data<float>());
}

TEST(TensorTest, ToDeviceAndDtype_MultiCUDA) {
  auto tensor = at::empty({3, 4});
  REQUIRE_TENSOR_OPTIONS(at::kCPU, -1, at::kFloat, at::kStrided);

  tensor = tensor.to({at::kCUDA, 1}, at::kInt);
  REQUIRE_TENSOR_OPTIONS(at::kCUDA, 1, at::kInt, at::kStrided);

  tensor = tensor.to(at::TensorOptions({at::kCUDA, 0}).dtype(at::kLong));
  REQUIRE_TENSOR_OPTIONS(at::kCUDA, 0, at::kLong, at::kStrided);

  tensor = tensor.to(at::TensorOptions({at::kCUDA, 1}).dtype(at::kDouble));
  REQUIRE_TENSOR_OPTIONS(at::kCUDA, 1, at::kDouble, at::kStrided);

  tensor = tensor.to(at::kCPU, at::kInt);
  REQUIRE_TENSOR_OPTIONS(at::kCPU, -1, at::kInt, at::kStrided);
}
