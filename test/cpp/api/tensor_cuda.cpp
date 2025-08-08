#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <torch/torch.h>

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
  auto tensor = at::empty(
      {3, 4}, at::TensorOptions(at::kFloat).device(at::Device("cuda")));
  auto hopefully_not_copy = tensor.to(tensor.options());
  ASSERT_EQ(hopefully_not_copy.data_ptr<float>(), tensor.data_ptr<float>());
  hopefully_not_copy = tensor.to(at::kFloat);
  ASSERT_EQ(hopefully_not_copy.data_ptr<float>(), tensor.data_ptr<float>());
  hopefully_not_copy = tensor.to("cuda");
  ASSERT_EQ(hopefully_not_copy.data_ptr<float>(), tensor.data_ptr<float>());
  hopefully_not_copy = tensor.to(at::TensorOptions("cuda"));
  ASSERT_EQ(hopefully_not_copy.data_ptr<float>(), tensor.data_ptr<float>());
  hopefully_not_copy = tensor.to(at::TensorOptions(at::kFloat));
  ASSERT_EQ(hopefully_not_copy.data_ptr<float>(), tensor.data_ptr<float>());
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

TEST(TensorTest, MagmaInitializesCorrectly_CUDA) {
  // Any tensor will work here as long as it's invertible
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
  float data[] = {1, 1, 1, 0, 0, 3, 1, 2, 2, 3, 1, 0, 1, 0, 2, 1};
  auto tensor =
      at::from_blob(data, {4, 4}, at::TensorOptions(at::kFloat)).cuda();
  if (at::hasMAGMA()) {
    at::inverse(tensor);
  }
}

#ifdef USE_CUDA
#include <ATen/cuda/CUDAConfig.h>
#if AT_CUDNN_ENABLED()
TEST(CuDNNBatchNormTest, OutVariantMatchesFunctional) {
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA is not available";
  }
  if (!at::Context::hasCuDNN()) {
    GTEST_SKIP() << "cuDNN is not available";
  }

  auto device = torch::device(torch::kCUDA);

  auto input = torch::rand({2, 3, 4, 4}, device);
  auto weight = torch::randn({3}, device);
  auto bias = torch::randn({3}, device);
  auto running_mean = torch::zeros({3}, device);
  auto running_var = torch::ones({3}, device);

  bool training = true;
  double exponential_average_factor = 0.1;
  double epsilon = 1e-5;

  auto output = torch::empty_like(input);
  auto save_mean = torch::empty({3}, device);
  auto save_var = torch::empty({3}, device);
  auto reserve = torch::empty({0}, device.dtype(torch::kByte));

  at::native::cudnn_batch_norm_out(
      input,
      weight,
      bias,
      running_mean,
      running_var,
      training,
      exponential_average_factor,
      epsilon,
      output,
      save_mean,
      save_var,
      reserve);

  auto ref_outputs = at::native::cudnn_batch_norm(
      input,
      weight,
      bias,
      running_mean,
      running_var,
      training,
      exponential_average_factor,
      epsilon);

  ASSERT_TRUE(torch::allclose(output, std::get<0>(ref_outputs)));
  ASSERT_TRUE(torch::allclose(save_mean, std::get<1>(ref_outputs)));
  ASSERT_TRUE(torch::allclose(save_var, std::get<2>(ref_outputs)));
  ASSERT_TRUE(torch::equal(reserve, std::get<3>(ref_outputs)));
}
#endif // AT_CUDNN_ENABLED()
#endif // USE_CUDA
