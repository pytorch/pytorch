#include <gtest/gtest.h>
#include <test/cpp/api/support.h>

#include <torch/types.h>

#include <ATen/ATen.h>

#include <cmath>
#include <cstddef>
#include <vector>

template <typename T>
bool exactly_equal(at::Tensor left, T right) {
  return left.item<T>() == right;
}

template <typename T>
bool almost_equal(at::Tensor left, T right, T tolerance = 1e-4) {
  return std::abs(left.item<T>() - right) < tolerance;
}

#define REQUIRE_TENSOR_OPTIONS(device_, index_, type_, layout_)            \
  ASSERT_TRUE(                                                             \
      tensor.device().type() == at::Device((device_), (index_)).type());   \
  ASSERT_TRUE(                                                             \
      tensor.device().index() == at::Device((device_), (index_)).index()); \
  ASSERT_EQ(tensor.dtype(), (type_));                                      \
  ASSERT_TRUE(tensor.layout() == (layout_))

TEST(TensorTest, ToDtype) {
  auto tensor = at::empty({3, 4});
  REQUIRE_TENSOR_OPTIONS(at::kCPU, -1, at::kFloat, at::kStrided);

  tensor = tensor.to(at::kInt);
  REQUIRE_TENSOR_OPTIONS(at::kCPU, -1, at::kInt, at::kStrided);

  tensor = tensor.to(at::kChar);
  REQUIRE_TENSOR_OPTIONS(at::kCPU, -1, at::kChar, at::kStrided);

  tensor = tensor.to(at::kDouble);
  REQUIRE_TENSOR_OPTIONS(at::kCPU, -1, at::kDouble, at::kStrided);

  tensor = tensor.to(at::TensorOptions(at::kInt));
  REQUIRE_TENSOR_OPTIONS(at::kCPU, -1, at::kInt, at::kStrided);

  tensor = tensor.to(at::TensorOptions(at::kChar));
  REQUIRE_TENSOR_OPTIONS(at::kCPU, -1, at::kChar, at::kStrided);

  tensor = tensor.to(at::TensorOptions(at::kDouble));
  REQUIRE_TENSOR_OPTIONS(at::kCPU, -1, at::kDouble, at::kStrided);
}

TEST(TensorTest, ToTensorAndTensorAttributes) {
  auto tensor = at::empty({3, 4});
  REQUIRE_TENSOR_OPTIONS(at::kCPU, -1, at::kFloat, at::kStrided);

  auto other = at::empty({3, 4}, at::kInt);
  tensor = tensor.to(other);
  REQUIRE_TENSOR_OPTIONS(at::kCPU, -1, at::kInt, at::kStrided);

  other = at::empty({3, 4}, at::kDouble);
  tensor = tensor.to(other.dtype());
  REQUIRE_TENSOR_OPTIONS(at::kCPU, -1, at::kDouble, at::kStrided);
  tensor = tensor.to(other.device());
  REQUIRE_TENSOR_OPTIONS(at::kCPU, -1, at::kDouble, at::kStrided);

  other = at::empty({3, 4}, at::kLong);
  tensor = tensor.to(other.device(), other.dtype());
  REQUIRE_TENSOR_OPTIONS(at::kCPU, -1, at::kLong, at::kStrided);

  other = at::empty({3, 4}, at::kInt);
  tensor = tensor.to(other.options());
  REQUIRE_TENSOR_OPTIONS(at::kCPU, -1, at::kInt, at::kStrided);
}

// Not currently supported.
// TEST(TensorTest, ToLayout) {
//   auto tensor = at::empty({3, 4});
//   REQUIRE_TENSOR_OPTIONS(at::kCPU, -1, at::kFloat, at::kStrided);
//
//   tensor = tensor.to(at::kSparse);
//   REQUIRE_TENSOR_OPTIONS(at::kCPU, -1, at::kFloat, at::kSparse);
//
//   tensor = tensor.to(at::kStrided);
//   REQUIRE_TENSOR_OPTIONS(at::kCPU, -1, at::kFloat, at::kStrided);
// }

TEST(TensorTest, ToOptionsWithRequiresGrad) {
  {
    // Respects requires_grad
    auto tensor = torch::empty({3, 4}, at::requires_grad());
    ASSERT_TRUE(tensor.requires_grad());

    tensor = tensor.to(at::kDouble);
    ASSERT_TRUE(tensor.requires_grad());

    // Throws if requires_grad is set in TensorOptions
    ASSERT_THROW(
        tensor.to(at::TensorOptions().requires_grad(true)), c10::Error);
    ASSERT_THROW(
        tensor.to(at::TensorOptions().requires_grad(false)), c10::Error);
  }
  {
    auto tensor = torch::empty({3, 4});
    ASSERT_FALSE(tensor.requires_grad());

    // Respects requires_grad
    tensor = tensor.to(at::kDouble);
    ASSERT_FALSE(tensor.requires_grad());

    // Throws if requires_grad is set in TensorOptions
    ASSERT_THROW(
        tensor.to(at::TensorOptions().requires_grad(true)), c10::Error);
    ASSERT_THROW(
        tensor.to(at::TensorOptions().requires_grad(false)), c10::Error);
  }
}

TEST(TensorTest, ToDoesNotCopyWhenOptionsAreAllTheSame) {
  {
    auto tensor = at::empty({3, 4}, at::kFloat);
    auto hopefully_not_copy = tensor.to(at::kFloat);
    ASSERT_EQ(hopefully_not_copy.data_ptr<float>(), tensor.data_ptr<float>());
  }
  {
    auto tensor = at::empty({3, 4}, at::kFloat);
    auto hopefully_not_copy = tensor.to(tensor.options());
    ASSERT_EQ(hopefully_not_copy.data_ptr<float>(), tensor.data_ptr<float>());
  }
  {
    auto tensor = at::empty({3, 4}, at::kFloat);
    auto hopefully_not_copy = tensor.to(tensor.dtype());
    ASSERT_EQ(hopefully_not_copy.data_ptr<float>(), tensor.data_ptr<float>());
  }
  {
    auto tensor = at::empty({3, 4}, at::kFloat);
    auto hopefully_not_copy = tensor.to(tensor.device());
    ASSERT_EQ(hopefully_not_copy.data_ptr<float>(), tensor.data_ptr<float>());
  }
  {
    auto tensor = at::empty({3, 4}, at::kFloat);
    auto hopefully_not_copy = tensor.to(tensor);
    ASSERT_EQ(hopefully_not_copy.data_ptr<float>(), tensor.data_ptr<float>());
  }
}

TEST(TensorTest, ContainsCorrectValueForSingleValue) {
  auto tensor = at::tensor(123);
  ASSERT_EQ(tensor.numel(), 1);
  ASSERT_EQ(tensor.dtype(), at::kInt);
  ASSERT_EQ(tensor[0].item<int32_t>(), 123);

  tensor = at::tensor(123.456f);
  ASSERT_EQ(tensor.numel(), 1);
  ASSERT_EQ(tensor.dtype(), at::kFloat);
  ASSERT_TRUE(almost_equal(tensor[0], 123.456f));

  tensor = at::tensor(123.456);
  ASSERT_EQ(tensor.numel(), 1);
  ASSERT_EQ(tensor.dtype(), at::kDouble);
  ASSERT_TRUE(almost_equal(tensor[0], 123.456));
}

TEST(TensorTest, ContainsCorrectValuesForManyValues) {
  auto tensor = at::tensor({1, 2, 3});
  ASSERT_EQ(tensor.numel(), 3);
  ASSERT_EQ(tensor.dtype(), at::kInt);
  ASSERT_TRUE(exactly_equal(tensor[0], 1));
  ASSERT_TRUE(exactly_equal(tensor[1], 2));
  ASSERT_TRUE(exactly_equal(tensor[2], 3));

  tensor = at::tensor({1.5, 2.25, 3.125});
  ASSERT_EQ(tensor.numel(), 3);
  ASSERT_EQ(tensor.dtype(), at::kDouble);
  ASSERT_TRUE(almost_equal(tensor[0], 1.5));
  ASSERT_TRUE(almost_equal(tensor[1], 2.25));
  ASSERT_TRUE(almost_equal(tensor[2], 3.125));
}

TEST(TensorTest, ContainsCorrectValuesForManyValuesVariable) {
  auto tensor = torch::tensor({1, 2, 3});
  ASSERT_TRUE(tensor.is_variable());
  ASSERT_EQ(tensor.numel(), 3);
  ASSERT_EQ(tensor.dtype(), at::kInt);
  ASSERT_TRUE(exactly_equal(tensor[0], 1));
  ASSERT_TRUE(exactly_equal(tensor[1], 2));
  ASSERT_TRUE(exactly_equal(tensor[2], 3));

  tensor = torch::tensor({1.5, 2.25, 3.125});
  ASSERT_TRUE(tensor.is_variable());
  ASSERT_EQ(tensor.numel(), 3);
  ASSERT_EQ(tensor.dtype(), at::kDouble);
  ASSERT_TRUE(almost_equal(tensor[0], 1.5));
  ASSERT_TRUE(almost_equal(tensor[1], 2.25));
  ASSERT_TRUE(almost_equal(tensor[2], 3.125));
}

TEST(TensorTest, ContainsCorrectValuesWhenConstructedFromVector) {
  std::vector<int> v = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  auto tensor = at::tensor(v);
  ASSERT_EQ(tensor.numel(), v.size());
  ASSERT_EQ(tensor.dtype(), at::kInt);
  for (size_t i = 0; i < v.size(); ++i) {
    ASSERT_TRUE(exactly_equal(tensor[i], v.at(i)));
  }

  std::vector<double> w = {1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.0};
  tensor = at::tensor(w);
  ASSERT_EQ(tensor.numel(), w.size());
  ASSERT_EQ(tensor.dtype(), at::kDouble);
  for (size_t i = 0; i < w.size(); ++i) {
    ASSERT_TRUE(almost_equal(tensor[i], w.at(i)));
  }
}

TEST(TensorTest, UsesOptionsThatAreSupplied) {
  auto tensor = at::tensor(123, at::dtype(at::kFloat)) + 0.5;
  ASSERT_EQ(tensor.numel(), 1);
  ASSERT_EQ(tensor.dtype(), at::kFloat);
  ASSERT_TRUE(almost_equal(tensor[0], 123.5));

  tensor = at::tensor({1.1, 2.2, 3.3}, at::dtype(at::kInt));
  ASSERT_EQ(tensor.numel(), 3);
  ASSERT_EQ(tensor.dtype(), at::kInt);
  ASSERT_EQ(tensor.layout(), at::kStrided);
  ASSERT_TRUE(exactly_equal(tensor[0], 1));
  ASSERT_TRUE(exactly_equal(tensor[1], 2));
  ASSERT_TRUE(exactly_equal(tensor[2], 3));
}

TEST(TensorTest, FromBlob) {
  std::vector<double> v = {1.0, 2.0, 3.0};
  auto tensor = torch::from_blob(
      v.data(), v.size(), torch::dtype(torch::kFloat64).requires_grad(true));
  ASSERT_TRUE(tensor.is_variable());
  ASSERT_TRUE(tensor.requires_grad());
  ASSERT_EQ(tensor.dtype(), torch::kFloat64);
  ASSERT_EQ(tensor.numel(), 3);
  ASSERT_EQ(tensor[0].item<double>(), 1);
  ASSERT_EQ(tensor[1].item<double>(), 2);
  ASSERT_EQ(tensor[2].item<double>(), 3);
}

TEST(TensorTest, FromBlobUsesDeleter) {
  bool called = false;
  {
    std::vector<int32_t> v = {1, 2, 3};
    auto tensor = torch::from_blob(
        v.data(),
        v.size(),
        /*deleter=*/[&called](void* data) { called = true; },
        torch::kInt32);
  }
  ASSERT_TRUE(called);
}

TEST(TensorTest, FromBlobWithStrides) {
  // clang-format off
  std::vector<int32_t> v = {
    1, 2, 3,
    4, 5, 6,
    7, 8, 9
  };
  // clang-format on
  auto tensor = torch::from_blob(
      v.data(),
      /*sizes=*/{3, 3},
      /*strides=*/{1, 3},
      torch::kInt32);
  ASSERT_TRUE(tensor.is_variable());
  ASSERT_EQ(tensor.dtype(), torch::kInt32);
  ASSERT_EQ(tensor.numel(), 9);
  const std::vector<int64_t> expected_strides = {1, 3};
  ASSERT_EQ(tensor.strides(), expected_strides);
  for (int64_t i = 0; i < tensor.size(0); ++i) {
    for (int64_t j = 0; j < tensor.size(1); ++j) {
      // NOTE: This is column major because the strides are swapped.
      EXPECT_EQ(tensor[i][j].item<int32_t>(), 1 + (j * tensor.size(1)) + i);
    }
  }
}

TEST(TensorTest, Item) {
  {
    torch::Tensor tensor = torch::tensor(3.14);
    torch::Scalar scalar = tensor.item();
    ASSERT_NEAR(scalar.to<float>(), 3.14, 1e-5);
  }
  {
    torch::Tensor tensor = torch::tensor(123);
    torch::Scalar scalar = tensor.item();
    ASSERT_EQ(scalar.to<int>(), 123);
  }
}

TEST(TensorTest, Item_CUDA) {
  {
    torch::Tensor tensor = torch::tensor(3.14, torch::kCUDA);
    torch::Scalar scalar = tensor.item();
    ASSERT_NEAR(scalar.to<float>(), 3.14, 1e-5);
  }
  {
    torch::Tensor tensor = torch::tensor(123, torch::kCUDA);
    torch::Scalar scalar = tensor.item();
    ASSERT_EQ(scalar.to<int>(), 123);
  }
}

TEST(TensorTest, DataPtr) {
  auto tensor = at::empty({3, 4}, at::kFloat);
  auto tensor_not_copy = tensor.to(tensor.options());
  ASSERT_EQ(tensor_not_copy.data_ptr<float>(), tensor.data_ptr<float>());
  ASSERT_EQ(tensor_not_copy.data_ptr(), tensor.data_ptr());
}

TEST(TensorTest, Data) {
  const auto tensor = torch::empty({3, 3});
  ASSERT_TRUE(torch::equal(tensor, tensor.data()));

  const auto tensor2 = at::empty({3, 3});
  ASSERT_THROW(tensor2.data(), c10::Error);
}

TEST(TensorTest, BackwardAndGrad) {
  auto x = torch::tensor({5}, at::TensorOptions().requires_grad(true));
  auto y = x * x;
  y.backward();
  ASSERT_EQ(x.grad().item<float>(), 10.0);

  x = at::tensor({5});
  y = x * x;
  ASSERT_THROWS_WITH(y.backward(), "backward is not implemented for Tensor");
  ASSERT_THROWS_WITH(x.grad(), "grad is not implemented for Tensor");
}

TEST(TensorTest, BackwardCreatesOnesGrad) {
  const auto x = torch::tensor({5}, at::TensorOptions().requires_grad(true));
  x.backward();
  ASSERT_TRUE(torch::equal(x.grad(),
              torch::ones_like(x)));
}

TEST(TensorTest, IsLeaf) {
  auto x = torch::tensor({5}, at::TensorOptions().requires_grad(true));
  auto y = x * x;
  ASSERT_TRUE(x.is_leaf());
  ASSERT_FALSE(y.is_leaf());

  x = at::tensor({5});
  y = x * x;
  const auto message = "is_leaf is not implemented for Tensor";
  ASSERT_THROWS_WITH(y.is_leaf(), message);
  ASSERT_THROWS_WITH(x.is_leaf(), message);
}

TEST(TensorTest, OutputNr) {
  auto x = torch::tensor({5}, at::TensorOptions().requires_grad(true));
  auto y = x * x;
  ASSERT_EQ(x.output_nr(), 0);
  ASSERT_EQ(y.output_nr(), 0);

  x = at::tensor({5});
  y = x * x;
  const auto message = "output_nr is not implemented for Tensor";
  ASSERT_THROWS_WITH(y.output_nr(), message);
  ASSERT_THROWS_WITH(x.output_nr(), message);
}
