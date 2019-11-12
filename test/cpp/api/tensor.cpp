#include <gtest/gtest.h>
#include <test/cpp/api/support.h>

#include <torch/torch.h>

#include <cmath>
#include <cstddef>
#include <vector>

#include <test/cpp/common/support.h>

using namespace torch::test;

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

TEST(TensorTest, AtTensorCtorScalar) {
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

  tensor = at::tensor(123, at::dtype(at::kFloat)) + 0.5;
  ASSERT_EQ(tensor.numel(), 1);
  ASSERT_EQ(tensor.dtype(), at::kFloat);
  ASSERT_TRUE(almost_equal(tensor[0], 123.5));
}

TEST(TensorTest, AtTensorCtorSingleDim) {
  auto tensor = at::tensor({1, 2, 3});
  ASSERT_EQ(tensor.numel(), 3);
  ASSERT_EQ(tensor.dtype(), at::kInt);
  ASSERT_TRUE(exactly_equal(tensor[0], 1));
  ASSERT_TRUE(exactly_equal(tensor[1], 2));
  ASSERT_TRUE(exactly_equal(tensor[2], 3));

  tensor = at::tensor(std::vector<int>({1, 2, 3}));
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

  tensor = at::tensor({1.1, 2.2, 3.3}, at::dtype(at::kInt));
  ASSERT_EQ(tensor.numel(), 3);
  ASSERT_EQ(tensor.dtype(), at::kInt);
  ASSERT_EQ(tensor.layout(), at::kStrided);
  ASSERT_TRUE(exactly_equal(tensor[0], 1));
  ASSERT_TRUE(exactly_equal(tensor[1], 2));
  ASSERT_TRUE(exactly_equal(tensor[2], 3));

  tensor = at::tensor(std::vector<double>({1.5, 2.25, 3.125}));
  ASSERT_EQ(tensor.numel(), 3);
  ASSERT_EQ(tensor.dtype(), at::kDouble);
  ASSERT_TRUE(almost_equal(tensor[0], 1.5));
  ASSERT_TRUE(almost_equal(tensor[1], 2.25));
  ASSERT_TRUE(almost_equal(tensor[2], 3.125));

  std::vector<int> v = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  tensor = at::tensor(v);
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

TEST(TensorTest, TorchTensorCtorScalarIntegralType) {
  auto tensor = torch::tensor(123);
  ASSERT_EQ(tensor.numel(), 1);
  ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({}));
  ASSERT_EQ(tensor.dtype(), at::kLong);
  ASSERT_EQ(tensor.item<int32_t>(), 123);
}

void test_TorchTensorCtorScalarFloatingType_expected_dtype(c10::ScalarType default_dtype) {
  AutoDefaultDtypeMode dtype_mode(default_dtype);

  auto tensor = torch::tensor(123.456f);
  ASSERT_EQ(tensor.numel(), 1);
  ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({}));
  ASSERT_EQ(tensor.dtype(), at::kFloat);
  ASSERT_TRUE(almost_equal(tensor, 123.456f));

  tensor = torch::tensor(123.456);
  ASSERT_EQ(tensor.numel(), 1);
  ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({}));
  ASSERT_EQ(tensor.dtype(), default_dtype);
  ASSERT_TRUE(almost_equal(tensor, 123.456));

  tensor = torch::tensor({123.456});
  ASSERT_EQ(tensor.numel(), 1);
  ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({1}));
  ASSERT_EQ(tensor.dtype(), default_dtype);
  ASSERT_TRUE(almost_equal(tensor[0], 123.456));
}

TEST(TensorTest, TorchTensorCtorScalarFloatingType) {
  test_TorchTensorCtorScalarFloatingType_expected_dtype(/*default_dtype=*/torch::kFloat);
  test_TorchTensorCtorScalarFloatingType_expected_dtype(/*default_dtype=*/torch::kDouble);
}

TEST(TensorTest, TorchTensorCtorScalarBoolType) {
  auto tensor = torch::tensor(true);
  ASSERT_EQ(tensor.numel(), 1);
  ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({}));
  ASSERT_EQ(tensor.dtype(), at::kBool);
  ASSERT_TRUE(exactly_equal(tensor, true));

  tensor = torch::tensor({true});
  ASSERT_EQ(tensor.numel(), 1);
  ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({1}));
  ASSERT_EQ(tensor.dtype(), at::kBool);
  ASSERT_TRUE(exactly_equal(tensor[0], true));
}

TEST(TensorTest, TorchTensorCtorSingleDimIntegralType) {
  auto tensor = torch::tensor({1, 2, 3});
  ASSERT_TRUE(tensor.is_variable());
  ASSERT_EQ(tensor.numel(), 3);
  ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({3}));
  ASSERT_EQ(tensor.dtype(), at::kLong);
  ASSERT_TRUE(exactly_equal(tensor[0], 1));
  ASSERT_TRUE(exactly_equal(tensor[1], 2));
  ASSERT_TRUE(exactly_equal(tensor[2], 3));

  tensor = torch::tensor(at::ArrayRef<int>({1, 2, 3}));
  ASSERT_TRUE(tensor.is_variable());
  ASSERT_EQ(tensor.numel(), 3);
  ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({3}));
  ASSERT_EQ(tensor.dtype(), at::kInt);
  ASSERT_TRUE(exactly_equal(tensor[0], 1));
  ASSERT_TRUE(exactly_equal(tensor[1], 2));
  ASSERT_TRUE(exactly_equal(tensor[2], 3));

  tensor = torch::tensor(std::vector<int>({1, 2, 3}));
  ASSERT_TRUE(tensor.is_variable());
  ASSERT_EQ(tensor.numel(), 3);
  ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({3}));
  ASSERT_EQ(tensor.dtype(), at::kInt);
  ASSERT_TRUE(exactly_equal(tensor[0], 1));
  ASSERT_TRUE(exactly_equal(tensor[1], 2));
  ASSERT_TRUE(exactly_equal(tensor[2], 3));

  tensor = torch::tensor(at::ArrayRef<int64_t>({1, 2, 3}));
  ASSERT_TRUE(tensor.is_variable());
  ASSERT_EQ(tensor.numel(), 3);
  ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({3}));
  ASSERT_EQ(tensor.dtype(), at::kLong);
  ASSERT_TRUE(exactly_equal(tensor[0], 1));
  ASSERT_TRUE(exactly_equal(tensor[1], 2));
  ASSERT_TRUE(exactly_equal(tensor[2], 3));

  tensor = torch::tensor(std::vector<int64_t>({1, 2, 3}));
  ASSERT_TRUE(tensor.is_variable());
  ASSERT_EQ(tensor.numel(), 3);
  ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({3}));
  ASSERT_EQ(tensor.dtype(), at::kLong);
  ASSERT_TRUE(exactly_equal(tensor[0], 1));
  ASSERT_TRUE(exactly_equal(tensor[1], 2));
  ASSERT_TRUE(exactly_equal(tensor[2], 3));
}

void test_TorchTensorCtorSingleDimFloatingType_expected_dtype(c10::ScalarType default_dtype) {
  AutoDefaultDtypeMode dtype_mode(default_dtype);

  auto tensor = torch::tensor({1.5, 2.25, 3.125});
  ASSERT_TRUE(tensor.is_variable());
  ASSERT_EQ(tensor.numel(), 3);
  ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({3}));
  ASSERT_EQ(tensor.dtype(), default_dtype);
  ASSERT_TRUE(almost_equal(tensor[0], 1.5));
  ASSERT_TRUE(almost_equal(tensor[1], 2.25));
  ASSERT_TRUE(almost_equal(tensor[2], 3.125));

  tensor = torch::tensor({1.5f, 2.25f, 3.125f});
  ASSERT_TRUE(tensor.is_variable());
  ASSERT_EQ(tensor.numel(), 3);
  ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({3}));
  ASSERT_EQ(tensor.dtype(), at::kFloat);
  ASSERT_TRUE(almost_equal(tensor[0], 1.5f));
  ASSERT_TRUE(almost_equal(tensor[1], 2.25f));
  ASSERT_TRUE(almost_equal(tensor[2], 3.125f));

  tensor = torch::tensor(at::ArrayRef<float>({1.5, 2.25, 3.125}));
  ASSERT_TRUE(tensor.is_variable());
  ASSERT_EQ(tensor.numel(), 3);
  ASSERT_EQ(tensor.dtype(), at::kFloat);
  ASSERT_TRUE(almost_equal(tensor[0], 1.5));
  ASSERT_TRUE(almost_equal(tensor[1], 2.25));
  ASSERT_TRUE(almost_equal(tensor[2], 3.125));

  tensor = torch::tensor(std::vector<float>({1.5, 2.25, 3.125}));
  ASSERT_TRUE(tensor.is_variable());
  ASSERT_EQ(tensor.numel(), 3);
  ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({3}));
  ASSERT_EQ(tensor.dtype(), at::kFloat);
  ASSERT_TRUE(almost_equal(tensor[0], 1.5));
  ASSERT_TRUE(almost_equal(tensor[1], 2.25));
  ASSERT_TRUE(almost_equal(tensor[2], 3.125));

  tensor = torch::tensor(at::ArrayRef<double>({1.5, 2.25, 3.125}));
  ASSERT_TRUE(tensor.is_variable());
  ASSERT_EQ(tensor.numel(), 3);
  ASSERT_EQ(tensor.dtype(), at::kDouble);
  ASSERT_TRUE(almost_equal(tensor[0], 1.5));
  ASSERT_TRUE(almost_equal(tensor[1], 2.25));
  ASSERT_TRUE(almost_equal(tensor[2], 3.125));

  tensor = torch::tensor(std::vector<double>({1.5, 2.25, 3.125}));
  ASSERT_TRUE(tensor.is_variable());
  ASSERT_EQ(tensor.numel(), 3);
  ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({3}));
  ASSERT_EQ(tensor.dtype(), at::kDouble);
  ASSERT_TRUE(almost_equal(tensor[0], 1.5));
  ASSERT_TRUE(almost_equal(tensor[1], 2.25));
  ASSERT_TRUE(almost_equal(tensor[2], 3.125));
}

TEST(TensorTest, TorchTensorCtorSingleDimFloatingType) {
  test_TorchTensorCtorSingleDimFloatingType_expected_dtype(/*default_dtype=*/torch::kFloat);
  test_TorchTensorCtorSingleDimFloatingType_expected_dtype(/*default_dtype=*/torch::kDouble);
}

TEST(TensorTest, TorchTensorCtorSingleDimBoolType) {
  auto tensor = torch::tensor({true, false, true});
  ASSERT_TRUE(tensor.is_variable());
  ASSERT_EQ(tensor.numel(), 3);
  ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({3}));
  ASSERT_EQ(tensor.dtype(), at::kBool);
  ASSERT_TRUE(exactly_equal(tensor[0], true));
  ASSERT_TRUE(exactly_equal(tensor[1], false));
  ASSERT_TRUE(exactly_equal(tensor[2], true));

  tensor = torch::tensor(at::ArrayRef<bool>({true, false, true}));
  ASSERT_TRUE(tensor.is_variable());
  ASSERT_EQ(tensor.numel(), 3);
  ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({3}));
  ASSERT_EQ(tensor.dtype(), at::kBool);
  ASSERT_TRUE(exactly_equal(tensor[0], true));
  ASSERT_TRUE(exactly_equal(tensor[1], false));
  ASSERT_TRUE(exactly_equal(tensor[2], true));
}

TEST(TensorTest, TorchTensorCtorMultiDimIntegralType) {
  {
    auto tensor = torch::tensor({{1, 2}});
    ASSERT_EQ(tensor.dtype(), torch::kLong);
    ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({1, 2}));
    ASSERT_TRUE(torch::allclose(tensor, torch::arange(1, 3, torch::kLong).view(tensor.sizes())));
    ASSERT_FALSE(tensor.requires_grad());
  }
  {
    auto tensor = torch::tensor({{1}, {2}});
    ASSERT_EQ(tensor.dtype(), torch::kLong);
    ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({2, 1}));
    ASSERT_TRUE(torch::allclose(tensor, torch::arange(1, 3, torch::kLong).view(tensor.sizes())));
    ASSERT_FALSE(tensor.requires_grad());
  }
  {
    auto tensor = torch::tensor({{{1, 2}}});
    ASSERT_EQ(tensor.dtype(), torch::kLong);
    ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({1, 1, 2}));
    ASSERT_TRUE(torch::allclose(tensor, torch::arange(1, 3, torch::kLong).view(tensor.sizes())));
    ASSERT_FALSE(tensor.requires_grad());
  }
  {
    auto tensor = torch::tensor({{{1}, {2}}});
    ASSERT_EQ(tensor.dtype(), torch::kLong);
    ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({1, 2, 1}));
    ASSERT_TRUE(torch::allclose(tensor, torch::arange(1, 3, torch::kLong).view(tensor.sizes())));
    ASSERT_FALSE(tensor.requires_grad());
  }
  {
    auto tensor = torch::tensor({{1, 2}, {3, 4}});
    ASSERT_EQ(tensor.dtype(), torch::kLong);
    ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({2, 2}));
    ASSERT_TRUE(torch::allclose(tensor, torch::arange(1, 5, torch::kLong).view(tensor.sizes())));
    ASSERT_FALSE(tensor.requires_grad());
  }
  {
    auto tensor = torch::tensor({{{{{{{{{{1}}}}}}}}}});
    ASSERT_EQ(tensor.dtype(), torch::kLong);
    ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({1, 1, 1, 1, 1, 1, 1, 1, 1, 1}));
    ASSERT_TRUE(torch::allclose(tensor, torch::full({1}, 1, torch::kLong).view(tensor.sizes())));
    ASSERT_FALSE(tensor.requires_grad());
  }
  {
    auto tensor = torch::tensor({{{{{{{{{{1, 2}}}}}}}}}});
    ASSERT_EQ(tensor.dtype(), torch::kLong);
    ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({1, 1, 1, 1, 1, 1, 1, 1, 1, 2}));
    ASSERT_TRUE(torch::allclose(tensor, torch::arange(1, 3, torch::kLong).view(tensor.sizes())));
    ASSERT_FALSE(tensor.requires_grad());
  }
}

void test_TorchTensorCtorMultiDimFloatingType_expected_dtype(c10::ScalarType default_dtype) {
  AutoDefaultDtypeMode dtype_mode(default_dtype);
  {
    auto tensor = torch::tensor({{1.0, 2.0}});
    ASSERT_EQ(tensor.dtype(), default_dtype);
    ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({1, 2}));
    ASSERT_TRUE(torch::allclose(tensor, torch::arange(1, 3, default_dtype).view(tensor.sizes())));
    ASSERT_FALSE(tensor.requires_grad());
  }
  {
    auto tensor = torch::tensor({{{{{{{{1.0, 2.0, 3.0}}}}}, {{{{{4.0, 5.0, 6.0}}}}}, {{{{{7.0, 8.0, 9.0}}}}}}}});
    ASSERT_EQ(tensor.dtype(), default_dtype);
    ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({1, 1, 3, 1, 1, 1, 1, 3}));
    ASSERT_TRUE(torch::allclose(tensor, torch::arange(1, 10, default_dtype).view(tensor.sizes())));
    ASSERT_FALSE(tensor.requires_grad());
  }
}

TEST(TensorTest, TorchTensorCtorMultiDimFloatingType) {
  test_TorchTensorCtorMultiDimFloatingType_expected_dtype(/*default_dtype=*/torch::kFloat);
  test_TorchTensorCtorMultiDimFloatingType_expected_dtype(/*default_dtype=*/torch::kDouble);
}

TEST(TensorTest, TorchTensorCtorMultiDimBoolType) {
  {
    auto tensor = torch::tensor({{true, false}});
    ASSERT_EQ(tensor.dtype(), torch::kBool);
    ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({1, 2}));
    auto expected = torch::empty(tensor.sizes(), torch::kBool);
    expected[0][0] = true;
    expected[0][1] = false;
    ASSERT_TRUE(torch::equal(tensor, expected));
    ASSERT_FALSE(tensor.requires_grad());
  }
  {
    auto tensor = torch::tensor({{true}, {false}});
    ASSERT_EQ(tensor.dtype(), torch::kBool);
    ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({2, 1}));
    auto expected = torch::empty(tensor.sizes(), torch::kBool);
    expected[0][0] = true;
    expected[1][0] = false;
    ASSERT_TRUE(torch::equal(tensor, expected));
    ASSERT_FALSE(tensor.requires_grad());
  }
}

TEST(TensorTest, TorchTensorCtorMultiDimWithOptions) {
  {
    auto tensor = torch::tensor({{1, 2}}, torch::dtype(torch::kInt));
    ASSERT_EQ(tensor.dtype(), torch::kInt);
    ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({1, 2}));
    ASSERT_TRUE(torch::allclose(tensor, torch::arange(1, 3, torch::kInt).view(tensor.sizes())));
    ASSERT_FALSE(tensor.requires_grad());
  }
  {
    auto tensor = torch::tensor({{1, 2}, {3, 4}}, torch::dtype(torch::kFloat).requires_grad(true));
    ASSERT_EQ(tensor.dtype(), torch::kFloat);
    ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({2, 2}));
    ASSERT_TRUE(torch::allclose(tensor, torch::arange(1, 5, torch::kFloat).view(tensor.sizes())));
    ASSERT_TRUE(tensor.requires_grad());
  }
}

TEST(TensorTest, TorchTensorCtorMultiDimErrorChecks) {
  {
    ASSERT_THROWS_WITH(torch::tensor({{{2, 3, 4}, {{5, 6}, {7}}}}),
      "Expected all sub-lists to have sizes: 2 (e.g. {5, 6}), but got sub-list {7} with sizes: 1");
  }
  {
    ASSERT_THROWS_WITH(torch::tensor({{{1, 2.0}, {1, 2.0}}}),
      "Expected all elements of the tensor to have the same scalar type: Long, but got element of scalar type: Float");
  }
  {
    ASSERT_THROWS_WITH(torch::tensor({{{true, 2.0, 3}, {true, 2.0, 3}}}),
      "Expected all elements of the tensor to have the same scalar type: Bool, but got element of scalar type: Float");
  }
  {
    ASSERT_THROWS_WITH(torch::tensor({{{true}, {2}}}),
      "Expected all elements of the tensor to have the same scalar type: Bool, but got element of scalar type: Long");
  }
  {
    ASSERT_THROWS_WITH(torch::tensor({{{true, 2}}}),
      "Expected all elements of the tensor to have the same scalar type: Bool, but got element of scalar type: Long");
  }
}

void test_TorchTensorCtorMultiDim_CUDA_expected_dtype(c10::ScalarType default_dtype) {
  AutoDefaultDtypeMode dtype_mode(default_dtype);

  auto tensor = torch::tensor(
    {{{{{{{{1.0, 2.0, 3.0}}}}}, {{{{{4.0, 5.0, 6.0}}}}}, {{{{{7.0, 8.0, 9.0}}}}}}}},
    torch::dtype(default_dtype).device(torch::kCUDA));
  ASSERT_TRUE(tensor.device().is_cuda());
  ASSERT_EQ(tensor.dtype(), default_dtype);
  ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({1, 1, 3, 1, 1, 1, 1, 3}));
  ASSERT_TRUE(torch::allclose(
    tensor,
    torch::arange(1, 10, default_dtype).view(tensor.sizes()).to(torch::kCUDA)));
  ASSERT_FALSE(tensor.requires_grad());
}

TEST(TensorTest, TorchTensorCtorMultiDim_CUDA) {
  test_TorchTensorCtorMultiDim_CUDA_expected_dtype(/*default_dtype=*/torch::kFloat);
  test_TorchTensorCtorMultiDim_CUDA_expected_dtype(/*default_dtype=*/torch::kDouble);
}

void test_TorchTensorCtorZeroSizedDim_expected_dtype(c10::ScalarType default_dtype) {
  AutoDefaultDtypeMode dtype_mode(default_dtype);
  {
    auto tensor = torch::tensor({});
    ASSERT_EQ(tensor.numel(), 0);
    ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({0}));
    ASSERT_EQ(tensor.dtype(), default_dtype);
    ASSERT_FALSE(tensor.requires_grad());
  }
  {
    auto tensor = torch::tensor({{}, {}});
    ASSERT_EQ(tensor.numel(), 0);
    ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({2, 0}));
    ASSERT_EQ(tensor.dtype(), default_dtype);
    ASSERT_FALSE(tensor.requires_grad());
  }
  {
    auto tensor = torch::tensor({{{}, {}}});
    ASSERT_EQ(tensor.numel(), 0);
    ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({1, 2, 0}));
    ASSERT_EQ(tensor.dtype(), default_dtype);
    ASSERT_FALSE(tensor.requires_grad());
  }
  {
    auto tensor = torch::tensor({{{}}});
    ASSERT_EQ(tensor.numel(), 0);
    ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({1, 1, 0}));
    ASSERT_EQ(tensor.dtype(), default_dtype);
    ASSERT_FALSE(tensor.requires_grad());
  }
  {
    auto tensor = torch::tensor({{{{{{{{}}}}}}}});
    ASSERT_EQ(tensor.numel(), 0);
    ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({1, 1, 1, 1, 1, 1, 1, 0}));
    ASSERT_EQ(tensor.dtype(), default_dtype);
    ASSERT_FALSE(tensor.requires_grad());
  }
  {
    auto tensor = torch::tensor({{{{{{{{}}}}, {{{{}}}}}}}});
    ASSERT_EQ(tensor.numel(), 0);
    ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({1, 1, 1, 2, 1, 1, 1, 0}));
    ASSERT_EQ(tensor.dtype(), default_dtype);
    ASSERT_FALSE(tensor.requires_grad());
  }
  {
    auto tensor = torch::tensor({{{{{{{{{{}}}}}}}}}});
    ASSERT_EQ(tensor.numel(), 0);
    ASSERT_EQ(tensor.sizes(), std::vector<int64_t>({1, 1, 1, 1, 1, 1, 1, 1, 1, 0}));
    ASSERT_EQ(tensor.dtype(), default_dtype);
    ASSERT_FALSE(tensor.requires_grad());
  }
}

TEST(TensorTest, TorchTensorCtorZeroSizedDim) {
  test_TorchTensorCtorZeroSizedDim_expected_dtype(/*default_dtype=*/torch::kFloat);
  test_TorchTensorCtorZeroSizedDim_expected_dtype(/*default_dtype=*/torch::kDouble);
}

void test_TorchTensorCtorWithoutSpecifyingDtype_expected_dtype(c10::ScalarType default_dtype) {
  AutoDefaultDtypeMode dtype_mode(default_dtype);

  ASSERT_EQ(torch::tensor({1., 2., 3.}).dtype(), default_dtype);
  ASSERT_EQ(torch::tensor({{1., 2., 3.}}).dtype(), default_dtype);   
  ASSERT_EQ(torch::tensor({1., 2., 3.}, torch::TensorOptions()).dtype(), default_dtype);
  ASSERT_EQ(torch::tensor({{1., 2., 3.}}, torch::TensorOptions()).dtype(), default_dtype);
}

TEST(TensorTest, TorchTensorCtorWithoutSpecifyingDtype) {
  ASSERT_EQ(torch::tensor({1, 2, 3}).dtype(), torch::kLong);
  ASSERT_EQ(torch::tensor({{1, 2, 3}}).dtype(), torch::kLong);
  ASSERT_EQ(torch::tensor({1, 2, 3}, torch::TensorOptions()).dtype(), torch::kLong);
  ASSERT_EQ(torch::tensor({{1, 2, 3}}, torch::TensorOptions()).dtype(), torch::kLong);

  test_TorchTensorCtorWithoutSpecifyingDtype_expected_dtype(/*default_dtype=*/torch::kFloat);
  test_TorchTensorCtorWithoutSpecifyingDtype_expected_dtype(/*default_dtype=*/torch::kDouble);
}

void test_Arange_expected_dtype(c10::ScalarType default_dtype) {
  AutoDefaultDtypeMode dtype_mode(default_dtype);

  ASSERT_EQ(torch::arange(0., 5).dtype(), default_dtype);
}

TEST(TensorTest, Arange) {
  {
    auto x = torch::arange(0, 5);
    ASSERT_EQ(x.dtype() == torch::kLong);
  }
  test_Arange_expected_dtype(torch::kFloat);
  test_Arange_expected_dtype(torch::kDouble);
}

TEST(TensorTest, PrettyPrintTensorDataContainer) {
  {
    ASSERT_EQ(
      c10::str(torch::detail::TensorDataContainer(1.1)),
      "1.1");
  }
  {
    ASSERT_EQ(
      c10::str(torch::detail::TensorDataContainer({1.1, 2.2})),
      "{1.1, 2.2}");
  }
  {
    ASSERT_EQ(
      c10::str(torch::detail::TensorDataContainer({{1, 2}, {3, 4}})),
      "{{1, 2}, {3, 4}}");
  }
  {
    ASSERT_EQ(
      c10::str(torch::detail::TensorDataContainer({{{{{{{{1.1, 2.2, 3.3}}}}}, {{{{{4.4, 5.5, 6.6}}}}}, {{{{{7.7, 8.8, 9.9}}}}}}}})),
      "{{{{{{{{1.1, 2.2, 3.3}}}}}, {{{{{4.4, 5.5, 6.6}}}}}, {{{{{7.7, 8.8, 9.9}}}}}}}}");
  }
  {
    ASSERT_EQ(
      c10::str(torch::detail::TensorDataContainer({{{{{{{{{{1}}}}}}}}}})),
      "{{{{{{{{{{1}}}}}}}}}}");
  }
  {
    ASSERT_EQ(
      c10::str(torch::detail::TensorDataContainer({{{{{{{{{{}}}}}}}}}})),
      "{{{{{{{{{{}}}}}}}}}}");
  }
  {
    ASSERT_EQ(
      c10::str(torch::detail::TensorDataContainer({{{{{{{{{{1, 2}}}}}}}}}})),
      "{{{{{{{{{{1, 2}}}}}}}}}}");
  }
  {
    ASSERT_EQ(
      c10::str(torch::detail::TensorDataContainer(at::ArrayRef<double>({1.1, 2.2}))),
      "{1.1, 2.2}");
  }
  {
    ASSERT_EQ(
      c10::str(torch::detail::TensorDataContainer(std::vector<double>({1.1, 2.2}))),
      "{1.1, 2.2}");
  }
}

TEST(TensorTest, TensorDataContainerCallingAccessorOfWrongType) {
  {
    ASSERT_THROWS_WITH(
      torch::detail::TensorDataContainer(1.1).init_list(),
      "Can only call `init_list()` on a TensorDataContainer that has `is_init_list() == true`");
    ASSERT_THROWS_WITH(
      torch::detail::TensorDataContainer(1.1).tensor(),
      "Can only call `tensor()` on a TensorDataContainer that has `is_tensor() == true`");
  }
  {
    ASSERT_THROWS_WITH(
      torch::detail::TensorDataContainer({1.1, 2.2}).scalar(),
      "Can only call `scalar()` on a TensorDataContainer that has `is_scalar() == true`");
    ASSERT_THROWS_WITH(
      torch::detail::TensorDataContainer({1.1, 2.2}).tensor(),
      "Can only call `tensor()` on a TensorDataContainer that has `is_tensor() == true`");
  }
  {
    ASSERT_THROWS_WITH(
      torch::detail::TensorDataContainer(at::ArrayRef<double>({1.1, 2.2})).scalar(),
      "Can only call `scalar()` on a TensorDataContainer that has `is_scalar() == true`");
    ASSERT_THROWS_WITH(
      torch::detail::TensorDataContainer(at::ArrayRef<double>({1.1, 2.2})).init_list(),
      "Can only call `init_list()` on a TensorDataContainer that has `is_init_list() == true`");
  }
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
  const auto tensor = torch::rand({3, 3});
  ASSERT_TRUE(torch::equal(tensor, tensor.data()));
}

TEST(TensorTest, BackwardAndGrad) {
  auto x = torch::tensor({5}, torch::dtype(torch::kFloat).requires_grad(true));
  auto y = x * x;
  y.backward();
  ASSERT_EQ(x.grad().item<float>(), 10.0);
}

TEST(TensorTest, BackwardCreatesOnesGrad) {
  const auto x = torch::tensor({5}, torch::dtype(torch::kFloat).requires_grad(true));
  x.backward();
  ASSERT_TRUE(torch::equal(x.grad(),
              torch::ones_like(x)));
}

TEST(TensorTest, BackwardNonScalarOutputs) {
  auto x = torch::randn({5, 5}, torch::requires_grad());
  auto y = x * x;
  ASSERT_THROWS_WITH(y.backward(),
    "grad can be implicitly created only for scalar outputs");
}

TEST(TensorTest, IsLeaf) {
  auto x = torch::tensor({5}, torch::dtype(torch::kFloat).requires_grad(true));
  auto y = x * x;
  ASSERT_TRUE(x.is_leaf());
  ASSERT_FALSE(y.is_leaf());
}

TEST(TensorTest, OutputNr) {
  auto x = torch::tensor({5}, torch::dtype(torch::kFloat).requires_grad(true));
  auto y = x * x;
  ASSERT_EQ(x.output_nr(), 0);
  ASSERT_EQ(y.output_nr(), 0);
}

TEST(TensorTest, Version) {
  auto x = torch::ones(3);
  ASSERT_EQ(x._version(), 0);
  x.mul_(2);
  ASSERT_EQ(x._version(), 1);
  x.add_(1);
  ASSERT_EQ(x._version(), 2);
}

TEST(TensorTest, Detach) {
  auto x = torch::tensor({5}, torch::dtype(torch::kFloat).requires_grad(true));
  auto y = x * x;
  const auto y_detached = y.detach();
  ASSERT_FALSE(y.is_leaf());
  ASSERT_TRUE(y_detached.is_leaf());
  ASSERT_FALSE(y_detached.requires_grad());
}

TEST(TensorTest, DetachInplace) {
  auto x = torch::tensor({5}, torch::dtype(torch::kFloat).requires_grad(true));
  auto y = x * x;
  auto y_detached = y.detach_();
  ASSERT_TRUE(y.is_leaf());
  ASSERT_FALSE(y.requires_grad());
  ASSERT_TRUE(y_detached.is_leaf());
  ASSERT_FALSE(y_detached.requires_grad());
}

TEST(TensorTest, SetData) {
  auto x = torch::randn({5});
  auto y = torch::randn({5});
  ASSERT_FALSE(torch::equal(x, y));
  ASSERT_NE(x.data_ptr<float>(), y.data_ptr<float>());

  x.set_data(y);
  ASSERT_TRUE(torch::equal(x, y));
  ASSERT_EQ(x.data_ptr<float>(), y.data_ptr<float>());
}

TEST(TensorTest, RequiresGradInplace) {
  auto x = torch::tensor({5.0});
  x.requires_grad_(true);
  ASSERT_TRUE(x.requires_grad());

  auto y = x * x;
  ASSERT_THROWS_WITH(y.requires_grad_(false),
    "you can only change requires_grad flags of leaf variables.");

  x.requires_grad_(false);
  ASSERT_FALSE(x.requires_grad());

  const auto int_tensor = torch::tensor({5}, at::TensorOptions().dtype(torch::kInt));
  ASSERT_THROWS_WITH(int_tensor.requires_grad_(true),
    "Only Tensors of floating point dtype can require gradients");
}
