#include <catch.hpp>

#include <torch/tensor.h>

#include <ATen/ATen.h>

#include <cmath>
#include <cstddef>
#include <vector>

template <typename T>
bool exactly_equal(at::Tensor left, T right) {
  return at::Scalar(left).to<T>() == right;
}

template <typename T>
bool almost_equal(at::Tensor left, T right, T tolerance = 1e-4) {
  return std::abs(at::Scalar(left).to<T>() - right) < tolerance;
}

#define REQUIRE_TENSOR_OPTIONS(device_, index_, type_, layout_)                \
  REQUIRE(tensor.device().type() == at::Device((device_), (index_)).type());   \
  REQUIRE(tensor.device().index() == at::Device((device_), (index_)).index()); \
  REQUIRE(tensor.dtype() == (type_));                                          \
  REQUIRE(tensor.layout() == (layout_))

TEST_CASE("Tensor/ToDtype") {
  auto tensor = at::empty({3, 4});
  REQUIRE_TENSOR_OPTIONS(at::kCPU, -1, at::kFloat, at::kStrided);

  tensor = tensor.to(at::kInt);
  REQUIRE_TENSOR_OPTIONS(at::kCPU, -1, at::kInt, at::kStrided);

  tensor = tensor.to(at::kChar);
  REQUIRE_TENSOR_OPTIONS(at::kCPU, -1, at::kChar, at::kStrided);

  tensor = tensor.to(at::kDouble);
  REQUIRE_TENSOR_OPTIONS(at::kCPU, -1, at::kDouble, at::kStrided);
}

// Not currently supported.
// TEST_CASE("Tensor/ToLayout") {
//   auto tensor = at::empty({3, 4});
//   REQUIRE_TENSOR_OPTIONS(at::kCPU, -1, at::kFloat, at::kStrided);
//
//   tensor = tensor.to(at::kSparse);
//   REQUIRE_TENSOR_OPTIONS(at::kCPU, -1, at::kFloat, at::kSparse);
//
//   tensor = tensor.to(at::kStrided);
//   REQUIRE_TENSOR_OPTIONS(at::kCPU, -1, at::kFloat, at::kStrided);
// }

TEST_CASE("Tensor/ToDevice", "[cuda]") {
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
}

TEST_CASE("Tensor/ToDeviceAndDtype", "[cuda]") {
  auto tensor = at::empty({3, 4});
  REQUIRE_TENSOR_OPTIONS(at::kCPU, -1, at::kFloat, at::kStrided);

  tensor = tensor.to({at::kCUDA, 1}, at::kInt);
  REQUIRE_TENSOR_OPTIONS(at::kCUDA, 1, at::kInt, at::kStrided);
}

TEST_CASE("Tensor/ToOptionsRespectsRequiresGrad") {
  {
    auto tensor = torch::empty({3, 4}, at::requires_grad());
    REQUIRE(tensor.requires_grad());

    tensor = tensor.to(at::kDouble);
    REQUIRE(tensor.requires_grad());
  }
  {
    auto tensor = torch::empty({3, 4});
    REQUIRE(!tensor.requires_grad());

    tensor = tensor.to(at::kDouble);
    REQUIRE(!tensor.requires_grad());
  }
}

TEST_CASE("Tensor/ToDoesNotCopyWhenOptionsAreAllTheSame") {
  auto tensor = at::empty({3, 4}, at::kFloat);
  auto hopefully_not_copy = tensor.to(at::kFloat);
  REQUIRE(hopefully_not_copy.data<float>() == tensor.data<float>());
}

TEST_CASE("Tensor/ContainsCorrectValueForSingleValue") {
  auto tensor = at::tensor(123);
  REQUIRE(tensor.numel() == 1);
  REQUIRE(tensor.dtype() == at::kInt);
  REQUIRE(tensor[0].toCInt() == 123);

  tensor = at::tensor(123.456f);
  REQUIRE(tensor.numel() == 1);
  REQUIRE(tensor.dtype() == at::kFloat);
  REQUIRE(almost_equal(tensor[0], 123.456f));

  tensor = at::tensor(123.456);
  REQUIRE(tensor.numel() == 1);
  REQUIRE(tensor.dtype() == at::kDouble);
  REQUIRE(almost_equal(tensor[0], 123.456));
}

TEST_CASE("Tensor/ContainsCorrectValuesForManyValues") {
  auto tensor = at::tensor({1, 2, 3});
  REQUIRE(tensor.numel() == 3);
  REQUIRE(tensor.dtype() == at::kInt);
  REQUIRE(exactly_equal(tensor[0], 1));
  REQUIRE(exactly_equal(tensor[1], 2));
  REQUIRE(exactly_equal(tensor[2], 3));

  tensor = at::tensor({1.5, 2.25, 3.125});
  REQUIRE(tensor.numel() == 3);
  REQUIRE(tensor.dtype() == at::kDouble);
  REQUIRE(almost_equal(tensor[0], 1.5));
  REQUIRE(almost_equal(tensor[1], 2.25));
  REQUIRE(almost_equal(tensor[2], 3.125));
}

TEST_CASE("Tensor/ContainsCorrectValuesForManyValuesVariable") {
  auto tensor = torch::tensor({1, 2, 3});
  REQUIRE(tensor.is_variable());
  REQUIRE(tensor.numel() == 3);
  REQUIRE(tensor.dtype() == at::kInt);
  REQUIRE(exactly_equal(tensor[0], 1));
  REQUIRE(exactly_equal(tensor[1], 2));
  REQUIRE(exactly_equal(tensor[2], 3));

  tensor = torch::tensor({1.5, 2.25, 3.125});
  REQUIRE(tensor.is_variable());
  REQUIRE(tensor.numel() == 3);
  REQUIRE(tensor.dtype() == at::kDouble);
  REQUIRE(almost_equal(tensor[0], 1.5));
  REQUIRE(almost_equal(tensor[1], 2.25));
  REQUIRE(almost_equal(tensor[2], 3.125));
}

TEST_CASE("Tensor/ContainsCorrectValuesWhenConstructedFromVector") {
  std::vector<int> v = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  auto tensor = at::tensor(v);
  REQUIRE(tensor.numel() == v.size());
  REQUIRE(tensor.dtype() == at::kInt);
  for (size_t i = 0; i < v.size(); ++i) {
    REQUIRE(exactly_equal(tensor[i], v.at(i)));
  }

  std::vector<float> w = {1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.0};
  tensor = at::tensor(w);
  REQUIRE(tensor.numel() == w.size());
  REQUIRE(tensor.dtype() == at::kFloat);
  for (size_t i = 0; i < w.size(); ++i) {
    REQUIRE(almost_equal(tensor[i], w.at(i)));
  }
}

TEST_CASE("Tensor/UsesOptionsThatAreSupplied") {
  auto tensor = at::tensor(123, dtype(at::kFloat)) + 0.5;
  REQUIRE(tensor.numel() == 1);
  REQUIRE(tensor.dtype() == at::kFloat);
  REQUIRE(almost_equal(tensor[0], 123.5));

  tensor = at::tensor({1.1, 2.2, 3.3}, dtype(at::kInt));
  REQUIRE(tensor.numel() == 3);
  REQUIRE(tensor.dtype() == at::kInt);
  REQUIRE(tensor.layout() == at::kStrided);
  REQUIRE(exactly_equal(tensor[0], 1));
  REQUIRE(exactly_equal(tensor[1], 2));
  REQUIRE(exactly_equal(tensor[2], 3));
}

TEST_CASE("FromBlob") {
  std::vector<int32_t> v = {1, 2, 3};
  auto tensor = torch::from_blob(v.data(), v.size(), torch::kInt32);
  REQUIRE(tensor.is_variable());
  REQUIRE(tensor.numel() == 3);
  REQUIRE(tensor[0].toCInt() == 1);
  REQUIRE(tensor[1].toCInt() == 2);
  REQUIRE(tensor[2].toCInt() == 3);
}
