#include <catch.hpp>

#include <torch/functions.h>

#include <ATen/ATen.h>

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
