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

TEST_CASE("Tensor/ToLayout") {
  // TODO: failing due to unexpected exception with message:
  // _th_indices is not implemented for type CPUFloatType (_th_indices at /home/
  // psag/pytorch/pytorch/tools/cpp_build/build/caffe2/aten/src/ATen/Type.cpp:1986)
  // 1986)

  // auto tensor = at::empty({3, 4});
  // REQUIRE_TENSOR_OPTIONS(at::kCPU, -1, at::kFloat, at::kStrided);
  //
  // tensor = tensor.to(at::kSparse);
  // REQUIRE_TENSOR_OPTIONS(at::kCPU, -1, at::kFloat, at::kSparse);
  //
  // tensor = tensor.to(at::kStrided);
  // REQUIRE_TENSOR_OPTIONS(at::kCPU, -1, at::kFloat, at::kStrided);
}

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

TEST_CASE("Tensor/ToOptions", "[cuda]") {
  auto tensor = at::empty({3, 4});
  REQUIRE_TENSOR_OPTIONS(at::kCPU, -1, at::kFloat, at::kStrided);

  tensor = tensor.to(at::dtype(at::kInt).device({at::kCUDA, 1}));
  REQUIRE_TENSOR_OPTIONS(at::kCUDA, 1, at::kInt, at::kStrided);
}

TEST_CASE("Tensor/ToOptionsRespectsRequiresGrad") {
  auto tensor = torch::empty({3, 4});
  REQUIRE(!tensor.requires_grad());

  // `requires_grad` will subsequently always be true because it results from a
  // copy operation, which gives it a `CopyBackwards` gradient function.

  tensor = tensor.to(at::TensorOptions(tensor).requires_grad(true));
  REQUIRE(tensor.requires_grad());

  tensor = tensor.to(at::TensorOptions(tensor).requires_grad(false));
  REQUIRE(tensor.requires_grad());
}
