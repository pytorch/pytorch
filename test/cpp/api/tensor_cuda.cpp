#include <catch.hpp>

#include <ATen/ATen.h>

#include <cmath>

TEST_CASE("Tensor/AllocatesTensorOnTheCorrectDevice", "[cuda]") {
  auto tensor = at::tensor({1, 2, 3}, at::device({at::kCUDA, 1}));
  REQUIRE(tensor.device().type() == at::Device::Type::CUDA);
  REQUIRE(tensor.device().index() == 1);
}
