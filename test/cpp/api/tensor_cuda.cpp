#include <catch.hpp>

#include <ATen/ATen.h>

#include <cmath>

TEST_CASE("Tensor/AllocatesTensorOnTheCorrectDevice", "[cuda]") {
  auto tensor = at::tensor({1, 2, 3}, at::device({at::kCUDA, 1}));
  REQUIRE(tensor.device() == at::Device(at::kCUDA, 1));
}
