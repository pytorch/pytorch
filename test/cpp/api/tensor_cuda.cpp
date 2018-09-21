#include "catch_utils.hpp"

#include <ATen/ATen.h>

#include <cmath>

CATCH_TEST_CASE("Tensor/AllocatesTensorOnTheCorrectDevice", "[multi-cuda]") {
  auto tensor = at::tensor({1, 2, 3}, at::device({at::kCUDA, 1}));
  CATCH_REQUIRE(tensor.device().type() == at::Device::Type::CUDA);
  CATCH_REQUIRE(tensor.device().index() == 1);
}
