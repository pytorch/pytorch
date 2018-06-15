#include "catch.hpp"

#include <ATen/Context.h>
#include <ATen/Functions.h>
#include <ATen/TensorOptions.h>

#include <ATen/DeviceGuard.h>

using namespace at;

// A macro so we don't lose location information when an assertion fails.
#define REQUIRE_OPTIONS(device_, index_, type_, layout_)                    \
  REQUIRE(options.device().type() == Device((device_), (index_)).type());   \
  REQUIRE(options.device().index() == Device((device_), (index_)).index()); \
  REQUIRE(options.dtype() == (type_));                                      \
  REQUIRE(options.layout() == (layout_))

TEST_CASE("TensorOptions/ConstructsWellFromCUDATypes", "[cuda]") {
  auto options = TensorOptions(CUDA(kFloat));
  REQUIRE_OPTIONS(kCUDA, -1, kFloat, kStrided);

  options = TensorOptions(CUDA(kInt));
  REQUIRE_OPTIONS(kCUDA, -1, kInt, kStrided);

  options = TensorOptions(getType(kSparseCUDA, kFloat));
  REQUIRE_OPTIONS(kCUDA, -1, kFloat, kSparse);

  options = TensorOptions(getType(kSparseCUDA, kByte));
  REQUIRE_OPTIONS(kCUDA, -1, kByte, kSparse);

  options = TensorOptions(CUDA(kFloat), /*device=*/5);
  REQUIRE_OPTIONS(kCUDA, 5, kFloat, kStrided);

  options = TensorOptions(getType(kSparseCUDA, kFloat), /*device=*/5);
  REQUIRE_OPTIONS(kCUDA, 5, kFloat, kSparse);
}

TEST_CASE("TensorOptions/ConstructsWellFromCUDATensors", "[cuda]") {
  auto options = TensorOptions(empty(5, device(kCUDA).dtype(kDouble)));
  REQUIRE_OPTIONS(kCUDA, 0, kDouble, kStrided);

  options = TensorOptions(empty(5, getType(kSparseCUDA, kByte)));
  REQUIRE_OPTIONS(kCUDA, 0, kByte, kSparse);

  if (at::globalContext().getNumGPUs() > 1) {
    Tensor tensor;
    {
      DeviceGuard guard(1);
      tensor = empty(5, device(kCUDA));
    }
    options = TensorOptions(tensor);
    REQUIRE_OPTIONS(kCUDA, 1, kFloat, kStrided);

    {
      DeviceGuard guard(1);
      tensor = empty(5, device(kCUDA).layout(kSparse));
    }
    options = TensorOptions(tensor);
    REQUIRE_OPTIONS(kCUDA, 1, kFloat, kSparse);
  }
}
