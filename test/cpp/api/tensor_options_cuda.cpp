#include "catch.hpp"

#include <ATen/Context.h>
#include <ATen/DeviceGuard.h>
#include <ATen/Functions.h>
#include <ATen/OptionsGuard.h>
#include <ATen/TensorOptions.h>

using namespace at;

// A macro so we don't lose location information when an assertion fails.
#define REQUIRE_OPTIONS(device_, index_, type_, layout_)                    \
  REQUIRE(options.device().type() == Device((device_), (index_)).type());   \
  REQUIRE(options.device().index() == Device((device_), (index_)).index()); \
  REQUIRE(options.dtype() == (type_));                                      \
  REQUIRE(options.layout() == (layout_))

#define REQUIRE_TENSOR_OPTIONS(device_, index_, type_, layout_)            \
  REQUIRE(tensor.device().type() == Device((device_), (index_)).type());   \
  REQUIRE(tensor.device().index() == Device((device_), (index_)).index()); \
  REQUIRE(tensor.type().scalarType() == (type_));                          \
  REQUIRE(tensor.type().layout() == (layout_))

TEST_CASE("TensorOptions/ConstructsWellFromCUDATypes", "[cuda]") {
  auto options = TensorOptions(CUDA(kFloat));
  REQUIRE_OPTIONS(kCUDA, -1, kFloat, kStrided);

  options = TensorOptions(CUDA(kInt));
  REQUIRE_OPTIONS(kCUDA, -1, kInt, kStrided);

  options = TensorOptions(getType(Backend::SparseCUDA, kFloat));
  REQUIRE_OPTIONS(kCUDA, -1, kFloat, kSparse);

  options = TensorOptions(getType(Backend::SparseCUDA, kByte));
  REQUIRE_OPTIONS(kCUDA, -1, kByte, kSparse);

  options = TensorOptions(CUDA(kFloat), /*device=*/5);
  REQUIRE_OPTIONS(kCUDA, 5, kFloat, kStrided);

  options = TensorOptions(getType(Backend::SparseCUDA, kFloat), /*device=*/5);
  REQUIRE_OPTIONS(kCUDA, 5, kFloat, kSparse);
}

TEST_CASE("TensorOptions/ConstructsWellFromCUDATensors", "[multi-cuda]") {
  auto options = TensorOptions(empty(5, device(kCUDA).dtype(kDouble)));
  REQUIRE_OPTIONS(kCUDA, 0, kDouble, kStrided);

  options = TensorOptions(empty(5, getType(Backend::SparseCUDA, kByte)));
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

TEST_CASE("OptionsGuardCUDA", "[multi-cuda]") {
  Tensor tensor;
  {
    OptionsGuard guard(device(kCUDA));
    tensor = at::empty({10});
  }
  REQUIRE_TENSOR_OPTIONS(kCUDA, 0, kFloat, kStrided);

  {
    OptionsGuard guard(device({kCUDA, 1}));
    tensor = at::empty({10});
  }
  REQUIRE_TENSOR_OPTIONS(kCUDA, 1, kFloat, kStrided);

  {
    OptionsGuard guard(device(kCUDA).dtype(kInt));
    tensor = at::empty({10});
  }
  REQUIRE_TENSOR_OPTIONS(kCUDA, 0, kInt, kStrided);
}

TEST_CASE("DeviceGuardOptionsGuardInteraction", "[multi-cuda]") {
  Tensor tensor;
  {
    // Check that OptionsGuard respects any active device before construction.
    DeviceGuard guard(1);
    {
      OptionsGuard guard(device(kCUDA));
      tensor = at::empty({10});
      REQUIRE_TENSOR_OPTIONS(kCUDA, 1, kFloat, kStrided);
      {
        // Check that OptionsGuard respects any active device after
        // construction.
        DeviceGuard guard(0);
        tensor = at::empty({10});
        REQUIRE_TENSOR_OPTIONS(kCUDA, 0, kFloat, kStrided);
        {
          OptionsGuard guard(device({kCUDA, 1}));
          tensor = at::empty({10});
          REQUIRE_TENSOR_OPTIONS(kCUDA, 1, kFloat, kStrided);
        }
      }
    }
  }
}

TEST_CASE("DeviceGuardIsMovable", "[cuda]") {
  DeviceGuard first(1);
  REQUIRE(first.original_index() == 0);
  REQUIRE(first.last_index() == 1);
  DeviceGuard second(std::move(first));
  REQUIRE(second.original_index() == 0);
  REQUIRE(second.last_index() == 1);
  REQUIRE(first.original_index() == -1);
  DeviceGuard third;
  third = std::move(second);
  REQUIRE(third.original_index() == 0);
  REQUIRE(third.last_index() == 1);
  REQUIRE(second.original_index() == -1);
}
