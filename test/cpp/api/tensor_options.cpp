#include "catch.hpp"

#include <torch/tensor.h>

#include <ATen/Context.h>
#include <ATen/Functions.h>
#include <ATen/OptionsGuard.h>
#include <ATen/TensorOptions.h>

#include <vector>
#include <string>

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

TEST_CASE("TensorOptions/DefaultsToTheRightValues") {
  TensorOptions options;
  REQUIRE_OPTIONS(kCPU, -1, kFloat, kStrided);
}

TEST_CASE("TensorOptions/ReturnsTheCorrectType") {
  auto options = TensorOptions().device(kCPU).dtype(kInt).layout(kSparse);
  REQUIRE(options.type() == getType(Backend::SparseCPU, kInt));
}

TEST_CASE("TensorOptions/UtilityFunctionsReturnTheRightTensorOptions") {
  auto options = dtype(kInt);
  REQUIRE_OPTIONS(kCPU, -1, kInt, kStrided);

  options = layout(kSparse);
  REQUIRE_OPTIONS(kCPU, -1, kFloat, kSparse);

  options = device({kCUDA, 1});
  REQUIRE_OPTIONS(kCUDA, 1, kFloat, kStrided);

  options = device_index(1);
  REQUIRE_OPTIONS(kCUDA, 1, kFloat, kStrided);

  options = dtype(kByte).layout(kSparse).device({kCUDA, 2}).device_index(3);
  REQUIRE_OPTIONS(kCUDA, 3, kByte, kSparse);
}

TEST_CASE("TensorOptions/ConstructsWellFromCPUTypes") {
  TensorOptions options;
  REQUIRE_OPTIONS(kCPU, -1, kFloat, kStrided);

  options = TensorOptions({kCPU, 0});
  REQUIRE_OPTIONS(kCPU, 0, kFloat, kStrided);

  options = TensorOptions(kInt);
  REQUIRE_OPTIONS(kCPU, -1, kInt, kStrided);

  options = TensorOptions(getType(Backend::SparseCPU, kFloat));
  REQUIRE_OPTIONS(kCPU, -1, kFloat, kSparse);

  options = TensorOptions(getType(Backend::SparseCPU, kByte));
  REQUIRE_OPTIONS(kCPU, -1, kByte, kSparse);
}

TEST_CASE("TensorOptions/ConstructsWellFromCPUTensors") {
  auto options = TensorOptions(empty(5, kDouble));
  REQUIRE_OPTIONS(kCPU, -1, kDouble, kStrided);

  options = TensorOptions(empty(5, getType(Backend::SparseCPU, kByte)));
  REQUIRE_OPTIONS(kCPU, -1, kByte, kSparse);
}

TEST_CASE("TensorOptions/ConstructsWellFromVariables") {
  auto options = TensorOptions(torch::empty(5));
  REQUIRE_OPTIONS(kCPU, -1, kFloat, kStrided);
  REQUIRE(!options.requires_grad());

  options = TensorOptions(torch::empty(5, at::requires_grad()));
  REQUIRE_OPTIONS(kCPU, -1, kFloat, kStrided);
  REQUIRE(!options.requires_grad());
}

TEST_CASE("Device/ParsesCorrectlyFromString") {
  Device device("cpu:0");
  REQUIRE(device == Device(kCPU, 0));

  device = Device("cpu");
  REQUIRE(device == Device(kCPU));

  device = Device("cuda:123");
  REQUIRE(device == Device(kCUDA, 123));

  device = Device("cuda");
  REQUIRE(device == Device(kCUDA));

  std::vector<std::string> badnesses = {
      "", "cud:1", "cuda:", "cpu::1", ":1", "3", "tpu:4", "??"};
  for (const auto& badness : badnesses) {
    REQUIRE_THROWS(Device(badness));
  }
}

TEST_CASE("OptionsGuard") {
  Tensor tensor;
  {
    OptionsGuard guard(TensorOptions{});
    tensor = at::empty({10});
  }
  REQUIRE_TENSOR_OPTIONS(kCPU, -1, kFloat, kStrided);

  {
    OptionsGuard guard(TensorOptions().dtype(kInt));
    tensor = at::empty({10});
  }
  REQUIRE_TENSOR_OPTIONS(kCPU, -1, kInt, kStrided);

  {
    OptionsGuard guard(TensorOptions().dtype(kInt).layout(kSparse));
    tensor = at::empty({10});
  }
  REQUIRE_TENSOR_OPTIONS(kCPU, -1, kInt, kSparse);

  {
    OptionsGuard guard(requires_grad(true));
    tensor = torch::empty({10});
  }
  REQUIRE_TENSOR_OPTIONS(kCPU, -1, kFloat, kStrided);
  REQUIRE(tensor.requires_grad());
}
