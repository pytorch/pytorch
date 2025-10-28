// targeting a version below 2.8.0 causes this file to fail to compile,
// as we expect :)
// #define TORCH_TARGET_VERSION (((0ULL + 2) << 56) | ((0ULL + 5) << 48))

// Uncomment to target V1 "2.8.0"
// #define TORCH_TARGET_VERSION (((0ULL + 2) << 56) | ((0ULL + 8) << 48))

#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/stableivalue_conversions.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/dummy.h>
#include <torch/headeronly/util/Exception.h>

using torch::stable::Tensor;

// Function that takes a tensor and dummy, calls the op from ops.h, and
// returns the tensor result
Tensor test_op_with_dummy(const Tensor& input_tensor) {
  // Create a dummy with id=42
  dummy_types::Dummy dummy(42);

  Tensor result = torch::stable::op(input_tensor, dummy);

  return result;
}

void boxed_test_op_with_dummy(
    StableIValue* stack,
    uint64_t num_args,
    uint64_t num_outputs) {
  Tensor res = test_op_with_dummy(torch::stable::detail::to<Tensor>(stack[0]));
  stack[0] = torch::stable::detail::from(res);
}

// Function that takes a tensor and dummy, calls the op from ops.h with scale=3,
// and returns the tensor result
Tensor test_op_with_dummy_scale3(const Tensor& input_tensor) {
  // Create a dummy with id=42
  dummy_types::Dummy dummy(42);

  Tensor result = torch::stable::op(input_tensor, dummy, 3);

  return result;
}

void boxed_test_op_with_dummy_scale3(
    StableIValue* stack,
    uint64_t num_args,
    uint64_t num_outputs) {
  Tensor res = test_op_with_dummy_scale3(to<Tensor>(stack[0]));
  stack[0] = from(res);
}

// Register the functions in the library
STABLE_TORCH_LIBRARY(ops_test, m) {
  m.def("test_op_with_dummy(Tensor input) -> Tensor");
  m.def("test_op_with_dummy_scale3(Tensor input) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(ops_test, CPU, m) {
  m.impl("test_op_with_dummy", &boxed_test_op_with_dummy);
  m.impl("test_op_with_dummy_scale3", &boxed_test_op_with_dummy_scale3);
}
