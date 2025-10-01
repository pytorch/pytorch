#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/stableivalue_conversions.h>
#include <torch/csrc/stable/tensor_struct.h>
#include <torch/headeronly/dummy.h>

#include <iostream>

namespace {

using torch::stable::Tensor;

// Implementation of test_fn that accesses id from Dummy and fills tensor
// with it
Tensor test_fn_impl(const Tensor& a, const dummy_types::Dummy& b) {
  int32_t id_value = b.get_id();
  int8_t foo = b.get_foo();

  STD_TORCH_CHECK(static_cast<int32_t>(foo) == 1, "foo must be 1");

  Tensor result = torch::stable::empty_like(a);
  torch::stable::fill_(result, static_cast<double>(id_value));

  return result;
}

// Boxed kernel function for test_fn
void boxed_test_fn(
    StableIValue* stack,
    uint64_t num_args,
    uint64_t num_outputs) {
  Tensor input_tensor = to<Tensor>(stack[0]);
  dummy_types::Dummy dummy_obj = to<dummy_types::Dummy>(stack[1]);
  Tensor result = test_fn_impl(input_tensor, dummy_obj);
  // Put result back on the stack
  stack[0] = from(result);
}

// Implementation of create_dummy that takes a Tensor and returns a Dummy with
// id 42
dummy_types::Dummy create_dummy_impl(const Tensor& input) {
  return dummy_types::Dummy(42);
}

// Boxed kernel function for create_dummy
void boxed_create_dummy(
    StableIValue* stack,
    uint64_t num_args,
    uint64_t num_outputs) {
  Tensor input_tensor = to<Tensor>(stack[0]);
  dummy_types::Dummy result = create_dummy_impl(input_tensor);
  // Put result back on the stack
  stack[0] = from(result);
}

} // namespace

// Register the function schema
STABLE_TORCH_LIBRARY(dummy_type_test, m) {
  m.def("test_fn(Tensor a, Dummy b) -> Tensor");
  m.def("create_dummy(Tensor a) -> Dummy");
}

// Register the implementation
STABLE_TORCH_LIBRARY_IMPL(dummy_type_test, CPU, m) {
  m.impl("test_fn", &boxed_test_fn);
  m.impl("create_dummy", &boxed_create_dummy);
}
