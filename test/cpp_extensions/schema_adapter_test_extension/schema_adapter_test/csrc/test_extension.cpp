#include <ATen/core/function_schema.h>
#include <pybind11/pybind11.h>
#include <torch/csrc/inductor/aoti_torch/c/private_utils.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/stableivalue_conversions.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/extension.h>

// Simulate version constants since we can't change TORCH_ABI_VERSION
constexpr uint64_t TORCH_VERSION_2_8_0 =
    ((uint64_t)2 << 56) | ((uint64_t)8 << 48);
constexpr uint64_t TORCH_VERSION_2_9_0 =
    ((uint64_t)2 << 56) | ((uint64_t)9 << 48);

using torch::stable::Tensor;

// Dummy operation implementation
// We pretend we had
// V1: dummy_op(Tensor input) -> Tensor
// V2: dummy_op(Tensor input, int a = 2) -> Tensor  (added defaulted kwarg a
// in 2.9.0) V1 fills tensor with 2, V2 fills tensor with the value of 'a'
Tensor dummy_op_v2_impl(const Tensor& input, int64_t a) {
  Tensor result = torch::stable::empty_like(input);
  torch::stable::fill_(result, static_cast<double>(a));
  return result;
}

// Adapter to read stack of v1 op
// Schema adapter for our dummy operation
// V1: dummy_op(Tensor input) -> Tensor
// V2: dummy_op(Tensor input, int a = 2) -> Tensor  (added defaulted kwarg a
// in 2.9.0)
torch::jit::Stack adapt_dummy_op_v1_to_v2(
    const c10::FunctionSchema& current_schema,
    const StableIValue* extension_stack,
    uint64_t extension_abi_version) {
  TORCH_CHECK(
      extension_abi_version < TORCH_VERSION_2_9_0,
      "Schema adapter adapt_dummy_op_v1_to_v2 should only be called for extension ABI version < 2.9.0, got: ",
      extension_abi_version);
  const auto num_returns = current_schema.returns().size();
  const auto num_arguments = current_schema.arguments().size();
  torch::jit::Stack ivalue_stack;
  ivalue_stack.reserve(std::max(num_arguments, num_returns));

  // Extension provides 1 arg (input), current schema expects 2 (input, a)
  // Convert the input StableIValue to IValue
  auto arg_type = current_schema.arguments()[0].type();
  ivalue_stack.push_back(to_ivalue(arg_type, extension_stack[0]));

  // Add the missing 2nd argument (a=2 as default)
  ivalue_stack.push_back(c10::IValue(static_cast<int64_t>(2)));

  return ivalue_stack;
}

void boxed_test_dummy_op_v1(
    StableIValue* stack,
    uint64_t num_args,
    uint64_t num_outputs) {
  Tensor input = to<Tensor>(stack[0]);

  // Create stack for dispatcher call with v1 extension version (1 argument)
  StableIValue dispatch_stack[1];
  dispatch_stack[0] = from(input);

  // Simulate calling with v1 extension version
  TORCH_ERROR_CODE_CHECK(aoti_torch_call_dispatcher_v2(
      "schema_adapter_test::dummy_op",
      "",
      dispatch_stack,
      TORCH_VERSION_2_8_0));

  stack[0] = from(to<Tensor>(dispatch_stack[0]));
}

void boxed_test_dummy_op_v2(
    StableIValue* stack,
    uint64_t num_args,
    uint64_t num_outputs) {
  Tensor input = to<Tensor>(stack[0]);
  int64_t a = to<int64_t>(stack[1]);

  StableIValue dispatch_stack[2];
  dispatch_stack[0] = from(input);
  dispatch_stack[1] = from<int64_t>(a);

  // Simulate calling with v2 extension version
  TORCH_ERROR_CODE_CHECK(aoti_torch_call_dispatcher_v2(
      "schema_adapter_test::dummy_op",
      "",
      dispatch_stack,
      TORCH_VERSION_2_9_0));

  stack[0] = from(to<Tensor>(dispatch_stack[0]));
}

// Function to register the schema adapter
void register_adapter() {
  TORCH_ERROR_CODE_CHECK(register_schema_adapter(
      "schema_adapter_test::dummy_op",
      TORCH_VERSION_2_9_0,
      reinterpret_cast<void*>(adapt_dummy_op_v1_to_v2)));
}

void boxed_dummy_op(
    StableIValue* stack,
    uint64_t num_args,
    uint64_t num_outputs) {
  Tensor result = dummy_op_v2_impl(to<Tensor>(stack[0]), to<int64_t>(stack[1]));
  stack[0] = from(result);
}

STABLE_TORCH_LIBRARY(schema_adapter_test, m) {
  m.def("dummy_op(Tensor input, int a=2) -> Tensor");
  m.def("test_dummy_op_v1(Tensor input) -> Tensor");
  m.def("test_dummy_op_v2(Tensor input, int a) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(schema_adapter_test, CPU, m) {
  m.impl("dummy_op", &boxed_dummy_op);
  m.impl("test_dummy_op_v1", &boxed_test_dummy_op_v1);
  m.impl("test_dummy_op_v2", &boxed_test_dummy_op_v2);
}

// register_adapter does not take any tensor arguments
// we use pybind to avoid an error that claims we need a fallback.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "register_adapter",
      &register_adapter,
      "Register the schema adapter for dummy_op");
}
