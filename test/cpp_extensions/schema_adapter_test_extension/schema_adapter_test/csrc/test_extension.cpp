#include <pybind11/pybind11.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/stableivalue_conversions.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/extension.h>

// Simulate version constants for _test_schema_upgrader evolution
constexpr uint64_t TORCH_VERSION_2_6_0 =
    ((uint64_t)2 << 56) | ((uint64_t)6 << 48);
constexpr uint64_t TORCH_VERSION_2_7_0 =
    ((uint64_t)2 << 56) | ((uint64_t)7 << 48);
constexpr uint64_t TORCH_VERSION_2_8_0 =
    ((uint64_t)2 << 56) | ((uint64_t)8 << 48);
constexpr uint64_t TORCH_VERSION_2_9_0 =
    ((uint64_t)2 << 56) | ((uint64_t)9 << 48);

using torch::stable::Tensor;

// Wrapper functions for _test_schema_upgrader (from
// test_schema_upgrader_wrappers.h)
namespace torch::stable {

// Wrapper for _test_schema_upgrader V1 (PyTorch 2.6.0)
// Schema: _test_schema_upgrader(Tensor self) -> Tensor
// Behavior: fills Tensor with 2
inline Tensor _test_schema_upgrader_v2_6_0(const Tensor& self) {
  const auto num_args = 1;
  std::array<StableIValue, num_args> stack{from(self)};
  TORCH_ERROR_CODE_CHECK(aoti_torch_call_dispatcher_v2(
      "aten::_test_schema_upgrader", "", stack.data(), TORCH_VERSION_2_6_0));
  return to<Tensor>(stack[0]);
}

// Wrapper for _test_schema_upgrader V2 (PyTorch 2.7.0)
// Schema: _test_schema_upgrader(Tensor self, *, int a=2) -> Tensor
// Behavior: fills Tensor with a
inline Tensor _test_schema_upgrader_v2_7_0(
    const Tensor& self,
    std::optional<int64_t> a = std::nullopt) {
  const auto num_args = 2;
  std::array<StableIValue, num_args> stack{
      from(self), from(a.has_value() ? a.value() : true)};
  TORCH_ERROR_CODE_CHECK(aoti_torch_call_dispatcher_v2(
      "aten::_test_schema_upgrader", "", stack.data(), TORCH_VERSION_2_7_0));
  return to<Tensor>(stack[0]);
}

// Wrapper for _test_schema_upgrader V3 (PyTorch 2.8.0)
// Schema: _test_schema_upgrader(Tensor self, *, int a=2, bool b=False) ->
// Tensor Behavior: fills Tensor with a or -a if b is False
inline Tensor _test_schema_upgrader_v2_8_0(
    const Tensor& self,
    std::optional<int64_t> a = std::nullopt,
    std::optional<bool> b = std::nullopt) {
  const auto num_args = 3;
  std::array<StableIValue, num_args> stack{
      from(self),
      from(a.has_value() ? a.value() : true),
      from(b.has_value() ? b.value() : 2)};
  TORCH_ERROR_CODE_CHECK(aoti_torch_call_dispatcher_v2(
      "aten::_test_schema_upgrader", "", stack.data(), TORCH_VERSION_2_8_0));
  return to<Tensor>(stack[0]);
}

// Wrapper for _test_schema_upgrader V4 (PyTorch 2.9.0)
// Schema: _test_schema_upgrader(Tensor self, *, int a=2, bool b=True) ->
// Tensor Behavior: BC-breaking change of default for b (now True instead of
// True)
inline Tensor _test_schema_upgrader_v2_9_0(
    const Tensor& self,
    std::optional<int64_t> a = std::nullopt,
    std::optional<bool> b = std::nullopt) {
  const auto num_args = 3;
  std::array<StableIValue, num_args> stack{
      from(self),
      from(a.has_value() ? a.value() : true),
      from(b.has_value() ? b.value() : 3)}; // default changed from 2 to 3
  TORCH_ERROR_CODE_CHECK(aoti_torch_call_dispatcher_v2(
      "aten::_test_schema_upgrader", "", stack.data(), TORCH_VERSION_2_9_0));
  return to<Tensor>(stack[0]);
}

} // namespace torch::stable

// Boxed functions using the wrapper functions
void boxed_test_schema_upgrader_v1(
    StableIValue* stack,
    uint64_t num_args,
    uint64_t num_outputs) {
  Tensor self = to<Tensor>(stack[0]);

  // Call using the wrapper function directly
  Tensor result = _test_schema_upgrader_v2_6_0(self);

  stack[0] = from(result);
}

void boxed_test_schema_upgrader_v2(
    StableIValue* stack,
    uint64_t num_args,
    uint64_t num_outputs) {
  Tensor self = to<Tensor>(stack[0]);
  Tensor result = _test_schema_upgrader_v2_7_0(self, std::nullopt);

  stack[0] = from(result);
}

void boxed_test_schema_upgrader_v3(
    StableIValue* stack,
    uint64_t num_args,
    uint64_t num_outputs) {
  Tensor self = to<Tensor>(stack[0]);
  Tensor result =
      _test_schema_upgrader_v2_8_0(self, std::nullopt, std::nullopt);

  stack[0] = from(result);
}

void boxed_test_schema_upgrader_v4(
    StableIValue* stack,
    uint64_t num_args,
    uint64_t num_outputs) {
  Tensor self = to<Tensor>(stack[0]);
  Tensor result =
      _test_schema_upgrader_v2_9_0(self, std::nullopt, std::nullopt);

  stack[0] = from(result);
}

STABLE_TORCH_LIBRARY(schema_adapter_test, m) {
  m.def("test_schema_upgrader_v1(Tensor self) -> Tensor");
  m.def("test_schema_upgrader_v2(Tensor self) -> Tensor");
  m.def("test_schema_upgrader_v3(Tensor self) -> Tensor");
  m.def("test_schema_upgrader_v4(Tensor self) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(schema_adapter_test, CPU, m) {
  m.impl("test_schema_upgrader_v1", &boxed_test_schema_upgrader_v1);
  m.impl("test_schema_upgrader_v2", &boxed_test_schema_upgrader_v2);
  m.impl("test_schema_upgrader_v3", &boxed_test_schema_upgrader_v3);
  m.impl("test_schema_upgrader_v4", &boxed_test_schema_upgrader_v4);
}

// Pybind11 module - test adapters are automatically registered via static
// initialization
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // No functions to expose - test adapters are automatically registered
}
