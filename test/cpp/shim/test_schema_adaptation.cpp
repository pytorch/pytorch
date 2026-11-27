#include "schema_adaptation_poc.h"

#include <ATen/ATen.h>
#include <gtest/gtest.h>
#include <torch/csrc/inductor/aoti_torch/utils.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/library.h>

#include <stdexcept>
#include <utility>

namespace {
using namespace schema_adaptation_poc;
using torch::stable::Tensor;

constexpr const char* kOldSchema = "example(Tensor x, int n) -> Tensor";
constexpr const char* kNewSchema =
    "example(Tensor x, int n, bool new_mode=False) -> Tensor";

template <class Wrapper, class = void>
struct HasNewMode : std::false_type {};

template <class Wrapper>
struct HasNewMode<
    Wrapper,
    std::void_t<decltype(std::declval<Wrapper&>().example(
        std::declval<const Tensor&>(),
        int64_t{},
        bool{}))>> : std::true_type {};

static_assert(!HasNewMode<Example<kPocV14, kPocV14>>::value);
static_assert(!HasNewMode<Example<kPocV16, kPocV14>>::value);
static_assert(!HasNewMode<Example<kPocV16, kPocV16>>::value);
static_assert(!HasNewMode<Example<kPocV17, kPocV14>>::value);
static_assert(!HasNewMode<Example<kPocV17, kPocV16>>::value);
static_assert(HasNewMode<Example<kPocV17, kPocV17>>::value);

const at::Tensor& native(const Tensor& tensor) {
  return *torch::aot_inductor::tensor_handle_to_tensor_pointer(tensor.get());
}

at::Tensor native_old(const at::Tensor& x, int64_t n) {
  return x.reshape({n, -1});
}

at::Tensor native_new(const at::Tensor& x, int64_t n, bool new_mode) {
  auto result = x.reshape({n, -1});
  return new_mode ? result.transpose(0, 1) : result;
}

template <uint64_t Build, uint64_t Target>
void check_row(uint64_t runtime, const Tensor& x) {
  SCOPED_TRACE(
      ::testing::Message() << "simulated B/T/R: " << Build << "/" << Target
                           << "/" << runtime);
  CallTrace trace;
  Example<Build, Target> wrapper(simulated_runtime(runtime), trace);
  for (int64_t n : {2, 3}) {
    auto output = wrapper.example(x, n);
    EXPECT_TRUE(native(output).equal(native(x).reshape({n, -1})));
    EXPECT_EQ(native(output).size(0), n);
    constexpr bool new_slots = Target >= kPocV17;
    const bool v2 = Build >= kPocV16 && runtime >= kPocV16;
    EXPECT_EQ(trace.slots, new_slots ? 3 : 2);
    EXPECT_EQ(trace.route, v2 ? ApiRoute::DynamicV2 : ApiRoute::Legacy);
    EXPECT_EQ(trace.build, Build);
    EXPECT_EQ(trace.target, v2 ? Target : 0);
    if constexpr (new_slots) {
      auto transposed = wrapper.example(x, n, true);
      EXPECT_TRUE(
          native(transposed).equal(native(x).reshape({n, -1}).transpose(0, 1)));
      EXPECT_EQ(native(transposed).size(0), 6 / n);
    }
  }
}

TEST(SchemaAdaptationPoc, SimulatedCallMatrix) {
  auto old_def = MAKE_TORCH_LIBRARY(_test_schema_poc_v14);
  auto new_def = MAKE_TORCH_LIBRARY(_test_schema_poc_v17);
  old_def.def(kOldSchema);
  new_def.def(kNewSchema);
  auto old_impl = MAKE_TORCH_LIBRARY_IMPL(_test_schema_poc_v14, CPU);
  auto new_impl = MAKE_TORCH_LIBRARY_IMPL(_test_schema_poc_v17, CPU);
  old_impl.impl("example", native_old);
  new_impl.impl("example", native_new);

  auto input = at::arange(6, at::TensorOptions().dtype(at::kLong));
  Tensor x(torch::aot_inductor::new_tensor_handle(at::Tensor(input)));
  for (auto runtime : {kPocV14, kPocV17}) {
    check_row<kPocV14, kPocV14>(runtime, x);
    check_row<kPocV16, kPocV14>(runtime, x);
    check_row<kPocV17, kPocV14>(runtime, x);
  }
  check_row<kPocV16, kPocV16>(kPocV17, x);
  check_row<kPocV17, kPocV16>(kPocV17, x);
  check_row<kPocV17, kPocV17>(kPocV17, x);

  // These fail before boxing or calling the C API. Build < target is a
  // compile-time failure in Example, analogous to stable/version.h.
  const auto references = input.use_count();
  CallTrace rejected;
  Example<kPocV16, kPocV16> target16(simulated_runtime(kPocV14), rejected);
  Example<kPocV17, kPocV17> target17(simulated_runtime(kPocV14), rejected);
  EXPECT_THROW(target16.example(x, 2), std::runtime_error);
  EXPECT_THROW(target17.example(x, 2), std::runtime_error);
  EXPECT_EQ(rejected.route, ApiRoute::None);
  EXPECT_EQ(rejected.slots, 0);
  EXPECT_EQ(input.use_count(), references);
  auto missing_v2 = simulated_runtime(kPocV17);
  missing_v2.has_dispatcher_v2 = false;
  Example<kPocV17, kPocV14> target14_without_v2(missing_v2, rejected);
  EXPECT_THROW(target14_without_v2.example(x, 2), std::runtime_error);
  EXPECT_EQ(rejected.route, ApiRoute::None);
  EXPECT_EQ(rejected.slots, 0);
  EXPECT_EQ(input.use_count(), references);

  // A direct legacy B17 caller has no known target and supplies THREE slots.
  // Treating target zero as old would incorrectly overwrite new_mode=True.
  {
    std::array<StableIValue, 3> stack{
        torch::stable::detail::from(x),
        torch::stable::detail::from(int64_t{3}),
        torch::stable::detail::from(true)};
    ASSERT_EQ(
        torch_call_dispatcher(
            "_test_schema_poc_v17::example", "", stack.data(), kPocV17),
        AOTI_TORCH_SUCCESS);
    auto result = torch::stable::detail::to<Tensor>(stack[0]);
    EXPECT_TRUE(native(result).equal(input.reshape({3, 2}).transpose(0, 1)));
  }
  EXPECT_EQ(input.use_count(), references);
}

int old_kernel_calls = 0;

Tensor old_cpp_kernel(Tensor x, int64_t n) {
  ++old_kernel_calls;
  return torch::stable::reshape(x, {n, -1});
}

// Handwritten native-schema bridge, NOT a generic registered-kernel adapter.
// x owns the consumed handle and is released even when new_mode is rejected.
Tensor bridge_new_schema(Tensor x, int64_t n, bool new_mode) {
  STD_TORCH_CHECK(!new_mode, "old kernel does not support new_mode=True");
  return old_cpp_kernel(std::move(x), n);
}

ApiRoute register_old_kernel(torch::Library& lib, RuntimeCapabilities runtime) {
  auto handle = reinterpret_cast<TorchLibraryHandle>(&lib); // Borrowed.
  if (runtime.version >= kPocV17) {
    STD_TORCH_CHECK(
        runtime.has_dispatcher_v2,
        "new-schema fixture requires v2 registration");
    auto impl_v2 = reinterpret_cast<decltype(&torch_library_impl_v2)>(
        torch::stable::detail::lookup_stable_symbol("torch_library_impl_v2"));
    STD_TORCH_CHECK(impl_v2 != nullptr, "fixture requires library impl v2");
    STABLE_TORCH_ERROR_CODE_CHECK(impl_v2(
        handle, "example", TORCH_BOX(bridge_new_schema), kPocV17, kPocV14));
    return ApiRoute::DynamicV2;
  }
  STABLE_TORCH_ERROR_CODE_CHECK(torch_library_impl(
      handle, "example", TORCH_BOX(old_cpp_kernel), kPocV17));
  return ApiRoute::Legacy;
}

TEST(SchemaAdaptationPoc, ManualRegisteredKernelBridge) {
  auto old_def = MAKE_TORCH_LIBRARY(_test_schema_poc_v14);
  auto new_def = MAKE_TORCH_LIBRARY(_test_schema_poc_v17);
  old_def.def(kOldSchema);
  new_def.def(kNewSchema);
  auto old_impl = MAKE_TORCH_LIBRARY_IMPL(_test_schema_poc_v14, CPU);
  auto new_impl = MAKE_TORCH_LIBRARY_IMPL(_test_schema_poc_v17, CPU);
  EXPECT_EQ(
      register_old_kernel(old_impl, simulated_runtime(kPocV14)),
      ApiRoute::Legacy);
  EXPECT_EQ(
      register_old_kernel(new_impl, simulated_runtime(kPocV17)),
      ApiRoute::DynamicV2);
  old_kernel_calls = 0;
  auto input = at::arange(6, at::TensorOptions().dtype(at::kLong));
  const auto references = input.use_count();
  for (auto runtime : {kPocV14, kPocV17}) {
    auto capabilities = simulated_runtime(runtime);
    auto op = c10::Dispatcher::singleton().findSchemaOrThrow(
        capabilities.op_name, "");
    for (int64_t n : {2, 3}) {
      torch::jit::Stack stack{input, n};
      if (runtime >= kPocV17) {
        stack.emplace_back(false);
      }
      op.callBoxed(stack);
      ASSERT_EQ(stack.size(), 1);
      EXPECT_TRUE(stack[0].toTensor().equal(input.reshape({n, -1})));
    }
  }
  EXPECT_EQ(input.use_count(), references);
  EXPECT_EQ(old_kernel_calls, 4);
  auto op = c10::Dispatcher::singleton().findSchemaOrThrow(
      "_test_schema_poc_v17::example", "");
  torch::jit::Stack unsupported{input, int64_t{2}, true};
  EXPECT_THROW(op.callBoxed(unsupported), std::runtime_error);
  EXPECT_EQ(old_kernel_calls, 4); // Rejected before entering the old body.
  EXPECT_EQ(input.use_count(), references); // Includes exception-path cleanup.
}

} // namespace
