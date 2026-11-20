// Checks the behaviour of:
//  torch_new_stable_ivalue
//  torch_delete_stable_ivalue

#include <gtest/gtest.h>
#include <torch/csrc/stable/c/shim.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/stableivalue_conversions.h>
#include <torch/library.h>
#include <iterator>
#include <limits>
#include <optional>
#include <vector>

TEST(TorchStableIValue, TestStableIValueUse) {
  StableIValue* a = nullptr;
  ASSERT_EQ(torch_new_stable_ivalue(&a), AOTI_TORCH_SUCCESS);
  // Check if it is now a valid pointer.
  ASSERT_NE(a, nullptr);
  // Assign to verify that we can actually write to the full u64 size, this will
  // show up in valgrind if we were to allocate less than the required size.
  *a = std::numeric_limits<StableIValue>::max();

  // Free the value again.
  EXPECT_EQ(torch_delete_stable_ivalue(a), AOTI_TORCH_SUCCESS);

  // Freeing a nullptr should result in a failure.
  StableIValue* b = nullptr;
  ASSERT_EQ(torch_delete_stable_ivalue(b), AOTI_TORCH_FAILURE);

  // Trying to allocate with a pointer into which can't be assigned is a
  // failure.
  StableIValue** ret_value = nullptr;
  ASSERT_EQ(torch_new_stable_ivalue(ret_value), AOTI_TORCH_FAILURE);
}

TEST(TorchStableIValue, TestStableIValueLargeAmount) {
  // Next, do a bunch of allocations and frees, such that if the delete isn't
  // working but the allocation is we see a clear memory leak in valgrind.
  StableIValue* v = nullptr;
  const size_t total_allocation_bytes = 10'000'000;
  const size_t number_of_allocations =
      total_allocation_bytes / sizeof(StableIValue);
  for (size_t i = 0; i < number_of_allocations; i++) {
    ASSERT_EQ(torch_new_stable_ivalue(&v), AOTI_TORCH_SUCCESS);
    ASSERT_NE(v, nullptr);
    ASSERT_EQ(torch_delete_stable_ivalue(v), AOTI_TORCH_SUCCESS);
    v = nullptr;
  }
}

TEST(TorchStableIValue, TestStringRoundtrip) {
  // This performs a simple roundtrip from std::string -> StableIValue ->
  // std::string
  const std::string our_string = "Hello World!";
  const StableIValue as_stable_ivalue = torch::stable::detail::from(our_string);
  const std::string string_back =
      torch::stable::detail::to<std::string>(as_stable_ivalue);

  ASSERT_EQ(string_back, our_string);
}

#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_16_0

TORCH_LIBRARY(test_stable_dispatcher, m) {
  m.def("identity(int value) -> int", [](int64_t value) { return value; });
  m.def(
      "list(int?[] values) -> int?[]",
      [](c10::List<std::optional<int64_t>> values) { return values; });
}

TEST(TorchStableIValue, DispatcherVersions) {
  using torch::stable::detail::from;
  using torch::stable::detail::to;
  using Dispatcher = AOTITorchError (*)(const char*, StableIValue*);
  const Dispatcher dispatchers[] = {
      +[](const char* op, StableIValue* stack) {
        return aoti_torch_call_dispatcher(op, "", stack);
      },
      +[](const char* op, StableIValue* stack) {
        return torch_call_dispatcher(op, "", stack, TORCH_VERSION_2_14_0);
      },
      +[](const char* op, StableIValue* stack) {
        return torch_call_dispatcher(op, "", stack, TORCH_ABI_VERSION);
      },
      +[](const char* op, StableIValue* stack) {
        return torch_call_dispatcher_v2(
            op, "", stack, TORCH_ABI_VERSION, TORCH_VERSION_2_14_0);
      },
      +[](const char* op, StableIValue* stack) {
        return torch_call_dispatcher_v2(
            op, "", stack, TORCH_ABI_VERSION, TORCH_FEATURE_VERSION);
      },
      +[](const char* op, StableIValue* stack) {
        return torch::stable::call_dispatcher(op, "", stack);
      }};
  const std::vector<std::optional<int64_t>> values = {
      INT64_MIN, std::nullopt, INT64_MAX};
  for (size_t i = 0; i < std::size(dispatchers); ++i) {
    SCOPED_TRACE(i);
    for (const int64_t value : {INT64_MIN, INT64_C(-1), INT64_MAX}) {
      SCOPED_TRACE(value);
      StableIValue stack[] = {from(value)};
      ASSERT_EQ(
          dispatchers[i]("test_stable_dispatcher::identity", stack),
          AOTI_TORCH_SUCCESS);
      EXPECT_EQ(to<int64_t>(stack[0]), value);
    }
    StableIValue stack[] = {from(values)};
    ASSERT_EQ(
        dispatchers[i]("test_stable_dispatcher::list", stack),
        AOTI_TORCH_SUCCESS);
    EXPECT_EQ(to<std::vector<std::optional<int64_t>>>(stack[0]), values);
  }
}

TEST(TorchStableIValue, DispatcherRejectsTargetAboveBuild) {
  StableIValue stack[] = {torch::stable::detail::from(INT64_C(7))};
  EXPECT_EQ(
      torch_call_dispatcher_v2(
          "test_stable_dispatcher::identity",
          "",
          stack,
          TORCH_VERSION_2_14_0,
          TORCH_VERSION_2_16_0),
      AOTI_TORCH_FAILURE);
  const auto* error = torch_exception_get_what_without_backtrace();
  ASSERT_NE(error, nullptr);
  EXPECT_NE(
      std::string(error).find(
          "extension_target_version must be <= extension_build_version"),
      std::string::npos);
  EXPECT_EQ(torch::stable::detail::to<int64_t>(stack[0]), 7);
}

#endif // TORCH_FEATURE_VERSION >= TORCH_VERSION_2_16_0
