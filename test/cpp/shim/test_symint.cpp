#include <gtest/gtest.h>

#include <c10/core/SymInt.h>
#include <c10/core/SymIntArrayRef.h>
#include <torch/csrc/stable/stableivalue_conversions.h>
#include <torch/library.h>

#include <cstring>
#include <limits>

namespace {

using torch::stable::SymInt;
using torch::stable::detail::from;
using torch::stable::detail::to;

class SymbolicIntNode final : public c10::SymNodeImpl {
 public:
  bool is_int() override {
    return true;
  }
};

SymInt stable_from_native(c10::SymInt value) {
  int64_t word;
  std::memcpy(&word, &value, sizeof(word));
  if (value.is_heap_allocated()) {
    std::move(value).release();
  }
  return SymInt::reclaim(word);
}

TORCH_LIBRARY(test_stable_symint, m) {
  m.def("identity(SymInt value) -> SymInt", [](c10::SymInt value) {
    return value;
  });
  m.def("list(SymInt[] values) -> SymInt[]", [](c10::SymIntArrayRef values) {
    return values.vec();
  });
  m.def(
      "optional(SymInt? value) -> SymInt?",
      [](std::optional<c10::SymInt> value) { return value; });
  m.def("fail(SymInt value) -> ()", [](c10::SymInt) {
    TORCH_CHECK(false, "SymInt dispatcher test failure");
  });
}

TEST(TorchSymInt, InlineRepresentation) {
  // Keep the expected words independent of c10's representation constants.
  for (const int64_t value :
       {INT64_C(0), INT64_C(1), INT64_C(-1), -(INT64_C(1) << 62), INT64_MAX}) {
    SCOPED_TRACE(value);
    auto stable = stable_from_native(c10::SymInt(value));
    EXPECT_EQ(stable.get(), value);
    EXPECT_EQ(SymInt(value).get(), value);
  }
}

TEST(TorchSymInt, HeapRepresentation) {
  c10::SymNode node = c10::make_intrusive<SymbolicIntNode>();
  auto stable = stable_from_native(c10::SymInt(node));
  const auto address =
      static_cast<uint64_t>(reinterpret_cast<uintptr_t>(node.get()));
  EXPECT_EQ(
      static_cast<uint64_t>(stable.get()),
      address | UINT64_C(0xa000000000000000));
}

TEST(TorchSymInt, HeapAllocatedIntegers) {
  for (const int64_t value :
       {-(INT64_C(1) << 62) - 1, std::numeric_limits<int64_t>::min()}) {
    SCOPED_TRACE(value);
    SymInt stable(value);
    StableIValue stack[] = {from(stable)};
    ASSERT_EQ(
        torch_call_dispatcher(
            "test_stable_symint::identity", "", stack, TORCH_ABI_VERSION),
        AOTI_TORCH_SUCCESS);
    stable = to<SymInt>(stack[0]);
    c10::SymInt native(c10::SymInt::UNCHECKED, SymInt(stable).release());
    EXPECT_TRUE(native.is_heap_allocated());
    EXPECT_EQ(native.expect_int(), value);
    const auto address = static_cast<uint64_t>(
        reinterpret_cast<uintptr_t>(native.toSymNodeImplUnowned()));
    EXPECT_EQ(
        static_cast<uint64_t>(stable.get()),
        address | UINT64_C(0xa000000000000000));
  }
}

TEST(TorchSymInt, Ownership) {
  c10::SymNode node = c10::make_intrusive<SymbolicIntNode>();
  {
    auto stable = stable_from_native(c10::SymInt(node));
    EXPECT_EQ(node.use_count(), 2);
    auto copy = stable;
    EXPECT_EQ(node.use_count(), 3);
    auto moved = std::move(copy);
    EXPECT_EQ(copy.get(), 0);
    EXPECT_EQ(node.use_count(), 3);

    SymInt assigned;
    assigned = stable;
    EXPECT_EQ(node.use_count(), 4);
    assigned = std::move(moved);
    EXPECT_EQ(moved.get(), 0);
    EXPECT_EQ(node.use_count(), 3);

    const auto word = from(stable);
    EXPECT_EQ(node.use_count(), 4);
    auto unboxed = to<SymInt>(word);
    EXPECT_EQ(node.use_count(), 4);

    assigned = SymInt(7);
    EXPECT_EQ(node.use_count(), 3);
  }
  EXPECT_EQ(node.use_count(), 1);
}

TEST(TorchSymInt, DispatcherOwnership) {
  c10::SymNode node = c10::make_intrusive<SymbolicIntNode>();
  {
    auto stable = stable_from_native(c10::SymInt(node));
    for (const auto version : {TORCH_VERSION_2_15_0, TORCH_ABI_VERSION}) {
      SCOPED_TRACE(version);
      StableIValue stack[] = {from(stable)};
      ASSERT_EQ(
          torch_call_dispatcher(
              "test_stable_symint::identity", "", stack, version),
          AOTI_TORCH_SUCCESS);
      auto result = to<SymInt>(stack[0]);
      EXPECT_EQ(node.use_count(), 3);
    }
    StableIValue stack[] = {from(stable)};
    EXPECT_EQ(
        torch_call_dispatcher(
            "test_stable_symint::fail", "", stack, TORCH_ABI_VERSION),
        AOTI_TORCH_FAILURE);
    EXPECT_EQ(node.use_count(), 2);
  }
  EXPECT_EQ(node.use_count(), 1);
}

TEST(TorchSymInt, DispatcherIntegerCompatibility) {
  for (const auto version : {TORCH_VERSION_2_15_0, TORCH_ABI_VERSION}) {
    SCOPED_TRACE(version);
    for (const int64_t value :
         {INT64_C(0), INT64_C(-1), -(INT64_C(1) << 62), INT64_MAX}) {
      SCOPED_TRACE(value);
      StableIValue stack[] = {from(value)};
      ASSERT_EQ(
          torch_call_dispatcher(
              "test_stable_symint::identity", "", stack, version),
          AOTI_TORCH_SUCCESS);
      EXPECT_EQ(to<int64_t>(stack[0]), value);
    }
  }
}

TEST(TorchSymInt, DispatcherContainerOwnership) {
  c10::SymNode node = c10::make_intrusive<SymbolicIntNode>();
  {
    auto stable = stable_from_native(c10::SymInt(node));
    StableIValue list[] = {from(std::vector<SymInt>{stable, SymInt(3)})};
    ASSERT_EQ(
        torch_call_dispatcher(
            "test_stable_symint::list", "", list, TORCH_ABI_VERSION),
        AOTI_TORCH_SUCCESS);
    auto result = to<std::vector<SymInt>>(list[0]);
    EXPECT_EQ(node.use_count(), 3);

    for (bool present : {false, true}) {
      const auto value = present ? std::make_optional(stable) : std::nullopt;
      StableIValue optional[] = {from(value)};
      ASSERT_EQ(
          torch_call_dispatcher(
              "test_stable_symint::optional", "", optional, TORCH_ABI_VERSION),
          AOTI_TORCH_SUCCESS);
      auto returned = to<std::optional<SymInt>>(optional[0]);
      ASSERT_EQ(returned.has_value(), present);
      EXPECT_EQ(node.use_count(), present ? 5 : 3);
    }
  }
  EXPECT_EQ(node.use_count(), 1);
}

} // namespace
