#include <gtest/gtest.h>

#include <c10/core/SymInt.h>
#include <c10/core/SymIntArrayRef.h>
#include <torch/csrc/stable/stableivalue_conversions.h>
#include <torch/library.h>

#include <cstring>
#include <limits>
#include <optional>

namespace {

using torch::stable::SymInt;
using torch::stable::detail::from;
using torch::stable::detail::to;

enum class IntegerAPI { Aoti, OldBuild, NewBuild, LowerTarget };

constexpr IntegerAPI integer_apis[] = {
    IntegerAPI::Aoti,
    IntegerAPI::OldBuild,
    IntegerAPI::NewBuild,
    IntegerAPI::LowerTarget};

AOTITorchError call_integer_dispatcher(
    const char* op,
    StableIValue* stack,
    IntegerAPI api) {
  switch (api) {
    case IntegerAPI::Aoti:
      return aoti_torch_call_dispatcher(op, "", stack);
    case IntegerAPI::OldBuild:
      return torch_call_dispatcher(op, "", stack, TORCH_VERSION_2_14_0);
    case IntegerAPI::NewBuild:
      return torch_call_dispatcher(op, "", stack, TORCH_ABI_VERSION);
    case IntegerAPI::LowerTarget:
      return torch_call_dispatcher_v2(
          op, "", stack, TORCH_ABI_VERSION, TORCH_VERSION_2_14_0);
  }
  return AOTI_TORCH_FAILURE;
}

AOTITorchError register_integer_kernel(
    torch::Library& library,
    const char* name,
    IntegerAPI api) {
  auto handle = reinterpret_cast<TorchLibraryHandle>(&library);
  auto fn = +[](StableIValue* stack, uint64_t args, uint64_t returns) {
    EXPECT_EQ(args, 1);
    EXPECT_EQ(returns, 1);
    stack[0] = from(to<int64_t>(stack[0]));
  };
  switch (api) {
    case IntegerAPI::Aoti:
      return aoti_torch_library_impl(handle, name, fn);
    case IntegerAPI::OldBuild:
      return torch_library_impl(handle, name, fn, TORCH_VERSION_2_14_0);
    case IntegerAPI::NewBuild:
      return torch_library_impl(handle, name, fn, TORCH_ABI_VERSION);
    case IntegerAPI::LowerTarget:
      return torch_library_impl_v2(
          handle, name, fn, TORCH_ABI_VERSION, TORCH_VERSION_2_14_0);
  }
  return AOTI_TORCH_FAILURE;
}

class SymbolicIntNode final : public c10::SymNodeImpl {
 public:
  explicit SymbolicIntNode(std::optional<int64_t> hint = std::nullopt)
      : hint_(hint) {}

  bool is_int() override {
    return true;
  }

  int64_t guard_int(const char*, int64_t) override {
    ++guards;
    TORCH_CHECK(hint_.has_value(), "unresolved symbolic integer");
    return *hint_;
  }

  int guards = 0;

 private:
  std::optional<int64_t> hint_;
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
        torch_call_dispatcher_v2(
            "test_stable_symint::identity",
            "",
            stack,
            TORCH_ABI_VERSION,
            TORCH_VERSION_2_16_0),
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
    {
      StableIValue stack[] = {from(stable)};
      ASSERT_EQ(
          torch_call_dispatcher_v2(
              "test_stable_symint::identity",
              "",
              stack,
              TORCH_ABI_VERSION,
              TORCH_VERSION_2_16_0),
          AOTI_TORCH_SUCCESS);
      auto result = to<SymInt>(stack[0]);
      EXPECT_EQ(node.use_count(), 3);
    }
    StableIValue stack[] = {from(stable)};
    EXPECT_EQ(
        torch_call_dispatcher_v2(
            "test_stable_symint::fail",
            "",
            stack,
            TORCH_ABI_VERSION,
            TORCH_VERSION_2_16_0),
        AOTI_TORCH_FAILURE);
    EXPECT_EQ(node.use_count(), 2);
  }
  EXPECT_EQ(node.use_count(), 1);
}

TEST(TorchSymInt, DispatcherIntegerCompatibility) {
  for (const auto api : integer_apis) {
    SCOPED_TRACE(static_cast<int>(api));
    for (const int64_t value :
         {INT64_C(0),
          INT64_C(-1),
          -(INT64_C(1) << 62),
          -(INT64_C(1) << 62) - 1,
          INT64_MIN,
          INT64_MAX}) {
      SCOPED_TRACE(value);
      StableIValue stack[] = {from(value)};
      ASSERT_EQ(
          call_integer_dispatcher("test_stable_symint::identity", stack, api),
          AOTI_TORCH_SUCCESS);
      EXPECT_EQ(to<int64_t>(stack[0]), value);
    }
  }
}

TEST(TorchSymInt, DispatcherIntegerContainers) {
  const std::vector<int64_t> values = {INT64_MIN, -1, INT64_MAX};
  for (const auto api : integer_apis) {
    SCOPED_TRACE(static_cast<int>(api));
    StableIValue list[] = {from(values)};
    ASSERT_EQ(
        call_integer_dispatcher("test_stable_symint::list", list, api),
        AOTI_TORCH_SUCCESS);
    EXPECT_EQ(to<std::vector<int64_t>>(list[0]), values);

    for (const auto& value :
         {std::optional<int64_t>(INT64_MIN), std::optional<int64_t>()}) {
      StableIValue optional[] = {from(value)};
      ASSERT_EQ(
          call_integer_dispatcher(
              "test_stable_symint::optional", optional, api),
          AOTI_TORCH_SUCCESS);
      EXPECT_EQ(to<std::optional<int64_t>>(optional[0]), value);
    }
  }
}

TEST(TorchSymInt, NativeSymbolicReturns) {
  for (const bool guardable : {false, true}) {
    auto node = c10::make_intrusive<SymbolicIntNode>(
        guardable ? std::optional<int64_t>(INT64_MIN) : std::nullopt);
    {
      auto library = MAKE_TORCH_LIBRARY(test_stable_symint_returns);
      library.def("value() -> SymInt", [node]() {
        return c10::SymInt(c10::SymNode(node));
      });
      EXPECT_EQ(node.use_count(), 2);
      for (const auto api : integer_apis) {
        SCOPED_TRACE(static_cast<int>(api));
        StableIValue stack[1];
        const auto result = call_integer_dispatcher(
            "test_stable_symint_returns::value", stack, api);
        if (guardable) {
          ASSERT_EQ(result, AOTI_TORCH_SUCCESS);
          EXPECT_EQ(to<int64_t>(stack[0]), INT64_MIN);
        } else {
          EXPECT_EQ(result, AOTI_TORCH_FAILURE);
        }
        EXPECT_EQ(node.use_count(), 2);
      }
      EXPECT_EQ(node->guards, 4);
      StableIValue stack[1];
      ASSERT_EQ(
          torch_call_dispatcher_v2(
              "test_stable_symint_returns::value",
              "",
              stack,
              TORCH_ABI_VERSION,
              TORCH_VERSION_2_16_0),
          AOTI_TORCH_SUCCESS);
      {
        auto result = to<SymInt>(stack[0]);
        EXPECT_EQ(node.use_count(), 3);
        EXPECT_EQ(node->guards, 4);
      }
      EXPECT_EQ(node.use_count(), 2);
    }
    EXPECT_EQ(node.use_count(), 1);
  }
}

TEST(TorchSymInt, RegisteredIntegerCallbacks) {
  for (const auto api : integer_apis) {
    SCOPED_TRACE(static_cast<int>(api));
    auto library = MAKE_TORCH_LIBRARY(test_stable_symint_callbacks);
    library.def("identity(SymInt value) -> SymInt");
    ASSERT_EQ(
        register_integer_kernel(library, "identity", api), AOTI_TORCH_SUCCESS);
    auto op = c10::Dispatcher::singleton().findSchemaOrThrow(
        "test_stable_symint_callbacks::identity", "");
    for (const int64_t value : {INT64_MIN, INT64_C(-1), INT64_MAX}) {
      torch::jit::Stack stack{c10::IValue(c10::SymInt(value))};
      op.callBoxed(stack);
      ASSERT_EQ(stack.size(), 1);
      EXPECT_EQ(stack[0].toSymInt().expect_int(), value);
    }
    for (const bool guardable : {false, true}) {
      auto node = c10::make_intrusive<SymbolicIntNode>(
          guardable ? std::optional<int64_t>(INT64_MIN) : std::nullopt);
      {
        torch::jit::Stack stack{c10::IValue(c10::SymInt(c10::SymNode(node)))};
        EXPECT_EQ(node.use_count(), 2);
        if (guardable) {
          op.callBoxed(stack);
          ASSERT_EQ(stack.size(), 1);
          EXPECT_EQ(stack[0].toSymInt().expect_int(), INT64_MIN);
        } else {
          EXPECT_THROW(op.callBoxed(stack), c10::Error);
        }
        EXPECT_EQ(node->guards, 1);
      }
      EXPECT_EQ(node.use_count(), 1);
    }
  }
}

TEST(TorchSymInt, RegisteredOwnership) {
  auto library = MAKE_TORCH_LIBRARY(test_stable_symint_owning_callback);
  library.def("identity(SymInt value) -> SymInt");
  auto fn = +[](StableIValue* stack, uint64_t, uint64_t) {
    stack[0] = from(to<SymInt>(stack[0]));
  };
  ASSERT_EQ(
      torch_library_impl_v2(
          reinterpret_cast<TorchLibraryHandle>(&library),
          "identity",
          fn,
          TORCH_ABI_VERSION,
          TORCH_VERSION_2_16_0),
      AOTI_TORCH_SUCCESS);
  auto op = c10::Dispatcher::singleton().findSchemaOrThrow(
      "test_stable_symint_owning_callback::identity", "");
  auto node = c10::make_intrusive<SymbolicIntNode>();
  {
    torch::jit::Stack stack{c10::IValue(c10::SymInt(c10::SymNode(node)))};
    op.callBoxed(stack);
    ASSERT_EQ(stack.size(), 1);
    EXPECT_TRUE(stack[0].isSymInt());
    EXPECT_EQ(node.use_count(), 2);
    EXPECT_EQ(node->guards, 0);
  }
  EXPECT_EQ(node.use_count(), 1);
}

TEST(TorchSymInt, RegisteredSharedListOwnership) {
  auto library = MAKE_TORCH_LIBRARY(test_stable_symint_shared_list);
  library.def("consume(SymInt[] left, SymInt[] right) -> ()");
  auto fn = +[](StableIValue* stack, uint64_t nargs, uint64_t nreturns) {
    EXPECT_EQ(nargs, 2);
    EXPECT_EQ(nreturns, 0);
    auto left = to<std::vector<SymInt>>(stack[0]);
    auto right = to<std::vector<SymInt>>(stack[1]);
    ASSERT_EQ(left.size(), 2);
    ASSERT_EQ(right.size(), 2);
    EXPECT_EQ(left[0].get(), right[0].get());
    EXPECT_EQ(left[1].get(), 3);
    EXPECT_EQ(right[1].get(), 3);
  };
  ASSERT_EQ(
      torch_library_impl_v2(
          reinterpret_cast<TorchLibraryHandle>(&library),
          "consume",
          fn,
          TORCH_ABI_VERSION,
          TORCH_VERSION_2_16_0),
      AOTI_TORCH_SUCCESS);
  auto op = c10::Dispatcher::singleton().findSchemaOrThrow(
      "test_stable_symint_shared_list::consume", "");
  auto node = c10::make_intrusive<SymbolicIntNode>();
  {
    auto values = c10::impl::GenericList(c10::SymIntType::get());
    values.push_back(c10::IValue(c10::SymInt(c10::SymNode(node))));
    values.push_back(c10::IValue(c10::SymInt(3)));
    EXPECT_EQ(node.use_count(), 2);
    torch::jit::Stack stack{c10::IValue(values), c10::IValue(values)};
    ASSERT_NO_THROW(op.callBoxed(stack));
    EXPECT_TRUE(stack.empty());
    ASSERT_EQ(values.size(), 2);
    ASSERT_TRUE(values.get(0).isSymInt());
    EXPECT_TRUE(values.get(0).toSymInt().is_symbolic());
    EXPECT_EQ(values.get(1).toSymInt().expect_int(), 3);
    EXPECT_EQ(node.use_count(), 2);
    EXPECT_EQ(node->guards, 0);
  }
  EXPECT_EQ(node.use_count(), 1);
}

TEST(TorchSymInt, DispatcherContainerOwnership) {
  c10::SymNode node = c10::make_intrusive<SymbolicIntNode>();
  {
    auto stable = stable_from_native(c10::SymInt(node));
    StableIValue list[] = {from(std::vector<SymInt>{stable, SymInt(3)})};
    ASSERT_EQ(
        torch_call_dispatcher_v2(
            "test_stable_symint::list",
            "",
            list,
            TORCH_ABI_VERSION,
            TORCH_VERSION_2_16_0),
        AOTI_TORCH_SUCCESS);
    auto result = to<std::vector<SymInt>>(list[0]);
    ASSERT_EQ(result.size(), 2);
    EXPECT_EQ(result[0].get(), stable.get());
    EXPECT_EQ(result[1].get(), 3);
    EXPECT_EQ(node.use_count(), 3);

    for (bool present : {false, true}) {
      const auto value = present ? std::make_optional(stable) : std::nullopt;
      StableIValue optional[] = {from(value)};
      ASSERT_EQ(
          torch_call_dispatcher_v2(
              "test_stable_symint::optional",
              "",
              optional,
              TORCH_ABI_VERSION,
              TORCH_VERSION_2_16_0),
          AOTI_TORCH_SUCCESS);
      auto returned = to<std::optional<SymInt>>(optional[0]);
      ASSERT_EQ(returned.has_value(), present);
      if (present) {
        EXPECT_EQ(returned->get(), stable.get());
      }
      EXPECT_EQ(node.use_count(), present ? 5 : 3);
    }
  }
  EXPECT_EQ(node.use_count(), 1);
}

} // namespace
