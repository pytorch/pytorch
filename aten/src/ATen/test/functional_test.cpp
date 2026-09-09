#include <ATen/core/functional.h>

#include <gtest/gtest.h>

#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

struct RefOverloadCallable {
  int operator()(int&) const {
    return 1;
  }

  int operator()(int&&) const {
    return 2;
  }
};

// Only move assignment is deleted (not the move constructor): std::vector
// growth requires MoveInsertable even when never actually triggered at
// runtime, and a deleted move ctor would break that; leaving it undeclared
// falls back to the copy ctor while still forcing is_move_assignable_v to
// false.
struct CopyOnly {
  explicit CopyOnly(int value) : value(value) {}

  CopyOnly(const CopyOnly&) = default;
  CopyOnly& operator=(const CopyOnly&) = default;
  CopyOnly& operator=(CopyOnly&&) = delete;

  int value;
};

} // namespace

TEST(FMapTest, RvalueVectorReusesStorageForSameType) {
  std::vector<int> inputs{1, 2, 3};
  const auto* data = inputs.data();

  auto result = c10::fmap(std::move(inputs), [](int input) {
    return input + 1;
  });

  static_assert(std::is_same_v<decltype(result), std::vector<int>>);
  EXPECT_EQ(result.data(), data);
  EXPECT_EQ(result, (std::vector<int>{2, 3, 4}));
}

TEST(FMapTest, RvalueBoolVectorSupportsProxyReferences) {
  std::vector<bool> inputs{true, false, true};

  auto result = c10::fmap(std::move(inputs), [](bool input) { return !input; });

  EXPECT_EQ(result, (std::vector<bool>{false, true, false}));
}

TEST(FMapTest, RvalueVectorConsumesElementsWhenCallableAcceptsValue) {
  std::vector<std::unique_ptr<int>> inputs;
  inputs.emplace_back(std::make_unique<int>(3));
  inputs.emplace_back(std::make_unique<int>(4));

  auto result = c10::fmap(
      std::move(inputs),
      [](std::unique_ptr<int> input) { return *input; });

  EXPECT_EQ(result, (std::vector<int>{3, 4}));
}

TEST(FMapTest, RvalueVectorSupportsConstReferenceCallable) {
  std::vector<int> inputs{1, 2, 3};

  auto result = c10::fmap(
      std::move(inputs),
      [](const int& input) { return static_cast<long>(input); });

  static_assert(std::is_same_v<decltype(result), std::vector<long>>);
  EXPECT_EQ(result, (std::vector<long>{1, 2, 3}));
}

TEST(FMapTest, RvalueVectorFallsBackToLvalueCallable) {
  std::vector<int> inputs{1, 2, 3};

  auto result = c10::fmap(std::move(inputs), [](int& input) {
    input += 1;
    return static_cast<long>(input);
  });

  EXPECT_EQ(result, (std::vector<long>{2, 3, 4}));
}

TEST(FMapTest, RvalueVectorPrefersRvalueOverload) {
  std::vector<int> inputs{1, 2, 3};

  auto result = c10::fmap(std::move(inputs), RefOverloadCallable{});

  EXPECT_EQ(result, (std::vector<int>{2, 2, 2}));
}

TEST(FMapTest, RvalueVectorFallsBackForCopyOnlyByValueCallable) {
  std::vector<CopyOnly> inputs{CopyOnly(5), CopyOnly(6)};

  auto result = c10::fmap(
      std::move(inputs), [](CopyOnly input) { return input.value; });

  EXPECT_EQ(result, (std::vector<int>{5, 6}));
}

TEST(FMapTest, RvalueVectorConstructorConsumesElements) {
  std::vector<std::unique_ptr<int>> inputs;
  inputs.emplace_back(std::make_unique<int>(7));
  inputs.emplace_back(std::make_unique<int>(8));

  auto result = c10::fmap<std::shared_ptr<int>>(std::move(inputs));

  ASSERT_EQ(result.size(), 2);
  EXPECT_EQ(*result[0], 7);
  EXPECT_EQ(*result[1], 8);
}

TEST(FMapTest, RvalueVectorConstructorReusesStorageForSameType) {
  std::vector<int> inputs{1, 2, 3};
  const auto* data = inputs.data();

  auto result = c10::fmap<int>(std::move(inputs));

  EXPECT_EQ(result.data(), data);
  EXPECT_EQ(result, (std::vector<int>{1, 2, 3}));
}

TEST(FilterTest, RvalueVectorReusesStorage) {
  std::vector<int> inputs{1, 2, 3, 4};
  const auto* data = inputs.data();

  auto result =
      c10::filter(std::move(inputs), [](int input) { return input % 2 == 0; });

  EXPECT_EQ(result.data(), data);
  EXPECT_EQ(result, (std::vector<int>{2, 4}));
}

TEST(FilterTest, RvalueVectorSupportsMoveOnlyElements) {
  std::vector<std::unique_ptr<int>> inputs;
  inputs.emplace_back(std::make_unique<int>(1));
  inputs.emplace_back(std::make_unique<int>(2));
  inputs.emplace_back(std::make_unique<int>(3));

  auto result = c10::filter(
      std::move(inputs),
      [](const std::unique_ptr<int>& input) { return *input != 2; });

  ASSERT_EQ(result.size(), 2);
  EXPECT_EQ(*result[0], 1);
  EXPECT_EQ(*result[1], 3);
}

TEST(FilterTest, RvalueVectorFallsBackForCopyOnlyElements) {
  std::vector<CopyOnly> inputs{CopyOnly(1), CopyOnly(2), CopyOnly(3)};

  auto result = c10::filter(
      std::move(inputs), [](const CopyOnly& input) { return input.value != 2; });

  ASSERT_EQ(result.size(), 2);
  EXPECT_EQ(result[0].value, 1);
  EXPECT_EQ(result[1].value, 3);
}

TEST(FilterTest, RvalueBoolVectorSupportsProxyReferences) {
  std::vector<bool> inputs{true, false, true};

  auto result =
      c10::filter(std::move(inputs), [](bool input) { return input; });

  EXPECT_EQ(result, (std::vector<bool>{true, true}));
}
