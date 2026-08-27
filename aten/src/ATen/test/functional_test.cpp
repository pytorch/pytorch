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

struct CopyOnly {
  explicit CopyOnly(int value) : value(value) {}

  CopyOnly(const CopyOnly&) = default;
  CopyOnly& operator=(const CopyOnly&) = default;
  CopyOnly(CopyOnly&&) = delete;
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
