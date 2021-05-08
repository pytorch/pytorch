#include <c10/util/Optional.h>

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <string>

namespace {

template <typename T>
class OptionalTest : public ::testing::Test {
 public:
  using optional = c10::optional<T>;
};

template <typename T>
T getSampleValue();

template <>
bool getSampleValue() {
  return true;
}

template <>
uint64_t getSampleValue() {
  return 42;
}

template <>
std::string getSampleValue() {
  return "hello";
}

using OptionalTypes = ::testing::Types<
    // 32-bit scalar optimization.
    bool,
    // Trivially destructible but not 32-bit scalar.
    uint64_t,
    // Non-trivial destructor.
    std::string>;

TYPED_TEST_CASE(OptionalTest, OptionalTypes);

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TYPED_TEST(OptionalTest, Empty) {
  typename TestFixture::optional empty;

  EXPECT_FALSE((bool)empty);
  EXPECT_FALSE(empty.has_value());

  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  EXPECT_THROW(empty.value(), c10::bad_optional_access);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TYPED_TEST(OptionalTest, Initialized) {
  using optional = typename TestFixture::optional;

  const auto val = getSampleValue<TypeParam>();
  optional opt((val));
  auto copy(opt), moveFrom1(opt), moveFrom2(opt);
  optional move(std::move(moveFrom1));
  optional copyAssign;
  copyAssign = opt;
  optional moveAssign;
  moveAssign = std::move(moveFrom2);

  std::array<typename TestFixture::optional*, 5> opts = {
      &opt, &copy, &copyAssign, &move, &moveAssign};
  for (auto* popt : opts) {
    auto& opt = *popt;
    EXPECT_TRUE((bool)opt);
    EXPECT_TRUE(opt.has_value());

    EXPECT_EQ(opt.value(), val);
    EXPECT_EQ(*opt, val);
  }
}

} // namespace
