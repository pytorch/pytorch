#include <c10/util/safe_conv.h>

#include <c10/util/TypeCast.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <tuple>

namespace c10 {
namespace {

TEST(safeConvTest, InRangePassesThrough) {
  EXPECT_EQ(safe_conv<int32_t>(int64_t{5}), 5);
  EXPECT_EQ(
      safe_conv<int32_t>(int64_t{std::numeric_limits<int32_t>::max()}),
      std::numeric_limits<int32_t>::max());
  EXPECT_EQ(
      safe_conv<int32_t>(int64_t{std::numeric_limits<int32_t>::min()}),
      std::numeric_limits<int32_t>::min());
}

TEST(safeConvTest, SameTypeIsNoOp) {
  EXPECT_EQ(safe_conv<int32_t>(int32_t{42}), 42);
  EXPECT_EQ(
      safe_conv<uint64_t>(std::numeric_limits<uint64_t>::max()),
      std::numeric_limits<uint64_t>::max());
}

TEST(safeConvTest, OverflowHighThrows) {
  constexpr int64_t tooBig = int64_t{std::numeric_limits<int32_t>::max()} + 1;
  EXPECT_THROW(std::ignore = safe_conv<int32_t>(tooBig), std::runtime_error);
}

TEST(safeConvTest, OverflowLowThrows) {
  constexpr int64_t tooSmall = int64_t{std::numeric_limits<int32_t>::min()} - 1;
  EXPECT_THROW(std::ignore = safe_conv<int32_t>(tooSmall), std::runtime_error);
}

// The key differentiator from unsafe_wrapping_convert: a negative value
// converted to an unsigned type is REJECTED, never wrapped to a large positive
// value.
TEST(safeConvTest, NegativeToUnsignedThrows) {
  EXPECT_THROW(
      std::ignore = safe_conv<uint8_t>(int32_t{-1}), std::runtime_error);
  EXPECT_THROW(
      std::ignore = safe_conv<uint32_t>(int64_t{-1}), std::runtime_error);
}

TEST(safeConvTest, UnsignedToUnsignedBoundary) {
  EXPECT_EQ(safe_conv<uint32_t>(int64_t{0}), 0u);
  EXPECT_EQ(
      safe_conv<uint32_t>(int64_t{std::numeric_limits<uint32_t>::max()}),
      std::numeric_limits<uint32_t>::max());
  constexpr int64_t tooBig = int64_t{std::numeric_limits<uint32_t>::max()} + 1;
  EXPECT_THROW(std::ignore = safe_conv<uint32_t>(tooBig), std::runtime_error);
}

TEST(safeConvTest, UnsignedToSignedOverflowThrows) {
  EXPECT_EQ(safe_conv<int8_t>(uint32_t{100}), 100);
  EXPECT_THROW(
      std::ignore = safe_conv<int8_t>(uint32_t{200}), std::runtime_error);
}

// unsafe_wrapping_convert preserves the historical signed->unsigned
// two's-complement wrap that safe_conv rejects.
TEST(safeConvTest, UnsafeWrappingConvertPreservesWrap) {
  EXPECT_EQ(unsafe_wrapping_convert<uint8_t>(int64_t{-1}, "uint8_t"), 255);
  EXPECT_EQ(unsafe_wrapping_convert<uint16_t>(int64_t{-1}, "uint16_t"), 65535);
  EXPECT_EQ(
      unsafe_wrapping_convert<uint32_t>(int64_t{-1}, "uint32_t"), 4294967295u);
  // Wrap applies only to unsigned targets; signed targets still range-check.
  constexpr int64_t tooBig = int64_t{std::numeric_limits<int32_t>::max()} + 1;
  EXPECT_THROW(
      unsafe_wrapping_convert<int32_t>(tooBig, "int32_t"), std::runtime_error);
}

TEST(safeConvTest, NamedOverflowMessage) {
  try {
    std::ignore = c10::safe_conv<uint8_t>(int64_t{-1}, "uint8_t");
    FAIL() << "Expected strict integer conversion overflow";
  } catch (const std::runtime_error& error) {
    EXPECT_STREQ(
        error.what(),
        "value cannot be converted to type uint8_t without overflow");
  }
}

TEST(safeConvTest, NamedConversionBounds) {
  EXPECT_EQ(c10::safe_conv<uint8_t>(int64_t{0}, "uint8_t"), 0);
  EXPECT_EQ(c10::safe_conv<uint8_t>(int64_t{255}, "uint8_t"), 255);
  EXPECT_THROW(
      std::ignore = c10::safe_conv<uint8_t>(int64_t{256}, "uint8_t"),
      std::runtime_error);
  EXPECT_THROW(
      std::ignore = c10::safe_conv<int8_t>(int64_t{-129}, "int8_t"),
      std::runtime_error);
}

TEST(safeConvTest, UnnamedOverflowMessage) {
  try {
    std::ignore = safe_conv<uint8_t>(int64_t{-1});
    FAIL() << "Expected strict integer conversion overflow";
  } catch (const std::runtime_error& error) {
    EXPECT_STREQ(
        error.what(), "value cannot be safely converted without overflow");
  }
}

TEST(safeConvTest, CheckedConvertRejectsNegativeUnsignedTargets) {
  EXPECT_THROW(checked_convert<uint8_t>(-1, "uint8_t"), std::runtime_error);
  EXPECT_THROW(checked_convert<uint16_t>(-1, "uint16_t"), std::runtime_error);
  EXPECT_THROW(checked_convert<uint32_t>(-1, "uint32_t"), std::runtime_error);
  EXPECT_THROW(checked_convert<uint64_t>(-1, "uint64_t"), std::runtime_error);
}

TEST(safeConvTest, CheckedConvertUnsignedBounds) {
  EXPECT_EQ(checked_convert<uint8_t>(0, "uint8_t"), 0);
  EXPECT_EQ(checked_convert<uint8_t>(255, "uint8_t"), 255);
  EXPECT_THROW(checked_convert<uint8_t>(256, "uint8_t"), std::runtime_error);
  EXPECT_EQ(
      checked_convert<uint64_t>(
          std::numeric_limits<uint64_t>::max(), "uint64_t"),
      std::numeric_limits<uint64_t>::max());
  EXPECT_THROW(
      checked_convert<int64_t>(std::numeric_limits<uint64_t>::max(), "int64_t"),
      std::runtime_error);
}

TEST(safeConvTest, CheckedConvertPreservesGeneralConversions) {
  EXPECT_EQ(checked_convert<int32_t>(3.9, "int32_t"), 3);
  EXPECT_THROW(checked_convert<uint8_t>(-1.0, "uint8_t"), std::runtime_error);
  EXPECT_EQ(
      checked_convert<int32_t>(c10::complex<double>{3.9, 0}, "int32_t"), 3);
  EXPECT_EQ(checked_convert<uint8_t>(true, "uint8_t"), 1);
  EXPECT_TRUE(checked_convert<bool>(-1, "bool"));
  EXPECT_THROW(checked_convert<int32_t>(1e18, "int32_t"), std::runtime_error);
}

TEST(safeConvTest, CheckedConvertPreservesExceptionMessage) {
  try {
    (void)checked_convert<uint8_t>(-1, "uint8_t");
    FAIL() << "Expected checked integer conversion overflow";
  } catch (const std::runtime_error& error) {
    EXPECT_STREQ(
        error.what(),
        "value cannot be converted to type uint8_t without overflow");
  }
}

TEST(safeConvTest, HeaderOnlyCheckedConvertIsStrict) {
  using torch::headeronly::checked_convert;
  EXPECT_THROW(checked_convert<uint8_t>(-1, "uint8_t"), std::runtime_error);
  EXPECT_EQ(checked_convert<uint8_t>(255, "uint8_t"), 255);
}

} // namespace
} // namespace c10
