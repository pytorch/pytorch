#include <gtest/gtest.h>

#include <c10/core/ConstantSymNodeImpl.h>
#include <c10/core/Scalar.h>
#include <c10/core/SymNodeImpl.h>

#include <limits>
#include <optional>

using namespace c10;

namespace {
class ConstantIntPretendingToBeSymbolicSymNodeImpl
    : public ConstantSymNodeImpl<int64_t> {
 public:
  using ConstantSymNodeImpl<int64_t>::ConstantSymNodeImpl;

  std::optional<int64_t> constant_int() override {
    return std::nullopt;
  }

  std::optional<int64_t> maybe_as_int() override {
    return std::nullopt;
  }

  c10::SymNode wrap_int(int64_t num) override {
    return SymNode(
        c10::make_intrusive<ConstantIntPretendingToBeSymbolicSymNodeImpl>(num));
  }

  c10::SymNode wrap_bool(bool b) override {
    return SymNode(c10::make_intrusive<ConstantSymNodeImpl<bool>>(b));
  }

  SymNode eq(const SymNode& other) override {
    return wrap_bool(int_() == other->int_());
  }
};

SymInt create_symbolic_symint(int64_t value) {
  return SymInt(
      SymNode(c10::make_intrusive<ConstantIntPretendingToBeSymbolicSymNodeImpl>(
          value)));
}
} // namespace

TEST(ScalarTest, UnsignedConstructor) {
  uint16_t x = 0xFFFF;
  uint32_t y = 0xFFFFFFFF;
  uint64_t z0 = 0;
  uint64_t z1 = 0x7FFFFFFFFFFFFFFF;
  uint64_t z2 = 0xFFFFFFFFFFFFFFFF;
  auto sx = Scalar(x);
  auto sy = Scalar(y);
  auto sz0 = Scalar(z0);
  auto sz1 = Scalar(z1);
  auto sz2 = Scalar(z2);
  ASSERT_TRUE(sx.isIntegral(false));
  ASSERT_TRUE(sy.isIntegral(false));
  ASSERT_TRUE(sz0.isIntegral(false));
  ASSERT_TRUE(sz1.isIntegral(false));
  ASSERT_TRUE(sz2.isIntegral(false));
  ASSERT_EQ(sx.type(), ScalarType::Long);
  ASSERT_EQ(sy.type(), ScalarType::Long);
  ASSERT_EQ(sz0.type(), ScalarType::Long);
  ASSERT_EQ(sz1.type(), ScalarType::Long);
  ASSERT_EQ(sz2.type(), ScalarType::UInt64);
  ASSERT_EQ(sx.toUInt16(), x);
  ASSERT_EQ(sx.toInt(), x);
  ASSERT_EQ(sy.toUInt32(), y);
  EXPECT_THROW(sy.toInt(), std::runtime_error); // overflows
  ASSERT_EQ(sy.toLong(), y);
  ASSERT_EQ(sz0.toUInt64(), z0);
  ASSERT_EQ(sz0.toInt(), z0);
  ASSERT_EQ(sz1.toUInt64(), z1);
  EXPECT_THROW(sz1.toInt(), std::runtime_error); // overflows
  ASSERT_EQ(sz1.toLong(), z1);
  ASSERT_EQ(sz2.toUInt64(), z2);
  EXPECT_THROW(sz2.toInt(), std::runtime_error); // overflows
  EXPECT_THROW(sz2.toLong(), std::runtime_error); // overflows
}

TEST(ScalarTest, Equality) {
  ASSERT_TRUE(Scalar(static_cast<uint64_t>(0xFFFFFFFFFFFFFFFF))
                  .equal(0xFFFFFFFFFFFFFFFF));
  ASSERT_FALSE(Scalar(0).equal(0xFFFFFFFFFFFFFFFF));
  // ensure that we don't incorrectly coerce bitrep
  ASSERT_FALSE(Scalar(-1).equal(0xFFFFFFFFFFFFFFFF));
}

TEST(ScalarTest, SymIntEquality) {
  auto scalar = Scalar(create_symbolic_symint(2));
  ASSERT_TRUE(scalar.isSymbolic());
  ASSERT_TRUE(scalar.equal(2));
  ASSERT_TRUE(scalar.equal(2.0));
  ASSERT_TRUE(scalar.equal(c10::complex<double>(2.0, 0.0)));
  ASSERT_FALSE(scalar.equal(3));
  ASSERT_FALSE(scalar.equal(2.5));
  ASSERT_FALSE(scalar.equal(std::numeric_limits<double>::infinity()));
  ASSERT_FALSE(scalar.equal(c10::complex<double>(2.0, 1.0)));

  auto min_scalar =
      Scalar(create_symbolic_symint(std::numeric_limits<int64_t>::min()));
  ASSERT_TRUE(min_scalar.equal(
      static_cast<double>(std::numeric_limits<int64_t>::min())));
  const auto int64_upper =
      -static_cast<double>(std::numeric_limits<int64_t>::min());
  ASSERT_FALSE(scalar.equal(int64_upper));
  ASSERT_FALSE(min_scalar.equal(int64_upper));
}

TEST(ScalarTest, LongsAndLongLongs) {
  Scalar longOne = 1L;
  Scalar longlongOne = 1LL;
  ASSERT_EQ(longOne.toInt(), longlongOne.toInt());
}
