#include <gtest/gtest.h>

#include <c10/core/ConstantSymNodeImpl.h>
#include <c10/core/Scalar.h>
#include <c10/core/SymBool.h>
#include <c10/core/SymInt.h>
#include <c10/core/SymNodeImpl.h>
#include <c10/util/intrusive_ptr.h>

#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>

using namespace c10;

#ifndef C10_MOBILE
namespace {

template <typename T>
class SymbolicConstantNode final : public ConstantSymNodeImpl<T> {
 public:
  using ConstantSymNodeImpl<T>::ConstantSymNodeImpl;

  std::optional<int64_t> constant_int() override {
    return std::nullopt;
  }
  std::optional<bool> constant_bool() override {
    return std::nullopt;
  }
  bool is_constant() override {
    return false;
  }
  bool is_symbolic() override {
    return true;
  }
};

} // namespace
#endif // C10_MOBILE

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

TEST(ScalarTest, SignedIntegerWrappingIsExplicit) {
  EXPECT_EQ(Scalar(-1).toByte(), 255);
  EXPECT_EQ(Scalar(-255).toByte(), 1);
  EXPECT_EQ(Scalar(-1).toUInt16(), 65535);
  EXPECT_EQ(Scalar(-65535).toUInt16(), 1);
  EXPECT_EQ(Scalar(-1).toUInt32(), 4294967295u);
  EXPECT_EQ(Scalar(-4294967295LL).toUInt32(), 1);
  EXPECT_EQ(Scalar(-1).toUInt64(), std::numeric_limits<uint64_t>::max());
}

TEST(ScalarTest, SignedIntegerWrappingRetainsOverflowRejection) {
  EXPECT_THROW(Scalar(-256).toByte(), std::runtime_error);
  EXPECT_THROW(Scalar(256).toByte(), std::runtime_error);
  EXPECT_THROW(Scalar(-65536).toUInt16(), std::runtime_error);
  EXPECT_THROW(Scalar(65536).toUInt16(), std::runtime_error);
  EXPECT_THROW(Scalar(-4294967296LL).toUInt32(), std::runtime_error);
  EXPECT_THROW(Scalar(4294967296LL).toUInt32(), std::runtime_error);
}

TEST(ScalarTest, BooleanUnsignedConversion) {
  EXPECT_EQ(Scalar(false).toByte(), 0);
  EXPECT_EQ(Scalar(true).toByte(), 1);
  EXPECT_EQ(Scalar(false).toUInt16(), 0);
  EXPECT_EQ(Scalar(true).toUInt32(), 1);
}

#ifndef C10_MOBILE
TEST(ScalarTest, SymbolicIntegerWrapping) {
  const SymNode node = make_intrusive<SymbolicConstantNode<int64_t>>(-1);
  const Scalar value{SymInt(node)};
  ASSERT_TRUE(value.isSymInt());
  EXPECT_EQ(value.toByte(), 255);
  EXPECT_EQ(value.toUInt16(), 65535);
  EXPECT_EQ(value.toUInt32(), 4294967295u);

  const SymNode overflowNode =
      make_intrusive<SymbolicConstantNode<int64_t>>(-256);
  const Scalar tooSmall{SymInt(overflowNode)};
  ASSERT_TRUE(tooSmall.isSymInt());
  EXPECT_THROW(tooSmall.toByte(), std::runtime_error);
}

TEST(ScalarTest, SymbolicBooleanUnsignedConversion) {
  for (const bool input : {false, true}) {
    const SymNode node = make_intrusive<SymbolicConstantNode<bool>>(input);
    const Scalar value{SymBool(node)};
    ASSERT_TRUE(value.isSymBool());
    EXPECT_EQ(value.toByte(), input);
    EXPECT_EQ(value.toUInt16(), input);
    EXPECT_EQ(value.toUInt32(), input);
    EXPECT_EQ(value.toUInt64(), input);
  }
}
#endif // C10_MOBILE

TEST(ScalarTest, Equality) {
  ASSERT_TRUE(Scalar(static_cast<uint64_t>(0xFFFFFFFFFFFFFFFF))
                  .equal(0xFFFFFFFFFFFFFFFF));
  ASSERT_FALSE(Scalar(0).equal(0xFFFFFFFFFFFFFFFF));
  // ensure that we don't incorrectly coerce bitrep
  ASSERT_FALSE(Scalar(-1).equal(0xFFFFFFFFFFFFFFFF));
}

TEST(ScalarTest, LongsAndLongLongs) {
  Scalar longOne = 1L;
  Scalar longlongOne = 1LL;
  ASSERT_EQ(longOne.toInt(), longlongOne.toInt());
}
