#include <gtest/gtest.h>

#include <c10/core/ConstantSymNodeImpl.h>
#include <c10/core/SymInt.h>
#include <c10/core/SymNodeImpl.h>
#include <c10/macros/Macros.h>

using namespace c10;
#ifndef C10_MOBILE
static void check(int64_t value) {
  const auto i = SymInt(value);
  EXPECT_EQ(i.maybe_as_int(), std::make_optional(value));
}

TEST(SymIntTest, ConcreteInts) {
  check(INT64_MAX);
  check(0);
  check(-1);
  check(-4611686018427387904LL);
  check(INT64_MIN);
}

TEST(SymIntTest, CheckRange) {
  EXPECT_FALSE(SymInt::check_range(INT64_MIN));
}

#if !C10_UBSAN_ENABLED
// This test fails signed-integer-overflow UBSAN check
TEST(SymIntTest, Overflows) {
  const auto x = SymInt(INT64_MAX);
  EXPECT_NE(-(x + 1), 0);

  const auto y = SymInt(INT64_MIN);
  EXPECT_NE(-y, 0);
  EXPECT_NE(0 - y, 0);
}
#endif

namespace {

// We need a SymNodeImpl that 1) has working arithmetic with
// predictable results and 2) causes SymInt::maybe_as_int to return
// nullopt so that we can hit all 4 cases (zero/one/both arguments
// have null maybe_as_int) in the operator implementations.
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
  // Needs to be implemented for arithmetic to actually
  // work. NestedIntSymNodeImpl does this, for example.
  c10::SymNode wrap_int(int64_t num) override {
    return SymNode(
        c10::make_intrusive<ConstantIntPretendingToBeSymbolicSymNodeImpl>(num));
  }

  c10::SymNode wrap_bool(bool b) override {
    return SymNode(c10::make_intrusive<ConstantSymNodeImpl<bool>>(b));
  }

  SymNode add(const SymNode& other) override {
    return wrap_int(int_() + other->int_());
  }

  SymNode sub(const SymNode& other) override {
    return wrap_int(int_() - other->int_());
  }

  SymNode mul(const SymNode& other) override {
    return wrap_int(int_() * other->int_());
  }

  SymNode floordiv(const SymNode& other) override {
    return wrap_int(int_() / other->int_());
  }

  SymNode sym_min(const SymNode& other) override {
    return wrap_int(std::min(int_(), other->int_()));
  }

  SymNode sym_max(const SymNode& other) override {
    return wrap_int(std::max(int_(), other->int_()));
  }

  SymNode mod(const SymNode& other) override {
    return wrap_int(int_() % other->int_());
  }

  SymNode eq(const SymNode& other) override {
    return wrap_bool(int_() == other->int_());
  }

  SymNode ne(const SymNode& other) override {
    return wrap_bool(int_() != other->int_());
  }

  SymNode lt(const SymNode& other) override {
    return wrap_bool(int_() < other->int_());
  }

  SymNode le(const SymNode& other) override {
    return wrap_bool(int_() <= other->int_());
  }

  SymNode gt(const SymNode& other) override {
    return wrap_bool(int_() > other->int_());
  }

  SymNode ge(const SymNode& other) override {
    return wrap_bool(int_() >= other->int_());
  }
};

SymInt create_symbolic_symint(int64_t value) {
  return SymInt(
      SymNode(c10::make_intrusive<ConstantIntPretendingToBeSymbolicSymNodeImpl>(
          value)));
}

auto unwrap(const SymInt& x) {
  return x.guard_int(__FILE__, __LINE__);
}

auto unwrap(bool b) {
  return b;
}

template <template <typename> class Op>
void test_operator() {
  for (const auto& arg1 : {SymInt(42), create_symbolic_symint(42)}) {
    for (const auto& arg2 : {SymInt(27), create_symbolic_symint(27)}) {
      EXPECT_EQ(unwrap(Op<SymInt>()(arg1, arg2)), Op<int64_t>()(42, 27));
    }
  }
}
} // namespace

TEST(SymIntTest, BinaryPlus) {
  test_operator<std::plus>();
}

TEST(SymIntTest, BinaryMinus) {
  test_operator<std::minus>();
}

TEST(SymIntTest, BinaryMultiplies) {
  test_operator<std::multiplies>();
}

TEST(SymIntTest, BinaryDivides) {
  test_operator<std::divides>();
}

TEST(SymIntTest, BinaryModulus) {
  test_operator<std::modulus>();
}

TEST(SymIntTest, BinaryComparisonOperators) {
  test_operator<std::equal_to>();
  test_operator<std::not_equal_to>();
  test_operator<std::less>();
  test_operator<std::less_equal>();
  test_operator<std::greater>();
  test_operator<std::greater_equal>();
}

template <typename T>
struct MinWrapper {
  auto operator()(T lhs, T rhs) const {
    return std::min(lhs, rhs);
  }
};

template <>
struct MinWrapper<SymInt> {
  auto operator()(const SymInt& lhs, const SymInt& rhs) const {
    return lhs.min(rhs);
  }
};

template <typename T>
struct MaxWrapper {
  auto operator()(T lhs, T rhs) const {
    return std::max(lhs, rhs);
  }
};

template <>
struct MaxWrapper<SymInt> {
  auto operator()(const SymInt& lhs, const SymInt& rhs) const {
    return lhs.max(rhs);
  }
};

TEST(SymIntTest, MinMax) {
  test_operator<MinWrapper>();
  test_operator<MaxWrapper>();
}
#endif
