#include <c10/util/Optional.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <string>

namespace {

using testing::Eq;
using testing::Ge;
using testing::Gt;
using testing::Le;
using testing::Lt;
using testing::Ne;
using testing::Not;

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
c10::IntArrayRef getSampleValue() {
  return {};
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
    // ArrayRef optimization.
    c10::IntArrayRef,
    // Non-trivial destructor.
    std::string>;

// This assert is also in Optional.cpp; including here too to make it
// more likely that we'll remember to port this optimization over when
// we move to std::optional.
static_assert(
    sizeof(c10::optional<c10::IntArrayRef>) == sizeof(c10::IntArrayRef),
    "c10::optional<IntArrayRef> should be size-optimized");

TYPED_TEST_CASE(OptionalTest, OptionalTypes);

TYPED_TEST(OptionalTest, Empty) {
  typename TestFixture::optional empty;

  EXPECT_FALSE((bool)empty);
  EXPECT_FALSE(empty.has_value());

  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  EXPECT_THROW(empty.value(), c10::bad_optional_access);
}

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

class SelfCompareTest : public testing::TestWithParam<c10::optional<int>> {};

TEST_P(SelfCompareTest, SelfCompare) {
  c10::optional<int> x = GetParam();
  EXPECT_THAT(x, Eq(x));
  EXPECT_THAT(x, Le(x));
  EXPECT_THAT(x, Ge(x));
  EXPECT_THAT(x, Not(Ne(x)));
  EXPECT_THAT(x, Not(Lt(x)));
  EXPECT_THAT(x, Not(Gt(x)));
}

INSTANTIATE_TEST_CASE_P(
    nullopt,
    SelfCompareTest,
    testing::Values(c10::nullopt));
INSTANTIATE_TEST_CASE_P(
    int,
    SelfCompareTest,
    testing::Values(c10::make_optional(2)));

TEST(OptionalTest, Nullopt) {
  c10::optional<int> x = 2;

  EXPECT_THAT(c10::nullopt, Not(Eq(x)));
  EXPECT_THAT(x, Not(Eq(c10::nullopt)));

  EXPECT_THAT(x, Ne(c10::nullopt));
  EXPECT_THAT(c10::nullopt, Ne(x));

  EXPECT_THAT(x, Not(Lt(c10::nullopt)));
  EXPECT_THAT(c10::nullopt, Lt(x));

  EXPECT_THAT(x, Not(Le(c10::nullopt)));
  EXPECT_THAT(c10::nullopt, Le(x));

  EXPECT_THAT(x, Gt(c10::nullopt));
  EXPECT_THAT(c10::nullopt, Not(Gt(x)));

  EXPECT_THAT(x, Ge(c10::nullopt));
  EXPECT_THAT(c10::nullopt, Not(Ge(x)));
}

// Ensure comparisons work...
using CmpTestTypes = testing::Types<
    // between two optionals
    std::pair<c10::optional<int>, c10::optional<int>>,

    // between an optional and a value
    std::pair<c10::optional<int>, int>,
    // between a value and an optional
    std::pair<int, c10::optional<int>>,

    // between an optional and a differently typed value
    std::pair<c10::optional<int>, long>,
    // between a differently typed value and an optional
    std::pair<long, c10::optional<int>>>;
template <typename T>
class CmpTest : public testing::Test {};
TYPED_TEST_CASE(CmpTest, CmpTestTypes);

TYPED_TEST(CmpTest, Cmp) {
  TypeParam pair = {2, 3};
  auto x = pair.first;
  auto y = pair.second;

  EXPECT_THAT(x, Not(Eq(y)));

  EXPECT_THAT(x, Ne(y));

  EXPECT_THAT(x, Lt(y));
  EXPECT_THAT(y, Not(Lt(x)));

  EXPECT_THAT(x, Le(y));
  EXPECT_THAT(y, Not(Le(x)));

  EXPECT_THAT(x, Not(Gt(y)));
  EXPECT_THAT(y, Gt(x));

  EXPECT_THAT(x, Not(Ge(y)));
  EXPECT_THAT(y, Ge(x));
}

// Identifies a comparison operator.
enum class Operator { eq, ne, lt, le, gt, ge };

// Implements the stringable interface for Operator.
std::string to_string(Operator op) {
  switch (op) {
  case Operator::eq: return "==";
  case Operator::ne: return "!=";
  case Operator::lt: return "<";
  case Operator::le: return ">=";
  case Operator::gt: return ">";
  case Operator::ge: return ">=";
  default:
    TORCH_INTERNAL_ASSERT("unreachable");
    return "";
  }
}

// Implements the printable interface for Operator.
std::ostream& operator<<(std::ostream& out, Operator op) {
  return out << to_string(op);
}

// A simple class that tracks the most recent comparison operator
// applied to it.
//
// This can only keep track of a single operator at a time and for
// sanity checking requires clearing the operator before a new one may
// be invoked.
//
// OperatorSpy has value semantics but only one possible value, the
// tracked operator is considered transient and not part of the
// objects value.
class OperatorSpy {
public:
  // Returns and releases the most recent operator.
  //
  // REQUIRES: an operator has been invoked on this object since the
  // constructor or most recent call to this function.
  Operator release_most_recent_operator() const {
    TORCH_INTERNAL_ASSERT(op_.has_value());
    Operator op = *op_;
    op_.reset();
    return op;
  }

  bool operator==(OperatorSpy const& that) const {
    set_operator(Operator::eq);
    that.set_operator(Operator::eq);
    return true;
  }

  bool operator!=(OperatorSpy const& that) const {
    set_operator(Operator::ne);
    that.set_operator(Operator::ne);
    return false;
  }

  bool operator<(OperatorSpy const& that) const {
    set_operator(Operator::lt);
    that.set_operator(Operator::lt);
    return false;
  }

  bool operator<=(OperatorSpy const& that) const {
    set_operator(Operator::le);
    that.set_operator(Operator::le);
    return true;
  }

  bool operator>(OperatorSpy const& that) const {
    set_operator(Operator::gt);
    that.set_operator(Operator::gt);
    return false;
  }

  bool operator>=(OperatorSpy const& that) const {
    set_operator(Operator::ge);
    that.set_operator(Operator::ge);
    return true;
  }

private:
  // Sets the operator.
  //
  // REQUIRES: no operator has been invoked since the constructor or
  // most recent call to release_most_recent_operator.
  void set_operator(Operator op) const {
    TORCH_INTERNAL_ASSERT(!op_.has_value());
    op_ = op;
  }

  mutable c10::optional<Operator> op_;  // transient data
};

// Ensure operators delegate properly...
using OperatorDelegationTestTypes = testing::Types<
    // between two optionals
    std::pair<c10::optional<OperatorSpy>, c10::optional<OperatorSpy>>,

    // between an optional and a value
    std::pair<c10::optional<OperatorSpy>, OperatorSpy>,
    // between a value and an optional
    std::pair<OperatorSpy, c10::optional<OperatorSpy>>>;

template <typename T>
class OperatorDelegationTest : public testing::Test {};
TYPED_TEST_CASE(OperatorDelegationTest, OperatorDelegationTestTypes);

// Get the underlying OperatorSpy, which may be wrapped in an
// optional.
OperatorSpy const& underlying_spy(OperatorSpy const& value) { return value; }
OperatorSpy const& underlying_spy(c10::optional<OperatorSpy> const& opt) { return *opt; }

// Matches that a pair of OperatorSpy instances both have the most
// recent operator invoked as specified by "op".
//
// ENSURES: args have released their most recent operators.
MATCHER_P(most_recent_op_is, op, "") {
  Operator x_op = underlying_spy(arg.first).release_most_recent_operator();
  Operator y_op = underlying_spy(arg.second).release_most_recent_operator();
  if (x_op != y_op) {
    *result_listener << "OperatorSpy instances have different operators: "
		     << x_op << " != " << y_op << '\n';
    return false;
  }
  if (x_op != op) {
    *result_listener << "OperatorSpy instances claim most recent operator is "
		     << x_op << "; wanted " << op << '\n';
    return false;
  }
  return true;
}


TYPED_TEST(OperatorDelegationTest, Basic) {
  TypeParam pair = {OperatorSpy(), OperatorSpy()};

  EXPECT_TRUE(pair.first == pair.second);
  EXPECT_THAT(pair, most_recent_op_is(Operator::eq));

  EXPECT_FALSE(pair.first != pair.second);
  EXPECT_THAT(pair, most_recent_op_is(Operator::ne));

  EXPECT_FALSE(pair.first < pair.second);
  EXPECT_THAT(pair, most_recent_op_is(Operator::lt));

  EXPECT_TRUE(pair.first <= pair.second);
  EXPECT_THAT(pair, most_recent_op_is(Operator::le));

  EXPECT_FALSE(pair.first > pair.second);
  EXPECT_THAT(pair, most_recent_op_is(Operator::gt));

  EXPECT_TRUE(pair.first >= pair.second);
  EXPECT_THAT(pair, most_recent_op_is(Operator::ge));
}

} // namespace
