#include <gtest/gtest.h>

#include <ATen/Dimname.h>
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>
#include <ATen/core/EnableNamedTensor.h>

#ifdef BUILD_NAMEDTENSOR
using at::NameType;
using at::Symbol;
using at::Dimname;

TEST(DimnameTest, isValidIdentifier) {
  ASSERT_TRUE(Dimname::isValidName("a"));
  ASSERT_TRUE(Dimname::isValidName("batch"));
  ASSERT_TRUE(Dimname::isValidName("N"));
  ASSERT_TRUE(Dimname::isValidName("CHANNELS"));
  ASSERT_TRUE(Dimname::isValidName("foo_bar_baz"));

  ASSERT_FALSE(Dimname::isValidName(""));
  ASSERT_FALSE(Dimname::isValidName(" "));
  ASSERT_FALSE(Dimname::isValidName(" a "));
  ASSERT_FALSE(Dimname::isValidName("batch1"));
  ASSERT_FALSE(Dimname::isValidName("foo_bar_1"));
  ASSERT_FALSE(Dimname::isValidName("?"));
  ASSERT_FALSE(Dimname::isValidName("-"));
}

TEST(DimnameTest, wildcardName) {
  Dimname wildcard = Dimname::wildcard();
  ASSERT_EQ(wildcard.type(), NameType::WILDCARD);
  ASSERT_EQ(wildcard.symbol(), Symbol::dimname("*"));
}

TEST(DimnameTest, createNormalName) {
  auto foo = Symbol::dimname("foo");
  auto dimname = Dimname::fromSymbol(foo);
  ASSERT_EQ(dimname.type(), NameType::BASIC);
  ASSERT_EQ(dimname.symbol(), foo);
  ASSERT_THROW(Dimname::fromSymbol(Symbol::dimname("inva.lid")), c10::Error);
  ASSERT_THROW(Dimname::fromSymbol(Symbol::dimname("invalid1")), c10::Error);
}

static void check_unify_and_match(
    const std::string& dimname,
    const std::string& other,
    at::optional<const std::string> expected) {
  auto dimname1 = Dimname::fromSymbol(Symbol::dimname(dimname));
  auto dimname2 = Dimname::fromSymbol(Symbol::dimname(other));
  auto result = dimname1.unify(dimname2);
  if (expected) {
    auto expected_result = Dimname::fromSymbol(Symbol::dimname(*expected));
    ASSERT_EQ(result->symbol(), expected_result.symbol());
    ASSERT_EQ(result->type(), expected_result.type());
    ASSERT_TRUE(dimname1.matches(dimname2));
  } else {
    ASSERT_FALSE(result);
    ASSERT_FALSE(dimname1.matches(dimname2));
  }
}

TEST(DimnameTest, unifyAndMatch) {
  check_unify_and_match("a", "a", "a");
  check_unify_and_match("a", "*", "a");
  check_unify_and_match("*", "a", "a");
  check_unify_and_match("*", "*", "*");
  check_unify_and_match("a", "b", c10::nullopt);
}
#endif
