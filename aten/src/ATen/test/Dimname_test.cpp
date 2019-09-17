#include <gtest/gtest.h>

#include <ATen/Dimname.h>
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>
#include <ATen/core/EnableNamedTensor.h>

#ifdef BUILD_NAMEDTENSOR
using at::is_valid_identifier;
using at::NameType;
using at::Symbol;
using at::Dimname;

TEST(DimnameTest, isValidIdentifier) {
  ASSERT_TRUE(is_valid_identifier("a"));
  ASSERT_TRUE(is_valid_identifier("batch"));
  ASSERT_TRUE(is_valid_identifier("N"));
  ASSERT_TRUE(is_valid_identifier("CHANNELS"));
  ASSERT_TRUE(is_valid_identifier("foo_bar_baz"));

  ASSERT_FALSE(is_valid_identifier(""));
  ASSERT_FALSE(is_valid_identifier(" "));
  ASSERT_FALSE(is_valid_identifier(" a "));
  ASSERT_FALSE(is_valid_identifier("batch1"));
  ASSERT_FALSE(is_valid_identifier("foo_bar_1"));
  ASSERT_FALSE(is_valid_identifier("?"));
  ASSERT_FALSE(is_valid_identifier("-"));
}

TEST(DimnameTest, wildcardName) {
  Dimname wildcard = Dimname::wildcard();
  ASSERT_EQ(wildcard.type(), NameType::WILDCARD);
  ASSERT_EQ(wildcard.full_name(), Symbol::dimname("*"));
  ASSERT_EQ(wildcard.untagged_name(), Symbol::dimname("*"));
}

TEST(DimnameTest, createNormalName) {
  auto foo = Symbol::dimname("foo");
  auto dimname = Dimname::fromSymbol(foo);
  ASSERT_EQ(dimname.type(), NameType::NORMAL);
  ASSERT_EQ(dimname.full_name(), foo);
  ASSERT_EQ(dimname.untagged_name(), foo);

  ASSERT_THROW(Dimname::fromSymbol(Symbol::dimname("invalid1")), c10::Error);
}

TEST(DimnameTest, createTaggedName) {
  auto foo_bar = Symbol::dimname("foo.bar");
  auto foo = Symbol::dimname("foo");
  auto dimname = Dimname::fromSymbol(foo_bar);
  ASSERT_EQ(dimname.type(), NameType::TAGGED);
  ASSERT_EQ(dimname.full_name(), foo_bar);
  ASSERT_EQ(dimname.untagged_name(), foo);

  ASSERT_THROW(Dimname::fromSymbol(Symbol::dimname(".bar")), c10::Error);
  ASSERT_THROW(Dimname::fromSymbol(Symbol::dimname("foo.")), c10::Error);
  ASSERT_THROW(Dimname::fromSymbol(Symbol::dimname("foo.bar.baz")), c10::Error);
}

static void check_unify_and_match(
    const std::string& dimname,
    const std::string& other,
    at::optional<const std::string> expected) {
  auto dimname1 = Dimname::fromSymbol(Symbol::dimname(dimname));
  auto dimname2 = Dimname::fromSymbol(Symbol::dimname(other));
  auto result = at::unify(dimname1, dimname2);
  if (expected) {
    auto expected_result = Dimname::fromSymbol(Symbol::dimname(*expected));
    ASSERT_EQ(result->full_name(), expected_result.full_name());
    ASSERT_EQ(result->type(), expected_result.type());
    ASSERT_EQ(result->untagged_name(), expected_result.untagged_name());
    ASSERT_TRUE(match(dimname1, dimname2));
  } else {
    ASSERT_FALSE(result);
    ASSERT_FALSE(match(dimname1, dimname2));
  }
}

TEST(DimnameTest, unifyAndMatch) {
  check_unify_and_match("a", "a", "a");
  check_unify_and_match("a", "*", "a");
  check_unify_and_match("*", "a", "a");
  check_unify_and_match("*", "*", "*");
  check_unify_and_match("a", "b", c10::nullopt);

  check_unify_and_match("*", "a.b", "a.b");
  check_unify_and_match("a", "a.b", "a");
  check_unify_and_match("c", "a.b", c10::nullopt);
  check_unify_and_match("a.b", "a.c", "a");
  check_unify_and_match("a.b", "a.b", "a.b");
  check_unify_and_match("c.b", "a.b", c10::nullopt);
  check_unify_and_match("c.b", "a", c10::nullopt);
}
#endif
