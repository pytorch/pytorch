#ifdef NAMEDTENSOR_ENABLED
#include <gtest/gtest.h>

#include <ATen/Dimname.h>
#include <c10/util/Exception.h>

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
  ASSERT_EQ(wildcard.name(), Symbol::dimname("*"));
  ASSERT_EQ(wildcard.untagged_name(), Symbol::dimname("*"));
}

TEST(DimnameTest, createNormalName) {
  auto foo = Symbol::dimname("foo");
  auto dimname = Dimname::fromSymbol(foo);
  ASSERT_EQ(dimname.type(), NameType::NORMAL);
  ASSERT_EQ(dimname.name(), foo);
  ASSERT_EQ(dimname.untagged_name(), foo);

  ASSERT_THROW(Dimname::fromSymbol(Symbol::dimname("invalid1")), c10::Error);
}

TEST(DimnameTest, createTaggedName) {
  auto foo_bar = Symbol::dimname("foo.bar");
  auto foo = Symbol::dimname("foo");
  auto dimname = Dimname::fromSymbol(foo_bar);
  ASSERT_EQ(dimname.type(), NameType::TAGGED);
  ASSERT_EQ(dimname.name(), foo_bar);
  ASSERT_EQ(dimname.untagged_name(), foo);

  ASSERT_THROW(Dimname::fromSymbol(Symbol::dimname(".bar")), c10::Error);
  ASSERT_THROW(Dimname::fromSymbol(Symbol::dimname("foo.")), c10::Error);
  ASSERT_THROW(Dimname::fromSymbol(Symbol::dimname("foo.bar.baz")), c10::Error);
}
#endif
