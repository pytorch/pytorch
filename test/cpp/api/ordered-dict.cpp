#include <gtest/gtest.h>

#include <test/cpp/api/support.h>
#include <torch/detail/ordered_dict.h>

template <typename T>
using OrderedDict = torch::detail::OrderedDict<std::string, T>;

TEST(OrderedDictTest, IsEmptyAfterDefaultConstruction) {
  OrderedDict<int> dict;
  ASSERT_EQ(dict.subject(), "Key");
  ASSERT_TRUE(dict.is_empty());
  ASSERT_EQ(dict.size(), 0);
}

TEST(OrderedDictTest, InsertAddsElementsWhenTheyAreYetNotPresent) {
  OrderedDict<int> dict;
  dict.insert("a", 1);
  dict.insert("b", 2);
  ASSERT_EQ(dict.size(), 2);
}

TEST(OrderedDictTest, GetReturnsValuesWhenTheyArePresent) {
  OrderedDict<int> dict;
  dict.insert("a", 1);
  dict.insert("b", 2);
  ASSERT_EQ(dict.get("a"), 1);
  ASSERT_EQ(dict.get("b"), 2);
}

TEST(OrderedDictTest, GetThrowsWhenPassedKeysThatAreNotPresent) {
  OrderedDict<int> dict;
  dict.insert("a", 1);
  dict.insert("b", 2);
  ASSERT_THROWS_WITH(dict.get("foo"), "Key 'foo' is not defined");
  ASSERT_THROWS_WITH(dict.get(""), "Key '' is not defined");
}

TEST(OrderedDictTest, CanInitializeFromList) {
  OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
  ASSERT_EQ(dict.size(), 2);
  ASSERT_EQ(dict.get("a"), 1);
  ASSERT_EQ(dict.get("b"), 2);
}

TEST(OrderedDictTest, InsertThrowsWhenPassedElementsThatArePresent) {
  OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
  ASSERT_THROWS_WITH(dict.insert("a", 1), "Key 'a' already defined");
  ASSERT_THROWS_WITH(dict.insert("b", 1), "Key 'b' already defined");
}

TEST(OrderedDictTest, FrontReturnsTheFirstItem) {
  OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
  ASSERT_EQ(dict.front().key, "a");
  ASSERT_EQ(dict.front().value, 1);
}

TEST(OrderedDictTest, BackReturnsTheLastItem) {
  OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
  ASSERT_EQ(dict.back().key, "b");
  ASSERT_EQ(dict.back().value, 2);
}

TEST(OrderedDictTest, FindReturnsPointersToValuesWhenPresent) {
  OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
  ASSERT_NE(dict.find("a"), nullptr);
  ASSERT_EQ(*dict.find("a"), 1);
  ASSERT_NE(dict.find("b"), nullptr);
  ASSERT_EQ(*dict.find("b"), 2);
}

TEST(OrderedDictTest, FindReturnsNullPointersWhenPasesdKeysThatAreNotPresent) {
  OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
  ASSERT_EQ(dict.find("bar"), nullptr);
  ASSERT_EQ(dict.find(""), nullptr);
}

TEST(OrderedDictTest, SubscriptOperatorThrowsWhenPassedKeysThatAreNotPresent) {
  OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
  ASSERT_EQ(dict["a"], 1);
  ASSERT_EQ(dict["b"], 2);
}

TEST(
    OrderedDictTest,
    SubscriptOperatorReturnsItemsPositionallyWhenPassedIntegers) {
  OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
  ASSERT_EQ(dict[0].key, "a");
  ASSERT_EQ(dict[0].value, 1);
  ASSERT_EQ(dict[1].key, "b");
  ASSERT_EQ(dict[1].value, 2);
}

TEST(OrderedDictTest, SubscriptOperatorsThrowswhenPassedKeysThatAreNotPresent) {
  OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
  ASSERT_THROWS_WITH(dict.get("foo"), "Key 'foo' is not defined");
  ASSERT_THROWS_WITH(dict.get(""), "Key '' is not defined");
}

TEST(OrderedDictTest, UpdateInsertsAllItemsFromAnotherOrderedDict) {
  OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
  OrderedDict<int> dict2 = {{"c", 3}};
  dict2.update(dict);
  ASSERT_EQ(dict2.size(), 3);
  ASSERT_NE(dict2.find("a"), nullptr);
  ASSERT_NE(dict2.find("b"), nullptr);
  ASSERT_NE(dict2.find("c"), nullptr);
}

TEST(OrderedDictTest, UpdateAlsoChecksForDuplicates) {
  OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
  OrderedDict<int> dict2 = {{"a", 1}};
  ASSERT_THROWS_WITH(dict2.update(dict), "Key 'a' already defined");
}

TEST(OrderedDictTest, CanIterateItems) {
  OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
  auto iterator = dict.begin();
  ASSERT_NE(iterator, dict.end());
  ASSERT_EQ(iterator->key, "a");
  ASSERT_EQ(iterator->value, 1);
  ++iterator;
  ASSERT_NE(iterator, dict.end());
  ASSERT_EQ(iterator->key, "b");
  ASSERT_EQ(iterator->value, 2);
  ++iterator;
  ASSERT_EQ(iterator, dict.end());
}

TEST(OrderedDictTest, ClearMakesTheDictEmpty) {
  OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
  ASSERT_FALSE(dict.is_empty());
  dict.clear();
  ASSERT_TRUE(dict.is_empty());
}

TEST(OrderedDictTest, CanCopyConstruct) {
  OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
  OrderedDict<int> copy = dict;
  ASSERT_EQ(copy.size(), 2);
  ASSERT_EQ(*copy[0], 1);
  ASSERT_EQ(*copy[1], 2);
}

TEST(OrderedDictTest, CanCopyAssign) {
  OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
  OrderedDict<int> copy = {{"c", 1}};
  ASSERT_NE(copy.find("c"), nullptr);
  copy = dict;
  ASSERT_EQ(copy.size(), 2);
  ASSERT_EQ(*copy[0], 1);
  ASSERT_EQ(*copy[1], 2);
  ASSERT_EQ(copy.find("c"), nullptr);
}

TEST(OrderedDictTest, CanMoveConstruct) {
  OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
  OrderedDict<int> copy = std::move(dict);
  ASSERT_EQ(copy.size(), 2);
  ASSERT_EQ(*copy[0], 1);
  ASSERT_EQ(*copy[1], 2);
}

TEST(OrderedDictTest, CanMoveAssign) {
  OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
  OrderedDict<int> copy = {{"c", 1}};
  ASSERT_NE(copy.find("c"), nullptr);
  copy = std::move(dict);
  ASSERT_EQ(copy.size(), 2);
  ASSERT_EQ(*copy[0], 1);
  ASSERT_EQ(*copy[1], 2);
  ASSERT_EQ(copy.find("c"), nullptr);
}

TEST(OrderedDictTest, CanInsertWithBraces) {
  OrderedDict<std::pair<int, int>> dict;
  dict.insert("a", {1, 2});
  ASSERT_FALSE(dict.is_empty());
  ASSERT_EQ(dict["a"].first, 1);
  ASSERT_EQ(dict["a"].second, 2);
}

TEST(OrderedDictTest, ErrorMessagesIncludeTheWhat) {
  OrderedDict<int> dict("Penguin");
  ASSERT_EQ(dict.subject(), "Penguin");
  dict.insert("a", 1);
  ASSERT_FALSE(dict.is_empty());
  ASSERT_THROWS_WITH(dict.get("b"), "Penguin 'b' is not defined");
  ASSERT_THROWS_WITH(dict.insert("a", 1), "Penguin 'a' already defined");
}
