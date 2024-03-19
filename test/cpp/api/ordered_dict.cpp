#include <gtest/gtest.h>

#include <test/cpp/api/support.h>
#include <torch/torch.h>

template <typename T>
using OrderedDict = torch::OrderedDict<std::string, T>;

TEST(OrderedDictTest, IsEmptyAfterDefaultConstruction) {
  OrderedDict<int> dict;
  ASSERT_EQ(dict.key_description(), "Key");
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
  ASSERT_EQ(dict["a"], 1);
  ASSERT_EQ(dict["b"], 2);
}

TEST(OrderedDictTest, GetThrowsWhenPassedKeysThatAreNotPresent) {
  OrderedDict<int> dict;
  dict.insert("a", 1);
  dict.insert("b", 2);
  ASSERT_THROWS_WITH(dict["foo"], "Key 'foo' is not defined");
  ASSERT_THROWS_WITH(dict[""], "Key '' is not defined");
}

TEST(OrderedDictTest, CanInitializeFromList) {
  OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
  ASSERT_EQ(dict.size(), 2);
  ASSERT_EQ(dict["a"], 1);
  ASSERT_EQ(dict["b"], 2);
}

TEST(OrderedDictTest, InsertThrowsWhenPassedElementsThatArePresent) {
  OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
  ASSERT_THROWS_WITH(dict.insert("a", 1), "Key 'a' already defined");
  ASSERT_THROWS_WITH(dict.insert("b", 1), "Key 'b' already defined");
}

TEST(OrderedDictTest, FrontReturnsTheFirstItem) {
  OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
  ASSERT_EQ(dict.front().key(), "a");
  ASSERT_EQ(dict.front().value(), 1);
}

TEST(OrderedDictTest, FrontThrowsWhenEmpty) {
  OrderedDict<int> dict;
  ASSERT_THROWS_WITH(dict.front(), "Called front() on an empty OrderedDict");
}

TEST(OrderedDictTest, BackReturnsTheLastItem) {
  OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
  ASSERT_EQ(dict.back().key(), "b");
  ASSERT_EQ(dict.back().value(), 2);
}

TEST(OrderedDictTest, BackThrowsWhenEmpty) {
  OrderedDict<int> dict;
  ASSERT_THROWS_WITH(dict.back(), "Called back() on an empty OrderedDict");
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
  ASSERT_EQ(dict[0].key(), "a");
  ASSERT_EQ(dict[0].value(), 1);
  ASSERT_EQ(dict[1].key(), "b");
  ASSERT_EQ(dict[1].value(), 2);
}

TEST(OrderedDictTest, SubscriptOperatorsThrowswhenPassedKeysThatAreNotPresent) {
  OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
  ASSERT_THROWS_WITH(dict["foo"], "Key 'foo' is not defined");
  ASSERT_THROWS_WITH(dict[""], "Key '' is not defined");
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
  ASSERT_EQ(iterator->key(), "a");
  ASSERT_EQ(iterator->value(), 1);
  ++iterator;
  ASSERT_NE(iterator, dict.end());
  ASSERT_EQ(iterator->key(), "b");
  ASSERT_EQ(iterator->value(), 2);
  ++iterator;
  ASSERT_EQ(iterator, dict.end());
}

TEST(OrderedDictTest, EraseWorks) {
  OrderedDict<int> dict = {{"a", 1}, {"b", 2}, {"c", 3}};
  dict.erase("b");
  ASSERT_FALSE(dict.contains("b"));
  ASSERT_EQ(dict["a"], 1);
  ASSERT_EQ(dict["c"], 3);
  dict.erase("a");
  ASSERT_FALSE(dict.contains("a"));
  ASSERT_EQ(dict["c"], 3);
  dict.erase("c");
  ASSERT_FALSE(dict.contains("c"));
  ASSERT_TRUE(dict.is_empty());
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

TEST(OrderedDictTest, ErrorMessagesIncludeTheKeyDescription) {
  OrderedDict<int> dict("Penguin");
  ASSERT_EQ(dict.key_description(), "Penguin");
  dict.insert("a", 1);
  ASSERT_FALSE(dict.is_empty());
  ASSERT_THROWS_WITH(dict["b"], "Penguin 'b' is not defined");
  ASSERT_THROWS_WITH(dict.insert("a", 1), "Penguin 'a' already defined");
}

TEST(OrderedDictTest, KeysReturnsAllKeys) {
  OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
  ASSERT_EQ(dict.keys(), std::vector<std::string>({"a", "b"}));
}

TEST(OrderedDictTest, ValuesReturnsAllValues) {
  OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
  ASSERT_EQ(dict.values(), std::vector<int>({1, 2}));
}

TEST(OrderedDictTest, ItemsReturnsAllItems) {
  OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
  std::vector<OrderedDict<int>::Item> items = dict.items();
  ASSERT_EQ(items.size(), 2);
  ASSERT_EQ(items[0].key(), "a");
  ASSERT_EQ(items[0].value(), 1);
  ASSERT_EQ(items[1].key(), "b");
  ASSERT_EQ(items[1].value(), 2);
}
