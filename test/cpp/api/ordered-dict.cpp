#include <gtest/gtest.h>

#include <test/cpp/api/support.h>
#include <torch/detail/ordered_dict.h>

template <typename T>
using OrderedDict = torch::detail::OrderedDict<std::string, T>;

TEST(OrderedDictTest, IsEmptyAfterDefaultConstruction) {
  OrderedDict<int> dict;
  ASSERT_TRUE(dict.subject() == "Key");
  ASSERT_TRUE(dict.is_empty());
  ASSERT_TRUE(dict.size() == 0);
}

TEST(OrderedDictTest, InsertAddsElementsWhenTheyAreYetNotPresent) {
  OrderedDict<int> dict;
  dict.insert("a", 1);
  dict.insert("b", 2);
  ASSERT_TRUE(dict.size() == 2);
}

TEST(OrderedDictTest, GetReturnsValuesWhenTheyArePresent) {
  OrderedDict<int> dict;
  dict.insert("a", 1);
  dict.insert("b", 2);
  ASSERT_TRUE(dict.get("a") == 1);
  ASSERT_TRUE(dict.get("b") == 2);
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
  ASSERT_TRUE(dict.size() == 2);
  ASSERT_TRUE(dict.get("a") == 1);
  ASSERT_TRUE(dict.get("b") == 2);
}

TEST(OrderedDictTest, InsertThrowsWhenPassedElementsThatArePresent) {
  OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
  ASSERT_THROWS_WITH(dict.insert("a", 1), "Key 'a' already defined");
  ASSERT_THROWS_WITH(dict.insert("b", 1), "Key 'b' already defined");
}

TEST(OrderedDictTest, FrontReturnsTheFirstItem) {
  OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
  ASSERT_TRUE(dict.front().key == "a");
  ASSERT_TRUE(dict.front().value == 1);
}

TEST(OrderedDictTest, BackReturnsTheLastItem) {
  OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
  ASSERT_TRUE(dict.back().key == "b");
  ASSERT_TRUE(dict.back().value == 2);
}

TEST(OrderedDictTest, FindReturnsPointersToValuesWhenPresent) {
  OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
  ASSERT_TRUE(dict.find("a") != nullptr);
  ASSERT_TRUE(*dict.find("a") == 1);
  ASSERT_TRUE(dict.find("b") != nullptr);
  ASSERT_TRUE(*dict.find("b") == 2);
}

TEST(OrderedDictTest, FindReturnsNullPointersWhenPasesdKeysThatAreNotPresent) {
  OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
  ASSERT_TRUE(dict.find("bar") == nullptr);
  ASSERT_TRUE(dict.find("") == nullptr);
}

TEST(OrderedDictTest, SubscriptOperatorThrowsWhenPassedKeysThatAreNotPresent) {
  OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
  ASSERT_TRUE(dict["a"] == 1);
  ASSERT_TRUE(dict["b"] == 2);
}

TEST(
    OrderedDictTest,
    SubscriptOperatorReturnsItemsPositionallyWhenPassedIntegers) {
  OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
  ASSERT_TRUE(dict[0].key == "a");
  ASSERT_TRUE(dict[0].value == 1);
  ASSERT_TRUE(dict[1].key == "b");
  ASSERT_TRUE(dict[1].value == 2);
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
  ASSERT_TRUE(dict2.size() == 3);
  ASSERT_TRUE(dict2.find("a") != nullptr);
  ASSERT_TRUE(dict2.find("b") != nullptr);
  ASSERT_TRUE(dict2.find("c") != nullptr);
}

TEST(OrderedDictTest, UpdateAlsoChecksForDuplicates) {
  OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
  OrderedDict<int> dict2 = {{"a", 1}};
  ASSERT_THROWS_WITH(dict2.update(dict), "Key 'a' already defined");
}

TEST(OrderedDictTest, CanIterateItems) {
  OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
  auto iterator = dict.begin();
  ASSERT_TRUE(iterator != dict.end());
  ASSERT_TRUE(iterator->key == "a");
  ASSERT_TRUE(iterator->value == 1);
  ++iterator;
  ASSERT_TRUE(iterator != dict.end());
  ASSERT_TRUE(iterator->key == "b");
  ASSERT_TRUE(iterator->value == 2);
  ++iterator;
  ASSERT_TRUE(iterator == dict.end());
}

TEST(OrderedDictTest, ClearMakesTheDictEmpty) {
  OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
  ASSERT_TRUE(!dict.is_empty());
  dict.clear();
  ASSERT_TRUE(dict.is_empty());
}

TEST(OrderedDictTest, CanCopyConstruct) {
  OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
  OrderedDict<int> copy = dict;
  ASSERT_TRUE(copy.size() == 2);
  ASSERT_TRUE(*copy[0] == 1);
  ASSERT_TRUE(*copy[1] == 2);
}

TEST(OrderedDictTest, CanCopyAssign) {
  OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
  OrderedDict<int> copy = {{"c", 1}};
  ASSERT_TRUE(copy.find("c") != nullptr);
  copy = dict;
  ASSERT_TRUE(copy.size() == 2);
  ASSERT_TRUE(*copy[0] == 1);
  ASSERT_TRUE(*copy[1] == 2);
  ASSERT_TRUE(copy.find("c") == nullptr);
}

TEST(OrderedDictTest, CanMoveConstruct) {
  OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
  OrderedDict<int> copy = std::move(dict);
  ASSERT_TRUE(copy.size() == 2);
  ASSERT_TRUE(*copy[0] == 1);
  ASSERT_TRUE(*copy[1] == 2);
}

TEST(OrderedDictTest, CanMoveAssign) {
  OrderedDict<int> dict = {{"a", 1}, {"b", 2}};
  OrderedDict<int> copy = {{"c", 1}};
  ASSERT_TRUE(copy.find("c") != nullptr);
  copy = std::move(dict);
  ASSERT_TRUE(copy.size() == 2);
  ASSERT_TRUE(*copy[0] == 1);
  ASSERT_TRUE(*copy[1] == 2);
  ASSERT_TRUE(copy.find("c") == nullptr);
}

TEST(OrderedDictTest, CanInsertWithBraces) {
  OrderedDict<std::pair<int, int>> dict;
  dict.insert("a", {1, 2});
  ASSERT_TRUE(!dict.is_empty());
  ASSERT_TRUE(dict["a"].first == 1);
  ASSERT_TRUE(dict["a"].second == 2);
}

TEST(OrderedDictTest, ErrorMessagesIncludeTheWhat) {
  OrderedDict<int> dict("Penguin");
  ASSERT_TRUE(dict.subject() == "Penguin");
  dict.insert("a", 1);
  ASSERT_TRUE(!dict.is_empty());
  ASSERT_THROWS_WITH(dict.get("b"), "Penguin 'b' is not defined");
  ASSERT_THROWS_WITH(dict.insert("a", 1), "Penguin 'a' already defined");
}
