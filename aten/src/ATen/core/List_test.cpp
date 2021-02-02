#include <ATen/core/List.h>
#include <gtest/gtest.h>

using namespace c10;

TEST(ListTest_IValueBasedList, givenEmptyList_whenCallingEmpty_thenReturnsTrue) {
    List<string> list;
    EXPECT_TRUE(list.empty());
}

TEST(ListTest_IValueBasedList, givenNonemptyList_whenCallingEmpty_thenReturnsFalse) {
    List<string> list({"3"});
    EXPECT_FALSE(list.empty());
}

TEST(ListTest_IValueBasedList, givenEmptyList_whenCallingSize_thenReturnsZero) {
    List<string> list;
    EXPECT_EQ(0, list.size());
}

TEST(ListTest_IValueBasedList, givenNonemptyList_whenCallingSize_thenReturnsNumberOfElements) {
    List<string> list({"3", "4"});
    EXPECT_EQ(2, list.size());
}

TEST(ListTest_IValueBasedList, givenNonemptyList_whenCallingClear_thenIsEmpty) {
  List<string> list({"3", "4"});
  list.clear();
  EXPECT_TRUE(list.empty());
}

TEST(ListTest_IValueBasedList, whenCallingGetWithExistingPosition_thenReturnsElement) {
  List<string> list({"3", "4"});
  EXPECT_EQ("3", list.get(0));
  EXPECT_EQ("4", list.get(1));
}

TEST(ListTest_IValueBasedList, whenCallingGetWithNonExistingPosition_thenThrowsException) {
  List<string> list({"3", "4"});
  EXPECT_THROW(list.get(2), std::out_of_range);
}

TEST(ListTest_IValueBasedList, whenCallingExtractWithExistingPosition_thenReturnsElement) {
  List<string> list({"3", "4"});
  EXPECT_EQ("3", list.extract(0));
  EXPECT_EQ("4", list.extract(1));
}

TEST(ListTest_IValueBasedList, whenCallingExtractWithExistingPosition_thenListElementBecomesInvalid) {
  List<string> list({"3", "4"});
  list.extract(0);
  EXPECT_EQ("", list.get(0));
}

TEST(ListTest_IValueBasedList, whenCallingExtractWithNonExistingPosition_thenThrowsException) {
  List<string> list({"3", "4"});
  EXPECT_THROW(list.extract(2), std::out_of_range);
}

TEST(ListTest_IValueBasedList, whenCallingCopyingSetWithExistingPosition_thenChangesElement) {
  List<string> list({"3", "4"});
  string value = "5";
  list.set(1, value);
  EXPECT_EQ("3", list.get(0));
  EXPECT_EQ("5", list.get(1));
}

TEST(ListTest_IValueBasedList, whenCallingMovingSetWithExistingPosition_thenChangesElement) {
  List<string> list({"3", "4"});
  string value = "5";
  list.set(1, std::move(value));
  EXPECT_EQ("3", list.get(0));
  EXPECT_EQ("5", list.get(1));
}

TEST(ListTest_IValueBasedList, whenCallingCopyingSetWithNonExistingPosition_thenThrowsException) {
  List<string> list({"3", "4"});
  string value = "5";
  EXPECT_THROW(list.set(2, value), std::out_of_range);
}

TEST(ListTest_IValueBasedList, whenCallingMovingSetWithNonExistingPosition_thenThrowsException) {
  List<string> list({"3", "4"});
  string value = "5";
  EXPECT_THROW(list.set(2, std::move(value)), std::out_of_range);
}

TEST(ListTest_IValueBasedList, whenCallingAccessOperatorWithExistingPosition_thenReturnsElement) {
  List<string> list({"3", "4"});
  EXPECT_EQ("3", static_cast<string>(list[0]));
  EXPECT_EQ("4", static_cast<string>(list[1]));
}

TEST(ListTest_IValueBasedList, whenAssigningToAccessOperatorWithExistingPosition_thenSetsElement) {
  List<string> list({"3", "4", "5"});
  list[1] = "6";
  EXPECT_EQ("3", list.get(0));
  EXPECT_EQ("6", list.get(1));
  EXPECT_EQ("5", list.get(2));
}

TEST(ListTest_IValueBasedList, whenAssigningToAccessOperatorFromAccessOperator_thenSetsElement) {
  List<string> list({"3", "4", "5"});
  list[1] = list[2];
  EXPECT_EQ("3", list.get(0));
  EXPECT_EQ("5", list.get(1));
  EXPECT_EQ("5", list.get(2));
}

TEST(ListTest_IValueBasedList, whenSwappingFromAccessOperator_thenSwapsElements) {
  List<string> list({"3", "4", "5"});
  swap(list[1], list[2]);
  EXPECT_EQ("3", list.get(0));
  EXPECT_EQ("5", list.get(1));
  EXPECT_EQ("4", list.get(2));
}

TEST(ListTest_IValueBasedList, whenCallingAccessOperatorWithNonExistingPosition_thenThrowsException) {
  List<string> list({"3", "4"});
  EXPECT_THROW(list[2], std::out_of_range);
}

TEST(ListTest_IValueBasedList, whenCallingInsertOnIteratorWithLValue_thenInsertsElement) {
  List<string> list({"3", "4", "6"});
  string v = "5";
  list.insert(list.begin() + 2, v);
  EXPECT_EQ(4, list.size());
  EXPECT_EQ("5", list.get(2));
}

TEST(ListTest_IValueBasedList, whenCallingInsertOnIteratorWithRValue_thenInsertsElement) {
  List<string> list({"3", "4", "6"});
  string v = "5";
  list.insert(list.begin() + 2, std::move(v));
  EXPECT_EQ(4, list.size());
  EXPECT_EQ("5", list.get(2));
}

TEST(ListTest_IValueBasedList, whenCallingInsertWithLValue_thenReturnsIteratorToNewElement) {
  List<string> list({"3", "4", "6"});
  string v = "5";
  List<string>::iterator result = list.insert(list.begin() + 2, v);
  EXPECT_EQ(list.begin() + 2, result);
}

TEST(ListTest_IValueBasedList, whenCallingInsertWithRValue_thenReturnsIteratorToNewElement) {
  List<string> list({"3", "4", "6"});
  string v = "5";
  List<string>::iterator result = list.insert(list.begin() + 2, std::move(v));
  EXPECT_EQ(list.begin() + 2, result);
}

TEST(ListTest_IValueBasedList, whenCallingEmplaceWithLValue_thenInsertsElement) {
  List<string> list({"3", "4", "6"});
  string v = "5";
  list.emplace(list.begin() + 2, v);
  EXPECT_EQ(4, list.size());
  EXPECT_EQ("5", list.get(2));
}

TEST(ListTest_IValueBasedList, whenCallingEmplaceWithRValue_thenInsertsElement) {
  List<string> list({"3", "4", "6"});
  string v = "5";
  list.emplace(list.begin() + 2, std::move(v));
  EXPECT_EQ(4, list.size());
  EXPECT_EQ("5", list.get(2));
}

TEST(ListTest_IValueBasedList, whenCallingEmplaceWithConstructorArg_thenInsertsElement) {
  List<string> list({"3", "4", "6"});
  list.emplace(list.begin() + 2, "5"); // const char* is a constructor arg to std::string
  EXPECT_EQ(4, list.size());
  EXPECT_EQ("5", list.get(2));
}

TEST(ListTest_IValueBasedList, whenCallingPushBackWithLValue_ThenInsertsElement) {
  List<string> list;
  string v = "5";
  list.push_back(v);
  EXPECT_EQ(1, list.size());
  EXPECT_EQ("5", list.get(0));
}

TEST(ListTest_IValueBasedList, whenCallingPushBackWithRValue_ThenInsertsElement) {
  List<string> list;
  string v = "5";
  list.push_back(std::move(v));
  EXPECT_EQ(1, list.size());
  EXPECT_EQ("5", list.get(0));
}

TEST(ListTest_IValueBasedList, whenCallingEmplaceBackWithLValue_ThenInsertsElement) {
  List<string> list;
  string v = "5";
  list.emplace_back(v);
  EXPECT_EQ(1, list.size());
  EXPECT_EQ("5", list.get(0));
}

TEST(ListTest_IValueBasedList, whenCallingEmplaceBackWithRValue_ThenInsertsElement) {
  List<string> list;
  string v = "5";
  list.emplace_back(std::move(v));
  EXPECT_EQ(1, list.size());
  EXPECT_EQ("5", list.get(0));
}

TEST(ListTest_IValueBasedList, whenCallingEmplaceBackWithConstructorArg_ThenInsertsElement) {
  List<string> list;
  list.emplace_back("5");  // const char* is a constructor arg to std::string
  EXPECT_EQ(1, list.size());
  EXPECT_EQ("5", list.get(0));
}

TEST(ListTest_IValueBasedList, givenEmptyList_whenIterating_thenBeginIsEnd) {
  List<string> list;
  const List<string> clist;
  EXPECT_EQ(list.begin(), list.end());
  EXPECT_EQ(list.begin(), list.end());
  EXPECT_EQ(clist.begin(), clist.end());
  EXPECT_EQ(clist.begin(), clist.end());
}

TEST(ListTest_IValueBasedList, whenIterating_thenFindsElements) {
  List<string> list({"3", "5"});
  bool found_first = false;
  bool found_second = false;
  for (List<string>::iterator iter = list.begin(); iter != list.end(); ++iter) {
    if (static_cast<string>(*iter) == "3") {
      EXPECT_FALSE(found_first);
      found_first = true;
    } else if (static_cast<string>(*iter) == "5") {
      EXPECT_FALSE(found_second);
      found_second = true;
    } else {
      ADD_FAILURE();
    }
  }
  EXPECT_TRUE(found_first);
  EXPECT_TRUE(found_second);
}

TEST(ListTest_IValueBasedList, whenIteratingWithForeach_thenFindsElements) {
  List<string> list({"3", "5"});
  bool found_first = false;
  bool found_second = false;
  for (const string& elem : list) {
    if (elem == "3") {
      EXPECT_FALSE(found_first);
      found_first = true;
    } else if (elem == "5") {
      EXPECT_FALSE(found_second);
      found_second = true;
    } else {
      ADD_FAILURE();
    }
  }
  EXPECT_TRUE(found_first);
  EXPECT_TRUE(found_second);
}

TEST(ListTest_IValueBasedList, givenOneElementList_whenErasing_thenListIsEmpty) {
  List<string> list({"3"});
  list.erase(list.begin());
  EXPECT_TRUE(list.empty());
}

TEST(ListTest_IValueBasedList, givenList_whenErasing_thenReturnsIterator) {
  List<string> list({"1", "2", "3"});
  List<string>::iterator iter = list.erase(list.begin() + 1);
  EXPECT_EQ(list.begin() + 1, iter);
}

TEST(ListTest_IValueBasedList, givenList_whenErasingFullRange_thenIsEmpty) {
  List<string> list({"1", "2", "3"});
  list.erase(list.begin(), list.end());
  EXPECT_TRUE(list.empty());
}

TEST(ListTest_IValueBasedList, whenCallingReserve_thenDoesntCrash) {
  List<string> list;
  list.reserve(100);
}

TEST(ListTest_IValueBasedList, whenCopyConstructingList_thenAreEqual) {
  List<string> list1({"3", "4"});

  List<string> list2(list1);

  EXPECT_EQ(2, list2.size());
  EXPECT_EQ("3", list2.get(0));
  EXPECT_EQ("4", list2.get(1));
}

TEST(ListTest_IValueBasedList, whenCopyAssigningList_thenAreEqual) {
  List<string> list1({"3", "4"});

  List<string> list2;
  list2 = list1;

  EXPECT_EQ(2, list2.size());
  EXPECT_EQ("3", list2.get(0));
  EXPECT_EQ("4", list2.get(1));
}

TEST(ListTest_IValueBasedList, whenCopyingList_thenAreEqual) {
  List<string> list1({"3", "4"});

  List<string> list2 = list1.copy();

  EXPECT_EQ(2, list2.size());
  EXPECT_EQ("3", list2.get(0));
  EXPECT_EQ("4", list2.get(1));
}

TEST(ListTest_IValueBasedList, whenMoveConstructingList_thenNewIsCorrect) {
  List<string> list1({"3", "4"});

  List<string> list2(std::move(list1));

  EXPECT_EQ(2, list2.size());
  EXPECT_EQ("3", list2.get(0));
  EXPECT_EQ("4", list2.get(1));
}

TEST(ListTest_IValueBasedList, whenMoveAssigningList_thenNewIsCorrect) {
  List<string> list1({"3", "4"});

  List<string> list2;
  list2 = std::move(list1);

  EXPECT_EQ(2, list2.size());
  EXPECT_EQ("3", list2.get(0));
  EXPECT_EQ("4", list2.get(1));
}

TEST(ListTest_IValueBasedList, whenMoveConstructingList_thenOldIsEmpty) {
  List<string> list1({"3", "4"});

  List<string> list2(std::move(list1));
  EXPECT_TRUE(list1.empty());
}

TEST(ListTest_IValueBasedList, whenMoveAssigningList_thenOldIsEmpty) {
  List<string> list1({"3", "4"});

  List<string> list2;
  list2 = std::move(list1);
  EXPECT_TRUE(list1.empty());
}

TEST(ListTest_IValueBasedList, givenIterator_whenPostfixIncrementing_thenMovesToNextAndReturnsOldPosition) {
  List<string> list({"3", "4"});

  List<string>::iterator iter1 = list.begin();
  List<string>::iterator iter2 = iter1++;
  EXPECT_NE("3", static_cast<string>(*iter1));
  EXPECT_EQ("3", static_cast<string>(*iter2));
}

TEST(ListTest_IValueBasedList, givenIterator_whenPrefixIncrementing_thenMovesToNextAndReturnsNewPosition) {
  List<string> list({"3", "4"});

  List<string>::iterator iter1 = list.begin();
  List<string>::iterator iter2 = ++iter1;
  EXPECT_NE("3", static_cast<string>(*iter1));
  EXPECT_NE("3", static_cast<string>(*iter2));
}

TEST(ListTest_IValueBasedList, givenIterator_whenPostfixDecrementing_thenMovesToNextAndReturnsOldPosition) {
  List<string> list({"3", "4"});

  List<string>::iterator iter1 = list.end() - 1;
  List<string>::iterator iter2 = iter1--;
  EXPECT_NE("4", static_cast<string>(*iter1));
  EXPECT_EQ("4", static_cast<string>(*iter2));
}

TEST(ListTest_IValueBasedList, givenIterator_whenPrefixDecrementing_thenMovesToNextAndReturnsNewPosition) {
  List<string> list({"3", "4"});

  List<string>::iterator iter1 = list.end() - 1;
  List<string>::iterator iter2 = --iter1;
  EXPECT_NE("4", static_cast<string>(*iter1));
  EXPECT_NE("4", static_cast<string>(*iter2));
}

TEST(ListTest_IValueBasedList, givenIterator_whenIncreasing_thenMovesToNextAndReturnsNewPosition) {
  List<string> list({"3", "4", "5"});

  List<string>::iterator iter1 = list.begin();
  List<string>::iterator iter2 = iter1 += 2;
  EXPECT_EQ("5", static_cast<string>(*iter1));
  EXPECT_EQ("5", static_cast<string>(*iter2));
}

TEST(ListTest_IValueBasedList, givenIterator_whenDecreasing_thenMovesToNextAndReturnsNewPosition) {
  List<string> list({"3", "4", "5"});

  List<string>::iterator iter1 = list.end();
  List<string>::iterator iter2 = iter1 -= 2;
  EXPECT_EQ("4", static_cast<string>(*iter1));
  EXPECT_EQ("4", static_cast<string>(*iter2));
}

TEST(ListTest_IValueBasedList, givenIterator_whenAdding_thenReturnsNewIterator) {
  List<string> list({"3", "4", "5"});

  List<string>::iterator iter1 = list.begin();
  List<string>::iterator iter2 = iter1 + 2;
  EXPECT_EQ("3", static_cast<string>(*iter1));
  EXPECT_EQ("5", static_cast<string>(*iter2));
}

TEST(ListTest_IValueBasedList, givenIterator_whenSubtracting_thenReturnsNewIterator) {
  List<string> list({"3", "4", "5"});

  List<string>::iterator iter1 = list.end() - 1;
  List<string>::iterator iter2 = iter1 - 2;
  EXPECT_EQ("5", static_cast<string>(*iter1));
  EXPECT_EQ("3", static_cast<string>(*iter2));
}

TEST(ListTest_IValueBasedList, givenIterator_whenCalculatingDifference_thenReturnsCorrectNumber) {
  List<string> list({"3", "4"});
  EXPECT_EQ(2, list.end() - list.begin());
}

TEST(ListTest_IValueBasedList, givenEqualIterators_thenAreEqual) {
  List<string> list({"3", "4"});

  List<string>::iterator iter1 = list.begin();
  List<string>::iterator iter2 = list.begin();
  EXPECT_TRUE(iter1 == iter2);
  EXPECT_FALSE(iter1 != iter2);
}

TEST(ListTest_IValueBasedList, givenDifferentIterators_thenAreNotEqual) {
  List<string> list({"3", "4"});

  List<string>::iterator iter1 = list.begin();
  List<string>::iterator iter2 = list.begin();
  iter2++;

  EXPECT_FALSE(iter1 == iter2);
  EXPECT_TRUE(iter1 != iter2);
}

TEST(ListTest_IValueBasedList, givenIterator_whenDereferencing_thenPointsToCorrectElement) {
  List<string> list({"3"});

  List<string>::iterator iter = list.begin();
  EXPECT_EQ("3", static_cast<string>(*iter));
}

TEST(ListTest_IValueBasedList, givenIterator_whenAssigningNewValue_thenChangesValue) {
  List<string> list({"3"});

  List<string>::iterator iter = list.begin();
  *iter = "4";
  EXPECT_EQ("4", list.get(0));
}

TEST(ListTest_IValueBasedList, givenIterator_whenAssigningNewValueFromIterator_thenChangesValue) {
  List<string> list({"3", "4"});

  List<string>::iterator iter = list.begin();
  *iter = *(iter + 1);
  EXPECT_EQ("4", list.get(0));
  EXPECT_EQ("4", list.get(1));
}

TEST(ListTest_IValueBasedList, givenIterator_whenSwappingValuesFromIterator_thenChangesValue) {
  List<string> list({"3", "4"});

  List<string>::iterator iter = list.begin();
  swap(*iter, *(iter + 1));
  EXPECT_EQ("4", list.get(0));
  EXPECT_EQ("3", list.get(1));
}

TEST(ListTest_IValueBasedList, givenOneElementList_whenCallingPopBack_thenIsEmpty) {
  List<string> list({"3"});
  list.pop_back();
  EXPECT_TRUE(list.empty());
}

TEST(ListTest_IValueBasedList, givenEmptyList_whenCallingResize_thenResizesAndSetsEmptyValue) {
  List<string> list;
  list.resize(2);
  EXPECT_EQ(2, list.size());
  EXPECT_EQ("", list.get(0));
  EXPECT_EQ("", list.get(1));
}

TEST(ListTest_IValueBasedList, givenEmptyList_whenCallingResizeWithValue_thenResizesAndSetsValue) {
  List<string> list;
  list.resize(2, "value");
  EXPECT_EQ(2, list.size());
  EXPECT_EQ("value", list.get(0));
  EXPECT_EQ("value", list.get(1));
}

TEST(ListTest_IValueBasedList, isReferenceType) {
  List<string> list1;
  List<string> list2(list1);
  List<string> list3;
  list3 = list1;

  list1.push_back("three");
  EXPECT_EQ(1, list1.size());
  EXPECT_EQ(1, list2.size());
  EXPECT_EQ(1, list3.size());
}

TEST(ListTest_IValueBasedList, copyHasSeparateStorage) {
  List<string> list1;
  List<string> list2(list1.copy());
  List<string> list3;
  list3 = list1.copy();

  list1.push_back("three");
  EXPECT_EQ(1, list1.size());
  EXPECT_EQ(0, list2.size());
  EXPECT_EQ(0, list3.size());
}

TEST(ListTest_IValueBasedList, givenEqualLists_thenIsEqual) {
  List<string> list1({"first", "second"});
  List<string> list2({"first", "second"});

  EXPECT_EQ(list1, list2);
}

TEST(ListTest_IValueBasedList, givenDifferentLists_thenIsNotEqual) {
  List<string> list1({"first", "second"});
  List<string> list2({"first", "not_second"});

  EXPECT_NE(list1, list2);
}

TEST(ListTest_NonIValueBasedList, givenEmptyList_whenCallingEmpty_thenReturnsTrue) {
    List<int64_t> list;
    EXPECT_TRUE(list.empty());
}

TEST(ListTest_NonIValueBasedList, givenNonemptyList_whenCallingEmpty_thenReturnsFalse) {
    List<int64_t> list({3});
    EXPECT_FALSE(list.empty());
}

TEST(ListTest_NonIValueBasedList, givenEmptyList_whenCallingSize_thenReturnsZero) {
    List<int64_t> list;
    EXPECT_EQ(0, list.size());
}

TEST(ListTest_NonIValueBasedList, givenNonemptyList_whenCallingSize_thenReturnsNumberOfElements) {
    List<int64_t> list({3, 4});
    EXPECT_EQ(2, list.size());
}

TEST(ListTest_NonIValueBasedList, givenNonemptyList_whenCallingClear_thenIsEmpty) {
  List<int64_t> list({3, 4});
  list.clear();
  EXPECT_TRUE(list.empty());
}

TEST(ListTest_NonIValueBasedList, whenCallingGetWithExistingPosition_thenReturnsElement) {
  List<int64_t> list({3, 4});
  EXPECT_EQ(3, list.get(0));
  EXPECT_EQ(4, list.get(1));
}

TEST(ListTest_NonIValueBasedList, whenCallingGetWithNonExistingPosition_thenThrowsException) {
  List<int64_t> list({3, 4});
  EXPECT_THROW(list.get(2), std::out_of_range);
}

TEST(ListTest_NonIValueBasedList, whenCallingExtractWithExistingPosition_thenReturnsElement) {
  List<int64_t> list({3, 4});
  EXPECT_EQ(3, list.extract(0));
  EXPECT_EQ(4, list.extract(1));
}

TEST(ListTest_NonIValueBasedList, whenCallingExtractWithNonExistingPosition_thenThrowsException) {
  List<int64_t> list({3, 4});
  EXPECT_THROW(list.extract(2), std::out_of_range);
}

TEST(ListTest_NonIValueBasedList, whenCallingCopyingSetWithExistingPosition_thenChangesElement) {
  List<int64_t> list({3, 4});
  int64_t value = 5;
  list.set(1, value);
  EXPECT_EQ(3, list.get(0));
  EXPECT_EQ(5, list.get(1));
}

TEST(ListTest_NonIValueBasedList, whenCallingMovingSetWithExistingPosition_thenChangesElement) {
  List<int64_t> list({3, 4});
  int64_t value = 5;
  list.set(1, std::move(value));
  EXPECT_EQ(3, list.get(0));
  EXPECT_EQ(5, list.get(1));
}

TEST(ListTest_NonIValueBasedList, whenCallingCopyingSetWithNonExistingPosition_thenThrowsException) {
  List<int64_t> list({3, 4});
  int64_t value = 5;
  EXPECT_THROW(list.set(2, value), std::out_of_range);
}

TEST(ListTest_NonIValueBasedList, whenCallingMovingSetWithNonExistingPosition_thenThrowsException) {
  List<int64_t> list({3, 4});
  int64_t value = 5;
  EXPECT_THROW(list.set(2, std::move(value)), std::out_of_range);
}

TEST(ListTest_NonIValueBasedList, whenCallingAccessOperatorWithExistingPosition_thenReturnsElement) {
  List<int64_t> list({3, 4});
  EXPECT_EQ(3, static_cast<int64_t>(list[0]));
  EXPECT_EQ(4, static_cast<int64_t>(list[1]));
}

TEST(ListTest_NonIValueBasedList, whenAssigningToAccessOperatorWithExistingPosition_thenSetsElement) {
  List<int64_t> list({3, 4, 5});
  list[1] = 6;
  EXPECT_EQ(3, list.get(0));
  EXPECT_EQ(6, list.get(1));
  EXPECT_EQ(5, list.get(2));
}

TEST(ListTest_NonIValueBasedList, whenAssigningToAccessOperatorFromAccessOperator_thenSetsElement) {
  List<int64_t> list({3, 4, 5});
  list[1] = list[2];
  EXPECT_EQ(3, list.get(0));
  EXPECT_EQ(5, list.get(1));
  EXPECT_EQ(5, list.get(2));
}

TEST(ListTest_NonIValueBasedList, whenSwappingFromAccessOperator_thenSwapsElements) {
  List<int64_t> list({3, 4, 5});
  swap(list[1], list[2]);
  EXPECT_EQ(3, list.get(0));
  EXPECT_EQ(5, list.get(1));
  EXPECT_EQ(4, list.get(2));
}

TEST(ListTest_NonIValueBasedList, whenCallingAccessOperatorWithNonExistingPosition_thenThrowsException) {
  List<int64_t> list({3, 4});
  EXPECT_THROW(list[2], std::out_of_range);
}

TEST(ListTest_NonIValueBasedList, whenCallingInsertOnIteratorWithLValue_thenInsertsElement) {
  List<int64_t> list({3, 4, 6});
  int64_t v = 5;
  list.insert(list.begin() + 2, v);
  EXPECT_EQ(4, list.size());
  EXPECT_EQ(5, list.get(2));
}

TEST(ListTest_NonIValueBasedList, whenCallingInsertOnIteratorWithRValue_thenInsertsElement) {
  List<int64_t> list({3, 4, 6});
  int64_t v = 5;
  list.insert(list.begin() + 2, std::move(v));
  EXPECT_EQ(4, list.size());
  EXPECT_EQ(5, list.get(2));
}

TEST(ListTest_NonIValueBasedList, whenCallingInsertWithLValue_thenReturnsIteratorToNewElement) {
  List<int64_t> list({3, 4, 6});
  int64_t v = 5;
  List<int64_t>::iterator result = list.insert(list.begin() + 2, v);
  EXPECT_EQ(list.begin() + 2, result);
}

TEST(ListTest_NonIValueBasedList, whenCallingInsertWithRValue_thenReturnsIteratorToNewElement) {
  List<int64_t> list({3, 4, 6});
  int64_t v = 5;
  List<int64_t>::iterator result = list.insert(list.begin() + 2, std::move(v));
  EXPECT_EQ(list.begin() + 2, result);
}

TEST(ListTest_NonIValueBasedList, whenCallingEmplaceWithLValue_thenInsertsElement) {
  List<int64_t> list({3, 4, 6});
  int64_t v = 5;
  list.emplace(list.begin() + 2, v);
  EXPECT_EQ(4, list.size());
  EXPECT_EQ(5, list.get(2));
}

TEST(ListTest_NonIValueBasedList, whenCallingEmplaceWithRValue_thenInsertsElement) {
  List<int64_t> list({3, 4, 6});
  int64_t v = 5;
  list.emplace(list.begin() + 2, std::move(v));
  EXPECT_EQ(4, list.size());
  EXPECT_EQ(5, list.get(2));
}

TEST(ListTest_NonIValueBasedList, whenCallingEmplaceWithConstructorArg_thenInsertsElement) {
  List<int64_t> list({3, 4, 6});
  list.emplace(list.begin() + 2, 5); // const char* is a constructor arg to std::int64_t
  EXPECT_EQ(4, list.size());
  EXPECT_EQ(5, list.get(2));
}

TEST(ListTest_NonIValueBasedList, whenCallingPushBackWithLValue_ThenInsertsElement) {
  List<int64_t> list;
  int64_t v = 5;
  list.push_back(v);
  EXPECT_EQ(1, list.size());
  EXPECT_EQ(5, list.get(0));
}

TEST(ListTest_NonIValueBasedList, whenCallingPushBackWithRValue_ThenInsertsElement) {
  List<int64_t> list;
  int64_t v = 5;
  list.push_back(std::move(v));
  EXPECT_EQ(1, list.size());
  EXPECT_EQ(5, list.get(0));
}

TEST(ListTest_NonIValueBasedList, whenCallingEmplaceBackWithLValue_ThenInsertsElement) {
  List<int64_t> list;
  int64_t v = 5;
  list.emplace_back(v);
  EXPECT_EQ(1, list.size());
  EXPECT_EQ(5, list.get(0));
}

TEST(ListTest_NonIValueBasedList, whenCallingEmplaceBackWithRValue_ThenInsertsElement) {
  List<int64_t> list;
  int64_t v = 5;
  list.emplace_back(std::move(v));
  EXPECT_EQ(1, list.size());
  EXPECT_EQ(5, list.get(0));
}

TEST(ListTest_NonIValueBasedList, whenCallingEmplaceBackWithConstructorArg_ThenInsertsElement) {
  List<int64_t> list;
  list.emplace_back(5);  // const char* is a constructor arg to std::int64_t
  EXPECT_EQ(1, list.size());
  EXPECT_EQ(5, list.get(0));
}

TEST(ListTest_NonIValueBasedList, givenEmptyList_whenIterating_thenBeginIsEnd) {
  List<int64_t> list;
  const List<int64_t> clist;
  EXPECT_EQ(list.begin(), list.end());
  EXPECT_EQ(list.begin(), list.end());
  EXPECT_EQ(clist.begin(), clist.end());
  EXPECT_EQ(clist.begin(), clist.end());
}

TEST(ListTest_NonIValueBasedList, whenIterating_thenFindsElements) {
  List<int64_t> list({3, 5});
  bool found_first = false;
  bool found_second = false;
  for (List<int64_t>::iterator iter = list.begin(); iter != list.end(); ++iter) {
    if (static_cast<int64_t>(*iter) == 3) {
      EXPECT_FALSE(found_first);
      found_first = true;
    } else if (static_cast<int64_t>(*iter) == 5) {
      EXPECT_FALSE(found_second);
      found_second = true;
    } else {
      ADD_FAILURE();
    }
  }
  EXPECT_TRUE(found_first);
  EXPECT_TRUE(found_second);
}

TEST(ListTest_NonIValueBasedList, whenIteratingWithForeach_thenFindsElements) {
  List<int64_t> list({3, 5});
  bool found_first = false;
  bool found_second = false;
  for (const int64_t& elem : list) {
    if (elem == 3) {
      EXPECT_FALSE(found_first);
      found_first = true;
    } else if (elem == 5) {
      EXPECT_FALSE(found_second);
      found_second = true;
    } else {
      ADD_FAILURE();
    }
  }
  EXPECT_TRUE(found_first);
  EXPECT_TRUE(found_second);
}

TEST(ListTest_NonIValueBasedList, givenOneElementList_whenErasing_thenListIsEmpty) {
  List<int64_t> list({3});
  list.erase(list.begin());
  EXPECT_TRUE(list.empty());
}

TEST(ListTest_NonIValueBasedList, givenList_whenErasing_thenReturnsIterator) {
  List<int64_t> list({1, 2, 3});
  List<int64_t>::iterator iter = list.erase(list.begin() + 1);
  EXPECT_EQ(list.begin() + 1, iter);
}

TEST(ListTest_NonIValueBasedList, givenList_whenErasingFullRange_thenIsEmpty) {
  List<int64_t> list({1, 2, 3});
  list.erase(list.begin(), list.end());
  EXPECT_TRUE(list.empty());
}

TEST(ListTest_NonIValueBasedList, whenCallingReserve_thenDoesntCrash) {
  List<int64_t> list;
  list.reserve(100);
}

TEST(ListTest_NonIValueBasedList, whenCopyConstructingList_thenAreEqual) {
  List<int64_t> list1({3, 4});

  List<int64_t> list2(list1);

  EXPECT_EQ(2, list2.size());
  EXPECT_EQ(3, list2.get(0));
  EXPECT_EQ(4, list2.get(1));
}

TEST(ListTest_NonIValueBasedList, whenCopyAssigningList_thenAreEqual) {
  List<int64_t> list1({3, 4});

  List<int64_t> list2;
  list2 = list1;

  EXPECT_EQ(2, list2.size());
  EXPECT_EQ(3, list2.get(0));
  EXPECT_EQ(4, list2.get(1));
}

TEST(ListTest_NonIValueBasedList, whenCopyingList_thenAreEqual) {
  List<int64_t> list1({3, 4});

  List<int64_t> list2 = list1.copy();

  EXPECT_EQ(2, list2.size());
  EXPECT_EQ(3, list2.get(0));
  EXPECT_EQ(4, list2.get(1));
}

TEST(ListTest_NonIValueBasedList, whenMoveConstructingList_thenNewIsCorrect) {
  List<int64_t> list1({3, 4});

  List<int64_t> list2(std::move(list1));

  EXPECT_EQ(2, list2.size());
  EXPECT_EQ(3, list2.get(0));
  EXPECT_EQ(4, list2.get(1));
}

TEST(ListTest_NonIValueBasedList, whenMoveAssigningList_thenNewIsCorrect) {
  List<int64_t> list1({3, 4});

  List<int64_t> list2;
  list2 = std::move(list1);

  EXPECT_EQ(2, list2.size());
  EXPECT_EQ(3, list2.get(0));
  EXPECT_EQ(4, list2.get(1));
}

TEST(ListTest_NonIValueBasedList, whenMoveConstructingList_thenOldIsEmpty) {
  List<int64_t> list1({3, 4});

  List<int64_t> list2(std::move(list1));
  EXPECT_TRUE(list1.empty());
}

TEST(ListTest_NonIValueBasedList, whenMoveAssigningList_thenOldIsEmpty) {
  List<int64_t> list1({3, 4});

  List<int64_t> list2;
  list2 = std::move(list1);
  EXPECT_TRUE(list1.empty());
}

TEST(ListTest_NonIValueBasedList, givenIterator_whenPostfixIncrementing_thenMovesToNextAndReturnsOldPosition) {
  List<int64_t> list({3, 4});

  List<int64_t>::iterator iter1 = list.begin();
  List<int64_t>::iterator iter2 = iter1++;
  EXPECT_NE(3, static_cast<int64_t>(*iter1));
  EXPECT_EQ(3, static_cast<int64_t>(*iter2));
}

TEST(ListTest_NonIValueBasedList, givenIterator_whenPrefixIncrementing_thenMovesToNextAndReturnsNewPosition) {
  List<int64_t> list({3, 4});

  List<int64_t>::iterator iter1 = list.begin();
  List<int64_t>::iterator iter2 = ++iter1;
  EXPECT_NE(3, static_cast<int64_t>(*iter1));
  EXPECT_NE(3, static_cast<int64_t>(*iter2));
}

TEST(ListTest_NonIValueBasedList, givenIterator_whenPostfixDecrementing_thenMovesToNextAndReturnsOldPosition) {
  List<int64_t> list({3, 4});

  List<int64_t>::iterator iter1 = list.end() - 1;
  List<int64_t>::iterator iter2 = iter1--;
  EXPECT_NE(4, static_cast<int64_t>(*iter1));
  EXPECT_EQ(4, static_cast<int64_t>(*iter2));
}

TEST(ListTest_NonIValueBasedList, givenIterator_whenPrefixDecrementing_thenMovesToNextAndReturnsNewPosition) {
  List<int64_t> list({3, 4});

  List<int64_t>::iterator iter1 = list.end() - 1;
  List<int64_t>::iterator iter2 = --iter1;
  EXPECT_NE(4, static_cast<int64_t>(*iter1));
  EXPECT_NE(4, static_cast<int64_t>(*iter2));
}

TEST(ListTest_NonIValueBasedList, givenIterator_whenIncreasing_thenMovesToNextAndReturnsNewPosition) {
  List<int64_t> list({3, 4, 5});

  List<int64_t>::iterator iter1 = list.begin();
  List<int64_t>::iterator iter2 = iter1 += 2;
  EXPECT_EQ(5, static_cast<int64_t>(*iter1));
  EXPECT_EQ(5, static_cast<int64_t>(*iter2));
}

TEST(ListTest_NonIValueBasedList, givenIterator_whenDecreasing_thenMovesToNextAndReturnsNewPosition) {
  List<int64_t> list({3, 4, 5});

  List<int64_t>::iterator iter1 = list.end();
  List<int64_t>::iterator iter2 = iter1 -= 2;
  EXPECT_EQ(4, static_cast<int64_t>(*iter1));
  EXPECT_EQ(4, static_cast<int64_t>(*iter2));
}

TEST(ListTest_NonIValueBasedList, givenIterator_whenAdding_thenReturnsNewIterator) {
  List<int64_t> list({3, 4, 5});

  List<int64_t>::iterator iter1 = list.begin();
  List<int64_t>::iterator iter2 = iter1 + 2;
  EXPECT_EQ(3, static_cast<int64_t>(*iter1));
  EXPECT_EQ(5, static_cast<int64_t>(*iter2));
}

TEST(ListTest_NonIValueBasedList, givenIterator_whenSubtracting_thenReturnsNewIterator) {
  List<int64_t> list({3, 4, 5});

  List<int64_t>::iterator iter1 = list.end() - 1;
  List<int64_t>::iterator iter2 = iter1 - 2;
  EXPECT_EQ(5, static_cast<int64_t>(*iter1));
  EXPECT_EQ(3, static_cast<int64_t>(*iter2));
}

TEST(ListTest_NonIValueBasedList, givenIterator_whenCalculatingDifference_thenReturnsCorrectNumber) {
  List<int64_t> list({3, 4});
  EXPECT_EQ(2, list.end() - list.begin());
}

TEST(ListTest_NonIValueBasedList, givenEqualIterators_thenAreEqual) {
  List<int64_t> list({3, 4});

  List<int64_t>::iterator iter1 = list.begin();
  List<int64_t>::iterator iter2 = list.begin();
  EXPECT_TRUE(iter1 == iter2);
  EXPECT_FALSE(iter1 != iter2);
}

TEST(ListTest_NonIValueBasedList, givenDifferentIterators_thenAreNotEqual) {
  List<int64_t> list({3, 4});

  List<int64_t>::iterator iter1 = list.begin();
  List<int64_t>::iterator iter2 = list.begin();
  iter2++;

  EXPECT_FALSE(iter1 == iter2);
  EXPECT_TRUE(iter1 != iter2);
}

TEST(ListTest_NonIValueBasedList, givenIterator_whenDereferencing_thenPointsToCorrectElement) {
  List<int64_t> list({3});

  List<int64_t>::iterator iter = list.begin();
  EXPECT_EQ(3, static_cast<int64_t>(*iter));
}

TEST(ListTest_NonIValueBasedList, givenIterator_whenAssigningNewValue_thenChangesValue) {
  List<int64_t> list({3});

  List<int64_t>::iterator iter = list.begin();
  *iter = 4;
  EXPECT_EQ(4, list.get(0));
}

TEST(ListTest_NonIValueBasedList, givenIterator_whenAssigningNewValueFromIterator_thenChangesValue) {
  List<int64_t> list({3, 4});

  List<int64_t>::iterator iter = list.begin();
  *iter = *(iter + 1);
  EXPECT_EQ(4, list.get(0));
  EXPECT_EQ(4, list.get(1));
}

TEST(ListTest_NonIValueBasedList, givenIterator_whenSwappingValuesFromIterator_thenChangesValue) {
  List<int64_t> list({3, 4});

  List<int64_t>::iterator iter = list.begin();
  swap(*iter, *(iter + 1));
  EXPECT_EQ(4, list.get(0));
  EXPECT_EQ(3, list.get(1));
}

TEST(ListTest_NonIValueBasedList, givenOneElementList_whenCallingPopBack_thenIsEmpty) {
  List<int64_t> list({3});
  list.pop_back();
  EXPECT_TRUE(list.empty());
}

TEST(ListTest_NonIValueBasedList, givenEmptyList_whenCallingResize_thenResizesAndSetsEmptyValue) {
  List<int64_t> list;
  list.resize(2);
  EXPECT_EQ(2, list.size());
  EXPECT_EQ(0, list.get(0));
  EXPECT_EQ(0, list.get(1));
}

TEST(ListTest_NonIValueBasedList, givenEmptyList_whenCallingResizeWithValue_thenResizesAndSetsValue) {
  List<int64_t> list;
  list.resize(2, 5);
  EXPECT_EQ(2, list.size());
  EXPECT_EQ(5, list.get(0));
  EXPECT_EQ(5, list.get(1));
}

TEST(ListTest_NonIValueBasedList, isReferenceType) {
  List<int64_t> list1;
  List<int64_t> list2(list1);
  List<int64_t> list3;
  list3 = list1;

  list1.push_back(3);
  EXPECT_EQ(1, list1.size());
  EXPECT_EQ(1, list2.size());
  EXPECT_EQ(1, list3.size());
}

TEST(ListTest_NonIValueBasedList, copyHasSeparateStorage) {
  List<int64_t> list1;
  List<int64_t> list2(list1.copy());
  List<int64_t> list3;
  list3 = list1.copy();

  list1.push_back(3);
  EXPECT_EQ(1, list1.size());
  EXPECT_EQ(0, list2.size());
  EXPECT_EQ(0, list3.size());
}

TEST(ListTest_NonIValueBasedList, givenEqualLists_thenIsEqual) {
  List<int64_t> list1({1, 3});
  List<int64_t> list2({1, 3});

  EXPECT_EQ(list1, list2);
}

TEST(ListTest_NonIValueBasedList, givenDifferentLists_thenIsNotEqual) {
  List<int64_t> list1({1, 3});
  List<int64_t> list2({1, 2});

  EXPECT_NE(list1, list2);
}

TEST(ListTest_NonIValueBasedList, isChecksIdentity) {
  List<int64_t> list1({1, 3});
  const auto list2 = list1;

  EXPECT_TRUE(list1.is(list2));
}

TEST(ListTest_NonIValueBasedList, sameValueDifferentStorage_thenIsReturnsFalse) {
  List<int64_t> list1({1, 3});
  const auto list2 = list1.copy();

  EXPECT_FALSE(list1.is(list2));
}

TEST(ListTest, canAccessStringByReference) {
  List<std::string> list({"one", "two"});
  const auto& listRef = list;
  static_assert(std::is_same<decltype(listRef[1]), const std::string&>::value,
                "const List<std::string> acccess should be by const reference");
  std::string str = list[1];
  const std::string& strRef = listRef[1];
  EXPECT_EQ("two", str);
  EXPECT_EQ("two", strRef);
}

TEST(ListTest, canAccessOptionalStringByReference) {
  List<c10::optional<std::string>> list({"one", "two", c10::nullopt});
  const auto& listRef = list;
  static_assert(
      std::is_same<decltype(listRef[1]), c10::optional<std::reference_wrapper<const std::string>>>::value,
      "List<c10::optional<std::string>> acccess should be by const reference");
  c10::optional<std::string> str1 = list[1];
  c10::optional<std::string> str2 = list[2];
  decltype(auto) strRef1 = listRef[1];
  decltype(auto) strRef2 = listRef[2];
  EXPECT_EQ("two", str1.value());
  EXPECT_FALSE(str2.has_value());
  EXPECT_EQ("two", strRef1.value().get());
  EXPECT_FALSE(strRef2.has_value());
}

TEST(ListTest, canAccessTensorByReference) {
  List<at::Tensor> list;
  const auto& listRef = list;
  static_assert(
      std::is_same<decltype(listRef[0]), const at::Tensor&>::value,
      "List<at::Tensor> access should be by const reference");
}
