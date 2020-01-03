#include <gtest/gtest.h>

#include <c10/core/TensorTypeSet.h>

using namespace c10;

TEST(TensorTypeSet, Empty) {
  TensorTypeSet empty_set;
  for (uint8_t i = 1; i < static_cast<uint8_t>(TensorTypeId::NumTensorIds); i++) {
    auto tid = static_cast<TensorTypeId>(i);
    ASSERT_FALSE(empty_set.has(tid));
  }
  ASSERT_TRUE(empty_set.empty());
  TensorTypeSet empty_set2;
  ASSERT_TRUE(empty_set == empty_set2);
  ASSERT_EQ(empty_set.highestPriorityTypeId(), TensorTypeId::UndefinedTensorId);
}

TEST(TensorTypeSet, Singleton) {
  for (uint8_t i = 1; i < static_cast<uint8_t>(TensorTypeId::NumTensorIds); i++) {
    auto tid = static_cast<TensorTypeId>(i);
    TensorTypeSet sing(tid);
    ASSERT_EQ(sing, sing);
    ASSERT_EQ(sing, TensorTypeSet().add(tid));
    ASSERT_EQ(sing, sing.add(tid));
    ASSERT_EQ(sing, sing | sing);
    ASSERT_FALSE(sing.empty());
    ASSERT_TRUE(sing.has(tid));
    ASSERT_EQ(sing.highestPriorityTypeId(), tid);
    ASSERT_EQ(sing.remove(tid), TensorTypeSet());
  }
}

TEST(TensorTypeSet, Doubleton) {
  for (uint8_t i = 1; i < static_cast<uint8_t>(TensorTypeId::NumTensorIds); i++) {
    for (uint8_t j = i + 1; j < static_cast<uint8_t>(TensorTypeId::NumTensorIds); j++) {
      ASSERT_LT(i, j);
      auto tid1 = static_cast<TensorTypeId>(i);
      auto tid2 = static_cast<TensorTypeId>(j);
      auto doub = TensorTypeSet(tid1).add(tid2);
      ASSERT_EQ(doub, TensorTypeSet(tid1) | TensorTypeSet(tid2));
      ASSERT_TRUE(doub.has(tid1));
      ASSERT_TRUE(doub.has(tid2));
      ASSERT_EQ(doub.highestPriorityTypeId(), tid2);  // relies on i < j
    }
  }
}

TEST(TensorTypeSet, Full) {
  TensorTypeSet full(TensorTypeSet::FULL);
  for (uint8_t i = 1; i < static_cast<uint8_t>(TensorTypeId::NumTensorIds); i++) {
    auto tid = static_cast<TensorTypeId>(i);
    ASSERT_TRUE(full.has(tid));
  }
}
