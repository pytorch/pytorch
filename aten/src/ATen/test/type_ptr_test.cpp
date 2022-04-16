#include <gtest/gtest.h>
#include <ATen/core/type_ptr.h>
#include <ATen/core/jit_type.h>

using c10::SingletonOrSharedTypePtr;

namespace {

TEST(SingletonOrSharedTypePtr, Empty) {
  SingletonOrSharedTypePtr<int> empty;
  EXPECT_TRUE(!empty);
  EXPECT_EQ(nullptr, empty.get());
  EXPECT_EQ(empty, nullptr);
  std::shared_ptr<int> emptyShared;
  EXPECT_EQ(emptyShared, empty);
}

TEST(SingletonOrSharedTypePtr, NonEmpty) {
  auto shared = std::make_shared<int>(42);
  SingletonOrSharedTypePtr<int> p(shared);
  EXPECT_EQ(42, *shared);
  EXPECT_TRUE(shared);
  EXPECT_EQ(42, *p);
  EXPECT_TRUE(p);
  EXPECT_NE(nullptr, p.get());
  EXPECT_NE(p, nullptr);
  EXPECT_EQ(shared, p);
  EXPECT_EQ(shared.get(), p.get());
}

TEST(SingletonOrSharedTypePtr, Comparison) {
  SingletonOrSharedTypePtr<int> empty;
  auto shared = std::make_shared<int>(42);
  SingletonOrSharedTypePtr<int> p(shared);
  auto shared2 = std::make_shared<int>(3);
  SingletonOrSharedTypePtr<int> p2(shared2);

  EXPECT_NE(empty, p);
  EXPECT_NE(p, p2);
}

TEST(SingletonOrSharedTypePtr, SingletonComparison) {
  EXPECT_NE(c10::StringType::get(), c10::NoneType::get());
  EXPECT_NE(c10::StringType::get(), c10::DeviceObjType::get());
  EXPECT_NE(c10::NoneType::get(), c10::DeviceObjType::get());

  c10::TypePtr type = c10::NoneType::get();
  EXPECT_NE(type, c10::StringType::get());
  EXPECT_NE(type, c10::DeviceObjType::get());
}


} // namespace
