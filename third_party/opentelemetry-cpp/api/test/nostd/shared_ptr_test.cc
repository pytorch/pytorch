// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

#include "opentelemetry/nostd/shared_ptr.h"

#include <gtest/gtest.h>

#include <algorithm>

using opentelemetry::nostd::shared_ptr;

class A
{
public:
  explicit A(bool &destructed) noexcept : destructed_{destructed} { destructed_ = false; }

  ~A() { destructed_ = true; }

private:
  bool &destructed_;
};

class B
{
public:
  int f() const { return 123; }
};

class C
{
public:
  virtual ~C() {}
};

class D : public C
{
public:
  virtual ~D() {}
};

TEST(SharedPtrTest, DefaultConstruction)
{
  shared_ptr<int> ptr1;
  EXPECT_EQ(ptr1.get(), nullptr);

  shared_ptr<int> ptr2{nullptr};
  EXPECT_EQ(ptr2.get(), nullptr);
}

TEST(SharedPtrTest, ExplicitConstruction)
{
  auto c = new C();
  shared_ptr<C> ptr1{c};
  EXPECT_EQ(ptr1.get(), c);

  auto d = new D();
  shared_ptr<C> ptr2{d};
  EXPECT_EQ(ptr2.get(), d);
}

TEST(SharedPtrTest, MoveConstruction)
{
  auto value = new int{123};
  shared_ptr<int> ptr1{value};
  shared_ptr<int> ptr2{std::move(ptr1)};
  EXPECT_EQ(ptr1.get(), nullptr);
  EXPECT_EQ(ptr2.get(), value);
}

TEST(SharedPtrTest, MoveConstructionFromDifferentType)
{
  auto value = new int{123};
  shared_ptr<int> ptr1{value};
  shared_ptr<const int> ptr2{std::move(ptr1)};
  EXPECT_EQ(ptr1.get(), nullptr);
  EXPECT_EQ(ptr2.get(), value);
}

TEST(SharedPtrTest, MoveConstructionFromStdSharedPtr)
{
  auto value = new int{123};
  std::shared_ptr<int> ptr1{value};
  shared_ptr<int> ptr2{std::move(ptr1)};
  EXPECT_EQ(ptr1.get(), nullptr);
  EXPECT_EQ(ptr2.get(), value);
}

TEST(SharedPtrTest, Destruction)
{
  bool was_destructed;
  shared_ptr<A>{new A{was_destructed}};
  EXPECT_TRUE(was_destructed);
}

TEST(SharedPtrTest, Assignment)
{
  auto value = new int{123};
  shared_ptr<int> ptr1;

  ptr1 = shared_ptr<int>(value);
  EXPECT_EQ(ptr1.get(), value);

  ptr1 = nullptr;
  EXPECT_EQ(ptr1.get(), nullptr);

  auto value2                = new int{234};
  const shared_ptr<int> ptr2 = shared_ptr<int>(value2);
  ptr1                       = ptr2;
  EXPECT_EQ(ptr1.get(), value2);

  auto value3 = new int{345};
  std::shared_ptr<int> ptr3(value3);
  ptr1 = ptr3;
  EXPECT_EQ(ptr1.get(), value3);
}

TEST(SharedPtrTest, BoolConversionOpertor)
{
  auto value = new int{123};
  shared_ptr<int> ptr1{value};

  EXPECT_TRUE(ptr1);
  EXPECT_FALSE(shared_ptr<int>{});
}

TEST(SharedPtrTest, PointerOperators)
{
  auto value = new int{123};
  shared_ptr<int> ptr1{value};

  EXPECT_EQ(&*ptr1, value);
  EXPECT_EQ(
      shared_ptr<B> { new B }->f(), 123);
}

TEST(SharedPtrTest, Swap)
{
  auto value1 = new int{123};
  shared_ptr<int> ptr1{value1};

  auto value2 = new int{456};
  shared_ptr<int> ptr2{value2};
  ptr1.swap(ptr2);

  EXPECT_EQ(ptr1.get(), value2);
  EXPECT_EQ(ptr2.get(), value1);
}

TEST(SharedPtrTest, Comparison)
{
  shared_ptr<int> ptr1{new int{123}};
  shared_ptr<int> ptr2{new int{456}};
  shared_ptr<int> ptr3{};

  EXPECT_EQ(ptr1, ptr1);
  EXPECT_NE(ptr1, ptr2);

  EXPECT_NE(ptr1, nullptr);
  EXPECT_NE(nullptr, ptr1);

  EXPECT_EQ(ptr3, nullptr);
  EXPECT_EQ(nullptr, ptr3);
}

static void SharedPtrTest_Sort(size_t size = 10)
{
  std::vector<shared_ptr<const int>> nums;

  for (int i = static_cast<int>(size); i > 0; i--)
  {
    nums.push_back(shared_ptr<int>(new int(i)));
  }

  auto nums2 = nums;

  std::sort(nums.begin(), nums.end(),
            [](shared_ptr<const int> a, shared_ptr<const int> b) { return *a < *b; });

  EXPECT_NE(nums, nums2);

  std::reverse(nums2.begin(), nums2.end());

  EXPECT_EQ(nums, nums2);
}

TEST(SharedPtrTest, Sort)
{
  SharedPtrTest_Sort();
}
