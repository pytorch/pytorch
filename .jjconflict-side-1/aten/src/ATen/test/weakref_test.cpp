#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/core/ivalue.h>

#include <iostream>
#include <chrono>
#include <sstream>

using at::Tensor;
using c10::WeakIValue;
using c10::IValue;

// Weak pointer tests
// gets invalidated
TEST(TestWeakPointer, WeakPointerGetsInvalidated) {
  IValue a = at::ones({2, 2});
  WeakIValue b = a;
  a = IValue();
  ASSERT_TRUE(b.lock().isNone());
}

// can successfully lock
TEST(TestWeakPointer, WeakPointerLock) {
  IValue a = at::ones({2, 2});
  WeakIValue b = a;
  auto c = b.lock();
  ASSERT_TRUE(c.isTensor());

  a = IValue();
  ASSERT_TRUE(!b.lock().isNone());
  c = IValue();
  ASSERT_TRUE(b.lock().isNone());
}

// updates refcounts correctly
TEST(TestWeakPointer, WeakUpdatesRefcountsTest) {
  at::Tensor a = at::ones({2, 2});
  ASSERT_EQ(a.use_count(), 1);
  ASSERT_EQ(a.weak_use_count(), 1);
  {
    WeakIValue b = IValue(a);
    ASSERT_EQ(a.use_count(), 1);
    ASSERT_EQ(a.weak_use_count(), 2);
  }
  ASSERT_EQ(a.use_count(), 1);
  ASSERT_EQ(a.weak_use_count(), 1);
  {
    WeakIValue b = IValue(a);
    ASSERT_EQ(a.use_count(), 1);
    auto locked = b.lock();
    ASSERT_FALSE(locked.isNone());
    ASSERT_EQ(a.use_count(), 2);
  }
  ASSERT_EQ(a.use_count(), 1);
  ASSERT_EQ(a.weak_use_count(), 1);
  {
    WeakIValue b = IValue(a);
    ASSERT_EQ(a.use_count(), 1);
    ASSERT_EQ(a.weak_use_count(), 2);
    a.reset();
    ASSERT_EQ(b.use_count(), 0);
    ASSERT_EQ(b.weak_use_count(), 1);
  }
}
