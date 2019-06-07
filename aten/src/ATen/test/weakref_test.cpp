#include <gtest/gtest.h>

#include <ATen/ATen.h>

#include <iostream>
#include <chrono>
#include <sstream>

using at::Tensor;
using c10::WeakIValue;

// Weak pointer tests
// gets invalidated
TEST(TestWeakPointer, WeakPointerGetsInvalidated) {
  IValue a = at::ones({2, 2});
  WeakIValue b = a;
  a.reset();
  ASSERT_FALSE(b.lock().defined());
}

// can successfully lock
TEST(TestWeakPointer, WeakPointerLock) {
  IValue a = at::ones({2, 2});
  WeakIValue b = a;
  auto c = b.lock();
  ASSERT_TRUE(c.defined());

  a.reset();
  ASSERT_TRUE(b.lock().defined());
  c.reset();
  ASSERT_FALSE(b.lock().defined());
}

// updates refcounts correctly
TEST(TestWeakPointer, WeakUpdatesRefcountsTest) {
  IValue a = at::ones({2, 2});
  ASSERT_EQ(a.use_count(), 1);
  ASSERT_EQ(a.weak_use_count(), 1);
  {
    WeakIValue b = a;
    ASSERT_EQ(a.use_count(), 1);
    ASSERT_EQ(a.weak_use_count(), 2);
  }
  ASSERT_EQ(a.use_count(), 1);
  ASSERT_EQ(a.weak_use_count(), 1);
  {
    WeakIValue b = a;
    ASSERT_EQ(a.use_count(), 1);
    auto locked = b.lock();
    ASSERT_TRUE(locked.defined());
    ASSERT_EQ(a.use_count(), 2);
  }
  ASSERT_EQ(a.use_count(), 1);
  ASSERT_EQ(a.weak_use_count(), 1);
  {
    WeakIValue b = a;
    ASSERT_EQ(a.use_count(), 1);
    ASSERT_EQ(a.weak_use_count(), 2);
    a.reset();
    ASSERT_EQ(b.use_count(), 0);
    ASSERT_EQ(b.weak_use_count(), 1);
  }
}
