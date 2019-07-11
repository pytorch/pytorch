#include <gtest/gtest.h>

#include <ATen/ATen.h>

#include <iostream>
#include <chrono>
#include <sstream>

using at::Tensor;
using at::WeakTensor;

// Weak pointer tests
// gets invalidated
TEST(TestWeakPointer, WeakPointerGetsInvalidated) {
  Tensor a = at::ones({2, 2});
  WeakTensor b = a;
  a.reset();
  ASSERT_FALSE(b.lock().defined());
}

// can successfully lock
TEST(TestWeakPointer, WeakPointerLock) {
  Tensor a = at::ones({2, 2});
  WeakTensor b = a;
  auto c = b.lock();
  ASSERT_TRUE(c.defined());

  a.reset();
  ASSERT_TRUE(b.lock().defined());
  c.reset();
  ASSERT_FALSE(b.lock().defined());
}

// updates refcounts correctly
TEST(TestWeakPointer, WeakUpdatesRefcountsTest) {
  Tensor a = at::ones({2, 2});
  ASSERT_EQ(a.use_count(), 1);
  ASSERT_EQ(a.weak_use_count(), 1);
  {
    WeakTensor b = a;
    ASSERT_EQ(a.use_count(), 1);
    ASSERT_EQ(a.weak_use_count(), 2);
  }
  ASSERT_EQ(a.use_count(), 1);
  ASSERT_EQ(a.weak_use_count(), 1);
  {
    WeakTensor b = a;
    ASSERT_EQ(a.use_count(), 1);
    auto locked = b.lock();
    ASSERT_TRUE(locked.defined());
    ASSERT_EQ(a.use_count(), 2);
  }
  ASSERT_EQ(a.use_count(), 1);
  ASSERT_EQ(a.weak_use_count(), 1);
  {
    WeakTensor b = a;
    ASSERT_EQ(a.use_count(), 1);
    ASSERT_EQ(a.weak_use_count(), 2);
    a.reset();
    ASSERT_EQ(b.use_count(), 0);
    ASSERT_EQ(b.weak_use_count(), 1);
  }
}
