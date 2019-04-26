#pragma once

#include <ATen/core/qualified_name.h>
#include <c10/util/Exception.h>
#include "test/cpp/jit/test_base.h"

using c10::QualifiedName;
using c10::QualifiedNamePtr;

namespace torch {
namespace jit {
namespace test {
void testQualifiedName() {
  {
    // Test toString()
    auto foo = QualifiedName::create("foo");
    auto bar = QualifiedName::create(foo, "bar");
    auto baz = QualifiedName::create(bar, "baz");
    ASSERT_EQ(baz->toString(), "foo.bar.baz");
    ASSERT_EQ(baz->prefix_->toString(), "foo.bar");
    ASSERT_EQ(baz->name_, "baz");
    auto empty = QualifiedName::create("");
    ASSERT_EQ(empty->toString(), "");
    ASSERT_EQ(empty->name_, "");
  }
  {
    // Test createFromDotted API
    auto foo = QualifiedName::createFromDotted("foo.bar.baz");
    ASSERT_EQ(foo->toString(), "foo.bar.baz");
    auto bar = QualifiedName::createFromDotted("bar");
    ASSERT_EQ(bar->toString(), "bar");
  }
  {
    // throw some bad inputs at it
    ASSERT_ANY_THROW(QualifiedName::create("foo.bar"));
    ASSERT_ANY_THROW(QualifiedName::createFromDotted("foo..bar"));
    ASSERT_ANY_THROW(QualifiedName::createFromDotted(".foo.bar"));
    ASSERT_ANY_THROW(QualifiedName::createFromDotted("foo.bar."));
  }
  {
    // test equality api
    auto foo1 = QualifiedName::createFromDotted("foo.bar.baz");
    auto foo2 = QualifiedName::createFromDotted("foo.bar.baz");
    auto foo3 = QualifiedName::createFromDotted("bar.bar.baz");
    ASSERT_TRUE(foo1->equals(foo2));
    ASSERT_FALSE(foo3->equals(foo1));
    auto bar1 = QualifiedName::create("sup");
    auto bar2 = QualifiedName::create("sup");
    ASSERT_TRUE(foo1->equals(foo2));
  }
}
} // namespace test
} // namespace jit
} // namespace torch
