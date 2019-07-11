#pragma once

#include <ATen/core/qualified_name.h>
#include <c10/util/Exception.h>
#include "test/cpp/jit/test_base.h"

using c10::QualifiedName;

namespace torch {
namespace jit {
namespace test {
void testQualifiedName() {
  {
    // Test prefix construction
    auto foo = QualifiedName("foo");
    auto bar = QualifiedName(foo, "bar");
    auto baz = QualifiedName(bar, "baz");
    ASSERT_EQ(baz.qualifiedName(), "foo.bar.baz");
    ASSERT_EQ(baz.prefix(), "foo.bar");
    ASSERT_EQ(baz.name(), "baz");
    auto nullstate = QualifiedName();
    ASSERT_EQ(nullstate.qualifiedName(), "");
    ASSERT_EQ(nullstate.prefix(), "");
    ASSERT_EQ(nullstate.name(), "");
  }
  {
    // Test dotted construction
    auto foo = QualifiedName("foo.bar.baz");
    ASSERT_EQ(foo.qualifiedName(), "foo.bar.baz");
    ASSERT_EQ(foo.prefix(), "foo.bar");
    ASSERT_EQ(foo.name(), "baz");

    auto bar = QualifiedName("bar");
    ASSERT_EQ(bar.qualifiedName(), "bar");
    ASSERT_EQ(bar.prefix(), "");
    ASSERT_EQ(bar.name(), "bar");
  }
  {
    // throw some bad inputs at it
    ASSERT_ANY_THROW(QualifiedName("foo..bar"));
    ASSERT_ANY_THROW(QualifiedName(".foo.bar"));
    ASSERT_ANY_THROW(QualifiedName("foo.bar."));
    ASSERT_ANY_THROW(QualifiedName(""));
  }
  {
    // test equality api
    auto foo1 = QualifiedName("foo.bar.baz");
    auto foo2 = QualifiedName("foo.bar.baz");
    auto foo3 = QualifiedName("bar.bar.baz");
    ASSERT_EQ(foo1, foo2);
    ASSERT_NE(foo1, foo3);
    auto bar1 = QualifiedName("sup");
    auto bar2 = QualifiedName("sup");
    ASSERT_EQ(foo1, foo2);
  }
}
} // namespace test
} // namespace jit
} // namespace torch
