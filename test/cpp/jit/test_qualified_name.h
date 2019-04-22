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
  }
  {
    // Test createFromDotted API
    auto foo = QualifiedName::createFromDotted("foo.bar.baz");
    ASSERT_EQ(foo->toString(), "foo.bar.baz");
    auto bar = QualifiedName::createFromDotted("bar");
    ASSERT_EQ(bar->toString(), "bar");
  }
  {
    // Can't create directly from a dotted string to avoid ambiguity.
    ASSERT_ANY_THROW(QualifiedName::create("foo.bar"));
  }
}
} // namespace test
} // namespace jit
} // namespace torch
