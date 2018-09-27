#include <gtest/gtest.h>
#include <iostream>
#include <memory>

#include "c10/util/Registry.h"

// Note: we use a different namespace to test if the macros defined in
// Registry.h actuall works with a different namespace from c10.
namespace c10_test {

class Foo {
 public:
  explicit Foo(int x) {
    // LOG(INFO) << "Foo " << x;
  }
};

C10_DECLARE_REGISTRY(FooRegistry, Foo, int);
C10_DEFINE_REGISTRY(FooRegistry, Foo, int);
#define REGISTER_FOO(clsname) C10_REGISTER_CLASS(FooRegistry, clsname, clsname)

class Bar : public Foo {
 public:
  explicit Bar(int x) : Foo(x) {
    // LOG(INFO) << "Bar " << x;
  }
};
REGISTER_FOO(Bar);

class AnotherBar : public Foo {
 public:
  explicit AnotherBar(int x) : Foo(x) {
    // LOG(INFO) << "AnotherBar " << x;
  }
};
REGISTER_FOO(AnotherBar);

TEST(RegistryTest, CanRunCreator) {
  std::unique_ptr<Foo> bar(FooRegistry()->Create("Bar", 1));
  EXPECT_TRUE(bar != nullptr) << "Cannot create bar.";
  std::unique_ptr<Foo> another_bar(FooRegistry()->Create("AnotherBar", 1));
  EXPECT_TRUE(another_bar != nullptr);
}

TEST(RegistryTest, ReturnNullOnNonExistingCreator) {
  EXPECT_EQ(FooRegistry()->Create("Non-existing bar", 1), nullptr);
}

} // namespace c10_test
