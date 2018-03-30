#include <iostream>
#include <memory>

#include "caffe2/core/registry.h"
#include <gtest/gtest.h>
#include "caffe2/core/logging.h"

namespace caffe2 {
namespace {

class Foo {
 public:
  explicit Foo(int x) { LOG(INFO) << "Foo " << x; }
};

CAFFE_DECLARE_REGISTRY(FooRegistry, Foo, int);
CAFFE_DEFINE_REGISTRY(FooRegistry, Foo, int);
#define REGISTER_FOO(clsname) \
  CAFFE_REGISTER_CLASS(FooRegistry, clsname, clsname)

class Bar : public Foo {
 public:
  explicit Bar(int x) : Foo(x) { LOG(INFO) << "Bar " << x; }
};
REGISTER_FOO(Bar);

class AnotherBar : public Foo {
 public:
  explicit AnotherBar(int x) : Foo(x) {
    LOG(INFO) << "AnotherBar " << x;
  }
};
REGISTER_FOO(AnotherBar);

TEST(RegistryTest, CanRunCreator) {
  unique_ptr<Foo> bar(FooRegistry()->Create("Bar", 1));
  EXPECT_TRUE(bar != nullptr) << "Cannot create bar.";
  unique_ptr<Foo> another_bar(FooRegistry()->Create("AnotherBar", 1));
  EXPECT_TRUE(another_bar != nullptr);
}

TEST(RegistryTest, ReturnNullOnNonExistingCreator) {
  EXPECT_EQ(FooRegistry()->Create("Non-existing bar", 1), nullptr);
}
}
}  // namespace caffe2
