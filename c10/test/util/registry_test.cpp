#include <gtest/gtest.h>
#include <iostream>
#include <memory>

#include <c10/util/Registry.h>

// Note: we use a different namespace to test if the macros defined in
// Registry.h actually works with a different namespace from c10.
namespace c10_test {

class Foo {
 public:
  explicit Foo(int x) {
    // LOG(INFO) << "Foo " << x;
  }
  virtual ~Foo() = default;
};

// NOLINTNEXTLINE(misc-use-internal-linkage)
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

// C10_REGISTER_CLASS_WITH_PRIORITY defines static variable
static void RegisterFooDefault() {
  C10_REGISTER_CLASS_WITH_PRIORITY(
      FooRegistry, FooWithPriority, c10::REGISTRY_DEFAULT, Foo);
}

static void RegisterFooDefaultAgain() {
  C10_REGISTER_CLASS_WITH_PRIORITY(
      FooRegistry, FooWithPriority, c10::REGISTRY_DEFAULT, Foo);
}

static void RegisterFooBarFallback() {
  C10_REGISTER_CLASS_WITH_PRIORITY(
      FooRegistry, FooWithPriority, c10::REGISTRY_FALLBACK, Bar);
}

static void RegisterFooBarPreferred() {
  C10_REGISTER_CLASS_WITH_PRIORITY(
      FooRegistry, FooWithPriority, c10::REGISTRY_PREFERRED, Bar);
}

TEST(RegistryTest, RegistryPriorities) {
  FooRegistry()->SetTerminate(false);
  RegisterFooDefault();

  // throws because Foo is already registered with default priority
  // NOLINTNEXTLINE(hicpp-avoid-goto,cppcoreguidelines-avoid-goto)
  EXPECT_THROW(RegisterFooDefaultAgain(), std::runtime_error);

#ifdef __GXX_RTTI
  // not going to register Bar because Foo is registered with Default priority
  RegisterFooBarFallback();
  std::unique_ptr<Foo> bar1(FooRegistry()->Create("FooWithPriority", 1));
  EXPECT_EQ(dynamic_cast<Bar*>(bar1.get()), nullptr);

  // will register Bar because of higher priority
  RegisterFooBarPreferred();
  std::unique_ptr<Foo> bar2(FooRegistry()->Create("FooWithPriority", 1));
  EXPECT_NE(dynamic_cast<Bar*>(bar2.get()), nullptr);
#endif
}

} // namespace c10_test
