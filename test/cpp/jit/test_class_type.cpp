#include <gtest/gtest.h>

#include <test/cpp/jit/test_utils.h>
#include <torch/csrc/jit/testing/file_check.h>
#include <torch/torch.h>

namespace torch {
namespace jit {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(ClassTypeTest, AddRemoveAttr) {
  auto cu = std::make_shared<CompilationUnit>();
  auto cls = ClassType::create("foo.bar", cu, true);
  cls->addAttribute("attr1", TensorType::get(), true);
  cls->addAttribute("attr2", TensorType::get());
  cls->addAttribute("attr3", TensorType::get());
  ASSERT_TRUE(cls->hasAttribute("attr1"));
  ASSERT_TRUE(cls->hasAttribute("attr2"));
  ASSERT_TRUE(cls->hasAttribute("attr3"));

  // removing attribute attr2
  cls->unsafeRemoveAttribute("attr2");
  ASSERT_TRUE(cls->hasAttribute("attr1"));
  ASSERT_FALSE(cls->hasAttribute("attr2"));
  ASSERT_TRUE(cls->hasAttribute("attr3"));

  // removing parameter attr1
  cls->unsafeRemoveAttribute("attr1");
  ASSERT_FALSE(cls->hasAttribute("attr1"));
  ASSERT_FALSE(cls->hasAttribute("attr2"));
  ASSERT_TRUE(cls->hasAttribute("attr3"));

  // check that we can still add a non-parameter attr1 with
  // different type
  cls->addAttribute("attr1", IntType::get());
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(ClassTypeTest, AddRemoveConstant) {
  auto cu = std::make_shared<CompilationUnit>();
  auto cls = ClassType::create("foo.bar", cu);
  cls->addConstant("const1", IValue(1));
  cls->addConstant("const2", IValue(2));
  cls->addConstant("const3", IValue(3));
  ASSERT_EQ(cls->numConstants(), 3);
  ASSERT_TRUE(cls->hasConstant("const1"));
  ASSERT_TRUE(cls->hasConstant("const2"));
  ASSERT_TRUE(cls->hasConstant("const3"));
  ASSERT_FALSE(cls->hasConstant("const4"));

  ASSERT_EQ(cls->getConstant("const1").toInt(), 1);
  ASSERT_EQ(cls->getConstant("const2").toInt(), 2);
  ASSERT_EQ(cls->getConstant("const3").toInt(), 3);

  cls->unsafeRemoveConstant("const2");
  ASSERT_TRUE(cls->hasConstant("const1"));
  ASSERT_FALSE(cls->hasConstant("const2"));
  ASSERT_TRUE(cls->hasConstant("const3"));
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(ClassTypeTest, IdenticalTypesDifferentCus) {
  auto cu1 = std::make_shared<CompilationUnit>();
  auto cu2 = std::make_shared<CompilationUnit>();

  // Create two identically named ClassTypes and put them
  // in separate compilation units.
  auto cls1 = ClassType::create("foo", cu1);
  auto cls2 = ClassType::create("foo", cu2);

  // Create a function that accepts "foo" (cls1) as input.
  Argument arg("arg", cls1);
  Argument ret("ret", IntType::get());

  FunctionSchema schema("fn", "", {arg}, {ret});

  jit::BuiltinOpFunction method(
      "method",
      std::move(schema),
      [](jit::Stack& stack) mutable -> void {
        pop(stack);
        push(stack, 0);
      },
      "");

  // Create an object of type cls2.
  Object obj(cu2, cls2);

  // Call method with the above object; this should
  // throw an error because the types have identical
  // names but are in different compilation units.
  Stack stack;
  push(stack, obj._ivalue());
  try {
    method(stack, {});
  } catch (const std::exception& e) {
    // Check that the exception contains the address of the compilation unit
    // in addition to the ClassType's name.
    testing::FileCheck()
        .check("foo (of Python compilation unit at: 0x")
        ->check_same(")")
        ->check("foo (of Python compilation unit at: 0x")
        ->check_same(")")
        ->run(e.what());

    return;
  }

  // This should never execute.
  ASSERT_TRUE(false);
}

} // namespace jit
} // namespace torch
