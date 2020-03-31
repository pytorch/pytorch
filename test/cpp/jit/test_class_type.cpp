#include <test/cpp/jit/test_base.h>
#include <test/cpp/jit/test_utils.h>
#include <torch/torch.h>

namespace torch {
namespace jit {

void testClassTypeAddRemoveAttr() {
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

void testClassTypeAddRemoveConstant() {
  auto cu = std::make_shared<CompilationUnit>();
  auto cls = ClassType::create("foo.bar", cu);
  cls->addConstant("const1", IValue(1));
  cls->addConstant("const2", IValue(2));
  cls->addConstant("const3", IValue(2));
  ASSERT_EQ(cls->numConstants(), 3);
  ASSERT_TRUE(cls->hasConstant("const1"));
  ASSERT_TRUE(cls->hasConstant("const2"));
  ASSERT_TRUE(cls->hasConstant("const3"));
  ASSERT_FALSE(cls->hasConstant("const4"));

  ASSERT_EQ(cls->getConstant("const1").toInt(), 1);
  ASSERT_EQ(cls->getConstant("const2").toInt(), 2);
  ASSERT_EQ(cls->getConstant("const2").toInt(), 3);

  cls->unsafeRemoveConstant("const2");
  ASSERT_TRUE(cls->hasConstant("const1"));
  ASSERT_FALSE(cls->hasConstant("const2"));
  ASSERT_TRUE(cls->hasConstant("const3"));
}

} // namespace jit
} // namespace torch
