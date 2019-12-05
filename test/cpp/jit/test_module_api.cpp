#include <test/cpp/jit/test_base.h>
#include <test/cpp/jit/test_utils.h>
#include <torch/torch.h>

namespace torch {
namespace jit {

using namespace torch::jit::script;

void testModuleCloneInstance() {
  auto cu = std::make_shared<CompilationUnit>();
  auto cls = ClassType::create("foo.bar", cu, true);
  auto attr_name = "attr";
  cls->addAttribute(attr_name, IntType::get());
  Module m(cu, cls);
  auto v = IValue(2);
  m.register_attribute(attr_name,
                       IntType::get(),
                       v,
                       false);

  Module m2 = m.clone();
  Module m3 = m.clone_instance();
  // Make sure copy works
  ASSERT_EQ(m2.attr(attr_name).toInt(), 2);
  ASSERT_EQ(m3.attr(attr_name).toInt(), 2);

  // clone will copy both type and data, therefore we'll have a
  // different type
  ASSERT_NE(m.type(), m2.type());
  // clone_instance only copies data, type is shared
  ASSERT_EQ(m.type(), m3.type());

  // change value of copied instance
  m3.register_attribute(attr_name,
                        IntType::get(),
                        IValue(3),
                        false);
  // Verify value of original instance doesn't change
  ASSERT_EQ(m2.attr(attr_name).toInt(), 2);
  ASSERT_EQ(m3.attr(attr_name).toInt(), 3);
}

void testModuleConstant() {
  auto cu = std::make_shared<CompilationUnit>();
  auto cls = ClassType::create("foo.bar", cu, true);
  auto attr_name = "attr";
  auto const_name = "const";
  cls->addAttribute(attr_name, IntType::get());
  cls->addConstant(const_name, IValue(3));
  Module m(cu, cls);
  auto v = IValue(2);
  m.register_attribute(attr_name,
                       IntType::get(),
                       v,
                       false);
  ASSERT_TRUE(m.hasattr(attr_name));
  ASSERT_TRUE(m.hasattr(const_name));
  ASSERT_EQ(m.attr(attr_name).toInt(), 2);
  ASSERT_EQ(m.attr(const_name).toInt(), 3);
}

} // namespace jit
} // namespace torch
