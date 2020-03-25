#include <test/cpp/jit/test_base.h>
#include <test/cpp/jit/test_utils.h>
#include <torch/torch.h>

namespace torch {
namespace jit {

void testModuleClone() {
  auto cu = std::make_shared<CompilationUnit>();
  auto parent = ClassType::create("parent", cu, true);
  // creating child module
  auto child = ClassType::create("child", cu, true);
  auto attr_name = "attr";
  child->addAttribute(attr_name, IntType::get());
  Module c1(cu, child);
  auto v1 = IValue(2);
  c1.register_attribute(attr_name,
                        IntType::get(),
                        v1,
                        false);
  Module c2(cu, child);
  auto v2 = IValue(3);
  c2.register_attribute(attr_name,
                        IntType::get(),
                        v2,
                        false);

  // attach two child module instance to parent that shares
  // ClassType
  Module p(cu, parent);
  p.register_attribute("c1", c1.type(), c1._ivalue(), false);
  p.register_attribute("c2", c2.type(), c2._ivalue(), false);

  // clone parent
  Module p2 = p.clone();
  // check the two child module has the same ClassType
  ASSERT_EQ(p2.attr("c1").type(), p2.attr("c2").type());
  // but different instances
  ASSERT_EQ(Module(p2.attr("c1").toObject()).attr(attr_name).toInt(), 2);
  ASSERT_EQ(Module(p2.attr("c2").toObject()).attr(attr_name).toInt(), 3);
}

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

void testModuleParameter() {
  auto cu = std::make_shared<CompilationUnit>();
  auto cls = ClassType::create("foo.bar", cu, true);
  Module m(cu, cls);
  // Tensor parameter
  m.register_parameter("tensor_param", at::empty({3}, at::kFloat), /* is_buffer */ false);
  // None parameter
  m.register_attribute("none_param", NoneType::get(), IValue(), /* is_param */ true);
  m.register_attribute("none_param2", NoneType::get(), IValue(), /* is_param */ true);
  auto param_list = m.parameters();
  ASSERT_EQ(param_list.size(), 1);
  ASSERT_TRUE(m.hasattr("tensor_param"));
  ASSERT_TRUE(m.hasattr("none_param"));
  ASSERT_TRUE(m.hasattr("none_param2"));
}

} // namespace jit
} // namespace torch
