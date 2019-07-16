#pragma once

#include <test/cpp/jit/test_base.h>
#include <test/cpp/jit/test_utils.h>

#include <ATen/core/qualified_name.h>
#include <torch/csrc/jit/import_source.h>
#include <torch/torch.h>

namespace torch {
namespace jit {
namespace script {

static const auto classSrcs1 = R"JIT(
op_version_set = 1
class FooNestedTest:
    def __init__(self, y):
        self.y = y

class FooNestedTest2:
    def __init__(self, y):
        self.y = y
        self.nested = __torch__.FooNestedTest(y)

class FooTest:
    def __init__(self, x):
        self.class_attr = __torch__.FooNestedTest(x)
        self.class_attr2 = __torch__.FooNestedTest2(x)
        self.x = self.class_attr.y + self.class_attr2.y
)JIT";

static const auto classSrcs2 = R"JIT(
op_version_set = 1
class FooTest:
    def __init__(self, x):
      self.dx = x
)JIT";

void testClassImport() {
  CompilationUnit cu1;
  CompilationUnit cu2;
  std::vector<at::Tensor> constantTable;
  // Import different versions of FooTest into two namespaces.
  import_libs(
      cu1,
      "__torch__",
      std::make_shared<Source>(classSrcs1),
      constantTable,
      nullptr);
  import_libs(
      cu2,
      "__torch__",
      std::make_shared<Source>(classSrcs2),
      constantTable,
      nullptr);

  // We should get the correct version of `FooTest` for whichever namespace we
  // are referencing
  c10::QualifiedName base("__torch__");
  auto classType1 = cu1.get_class(c10::QualifiedName(base, "FooTest"));
  ASSERT_TRUE(classType1->hasAttribute("x"));
  ASSERT_FALSE(classType1->hasAttribute("dx"));

  auto classType2 = cu2.get_class(c10::QualifiedName(base, "FooTest"));
  ASSERT_TRUE(classType2->hasAttribute("dx"));
  ASSERT_FALSE(classType2->hasAttribute("x"));

  // We should only see FooNestedTest in the first namespace
  auto c = cu1.get_class(c10::QualifiedName(base, "FooNestedTest"));
  ASSERT_TRUE(c);

  c = cu2.get_class(c10::QualifiedName(base, "FooNestedTest"));
  ASSERT_FALSE(c);
}

void testScriptObject() {
  Module m1("m1");
  Module m2("m2");
  std::vector<at::Tensor> constantTable;
  import_libs(
      *m1.class_compilation_unit(),
      "__torch__",
      std::make_shared<Source>(classSrcs1),
      constantTable,
      nullptr);
  import_libs(
      *m2.class_compilation_unit(),
      "__torch__",
      std::make_shared<Source>(classSrcs2),
      constantTable,
      nullptr);

  // Incorrect arguments for constructor should throw
  c10::QualifiedName base("__torch__");
  ASSERT_ANY_THROW(m1.create_class(c10::QualifiedName(base, "FooTest"), {1}));
  auto x = torch::ones({2, 3});
  auto obj = m2.create_class(c10::QualifiedName(base, "FooTest"), x).toObject();
  auto dx = obj->getAttr("dx");
  ASSERT_TRUE(test::almostEqual(x, dx.toTensor()));

  auto new_x = torch::rand({2, 3});
  obj->setAttr("dx", new_x);
  auto new_dx = obj->getAttr("dx");
  ASSERT_TRUE(test::almostEqual(new_x, new_dx.toTensor()));
}

} // namespace script
} // namespace jit
} // namespace torch
