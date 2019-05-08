#pragma once

#include <ATen/core/qualified_name.h>
#include <test/cpp/jit/test_base.h>
#include <torch/csrc/jit/import_source.h>

namespace torch {
namespace jit {
namespace script {

static const auto classSrcs1 = R"JIT(
op_version_set = 0
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
op_version_set = 0
class FooTest:
    def __init__(self, x):
      self.dx = x
)JIT";

void testClassImport() {
  CompilationUnit cu1;
  CompilationUnit cu2;
  std::vector<at::Tensor> constantTable;
  // Import different versions of FooTest into two namespaces.
  import_libs(cu1, "__torch__", classSrcs1, constantTable, nullptr);
  import_libs(cu2, "__torch__", classSrcs2, constantTable, nullptr);

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

} // namespace script
} // namespace jit
} // namespace torch
