#pragma once

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
        self.nested = FooNestedTest(y)

class FooTest:
    def __init__(self, x):
        self.class_attr = FooNestedTest(x)
        self.class_attr2 = FooNestedTest2(x)
        self.x = self.class_attr.y + self.class_attr2.y
)JIT";

static const auto classSrcs2 = R"JIT(
op_version_set = 0

class FooTest:
    def __init__(self, x):
      self.dx = x
)JIT";

void testClassImport() {
  std::vector<at::Tensor> constantTable;
  // Import different versions of FooTest into two namespaces.
  import_libs("namespace1", classSrcs1, constantTable);
  import_libs("namespace2", classSrcs2, constantTable);

  // We should get the correct version of `FooTest` for whichever namespace we
  // are referencing
  auto classType1 = ClassType::get("namespace1", "FooTest");
  ASSERT_TRUE(classType1->hasAttribute("x"));
  ASSERT_FALSE(classType1->hasAttribute("dx"));

  auto classType2 = ClassType::get("namespace2", "FooTest");
  ASSERT_TRUE(classType2->hasAttribute("dx"));
  ASSERT_FALSE(classType2->hasAttribute("x"));

  // We should only see FooNestedTest in the first namespace
  auto c = ClassType::get("namespace1", "FooNestedTest");
  ASSERT_TRUE(c);

  c = ClassType::get("namespace2", "FooNestedTest");
  ASSERT_FALSE(c);
}

}
} // namespace jit
} // namespace torch
