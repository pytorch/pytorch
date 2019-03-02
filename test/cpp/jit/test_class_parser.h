#pragma once

#include <test/cpp/jit/test_base.h>
#include <torch/csrc/jit/script/parser.h>

namespace torch {
namespace jit {
namespace script {
const auto testSource = R"JIT(
  class FooTest:
    def __init__(self, x):
      self.x = x

    def get_x(self):
      return self.x
)JIT";

void testClassParser() {
  auto cu = std::make_shared<Module>();
  Parser p(testSource);
  std::vector<Def> definitions;
  std::vector<Resolver> resolvers;

  const auto classDef = ClassDef(p.parseClass());
  p.lexer().expect(TK_EOF);

  const auto type = ClassType::create(classDef.name().name(), cu);
  for (const auto& def : classDef.defs()) {
    definitions.push_back(def);
    resolvers.push_back(nativeResolver);
  }
  defineMethodsInModule(cu, definitions, resolvers, Self(type));

  const auto classType = ClassType::get("FooTest");
  ASSERT_TRUE(classType);
  ASSERT_TRUE(classType->numAttributes() == 1);
  ASSERT_TRUE(classType->getAttributeSlot("x") == 0);
  auto method = classType->getMethod("get_x");
  ASSERT_TRUE(method);
}
} // namespace script
} // namespace jit
} // namespace torch
