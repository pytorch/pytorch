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
  Parser p(std::make_shared<Source>(testSource));
  std::vector<Def> definitions;
  std::vector<Resolver> resolvers;

  const auto classDef = ClassDef(p.parseClassLike());
  p.lexer().expect(TK_EOF);

  ASSERT_EQ(classDef.name().name(), "FooTest");
  ASSERT_EQ(classDef.defs().size(), 2);
  ASSERT_EQ(classDef.defs()[0].name().name(), "__init__");
  ASSERT_EQ(classDef.defs()[1].name().name(), "get_x");
}
} // namespace script
} // namespace jit
} // namespace torch
