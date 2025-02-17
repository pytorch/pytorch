#include <gtest/gtest.h>

#include <torch/csrc/jit/frontend/parser.h>
#include <torch/csrc/jit/frontend/resolver.h>

namespace torch {
namespace jit {
constexpr std::string_view testSource = R"JIT(
  class FooTest:
    def __init__(self, x):
      self.x = x

    def get_x(self):
      return self.x

    an_attribute : Tensor
)JIT";

TEST(ClassParserTest, Basic) {
  Parser p(std::make_shared<Source>(testSource));
  std::vector<Def> definitions;
  std::vector<Resolver> resolvers;

  const auto classDef = ClassDef(p.parseClass());
  p.lexer().expect(TK_EOF);

  ASSERT_EQ(classDef.name().name(), "FooTest");
  ASSERT_EQ(classDef.body().size(), 3);
  ASSERT_EQ(Def(classDef.body()[0]).name().name(), "__init__");
  ASSERT_EQ(Def(classDef.body()[1]).name().name(), "get_x");
  ASSERT_EQ(
      Var(Assign(classDef.body()[2]).lhs()).name().name(), "an_attribute");
  ASSERT_FALSE(Assign(classDef.body()[2]).rhs().present());
  ASSERT_TRUE(Assign(classDef.body()[2]).type().present());
}
} // namespace jit
} // namespace torch
