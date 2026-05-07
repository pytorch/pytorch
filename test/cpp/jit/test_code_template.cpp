#include <gtest/gtest.h>

#include <ATen/code_template.h>
#include <test/cpp/jit/test_utils.h>

namespace torch {
namespace jit {

static const auto ct = at::jit::CodeTemplate(R"(
  int foo($args) {

      $bar
          $bar
      $a+$b
  }
  int commatest(int a${,stuff})
  int notest(int a${,empty,})
  )");
static const auto ct_expect = R"(
  int foo(hi, 8) {

      what
      on many
      lines...
      7
          what
          on many
          lines...
          7
      3+4
  }
  int commatest(int a, things..., others)
  int notest(int a)
  )";

TEST(TestCodeTemplate, Copying) {
  at::jit::TemplateEnv e;
  e.s("hi", "foo");
  e.v("what", {"is", "this"});
  at::jit::TemplateEnv c(e);
  c.s("hi", "foo2");
  ASSERT_EQ(e.s("hi"), "foo");
  ASSERT_EQ(c.s("hi"), "foo2");
  ASSERT_EQ(e.v("what")[0], "is");
}

TEST(TestCodeTemplate, Formatting) {
  at::jit::TemplateEnv e;
  e.v("args", {"hi", "8"});
  e.v("bar", {"what\non many\nlines...", "7"});
  e.s("a", "3");
  e.s("b", "4");
  e.v("stuff", {"things...", "others"});
  e.v("empty", {});
  auto s = ct.format(e);
  // std::cout << "'" << s << "'\n";
  // std::cout << "'" << ct_expect << "'\n";
  ASSERT_EQ(s, ct_expect);
}

} // namespace jit
} // namespace torch
