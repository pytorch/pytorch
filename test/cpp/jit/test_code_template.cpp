#include "test/cpp/jit/test_base.h"
#include "test/cpp/jit/test_utils.h"

#include "torch/csrc/jit/frontend/code_template.h"

namespace torch {
namespace jit {

static const auto ct = CodeTemplate(R"(
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

void testCodeTemplate() {
  {
    TemplateEnv e;
    e.s("hi", "foo");
    e.v("what", {"is", "this"});
    TemplateEnv c(e);
    c.s("hi", "foo2");
    ASSERT_EQ(e.s("hi"), "foo");
    ASSERT_EQ(c.s("hi"), "foo2");
    ASSERT_EQ(e.v("what")[0], "is");
  }

  {
    TemplateEnv e;
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
}

} // namespace jit
} // namespace torch
