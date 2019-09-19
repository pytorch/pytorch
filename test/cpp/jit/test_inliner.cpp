#include <test/cpp/jit/test_base.h>

#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/script/compilation_unit.h>
#include <torch/csrc/jit/script/module.h>
#include <torch/csrc/jit/testing/file_check.h>

const auto testSource = R"JIT(
def foo1(x):
    print("one")
    return x

def foo2(x):
    print("two")
    return foo1(x)

def foo3(x):
    print("three")
    return foo2(x)
)JIT";

namespace torch {
namespace jit {
using namespace script;
using namespace testing;

struct InlinerGuard {
  explicit InlinerGuard(bool shouldInline)
      : oldState_(getInlineEverythingMode()) {
    getInlineEverythingMode() = shouldInline;
  }

  ~InlinerGuard() {
    getInlineEverythingMode() = oldState_;
  }

  bool oldState_;
};

void testInliner() {
  {
    // disable automatic inlining so we can test it manually
    InlinerGuard guard(/*shouldInline=*/false);

    CompilationUnit cu(testSource);
    auto& fn = cu.get_function("foo3");

    auto g = fn.graph();
    Inline(*g);
    FileCheck()
        .check("three")
        ->check("two")
        ->check_count("prim::CallFunction", 1)
        ->run(*g);
  }
}
} // namespace jit
} // namespace torch
