#include <test/cpp/jit/test_base.h>
#include <test/cpp/jit/test_utils.h>

#include <torch/csrc/jit/script/error_report.h>

namespace torch {
namespace jit {

static void testBasic() {
  static const std::string src = R"(
def foo(x):
    abc def
    some more code()
    alas, even more code
)";
  auto source = std::make_shared<Source>(src);
  const std::string range = "abc";
  const auto pos = src.find(range);
  SourceRange sourceRange(std::move(source), pos, pos + range.size());

  script::ErrorReport err(sourceRange);
  err << "my error message";

  testing::FileCheck()
      // error message properly displayed
      .check("my error message")
      // Some context is shown
      ->check("def foo")
      // The actual line
      ->check_next("abc def")
      // Squigglies
      ->check_next("^^^")
      // There's no CallStack, so we shouldn't have a traceback
      ->check_not("Traceback:")
      ->run(err.what());
}

static void testCallStackError() {
  static const std::string src = R"(
def h(x):
    Lorem ipsum dolor sit amet, consectetur adipiscing elit.
    Nulla ut justo eu neque bibendum cursus.
    Aliquam ut lorem ac leo faucibus finibus vel non diam.

def g(x):
    Nunc elementum lorem a ligula molestie aliquam.
    Donec quis mauris sodales, congue diam non, porta odio.
    Donec eu massa id diam mattis dapibus sit amet ut quam.

def f(x):
    Cras in odio hendrerit, mattis nisl nec, aliquam mauris.
    Quisque malesuada mauris id finibus congue.
    Suspendisse volutpat leo sit amet purus ornare, a egestas diam suscipit.
)";
  const auto source =
      std::make_shared<Source>(src, "lipsum.py", /*starting_line_no=*/0);

  // These should be unique, to avoid confusion
  const std::string range1 = "cursus";
  const std::string range2 = "sodales";
  const std::string range3 = "ornare";

  auto rangeBegin = src.find(range1);
  SourceRange sourceRange1(source, rangeBegin, rangeBegin + range1.size());
  rangeBegin = src.find(range2);
  SourceRange sourceRange2(source, rangeBegin, rangeBegin + range2.size());
  rangeBegin = src.find(range3);
  SourceRange sourceRange3(source, rangeBegin, rangeBegin + range3.size());

  // Build up an error stack
  script::ErrorReport::CallStack call1("h");
  script::ErrorReport::CallStack::update_pending_range(sourceRange1);
  script::ErrorReport::CallStack call2("g");
  script::ErrorReport::CallStack::update_pending_range(sourceRange2);
  script::ErrorReport::CallStack call3("f");
  script::ErrorReport::CallStack::update_pending_range(sourceRange3);
  throw script::ErrorReport(sourceRange3) << "my error message";
}

void testErrorReporting() {
    testBasic();
    testCallStackError();
}
} // namespace jit
} // namespace torch
