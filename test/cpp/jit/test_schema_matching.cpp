#include <gtest/gtest.h>

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/testing/file_check.h>
#include <torch/jit.h>

#include <sstream>
#include <string>

namespace torch {
namespace jit {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(SchemaMatchingTest, VarType) {
  RegisterOperators reg({
      Operator(
          "aten::test_vartype(t[] a, t b) -> (t)",
          [](Stack* stack) {
            c10::List<double> list;
            // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
            double a;
            pop(stack, list, a);
            push(stack, a);
          },
          c10::AliasAnalysisKind::FROM_SCHEMA),
  });
  Module m("m");
  m.define(R"(
      def test(self):
        a = (1.0, 2.0)
        return torch.test_vartype(a, 2.0)
    )");
  auto result = m.run_method("test");
  TORCH_INTERNAL_ASSERT(result.toDouble() == 2.0);

  const std::string error_example = R"JIT(
      def test_2(self):
          a = (1.0, 2.0)
          non_float = (1, 1)
          return torch.test_vartype(a, non_float)
    )JIT";

  std::string err = "";
  try {
    m.define(error_example);
  } catch (const std::exception& e) {
    err = e.what();
  }
  TORCH_INTERNAL_ASSERT(
      err.find("previously matched to type") != std::string::npos);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
TEST(SchemaMatchingTest, VarType2) {
  RegisterOperators reg({
      Operator(
          "aten::test_vartype2(t a, t[] b) -> (t[])",
          [](Stack* stack) {
            // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
            double a;
            c10::List<double> list;
            pop(stack, a, list);
            push(stack, a);
          },
          AliasAnalysisKind::FROM_SCHEMA),
  });
  Module m("m");
  m.define(R"JIT(
      def test(self):
          a = (1.0, 2.0)
          return torch.test_vartype2(3.0, a)
    )JIT");
  auto result = m.run_method("test");
  TORCH_INTERNAL_ASSERT(result.toDouble() == 3.0);

  static const auto error_exam2 = R"JIT(
      def test_2(self):
          a = (1, 2)
          return torch.test_vartype2(3.0, a)
    )JIT";

  std::string err = "";
  try {
    m.define(error_exam2);
  } catch (const std::exception& e) {
    err = e.what();
  }
  TORCH_INTERNAL_ASSERT(
      err.find("previously matched to type") != std::string::npos);
}
} // namespace jit
} // namespace torch
