#pragma once

/**
 * See README.md for instructions on how to add a new test.
 */
#include <c10/macros/Export.h>
#include <torch/csrc/WindowsTorchApiMacro.h>

namespace torch {
namespace jit {
#define TH_FORALL_TESTS(_)      \
  _(ExprBasicValueTest)         \
  _(ExprBasicValueTest02)       \
  _(ExprLetTest01)              \
  _(ExprLetTest02)              \
  _(ExprVectorAdd01)            \
  _(ExprCompareSelectEQ)        \
  _(ExprSubstitute01)           \
  _(ExprDynamicShapeAdd)        \
  _(IRPrinterBasicValueTest)    \
  _(IRPrinterBasicValueTest02)  \
  _(IRPrinterLetTest01)         \
  _(IRPrinterLetTest02)         \
  _(IRPrinterCastTest)          \
  _(TypeTest01)                 \
  _(IfThenElse01)               \
  _(IfThenElse02)               \

#define TH_FORALL_TESTS_CUDA(_) \

#define DECLARE_TENSOREXPR_TEST(name) void test##name();
TH_FORALL_TESTS(DECLARE_TENSOREXPR_TEST)
#ifdef USE_CUDA
TH_FORALL_TESTS_CUDA(DECLARE_TENSOREXPR_TEST)
#endif
#undef DECLARE_TENSOREXPR_TEST

} // namespace jit
} // namespace torch
