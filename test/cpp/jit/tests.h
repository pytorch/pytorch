#pragma once

/**
 * See README.md for instructions on how to add a new test.
 */
#include <torch/csrc/WindowsTorchApiMacro.h>
#include <c10/macros/Export.h>

namespace torch {
namespace jit {
#define TH_FORALL_TESTS(_)             \
  _(LiteInterpreter)

#define TH_FORALL_TESTS_CUDA(_) \
  _(ArgumentSpec)               \
  _(CompleteArgumentSpec)       \
  _(Fusion)                     \
  _(GraphExecutor)              \
  _(ModuleConversion)           \
  _(Interp)

#define DECLARE_JIT_TEST(name) void test##name();
TH_FORALL_TESTS(DECLARE_JIT_TEST)
TH_FORALL_TESTS_CUDA(DECLARE_JIT_TEST)
#undef DECLARE_JIT_TEST

// This test is special since it requires prior setup in python.
// So it is not part of the general test list (which is shared between the gtest
// and python test runners), but is instead invoked manually by the
// torch_python_test.cpp
void testEvalModeForLoadedModule();

} // namespace jit
} // namespace torch
