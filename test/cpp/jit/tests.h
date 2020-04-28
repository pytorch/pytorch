#pragma once

/**
 * See README.md for instructions on how to add a new test.
 */
#include <c10/macros/Export.h>
#include <torch/csrc/WindowsTorchApiMacro.h>

namespace torch {
namespace jit {
#define TH_FORALL_TESTS(_)             \
  _(MobileTypeParser)\
  _(LiteInterpreterDict)

#if defined(USE_CUDA)
#define TH_FORALL_TESTS_CUDA(_)  \
  _(ArgumentSpec)                \
  _(CompleteArgumentSpec)        \
  _(Fusion)                      \
  _(GraphExecutor)               \
  _(ModuleConversion)            \
  _(Interp)                      \
  _(GPU_FusionDispatch)          \
  _(GPU_FusionSimpleArith)       \
  _(GPU_FusionSimpleTypePromote) \
  _(GPU_FusionCastOp)            \
  _(GPU_FusionMutator)           \
  _(GPU_FusionRegister)          \
  _(GPU_FusionTopoSort)          \
  _(GPU_FusionTensor)            \
  _(GPU_FusionTensorContiguity)  \
  _(GPU_FusionTVSplit)           \
  _(GPU_FusionTVMerge)           \
  _(GPU_FusionTVReorder)         \
  _(GPU_FusionEquality)          \
  _(GPU_FusionReplaceAll)        \
  _(GPU_FusionParser)            \
  _(GPU_FusionDependency)        \
  _(GPU_FusionCodeGen)           \
  _(GPU_FusionCodeGen2)          \
  _(GPU_FusionSimplePWise)       \
  _(GPU_FusionExecKernel)        \
  _(GPU_FusionForLoop)           \
  _(GPU_FusionLoopUnroll)
#else
#define TH_FORALL_TESTS_CUDA(_) \
  _(ArgumentSpec)               \
  _(CompleteArgumentSpec)       \
  _(Fusion)                     \
  _(GraphExecutor)              \
  _(ModuleConversion)           \
  _(Interp)
#endif

#define DECLARE_JIT_TEST(name) void test##name();
TH_FORALL_TESTS(DECLARE_JIT_TEST)
TH_FORALL_TESTS_CUDA(DECLARE_JIT_TEST)
#undef DECLARE_JIT_TEST

// This test is special since it requires prior setup in python.
// So it is not part of the general test list (which is shared between the gtest
// and python test runners), but is instead invoked manually by the
// torch_python_test.cpp
void testEvalModeForLoadedModule();
void testSerializationInterop();
void testTorchSaveError();

} // namespace jit
} // namespace torch
