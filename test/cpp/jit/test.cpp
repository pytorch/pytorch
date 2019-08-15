#if defined(USE_GTEST)
#include <gtest/gtest.h>
#endif

#include <c10/macros/Export.h>

// To add a new test file:
// 1. Add a test_foo.h file in this directory
// 2. include test_base.h
// 3. Write your tests as pure functions starting with "test", like "testFoo"
// 4. Include test_foo.h here and add it to the appropriate macro listing
#include <test/cpp/jit/test_alias_analysis.h>
#include <test/cpp/jit/test_argument_spec.h>
#include <test/cpp/jit/test_autodiff.h>
#include <test/cpp/jit/test_class_import.h>
#include <test/cpp/jit/test_class_parser.h>
#include <test/cpp/jit/test_code_template.h>
#include <test/cpp/jit/test_constant_pooling.h>
#include <test/cpp/jit/test_constant_propagation.h>
#include <test/cpp/jit/test_create_autodiff_subgraphs.h>
#include <test/cpp/jit/test_custom_operators.h>
#include <test/cpp/jit/test_dce.h>
#include <test/cpp/jit/test_dynamic_dag.h>
#include <test/cpp/jit/test_fuser.h>
#include <test/cpp/jit/test_graph_executor.h>
#include <test/cpp/jit/test_interpreter.h>
#include <test/cpp/jit/test_ir.h>
#include <test/cpp/jit/test_irparser.h>
#include <test/cpp/jit/test_ivalue.h>
#include <test/cpp/jit/test_misc.h>
#include <test/cpp/jit/test_netdef_converter.h>
#include <test/cpp/jit/test_peephole_optimize.h>
#include <test/cpp/jit/test_qualified_name.h>
#include <test/cpp/jit/test_save_load.h>
#include <test/cpp/jit/test_subgraph_matcher.h>
#include <test/cpp/jit/test_subgraph_utils.h>
#include <torch/csrc/WindowsTorchApiMacro.h>

using namespace torch::jit::script;
using namespace torch::jit::test;

namespace torch {
namespace jit {
#define TH_FORALL_TESTS(_)             \
  _(ADFormulas)                        \
  _(Attributes)                        \
  _(Blocks)                            \
  _(CodeTemplate)                      \
  _(ControlFlow)                       \
  _(CreateAutodiffSubgraphs)           \
  _(CustomOperators)                   \
  _(CustomOperatorAliasing)            \
  _(IValueKWargs)                      \
  _(CustomFusion)                      \
  _(Differentiate)                     \
  _(DifferentiateWithRequiresGrad)     \
  _(DynamicDAG)                        \
  _(FromQualString)                    \
  _(InternedStrings)                   \
  _(IValue)                            \
  _(PassManagement)                    \
  _(Proto)                             \
  _(RegisterFusionCachesKernel)        \
  _(SchemaParser)                      \
  _(TopologicalIndex)                  \
  _(TopologicalMove)                   \
  _(SubgraphUtils)                     \
  _(AliasAnalysis)                     \
  _(ContainerAliasing)                 \
  _(AliasRegistration)                 \
  _(WriteTracking)                     \
  _(Wildcards)                         \
  _(MemoryDAG)                         \
  _(IRParser)                          \
  _(ConstantPooling)                   \
  _(ConstantPropagation)               \
  _(NetDefConverter)                   \
  _(THNNConv)                          \
  _(ATenNativeBatchNorm)               \
  _(NoneSchemaMatch)                   \
  _(ClassParser)                       \
  _(Profiler)                          \
  _(InsertAndEliminateRedundantGuards) \
  _(InsertBailOuts)                    \
  _(PeepholeOptimize)                  \
  _(RecordFunction)                    \
  _(ThreadLocalDebugInfo)              \
  _(SubgraphMatching)                  \
  _(ModuleDefine)                      \
  _(QualifiedName)                     \
  _(ClassImport)                       \
  _(ProfiledTensorTypeHashing)         \
  _(ScriptObject)                      \
  _(SaveExtraFilesHook)                \
  _(InsertConstant)                    \
  _(DCE)                               \
  _(CustomFusionNestedBlocks)          \
  _(ImportTooNew)

#define TH_FORALL_TESTS_CUDA(_) \
  _(ArgumentSpec)               \
  _(CompleteArgumentSpec)       \
  _(Fusion)                     \
  _(GraphExecutor)              \
  _(ModuleConversion)           \
  _(Interp)

#if defined(USE_GTEST)

#define JIT_GTEST(name) \
  TEST(JitTest, name) { \
    test##name();       \
  }
TH_FORALL_TESTS(JIT_GTEST)
#undef JIT_TEST

#define JIT_GTEST_CUDA(name)   \
  TEST(JitTest, name##_CUDA) { \
    test##name();              \
  }
TH_FORALL_TESTS_CUDA(JIT_GTEST_CUDA)
#undef JIT_TEST_CUDA
#endif

#define JIT_TEST(name) test##name();
TORCH_API void runJITCPPTests(bool runCuda) {
  TH_FORALL_TESTS(JIT_TEST)
  if (runCuda) {
    TH_FORALL_TESTS_CUDA(JIT_TEST)
  }

  // This test is special since it requires prior setup in python.
  // So it's included here but not in the pure cpp gtest suite
  testEvalModeForLoadedModule();
}
#undef JIT_TEST
} // namespace jit
} // namespace torch
