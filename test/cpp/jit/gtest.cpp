#include <gtest/gtest.h>

#include <test/cpp/jit/test_alias_analysis.h>
#include <test/cpp/jit/test_misc.h>

using namespace torch;
using namespace torch::jit;

#define JIT_TEST(name)  \
  TEST(JitTest, name) { \
    test##name();       \
  }

JIT_TEST(ADFormulas)
JIT_TEST(Attributes)
JIT_TEST(Blocks)
JIT_TEST(CodeTemplate)
JIT_TEST(ControlFlow)
JIT_TEST(CreateAutodiffSubgraphs)
JIT_TEST(CustomOperators)
JIT_TEST(Differentiate)
JIT_TEST(DifferentiateWithRequiresGrad)
JIT_TEST(DynamicDAG)
JIT_TEST(EvalModeForLoadedModule)
JIT_TEST(FromQualString)
JIT_TEST(InternedStrings)
JIT_TEST(IValue)
JIT_TEST(Proto)
JIT_TEST(RegisterFusionCachesKernel)
JIT_TEST(SchemaParser)
JIT_TEST(TopologicalIndex)
JIT_TEST(TopologicalMove)
JIT_TEST(SubgraphUtils)
JIT_TEST(AliasAnalysis)
JIT_TEST(AliasTracker)

JIT_TEST(THNNConv)
JIT_TEST(ATenNativeBatchNorm)

#define JIT_TEST_CUDA(name)    \
  TEST(JitTest, name##_CUDA) { \
    test##name();              \
  }

JIT_TEST_CUDA(ArgumentSpec)
JIT_TEST_CUDA(Fusion)
JIT_TEST_CUDA(GraphExecutor)
JIT_TEST_CUDA(Interp)
