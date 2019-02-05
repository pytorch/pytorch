#include <test/cpp/jit/test_alias_analysis.h>
#include <test/cpp/jit/test_misc.h>

#include <sstream>
#include <string>

namespace torch {
namespace jit {
std::string runJITCPPTests() {
  std::stringstream out;
  testAutogradProfiler();
  testADFormulas();
  testArgumentSpec();
  testAttributes();
  testBlocks(out);
  testCodeTemplate();
  testControlFlow();
  testCreateAutodiffSubgraphs(out);
  testCustomOperators();
  testDifferentiate(out);
  testDifferentiateWithRequiresGrad(out);
  testDynamicDAG();
  testEvalModeForLoadedModule();
  testFromQualString();
  testFusion();
  testGraphExecutor();
  testInternedStrings();
  testInterp();
  testIValue();
  testProto();
  testSchemaParser();
  testTopologicalIndex();
  testTopologicalMove();
  testSubgraphUtils();
  testTHNNConv();
  testATenNativeBatchNorm();
  testRegisterFusionCachesKernel();
  testAliasAnalysis();
  testAliasTracker();
  return out.str();
}

} // namespace jit
} // namespace torch
