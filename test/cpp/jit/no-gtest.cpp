#include <test/cpp/jit/test_alias_analysis.h>
#include <test/cpp/jit/test_irparser.h>
#include <test/cpp/jit/test_misc.h>
#include <test/cpp/jit/test_netdef_converter.h>

#include <sstream>
#include <string>

namespace torch {
namespace jit {
std::string runJITCPPTests() {
  std::stringstream out;
  testNoneSchemaMatch();
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
  testNetDefConverter(out);
  testIRParser(out);
  return out.str();
}

} // namespace jit
} // namespace torch
