#include <test/cpp/jit/test_alias_analysis.h>
#include <test/cpp/jit/test_constant_pooling.h>
#include <test/cpp/jit/test_class_parser.h>
#include <test/cpp/jit/test_irparser.h>
#include <test/cpp/jit/test_misc.h>
#include <test/cpp/jit/test_netdef_converter.h>

#include <sstream>
#include <string>

using namespace torch::jit::script;
namespace torch {
namespace jit {
void runJITCPPTests() {
  testNoneSchemaMatch();
  testAutogradProfiler();
  testADFormulas();
  testArgumentSpec();
  testAttributes();
  testBlocks();
  testCodeTemplate();
  testControlFlow();
  testCreateAutodiffSubgraphs();
  testCustomOperators();
  testDifferentiate();
  testDifferentiateWithRequiresGrad();
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
  testWriteTracking();
  testWildcards();
  testMemoryDAG();
  testNetDefConverter();
  testIRParser();
  testConstantPooling();
  testClassParser();
}

} // namespace jit
} // namespace torch
