#include <test/cpp/jit/tests.h>

#include <sstream>
#include <string>

namespace torch {
namespace jit {
std::string runJITCPPTests() {
  std::stringstream out;
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
  return out.str();
}

} // namespace jit
} // namespace torch
