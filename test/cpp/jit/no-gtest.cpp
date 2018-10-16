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
  testSchemaParser();
  testDifferentiate(out);
  testDifferentiateWithRequiresGrad(out);
  testFromQualString();
  testFusion();
  testGraphExecutor();
  testInternedStrings();
  testInterp();
  testIValue();
  testProto();
  return out.str();
}
} // namespace jit
} // namespace torch
