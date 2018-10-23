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
  return out.str();
}


at::Tensor leaky_relu(at::Tensor tensor, double scalar) {
  return at::leaky_relu(tensor, scalar);
}
at::Tensor cat(std::vector<at::Tensor> tensors) {
  return at::cat(tensors);
}

static auto registry =
    torch::jit::RegisterOperators()
        .op("_test::leaky_relu(Tensor self, float v=0.01) -> Tensor", &leaky_relu)
        .op("_test::cat(Tensor[] inputs) -> Tensor", &cat);


} // namespace jit
} // namespace torch
