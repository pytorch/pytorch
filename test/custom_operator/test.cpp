#include "op.h"

#include <cassert>
#include <vector>

int main() {
  auto& ops = torch::jit::getAllOperatorsFor(
      torch::jit::Symbol::fromQualString("custom::op"));
  assert(ops.size() == 1);

  auto& op = ops.front();
  assert(op->schema().name == "custom::op");

  torch::jit::Stack stack;
  torch::jit::push(stack, torch::ones(5), 2.0, 3);
  op->getOperation()(stack);
  std::vector<at::Tensor> output;
  torch::jit::pop(stack, output);

  assert(output.size() == 3);
  for (const auto& tensor : output) {
    assert(tensor.allclose(torch::ones(5) * 2));
  }
  std::cout << "success" << std::endl;
}
