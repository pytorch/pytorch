#include <torch/op.h>

#include "op.h"

#include <cassert>
#include <memory>
#include <vector>

#include <iostream>

void get_operator_from_registry_and_execute() {
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
}

void load_serialized_module_with_custom_op_and_execute(
    const char* path_to_exported_script_module) {
  std::shared_ptr<torch::jit::script::Module> module =
      torch::jit::load(path_to_exported_script_module);
  assert(module != nullptr);

  torch::jit::script::Method& forward = module->get_method("forward");

  torch::jit::Stack stack;
  torch::jit::push(stack, torch::ones(5));
  forward.run(stack);
  at::Tensor output;
  torch::jit::pop(stack, output);

  assert(output.allclose(torch::ones(5) + 1));
}

int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: test_custom_ops <path-to-exported-script-module>\n";
    return -1;
  }
  get_operator_from_registry_and_execute();
  load_serialized_module_with_custom_op_and_execute(argv[1]);
  std::cout << "ok\n";
}
