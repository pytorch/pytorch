#include <torch/script.h>

#include "op.h"

#include <memory>
#include <string>
#include <vector>

#include <iostream>

namespace helpers {
template <typename Predicate>
void check_all_parameters(
    const torch::jit::script::Module& module,
    Predicate predicate) {
  for (const auto& parameter : module.get_parameters()) {
    AT_ASSERT(predicate(*parameter->slot()));
  }
  for (const auto& child : module.get_modules()) {
    check_all_parameters(*child->module, predicate);
  }
}
} // namespace helpers

void get_operator_from_registry_and_execute() {
  auto& ops = torch::jit::getAllOperatorsFor(
      torch::jit::Symbol::fromQualString("custom::op"));
  AT_ASSERT(ops.size() == 1);

  auto& op = ops.front();
  AT_ASSERT(op->schema().name() == "custom::op");

  torch::jit::Stack stack;
  torch::jit::push(stack, torch::ones(5), 2.0, 3);
  op->getOperation()(stack);
  std::vector<torch::Tensor> output;
  torch::jit::pop(stack, output);

  const auto manual = custom_op(torch::ones(5), 2.0, 3);

  AT_ASSERT(output.size() == 3);
  for (size_t i = 0; i < output.size(); ++i) {
    AT_ASSERT(output[i].allclose(torch::ones(5) * 2));
    AT_ASSERT(output[i].allclose(manual[i]));
  }
}

void load_serialized_module_with_custom_op_and_execute(
    const std::string& path_to_exported_script_module) {
  std::shared_ptr<torch::jit::script::Module> module =
      torch::jit::load(path_to_exported_script_module);
  AT_ASSERT(module != nullptr);

  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(torch::ones(5));
  auto output = module->forward(inputs).toTensor();

  AT_ASSERT(output.allclose(torch::ones(5) + 1));
}

void test_argument_checking_for_serialized_modules(
    const std::string& path_to_exported_script_module) {
  std::shared_ptr<torch::jit::script::Module> module =
      torch::jit::load(path_to_exported_script_module);
  AT_ASSERT(module != nullptr);

  try {
    module->forward({torch::jit::IValue(1), torch::jit::IValue(2)});
    AT_ASSERT(false);
  } catch (const c10::Error& error) {
    AT_ASSERT(
        std::string(error.what_without_backtrace())
            .find("Expected at most 1 argument(s) for operator 'forward', "
                  "but received 2 argument(s)") == 0);
  }

  try {
    module->forward({torch::jit::IValue(5)});
    AT_ASSERT(false);
  } catch (const c10::Error& error) {
    AT_ASSERT(
        std::string(error.what_without_backtrace())
            .find("Expected value of type Dynamic for argument 'input' in "
                  "position 0, but instead got value of type int") == 0);
  }

  try {
    module->forward({});
    AT_ASSERT(false);
  } catch (const c10::Error& error) {
    AT_ASSERT(
        std::string(error.what_without_backtrace())
            .find("forward() is missing value for argument 'input'") == 0);
  }
}

void test_move_to_device(const std::string& path_to_exported_script_module) {
  std::shared_ptr<torch::jit::script::Module> module =
      torch::jit::load(path_to_exported_script_module);
  AT_ASSERT(module != nullptr);

  helpers::check_all_parameters(*module, [](const torch::Tensor& tensor) {
    return tensor.device().is_cpu();
  });

  module->to(torch::kCUDA);

  helpers::check_all_parameters(*module, [](const torch::Tensor& tensor) {
    return tensor.device().is_cuda();
  });

  module->to(torch::kCPU);

  helpers::check_all_parameters(*module, [](const torch::Tensor& tensor) {
    return tensor.device().is_cpu();
  });
}

void test_move_to_dtype(const std::string& path_to_exported_script_module) {
  std::shared_ptr<torch::jit::script::Module> module =
      torch::jit::load(path_to_exported_script_module);
  AT_ASSERT(module != nullptr);

  module->to(torch::kInt);

  helpers::check_all_parameters(*module, [](const torch::Tensor& tensor) {
    return tensor.dtype() == torch::kInt;
  });

  module->to(torch::kDouble);

  helpers::check_all_parameters(*module, [](const torch::Tensor& tensor) {
    return tensor.dtype() == torch::kDouble;
  });
}

int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: test_custom_ops <path-to-exported-script-module>\n";
    return -1;
  }
  const std::string path_to_exported_script_module = argv[1];

  get_operator_from_registry_and_execute();
  load_serialized_module_with_custom_op_and_execute(
      path_to_exported_script_module);
  test_argument_checking_for_serialized_modules(path_to_exported_script_module);
  test_move_to_dtype(path_to_exported_script_module);

  if (torch::globalContext().getNumGPUs() > 0) {
    test_move_to_device(path_to_exported_script_module);
  }

  std::cout << "ok\n";
}
