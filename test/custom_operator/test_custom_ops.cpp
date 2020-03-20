#include <torch/script.h>
#include <torch/cuda.h>

#include "op.h"

#include <memory>
#include <string>
#include <vector>

#include <iostream>

namespace helpers {
template <typename Predicate>
void check_all_parameters(
    const torch::jit::Module& module,
    Predicate predicate) {
  for (at::Tensor parameter : module.parameters()) {
    AT_ASSERT(predicate(parameter));
  }
}

template<class Result, class... Args>
Result get_operator_from_registry_and_execute(const char* op_name, Args&&... args) {
  auto& ops = torch::jit::getAllOperatorsFor(
      torch::jit::Symbol::fromQualString(op_name));
  TORCH_INTERNAL_ASSERT(ops.size() == 1);

  auto& op = ops.front();
  TORCH_INTERNAL_ASSERT(op->schema().name() == op_name);

  torch::jit::Stack stack;
  torch::jit::push(stack, std::forward<Args>(args)...);
  op->getOperation()(stack);

  TORCH_INTERNAL_ASSERT(1 == stack.size());
  return torch::jit::pop(stack).to<Result>();
}
} // namespace helpers

void get_operator_from_registry_and_execute() {
  std::vector<torch::Tensor> output =
    helpers::get_operator_from_registry_and_execute<std::vector<torch::Tensor>>("custom::op", torch::ones(5), 2.0, 3);

  const auto manual = custom_op(torch::ones(5), 2.0, 3);

  TORCH_INTERNAL_ASSERT(output.size() == 3);
  for (size_t i = 0; i < output.size(); ++i) {
    TORCH_INTERNAL_ASSERT(output[i].allclose(torch::ones(5) * 2));
    TORCH_INTERNAL_ASSERT(output[i].allclose(manual[i]));
  }
}

void get_autograd_operator_from_registry_and_execute() {
  torch::Tensor x = torch::randn({5,5}, torch::requires_grad());
  torch::Tensor y = torch::randn({5,5}, torch::requires_grad());

  torch::Tensor output =
    helpers::get_operator_from_registry_and_execute<torch::Tensor>("custom::op_with_autograd", x, 2, y);

  TORCH_INTERNAL_ASSERT(output.allclose(x + 2*y + x*y));
  auto go = torch::ones({}, torch::requires_grad());
  output.sum().backward(go, false, true);

  TORCH_INTERNAL_ASSERT(torch::allclose(x.grad(), y + torch::ones({5,5})));
  TORCH_INTERNAL_ASSERT(torch::allclose(y.grad(), x + torch::ones({5,5})*2));
}

void get_autograd_operator_from_registry_and_execute_in_nograd_mode() {
  at::AutoNonVariableTypeMode _var_guard(true);

  torch::Tensor x = torch::randn({5,5}, torch::requires_grad());
  torch::Tensor y = torch::randn({5,5}, torch::requires_grad());

  torch::Tensor output =
    helpers::get_operator_from_registry_and_execute<torch::Tensor>("custom::op_with_autograd", x, 2, y);

  TORCH_INTERNAL_ASSERT(output.allclose(x + 2*y + x*y));
}

void load_serialized_module_with_custom_op_and_execute(
    const std::string& path_to_exported_script_module) {
  torch::jit::Module module =
      torch::jit::load(path_to_exported_script_module);
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(torch::ones(5));
  auto output = module.forward(inputs).toTensor();

  AT_ASSERT(output.allclose(torch::ones(5) + 1));
}

void test_argument_checking_for_serialized_modules(
    const std::string& path_to_exported_script_module) {
  torch::jit::Module module =
      torch::jit::load(path_to_exported_script_module);

  try {
    module.forward({torch::jit::IValue(1), torch::jit::IValue(2)});
    AT_ASSERT(false);
  } catch (const c10::Error& error) {
    AT_ASSERT(
        std::string(error.what_without_backtrace())
            .find("Expected at most 2 argument(s) for operator 'forward', "
                  "but received 3 argument(s)") == 0);
  }

  try {
    module.forward({torch::jit::IValue(5)});
    AT_ASSERT(false);
  } catch (const c10::Error& error) {
    AT_ASSERT(
        std::string(error.what_without_backtrace())
            .find("forward() Expected a value of type 'Tensor' "
                  "for argument 'input' but instead found type 'int'") == 0);
  }

  try {
    module.forward({});
    AT_ASSERT(false);
  } catch (const c10::Error& error) {
    AT_ASSERT(
        std::string(error.what_without_backtrace())
            .find("forward() is missing value for argument 'input'") == 0);
  }
}

void test_move_to_device(const std::string& path_to_exported_script_module) {
  torch::jit::Module module =
      torch::jit::load(path_to_exported_script_module);

  helpers::check_all_parameters(module, [](const torch::Tensor& tensor) {
    return tensor.device().is_cpu();
  });

  module.to(torch::kCUDA);

  helpers::check_all_parameters(module, [](const torch::Tensor& tensor) {
    return tensor.device().is_cuda();
  });

  module.to(torch::kCPU);

  helpers::check_all_parameters(module, [](const torch::Tensor& tensor) {
    return tensor.device().is_cpu();
  });
}

void test_move_to_dtype(const std::string& path_to_exported_script_module) {
  torch::jit::Module module =
      torch::jit::load(path_to_exported_script_module);

  module.to(torch::kInt);

  helpers::check_all_parameters(module, [](const torch::Tensor& tensor) {
    return tensor.dtype() == torch::kInt;
  });

  module.to(torch::kDouble);

  helpers::check_all_parameters(module, [](const torch::Tensor& tensor) {
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
  get_autograd_operator_from_registry_and_execute();
  get_autograd_operator_from_registry_and_execute_in_nograd_mode();
  load_serialized_module_with_custom_op_and_execute(
      path_to_exported_script_module);
  test_argument_checking_for_serialized_modules(path_to_exported_script_module);
  test_move_to_dtype(path_to_exported_script_module);

  if (torch::cuda::device_count() > 0) {
    test_move_to_device(path_to_exported_script_module);
  }

  std::cout << "ok\n";
}
