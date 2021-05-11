#include <ATen/core/ivalue.h>
#include <ATen/core/stack.h>
#include <ATen/core/boxing/KernelFunction.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>
#include <ATen/Functions.h>
#include <torch/library.h>

namespace at { namespace native {

// convenience helper for converting tensors to cpu

std::vector<at::Tensor> to_cpu(const at::TensorList& tensors) {
    // We can't just call at::to_cpu() on the entire list of Tensors
    // Because it will break on undefined tensors. Separate out undefined tensors first.
    std::vector<at::Tensor> cpu_tensors(tensors.size());
    std::vector<at::Tensor> valid_tensors;
    std::vector<bool> to_translate(tensors.size());
    for (size_t i = 0; i < tensors.size(); ++i) {
        const at::Tensor& tensor = tensors[i];
        if (tensor.defined()) {
            to_translate[i] = true;
            valid_tensors.push_back(tensor);
        } else {
            cpu_tensors[i] = tensor;
        }
    }
    auto cpu_valid_tensors = at::_to_cpu(valid_tensors);
    for (size_t i = 0, defined_pos = 0; i < tensors.size(); ++i) {
        if (to_translate[i]) {
            cpu_tensors[i] = std::move(cpu_valid_tensors[defined_pos++]);
        }
    }
  return cpu_tensors;
}


void cpu_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  auto& args = op.schema().arguments();
  const auto num_arguments = op.schema().arguments().size();
  auto arguments = torch::jit::last(stack, num_arguments);
  const auto arguments_begin = stack->size() - num_arguments;

  std::vector<at::Tensor> tensor_args;
  std::vector<int> tensor_args_indices;

  for (int64_t idx = 0; idx < arguments.size(); ++idx) {
    const auto& ivalue = arguments[idx];
    if (ivalue.isTensor()) {
      tensor_args.push_back(ivalue.toTensor());
      tensor_args_indices.push_back(idx);
    } else if (ivalue.isTensorList()) {
      // I'm worried about the ivalue storing a reference to the std::vector temporary that I create.
      // To be extra safe I'm converting to a c10::List, IValue will take ownership of the items in the list.
      auto cpu_ivalue = c10::IValue(c10::List<at::Tensor>(to_cpu(ivalue.toTensorList().vec())));
      (*stack)[arguments_begin + idx] = std::move(cpu_ivalue);
    }
  }
  auto cpu_tensors = to_cpu(tensor_args);

  for (auto i = 0; i < tensor_args_indices.size(); ++i) {
    auto cpu_ivalue = c10::IValue(cpu_tensors[i]);
    auto idx = tensor_args_indices[i];
    (*stack)[arguments_begin + idx] = std::move(cpu_ivalue);
  }

  op.redispatchBoxed(c10::DispatchKeySet(c10::DispatchKey::CPU), stack);

  const auto& num_returns = op.schema().returns().size();
  auto returns = torch::jit::last(stack, num_returns);
  const auto returns_begin = stack->size() - num_returns;

    std::cout << "START OUTER" << std::endl;
  for (int64_t idx = 0; idx < returns.size(); ++idx) {
    if (returns[idx].isTensor()) {
      const auto& return_tens = returns[idx].toTensor();
      std::cout << "return[" << idx << "] device = " << return_tens.device() << std::endl;
      for (int64_t i = 0; i < cpu_tensors.size(); ++i) {
        std::cout << "cpu_tensors[" << i << "] device = " << cpu_tensors[i].device() << std::endl;
        if (return_tens.is_alias_of(cpu_tensors[i])) {
          at::_copy_from_and_resize(return_tens, tensor_args[i]);
          auto moved_ivalue = c10::IValue(tensor_args[i]);
          (*stack)[returns_begin + idx] = std::move(moved_ivalue);
            std::cout << "break" << std::endl;
          break;
        }
      }
    }
  }
    std::cout << "END OUTER" << std::endl;

    // find each  tensor arg, gather into a vector<Tensor>
    // temporarily move those tensors off the stack, probably store them in a temp vector
    // convert to cpu: at::_to_cpu(vec);
    // box cpu tensors back into ivalues, put them onto the stack, exactly where the XLA versions used to be
    // call op.redispatch(cpu)
    //
    // IMPROVED LOGIC:
    // - convert all input tensors to CPU, call the (boxed) CPU function
    // - for every input AND output to the CPU function call:
    //   - if it's a mutable annotation, at::copy_ its contents to the correspnoding XLA tensor
    // - for each return:
    //    - if it's a mutable annotation, replace it (in the stakc) with the original input tensor (usually self or out)
    //    - otherwise: call to(cpu) on the ivalue, leave it in the stack
}

} // namespace native
} // namespace at
