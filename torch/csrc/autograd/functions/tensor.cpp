#include "tensor.h"

#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/autograd/functions/basic_ops.h"
#include "torch/csrc/autograd/functions/utils.h"
#include "torch/csrc/utils/auto_gpu.h"

namespace torch { namespace autograd {

auto Identity::apply(const variable_list& inputs) -> variable_list {
  return inputs;
};

auto Clone::apply(const variable_list& inputs) -> variable_list {
  check_input_variables("Clone", inputs, 1);
  auto& input = inputs[0]->data;
  AutoGPU guard(input);

  at::Tensor output = input.clone();

  return wrap_outputs(inputs, as_tensor_list(std::move(output)), [&](FunctionFlags f) {
    return std::make_shared<Identity>(std::move(f));
  });
};

auto Contiguous::apply(const variable_list& inputs) -> variable_list {
  check_input_variables("Contiguous", inputs, 1);
  auto& input = inputs[0]->data;
  AutoGPU guard(input);

  at::Tensor output = input.contiguous();

  return wrap_outputs(inputs, as_tensor_list(std::move(output)), [&](FunctionFlags f) {
    return std::make_shared<Identity>(std::move(f));
  });
};

auto Transpose::apply(const variable_list& inputs) -> variable_list {
  check_input_variables("Transpose", inputs, 1);

  auto& input = inputs[0]->data;
  AutoGPU guard(input);

  at::Tensor output = input.transpose(dim1, dim2);

  return wrap_outputs(inputs, as_tensor_list(std::move(output)), [&](FunctionFlags f) {
    return std::make_shared<Transpose>(dim1, dim2);
  });
}

auto View::apply(const variable_list& inputs) -> variable_list {
  check_input_variables("View", inputs, 1);

  auto& input = inputs[0]->data;
  AutoGPU guard(input);

  at::Tensor output = input.view(size);

  return wrap_outputs(inputs, as_tensor_list(std::move(output)), [&](FunctionFlags f) {
    return std::make_shared<View>(input.sizes());
  });
}

auto Expand::apply(const variable_list& inputs) -> variable_list {
  check_input_variables("Expand", inputs, 1);

  auto& input = inputs[0]->data;
  AutoGPU guard(input);

  at::Tensor output = input.expand(size);

  return wrap_outputs(inputs, as_tensor_list(std::move(output)), [&](FunctionFlags f) {
    return std::make_shared<Error>("Expand is not differentiable", std::move(f));
  });
}

auto Narrow::apply(const variable_list& inputs) -> variable_list {
  check_input_variables("Narrow", inputs, 1);

  auto& input = inputs[0]->data;
  AutoGPU guard(input);

  at::Tensor output = input.narrow(dim, start, size);

  return wrap_outputs(inputs, as_tensor_list(std::move(output)), [&](FunctionFlags f) {
    return std::make_shared<Error>("Narrow is not differentiable", std::move(f));
  });
}

auto Cat::apply(const variable_list& inputs) -> variable_list {
  int num_inputs = inputs.size();
  if (num_inputs == 0) {
    throw std::runtime_error("Cat operation expect at least one argument.");
  }

  auto& input = inputs[0]->data;
  AutoGPU guard(input);

  std::vector<at::Tensor> tensors(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    tensors[i] = inputs[i]->data;
  }
  auto output = input.type().cat(tensors, dim);

  return wrap_outputs(inputs, as_tensor_list(output), [&](FunctionFlags f) {
    return std::make_shared<Error>("Cat is not differentiable", std::move(f));
  });
}

}} // namespace torch::autograd
