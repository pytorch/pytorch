#include "tensor.h"

#include "torch/csrc/autograd/variable.h"
#include "torch/csrc/autograd/functions/basic_ops.h"
#include "torch/csrc/autograd/functions/utils.h"
#include "torch/csrc/autograd/generated/Functions.h"
#include "torch/csrc/utils/auto_gpu.h"

namespace torch { namespace autograd {

auto Identity::apply(const variable_list& inputs) -> variable_list {
  return inputs;
};

auto CopyBackwards::apply(const variable_list& grads) -> variable_list {
  check_input_variables("CopyBackwards", grads, 1);
  auto& grad = grads[0];
  variable_list grad_inputs(2);
  if (should_compute_output(0)) {
    grad_inputs[0] = at::zeros_like(grad);
  }
  if (should_compute_output(1)) {
    AutoGPU autoGPU(src_device);
    if (grad.is_cuda() && grad.get_device() != src_device) {
      grad_inputs[1] = src_type->copy(grad);
    } else {
      grad_inputs[1] = grad.toType(*src_type);
    }
  }
  return grad_inputs;
};

auto Clone::apply(const variable_list& inputs) -> variable_list {
  check_input_variables("Clone", inputs, 1);
  auto& input = inputs[0].data();
  AutoGPU guard(input);

  at::Tensor output = input.clone();

  return wrap_outputs(inputs, as_tensor_list(std::move(output)), [&](FunctionFlags f) {
    return std::make_shared<Identity>(std::move(f));
  });
};

auto Contiguous::apply(const variable_list& inputs) -> variable_list {
  check_input_variables("Contiguous", inputs, 1);
  auto& input = inputs[0].data();
  AutoGPU guard(input);

  at::Tensor output = input.contiguous();

  return wrap_outputs(inputs, as_tensor_list(std::move(output)), [&](FunctionFlags f) {
    return std::make_shared<Identity>(std::move(f));
  });
};

auto Transpose::apply(const variable_list& inputs) -> variable_list {
  check_input_variables("Transpose", inputs, 1);

  auto& input = inputs[0].data();
  AutoGPU guard(input);

  at::Tensor output = input.transpose(dim1, dim2);

  return wrap_outputs(inputs, as_tensor_list(std::move(output)), [&](FunctionFlags f) {
    return std::make_shared<Transpose>(dim1, dim2);
  });
}

auto View::apply(const variable_list& inputs) -> variable_list {
  check_input_variables("View", inputs, 1);

  auto& input = inputs[0].data();
  AutoGPU guard(input);

  at::Tensor output = input.view(size);

  return wrap_outputs(inputs, as_tensor_list(std::move(output)), [&](FunctionFlags f) {
    return std::make_shared<View>(input.sizes());
  });
}

auto Expand::apply(const variable_list& inputs) -> variable_list {
  check_input_variables("Expand", inputs, 1);

  auto& input = inputs[0].data();
  AutoGPU guard(input);

  at::Tensor output = input.expand(size);

  return wrap_outputs(inputs, as_tensor_list(std::move(output)), [&](FunctionFlags f) {
    return std::make_shared<Error>("Expand is not differentiable", std::move(f));
  });
}

auto Narrow::apply(const variable_list& inputs) -> variable_list {
  check_input_variables("Narrow", inputs, 1);

  auto& input = inputs[0].data();
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

  auto& input = inputs[0].data();
  AutoGPU guard(input);

  std::vector<at::Tensor> tensors(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    tensors[i] = inputs[i].data();
  }
  auto output = input.type().cat(tensors, (int) dim);

  return wrap_outputs(inputs, as_tensor_list(output), [&](FunctionFlags f) {
    return std::make_shared<Error>("Cat is not differentiable", std::move(f));
  });
}

auto Chunk::apply(const variable_list& inputs) -> variable_list {
  auto outputs = chunk(inputs[0].data(), chunks, dim);
  return wrap_outputs(inputs, std::move(outputs), [](FunctionFlags f) {
    return std::make_shared<Error>("Chunk is not differentiable", std::move(f));
  });
}

CopySlices::CopySlices(const Variable& base_var, at::TensorGeometry view_, std::shared_ptr<Function> fn_)
  : base(base_var)
  , view(std::move(view_))
  , fn(std::move(fn_))
{
  num_inputs = 1;

  // Take the next_functions of fn as our own, except for index 0 which goes
  // to base instead of the view.
  next_functions.resize(fn->next_functions.size());
  next_functions[0] = std::make_pair(base_var.grad_fn(), base_var.output_nr());
  for (size_t i = 1; i < next_functions.size(); i++) {
    next_functions[i] = fn->next_functions[i];
  }
}

auto CopySlices::apply(const variable_list& inputs) -> variable_list {
  check_input_variables("CopySlices", inputs, 1);
  auto& grad = inputs[0];

  if (!fn) {
    throw std::runtime_error(ERR_BACKWARD_TWICE);
  }

  auto result = grad.type().tensor(base.sizes(), base.strides());
  result.copy_(grad);

  auto offset = view.storage_offset() - base.storage_offset();
  auto grad_slice = result.as_strided(view.sizes(), view.strides(), offset);

  // TODO: We clone grad_slice because we modify it below and "fn" might save
  // it for the backward of res. We might be able to avoid the clone() if
  // grad_slice is volatile.
  auto res = (*fn)({ grad_slice.clone() });

  variable_list grad_inputs(next_functions.size());
  for (size_t i = 0; i < res.size(); i++) {
    if (should_compute_output(i)) {
      TORCH_ASSERT(res[i].defined());
      if (i == 0) {
        grad_slice.copy_(res[i]);
        grad_inputs[i] = std::move(result);
      } else {
        grad_inputs[i] = std::move(res[i]);
      }
    }
  }

  return grad_inputs;
}

void CopySlices::releaseVariables() {
  fn = nullptr;
}

}} // namespace torch::autograd
