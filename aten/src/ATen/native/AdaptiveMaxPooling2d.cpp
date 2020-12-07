#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/AdaptivePooling.h>


namespace at {
namespace native {

namespace {

void adaptive_max_pool2d_out_cpu_template(
          Tensor& output,
          Tensor& indices,
          const Tensor& input,
          IntArrayRef output_size)
{
  int64_t ndim = input.ndimension();
  for (int64_t i = 0; i < ndim; i++) {
    TORCH_CHECK(input.size(i) > 0,
      "adaptive_max_pool2d: expected input to have non-empty spatial dimensions, "
      "but input has sizes ", input.sizes(), " with dimension ", i, " being "
      "empty");
  }

  TORCH_CHECK((ndim == 3 || ndim == 4),
    "non-empty 3D or 4D (batch mode) tensor expected for input");

  TORCH_CHECK(output_size.size() == 2,
    "adaptive_max_pool2d: internal error: output_size.size() must be 2");

  TORCH_CHECK(input.dtype() == output.dtype(),
    "expected dtype ", input.dtype(), " for `output` but got dtype ", output.dtype());

  int64_t channels  = input.size(-3);
  int64_t input_height = input.size(-2);
  int64_t input_width = input.size(-1);
  int64_t output_height = output_size[0];
  int64_t output_width = output_size[1];

  if (ndim == 3) {
    output.resize_({channels, output_height, output_width});
    indices.resize_({channels, output_height, output_width});
  } else {
    int64_t nbatch = input.size(0);
    output.resize_({nbatch, channels, output_height, output_width}, input.suggest_memory_format());
    indices.resize_({nbatch, channels, output_height, output_width}, input.suggest_memory_format());
  }

  adaptive_max_pool2d_kernel(kCPU, output, indices, input, output_size);
}

Tensor& adaptive_max_pool2d_backward_out_cpu_template(
          Tensor& grad_input,
          const Tensor& grad_output,
          const Tensor& input,
          const Tensor& indices)
{
  int64_t ndim = grad_output.ndimension();
  for (int64_t i = 0; i < ndim; i++) {
    TORCH_CHECK(grad_output.size(i) > 0,
      "adaptive_max_pooling2d_backward(): expected grad_output to have non-empty spatial dimensions, "
      "but grad_output has sizes ", grad_output.sizes(), " with dimension ", i, " being "
      "empty");
  }

  TORCH_CHECK((ndim == 3 || ndim == 4),
    "non-empty 3D or 4D (batch mode) tensor expected for grad_output");
  TORCH_CHECK(input.dtype() == grad_output.dtype(),
    "expected dtype ", input.dtype(), " for `grad_output` but got dtype ", grad_output.dtype());
  TORCH_CHECK(input.dtype() == grad_input.dtype(),
    "expected dtype ", input.dtype(), " for `grad_input` but got dtype ", grad_input.dtype());

  grad_input.resize_(input.sizes(), input.suggest_memory_format());
  grad_input.zero_();

  adaptive_max_pool2d_backward_kernel(kCPU, grad_input, grad_output, indices);
  return grad_input;
}

} // namespace

std::tuple<Tensor&, Tensor&> adaptive_max_pool2d_out_cpu(
  Tensor& output,
  Tensor& indices,
  const Tensor& input,
  IntArrayRef output_size)
{
  adaptive_max_pool2d_out_cpu_template(
    output,
    indices,
    input,
    output_size);
  return std::tuple<Tensor&, Tensor&>(output, indices);
}

std::tuple<Tensor, Tensor> adaptive_max_pool2d_cpu(
  const Tensor& input,
  IntArrayRef output_size)
{
  Tensor output = at::empty({0}, input.options());
  Tensor indices = at::empty({0}, input.options().dtype(kLong));
  adaptive_max_pool2d_out_cpu_template(
    output,
    indices,
    input,
    output_size);
  return std::tuple<Tensor, Tensor>(output, indices);
}

Tensor& adaptive_max_pool2d_backward_out_cpu(
  Tensor& grad_input,
  const Tensor& grad_output,
  const Tensor& input,
  const Tensor& indices)
{
  adaptive_max_pool2d_backward_out_cpu_template(
    grad_input,
    grad_output,
    input,
    indices);
  return grad_input;
}

Tensor adaptive_max_pool2d_backward_cpu(
  const Tensor& grad_output,
  const Tensor& input,
  const Tensor& indices)
{
  auto grad_input = at::empty({0}, input.options());
  adaptive_max_pool2d_backward_out_cpu_template(
    grad_input,
    grad_output,
    input,
    indices);
  return grad_input;
}

DEFINE_DISPATCH(adaptive_max_pool2d_kernel);
DEFINE_DISPATCH(adaptive_max_pool2d_backward_kernel);

} // at::native
} // at
