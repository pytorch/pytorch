#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>
#include <ATen/TensorUtils.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/native/cpu/SoftmaxKernel.h>
#ifdef BUILD_NAMEDTENSOR
#include <ATen/NamedTensorUtils.h>
#endif

namespace at {
namespace native {
namespace {

template <typename scalar_t, bool LogSoftMax>
void host_softmax(Tensor output, const Tensor& input, const int64_t dim) {
  int64_t outer_size = 1;
  int64_t dim_size = input.size(dim);
  int64_t inner_size = 1;
  for (int64_t i = 0; i < dim; ++i)
    outer_size *= input.size(i);
  for (int64_t i = dim + 1; i < input.dim(); ++i)
    inner_size *= input.size(i);
  int64_t dim_stride = inner_size;
  int64_t outer_stride = dim_size * dim_stride;
  scalar_t* input_data_base = input.data<scalar_t>();
  scalar_t* output_data_base = output.data<scalar_t>();
  int64_t grain_size = std::min(internal::GRAIN_SIZE / dim_size, (int64_t)1);
  parallel_for(
      0, outer_size * inner_size, grain_size,
      [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; i++) {
          int64_t outer_idx = i / inner_size;
          int64_t inner_idx = i % inner_size;
          scalar_t* input_data =
              input_data_base + outer_idx * outer_stride + inner_idx;
          scalar_t* output_data =
              output_data_base + outer_idx * outer_stride + inner_idx;
          scalar_t max_input = input_data[0];
          for (int64_t d = 1; d < dim_size; d++)
            max_input = std::max(max_input, input_data[d * dim_stride]);

          scalar_t tmpsum = 0;
          for (int64_t d = 0; d < dim_size; d++) {
            scalar_t z = std::exp(input_data[d * dim_stride] - max_input);
            if (!LogSoftMax) {
              output_data[d * dim_stride] = z;
            }
            tmpsum += z;
          }

          if (LogSoftMax)
            tmpsum = std::log(tmpsum);
          else
            tmpsum = 1 / tmpsum;

          for (int64_t d = 0; d < dim_size; d++)
            if (LogSoftMax)
              output_data[d * dim_stride] =
                  input_data[d * dim_stride] - max_input - tmpsum;
            else
              output_data[d * dim_stride] *= tmpsum;
        }
      });
}

template <typename scalar_t, bool LogSoftMax>
void host_softmax_backward(
    Tensor& gI,
    const Tensor& grad,
    const Tensor& output,
    int64_t dim) {

  int64_t outer_size = 1;
  int64_t dim_size = grad.size(dim);
  int64_t inner_size = 1;
  for (int64_t i = 0; i < dim; ++i)
    outer_size *= grad.size(i);
  for (int64_t i = dim + 1; i < grad.dim(); ++i)
    inner_size *= grad.size(i);
  int64_t dim_stride = inner_size;
  int64_t outer_stride = dim_size * dim_stride;
  scalar_t* gradInput_data_base = gI.data<scalar_t>();
  scalar_t* output_data_base = output.data<scalar_t>();
  scalar_t* gradOutput_data_base = grad.data<scalar_t>();
  int64_t grain_size = std::min(internal::GRAIN_SIZE / dim_size, (int64_t)1);
  parallel_for(
      0, outer_size * inner_size, grain_size, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; i++) {
          int64_t outer_idx = i / inner_size;
          int64_t inner_idx = i % inner_size;
          scalar_t* gradInput_data =
              gradInput_data_base + outer_idx * outer_stride + inner_idx;
          scalar_t* output_data =
              output_data_base + outer_idx * outer_stride + inner_idx;
          const scalar_t* gradOutput_data =
              gradOutput_data_base + outer_idx * outer_stride + inner_idx;

          scalar_t sum = 0; // TODO was accreal here
          for (int64_t d = 0; d < dim_size; d++)
            if (LogSoftMax)
              sum += gradOutput_data[d * dim_stride];
            else
              sum +=
                  gradOutput_data[d * dim_stride] * output_data[d * dim_stride];

          for (int64_t d = 0; d < dim_size; d++) {
            if (LogSoftMax) {
              gradInput_data[d * dim_stride] = gradOutput_data[d * dim_stride] -
                  std::exp(output_data[d * dim_stride]) * sum;
            } else {
              gradInput_data[d * dim_stride] = output_data[d * dim_stride] *
                  (gradOutput_data[d * dim_stride] - sum);
            }
          }
        }
      });
}
} // namespace

Tensor softmax_cpu(const Tensor& input_, const int64_t dim_, const bool half_to_float) {
  AT_ASSERTM(!half_to_float, "softmax with half to float conversion is not supported on CPU");
  auto input = input_.contiguous();
  Tensor output = at::native::empty_like(input);
  int64_t dim = maybe_wrap_dim(dim_, input.dim());

  if (input.numel() == 0) {
    return output;
  }
 if (input.dim() == 0)
    input = input.view(1);
  TORCH_CHECK(
      dim >= 0 && dim < input.dim(),
      "dim must be non-negative and less than input dimensions");
  if (input.ndimension() > 0 && dim == input.ndimension() - 1) {
    softmax_lastdim_kernel(kCPU, output, input);
  } else {
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "softmax", [&] {
      host_softmax<scalar_t, false>(output, input, dim);
    });
  }
  return output;
}

Tensor log_softmax_cpu(const Tensor& input_, const int64_t dim_, const bool half_to_float) {
  AT_ASSERTM(!half_to_float, "softmax with half to float conversion is not supported on CPU");
  auto input = input_.contiguous();
  Tensor output = at::native::empty_like(input);
  int64_t dim = maybe_wrap_dim(dim_, input.dim());

  if (input.numel() == 0) {
    return output;
  }
  if (input.dim() == 0)
    input = input.view(1);
  TORCH_CHECK(
      dim >= 0 && dim < input.dim(),
      "dim must be non-negative and less than input dimensions");
  if (input.ndimension() > 0 && dim == input.ndimension() - 1) {
    log_softmax_lastdim_kernel(kCPU, output, input);
  } else {
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "log_softmax", [&] {
      host_softmax<scalar_t, true>(output, input, dim);
    });
  }
  return output;
}

Tensor softmax_backward_cpu(
    const Tensor& grad_,
    const Tensor& output_,
    int64_t dim_,
    const Tensor& input_) {
  TensorArg grad_arg{grad_, "grad", 1}, output_arg{output_, "output", 2};
  checkSameSize("softmax_backward", grad_arg, output_arg);
  int64_t dim = maybe_wrap_dim(dim_, grad_.dim());
  auto grad = grad_.contiguous();
  auto output = output_.contiguous();
  Tensor grad_input = at::native::empty_like(grad);

  if (output.numel() == 0) {
    return grad_input;
  }
  if (grad.dim() == 0)
    grad = grad.view(1);
  if (output.dim() == 0)
    output = output.view(1);
  TORCH_CHECK(
      dim >= 0 && dim < grad.dim(),
      "dim must be non-negative and less than input dimensions");
  if (grad.ndimension() > 0 && dim == grad.ndimension() - 1) {
    softmax_backward_lastdim_kernel(kCPU, grad_input, grad, output);
  } else {
    AT_DISPATCH_FLOATING_TYPES(grad.scalar_type(), "softmax_backward", [&] {
      host_softmax_backward<scalar_t, false>(grad_input, grad, output, dim);
    });
  }
  return grad_input;
}

Tensor log_softmax_backward_cpu(
    const Tensor& grad_,
    const Tensor& output_,
    int64_t dim_,
    const Tensor& input_) {
  TensorArg grad_arg{grad_, "grad", 1}, output_arg{output_, "output", 2};
  checkSameSize("log_softmax_backward", grad_arg, output_arg);
  int64_t dim = maybe_wrap_dim(dim_, grad_.dim());
  auto grad = grad_.contiguous();
  auto output = output_.contiguous();
  Tensor grad_input = at::native::empty_like(grad);

  if (output.numel() == 0) {
    return grad_input;
  }
  if (grad.dim() == 0)
    grad = grad.view(1);
  if (output.dim() == 0)
    output = output.view(1);
  TORCH_CHECK(
      dim >= 0 && dim < grad.dim(),
      "dim must be non-negative and less than input dimensions");
  if (grad.ndimension() > 0 && dim == grad.ndimension() - 1) {
    log_softmax_backward_lastdim_kernel(kCPU, grad_input, grad, output);
  } else {
    AT_DISPATCH_FLOATING_TYPES(grad.scalar_type(), "log_softmax_backward", [&] {
      host_softmax_backward<scalar_t, true>(grad_input, grad, output, dim);
    });
  }
  return grad_input;
}

Tensor softmax(const Tensor& input_, const int64_t dim_) {
  return at::_softmax(input_, dim_, false);
}

Tensor softmax(const Tensor& input_, const int64_t dim_, c10::optional<ScalarType> dtype) {
  if (input_.is_cuda() && input_.scalar_type() == ScalarType::Half && dtype == ScalarType::Float){
      return at::_softmax(input_, dim_, true);
  } else {
      Tensor converted = dtype.has_value() ? input_.toType(dtype.value()) : input_;
      return at::_softmax(converted, dim_, false);
  }
}

Tensor log_softmax(const Tensor& input_, const int64_t dim_) {
  return at::_log_softmax(input_, dim_, false);
}

Tensor log_softmax(const Tensor& input_, const int64_t dim_, c10::optional<ScalarType> dtype) {
  if (input_.is_cuda() && input_.scalar_type() == ScalarType::Half && dtype == ScalarType::Float){
      return at::_log_softmax(input_, dim_, true);
  } else {
      Tensor converted = dtype.has_value()? input_.toType(dtype.value()) : input_;
      return at::_log_softmax(converted, dim_, false);
  }
}

DEFINE_DISPATCH(softmax_lastdim_kernel);
DEFINE_DISPATCH(log_softmax_lastdim_kernel);
DEFINE_DISPATCH(softmax_backward_lastdim_kernel);
DEFINE_DISPATCH(log_softmax_backward_lastdim_kernel);

#ifdef BUILD_NAMEDTENSOR
Tensor softmax(const Tensor& self, Dimname dim, optional<ScalarType> dtype) {
  return at::softmax(self, dimname_to_position(self, dim), dtype);
}

Tensor log_softmax(const Tensor& self, Dimname dim, optional<ScalarType> dtype) {
  return at::log_softmax(self, dimname_to_position(self, dim), dtype);
}
#endif

}
}
