#include <iostream>
#include "ATen/ATen.h"
#include "ATen/AccumulateType.h"
#include "ATen/NativeFunctions.h"
#include "ATen/Parallel.h"
#include "ATen/TensorUtils.h"
#include "ATen/WrapDimUtils.h"
#include "ATen/native/cpu/SoftmaxKernel.h"

namespace at {
namespace native {
namespace {

static tbb::affinity_partitioner ap;

template <typename scalar_t, bool LogSoftMax>
void host_softmax(Tensor output, const Tensor& input, const int64_t dim) {
  internal::init_tbb_num_threads();
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
  int64_t grain_size = std::min(internal::TBB_GRAIN_SIZE / dim_size, (int64_t)1);
  tbb::parallel_for(
      tbb::blocked_range<int64_t>(0, outer_size * inner_size, grain_size),
      [&](const tbb::blocked_range<int64_t>& r) {
        for (int64_t i = r.begin(); i < r.end(); i++) {
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
            tmpsum = max_input + std::log(tmpsum);
          else
            tmpsum = 1 / tmpsum;

          for (int64_t d = 0; d < dim_size; d++)
            if (LogSoftMax)
              output_data[d * dim_stride] = input_data[d * dim_stride] - tmpsum;
            else
              output_data[d * dim_stride] *= tmpsum;
        }
      },
      ap);
}

template <typename scalar_t, bool LogSoftMax>
void host_softmax_backward(
    Tensor& gI,
    const Tensor& grad,
    const Tensor& output,
    int64_t dim) {
  internal::init_tbb_num_threads();

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
  int64_t grain_size = std::min(internal::TBB_GRAIN_SIZE / dim_size, (int64_t)1);
  tbb::parallel_for(
      tbb::blocked_range<int64_t>(0, outer_size * inner_size, grain_size),
      [&](const tbb::blocked_range<int64_t>& r) {
        for (int64_t i = r.begin(); i < r.end(); i++) {
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
      },
      ap);
}
} // namespace

Tensor softmax_cpu(const Tensor& input_, const int64_t dim_) {
  auto input = input_.contiguous();
  Tensor output = at::native::empty_like(input);
  int64_t dim = maybe_wrap_dim(dim_, input.dim());
  if (input.dim() == 0)
    input = input.view(1);
  AT_CHECK(
      dim >= 0 && dim < input.dim(),
      "dim must be non-negative and less than input dimensions");
  if (input.ndimension() > 0 && dim == input.ndimension() - 1) {
    softmax_lastdim_kernel(output, input);
  } else {
    AT_DISPATCH_FLOATING_TYPES(input.type(), "softmax", [&] {
      host_softmax<scalar_t, false>(output, input, dim);
    });
  }
  return output;
}

Tensor log_softmax_cpu(const Tensor& input_, const int64_t dim_) {
  auto input = input_.contiguous();
  Tensor output = at::native::empty_like(input);
  int64_t dim = maybe_wrap_dim(dim_, input.dim());
  if (input.dim() == 0)
    input = input.view(1);
  AT_CHECK(
      dim >= 0 && dim < input.dim(),
      "dim must be non-negative and less than input dimensions");
  if (input.ndimension() > 0 && dim == input.ndimension() - 1) {
    log_softmax_lastdim_kernel(output, input);
  } else {
    AT_DISPATCH_FLOATING_TYPES(input.type(), "log_softmax", [&] {
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

  if (grad.dim() == 0)
    grad = grad.view(1);
  if (output.dim() == 0)
    output = output.view(1);
  AT_CHECK(
      dim >= 0 && dim < grad.dim(),
      "dim must be non-negative and less than input dimensions");
  if (grad.ndimension() > 0 && dim == grad.ndimension() - 1) {
    softmax_backward_lastdim_kernel(grad_input, grad, output);
  } else {
    AT_DISPATCH_FLOATING_TYPES(grad.type(), "softmax_backward", [&] {
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

  if (grad.dim() == 0)
    grad = grad.view(1);
  if (output.dim() == 0)
    output = output.view(1);
  AT_CHECK(
      dim >= 0 && dim < grad.dim(),
      "dim must be non-negative and less than input dimensions");
  if (grad.ndimension() > 0 && dim == grad.ndimension() - 1) {
    log_softmax_backward_lastdim_kernel(grad_input, grad, output);
  } else {
    AT_DISPATCH_FLOATING_TYPES(grad.type(), "log_softmax_backward", [&] {
      host_softmax_backward<scalar_t, true>(grad_input, grad, output, dim);
    });
  }
  return grad_input;
}
}
}
