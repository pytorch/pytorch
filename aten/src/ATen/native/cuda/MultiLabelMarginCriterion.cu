#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <c10/macros/Macros.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/block_reduce.cuh>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/CUDAFunctions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/zeros_like.h>
#include <ATen/ops/sum_cuda_dispatch.h>
#include <ATen/ops/multilabel_margin_loss.h>
#endif


namespace at {
namespace native {

namespace {
const int MULTILABELMARGIN_THREADS = 128;

void check_shape(const Tensor& input, const Tensor& target) {
  int64_t ndims = input.dim();
  bool valid_inputs = (ndims == 2 && input.size(1) != 0) ||
      (ndims == 1 && input.size(0) != 0) || (ndims == 0);
  TORCH_CHECK(
      valid_inputs,
      "Expected non-empty vector or matrix with optional 0-dim batch size, but got: ",
      input.sizes());

  if (ndims <= 1) {
    int dim = input.dim() == 0 ? 1 : input.size(0);
    TORCH_CHECK(
        valid_inputs && target.dim() <= 1 && target.numel() == dim,
        "inconsistent target size: ",
        target.sizes(),
        " for input of size: ",
        input.sizes());
  } else if (ndims == 2) {
    int nframe = input.size(0);
    int dim = input.size(1);
    TORCH_CHECK(
        valid_inputs && target.dim() == 2 && target.size(0) == nframe &&
            target.size(1) == dim,
        "inconsistent target size: ",
        target.sizes(),
        " for input of size: ",
        input.sizes());
  } else {
    TORCH_CHECK(false, "Expected input of ndims <= 2, but got ndims: ", ndims);
  }
}

template <typename scalar_t, typename accscalar_t>
C10_LAUNCH_BOUNDS_1(MULTILABELMARGIN_THREADS)
__global__ void multilabel_margin_loss_forward_kernel(
    scalar_t* output,
    scalar_t* input,
    int64_t* target,
    scalar_t* is_target,
    int nframe,
    int dim,
    bool size_average) {

  // vectors:
  int k = blockIdx.x;
  scalar_t* input_k = input + k * dim;
  int64_t* target_k = target + k * dim;
  scalar_t* output_k = output + k;
  scalar_t* is_target_k = is_target + k * dim;

  // zero is_target
  for (int d = threadIdx.x; d < dim; d += blockDim.x) {
    is_target_k[d] = static_cast<scalar_t>(0);
  }
  __syncthreads();

  // mark targets in is_target
  if (threadIdx.x == 0) {
    for (int dt = 0; dt < dim; dt++) {
      int target_idx = target_k[dt];
      if (target_idx < 0) {
        break;
      }
      is_target_k[target_idx] = static_cast<scalar_t>(1);
    }
  }
  __syncthreads();

  // iterate over targets
  accscalar_t sum = 0;
  for (int dt = 0; dt < dim; dt++) {
    // next target:
    int target_idx = target_k[dt];
    if (target_idx < 0) {
      break;
    }

    // current value for target
    scalar_t input_target_k = input_k[target_idx];

    // compare to all inputs (multithreaded):
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
      // contribute to loss only if not a target
      if (!static_cast<int>(is_target_k[d])) {
        scalar_t z = 1 - input_target_k + input_k[d];
        if (z > 0) {
          sum += z;
        }
      }
    }
  }

  // Temporary sums (for mapreduce)
  __shared__ accscalar_t smem[MULTILABELMARGIN_THREADS];
  accscalar_t total_sum = cuda_utils::BlockReduceSum(sum, smem);
  if (threadIdx.x == 0) {
    if (size_average) {
      *output_k = static_cast<scalar_t>((total_sum / dim) / nframe);
    } else {
      *output_k = static_cast<scalar_t>(total_sum / dim);
    }
  }
}

template <typename scalar_t, typename accscalar_t>
C10_LAUNCH_BOUNDS_1(MULTILABELMARGIN_THREADS)
__global__ void multilabel_margin_loss_backward_kernel(
    scalar_t* grad_input,
    scalar_t* grad_output,
    scalar_t* input,
    int64_t* target,
    scalar_t* is_target,
    int nframe,
    int dim,
    bool size_average,
    bool reduce) {

  int k = blockIdx.x;
  scalar_t* input_k = input + k * dim;
  scalar_t* grad_input_k = grad_input + k * dim;
  int64_t* target_k = target + k * dim;
  scalar_t* is_target_k = is_target + k * dim;

  scalar_t* grad_output_k = grad_output;
  if (!reduce) {
    grad_output_k += k;
  }

  // gain:
  scalar_t g = static_cast<scalar_t>(
      size_average && reduce ? 1. / static_cast<accscalar_t>(nframe * dim)
                             : 1. / static_cast<accscalar_t>(dim));

  // zero gradients:
  for (int d = threadIdx.x; d < dim; d += blockDim.x) {
    grad_input_k[d] = static_cast<scalar_t>(0);
  }
  __syncthreads();

  // iterate over targets
  for (int dt = 0; dt < dim; dt++) {
    // next target:
    int target_idx = static_cast<int>(target_k[dt]);
    if (target_idx < 0) {
      break;
    }

    // current value for target
    scalar_t input_target_k = input_k[target_idx];

    // compare to all inputs (multithreaded):
    accscalar_t sum = 0;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
      // contribute to loss only if not a target
      if (!static_cast<int>(is_target_k[d])) {
        scalar_t z = 1 - input_target_k + input_k[d];
        if (z > 0) {
          sum -= g;
          grad_input_k[d] += g;
        }
      }
    }
    __syncthreads();

    // Temporary sums (for mapreduce)
    __shared__ accscalar_t smem[MULTILABELMARGIN_THREADS];
    accscalar_t total_sum = cuda_utils::BlockReduceSum(sum, smem);
    if (threadIdx.x == 0) {
      grad_input_k[target_idx] += static_cast<scalar_t>(total_sum);
    }
  }

  for (int d = threadIdx.x; d < dim; d += blockDim.x) {
    grad_input_k[d] *= *grad_output_k;
  }
}

void multilabel_margin_loss_forward_out_cuda_template(
    const Tensor& input,
    const Tensor& target,
    int64_t reduction,
    Tensor& output,
    Tensor& is_target) {
  check_shape(input, target);
  if (input.numel() == 0) {
    return;
  }

  auto input_ = input.contiguous();
  auto target_ = target.contiguous();
  auto is_target_ = is_target.contiguous();
  is_target_.resize_as_(target);

  if (input.dim() <= 1) {
    int dim = input.dim() == 0 ? 1 : input.size(0);
    output.resize_({});

    dim3 blocks(1);
    dim3 threads(MULTILABELMARGIN_THREADS);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        input.scalar_type(),
        "multilabel_margin_loss_forward_kernel",
        [&] {
          using accscalar_t = at::acc_type<scalar_t, true>;
          multilabel_margin_loss_forward_kernel<scalar_t, accscalar_t>
              <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                  output.data_ptr<scalar_t>(),
                  input_.data_ptr<scalar_t>(),
                  target_.data_ptr<int64_t>(),
                  is_target_.data_ptr<scalar_t>(),
                  1,
                  dim,
                  reduction == at::Reduction::Mean);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
  } else if (input.dim() == 2) {
    int nframe = input.size(0);
    int dim = input.size(1);
    dim3 blocks(input.size(0));
    dim3 threads(MULTILABELMARGIN_THREADS);

    if (reduction != at::Reduction::None) {
      auto output_tmp = at::empty({input_.size(0)}, input_.options());
      output.resize_({});
      AT_DISPATCH_FLOATING_TYPES_AND2(
          at::ScalarType::Half,
          at::ScalarType::BFloat16,
          input.scalar_type(),
          "multilabel_margin_loss_forward_kernel",
          [&] {
            using accscalar_t = at::acc_type<scalar_t, true>;
            multilabel_margin_loss_forward_kernel<scalar_t, accscalar_t>
                <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                    output_tmp.data_ptr<scalar_t>(),
                    input_.data_ptr<scalar_t>(),
                    target_.data_ptr<int64_t>(),
                    is_target_.data_ptr<scalar_t>(),
                    nframe,
                    dim,
                    reduction == at::Reduction::Mean);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
          });
      at::cuda::sum_out(
          output,
          output_tmp,
          at::IntArrayRef(std::vector<int64_t>{}),
          false,
          output.scalar_type());
    } else {
      output.resize_({input.size(0)});
      AT_DISPATCH_FLOATING_TYPES_AND2(
          at::ScalarType::Half,
          at::ScalarType::BFloat16,
          input.scalar_type(),
          "multilabel_margin_loss_forward_kernel",
          [&] {
            using accscalar_t = at::acc_type<scalar_t, true>;
            multilabel_margin_loss_forward_kernel<scalar_t, accscalar_t>
                <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                    output.data_ptr<scalar_t>(),
                    input_.data_ptr<scalar_t>(),
                    target_.data_ptr<int64_t>(),
                    is_target_.data_ptr<scalar_t>(),
                    nframe,
                    dim,
                    false);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
          });
    }

  } else {
    TORCH_CHECK(
        false,
        "Expected 2D input with optional zero batch dim, or 1D input with non-zero dims, but got sizes: ",
        input.sizes());
  }
}

void multilabel_margin_loss_backward_cuda_out_template(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& target,
    int64_t reduction,
    const Tensor& is_target,
    Tensor& grad_input) {
  check_shape(input, target);
  auto input_ = input.contiguous();
  if (input_.numel() == 0) {
    return;
  }

  grad_input.resize_as_(input_);
  auto target_ = target.contiguous();
  auto is_target_ = is_target.contiguous();
  auto grad_output_ = grad_output.contiguous();

  if (grad_input.dim() <= 1) {
    int dim = grad_input.dim() == 0 ? 1 : grad_input.size(0);
    int target_size = target_.dim() == 0 ? 1 : target_.size(0);
    TORCH_CHECK(
        (target_.numel() != 0) && (target_.dim() <= 1) && (target_size == dim),
        "inconsistent target size");
    TORCH_CHECK(
        target_.sizes() == is_target_.sizes(), "inconsistent is_target size");
    dim3 blocks(1);
    dim3 threads(MULTILABELMARGIN_THREADS);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        input.scalar_type(),
        "multilabel_margin_loss_backward_kernel",
        [&] {
          using accscalar_t = at::acc_type<scalar_t, true>;
          multilabel_margin_loss_backward_kernel<scalar_t, accscalar_t>
              <<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
                  grad_input.data_ptr<scalar_t>(),
                  grad_output_.data_ptr<scalar_t>(),
                  input_.data_ptr<scalar_t>(),
                  target_.data_ptr<int64_t>(),
                  is_target_.data_ptr<scalar_t>(),
                  1,
                  dim,
                  reduction == at::Reduction::Mean,
                  reduction != at::Reduction::None);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
  } else if (grad_input.dim() == 2) {
    int nframe = grad_input.size(0);
    int dim = grad_input.size(1);
    TORCH_CHECK(
        (input_.size(1) != 0) && (target_.dim() == 2) &&
            (target_.size(0) == nframe) && (target_.size(1) == dim),
        "inconsistent target size");
    TORCH_CHECK(target_.sizes() == is_target_.sizes(), "inconsistent is_target size");
    dim3 blocks(grad_input.size(0));
    dim3 threads(MULTILABELMARGIN_THREADS);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        input.scalar_type(),
        "multilabel_margin_loss_backward_kernel",
        [&] {
          using accscalar_t = at::acc_type<scalar_t, true>;
          multilabel_margin_loss_backward_kernel<scalar_t, accscalar_t>
              <<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream()>>>(
                  grad_input.data_ptr<scalar_t>(),
                  grad_output_.data_ptr<scalar_t>(),
                  input_.data_ptr<scalar_t>(),
                  target_.data_ptr<int64_t>(),
                  is_target_.data_ptr<scalar_t>(),
                  grad_input.size(0),
                  grad_input.size(1),
                  reduction == at::Reduction::Mean,
                  reduction != at::Reduction::None);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
  } else {
    TORCH_CHECK(
        false,
        "Expected 2D input with optional zero batch dim, or 1D input with non-zero dims, but got sizes: ",
        grad_input.sizes());
  }
}

} // namespace

std::tuple<Tensor&, Tensor&> multilabel_margin_loss_forward_out_cuda(
    const Tensor& self,
    const Tensor& target,
    int64_t reduction,
    Tensor& output,
    Tensor& is_target) {
  multilabel_margin_loss_forward_out_cuda_template(
      self, target, reduction, output, is_target);
  return std::tuple<Tensor&, Tensor&>(output, is_target);
}

std::tuple<Tensor, Tensor> multilabel_margin_loss_forward_cuda(
    const Tensor& self,
    const Tensor& target,
    int64_t reduction) {
  auto output = at::empty({0}, self.options());
  auto is_target = at::empty({0}, self.options());
  multilabel_margin_loss_forward_out_cuda_template(
      self, target, reduction, output, is_target);
  return std::make_tuple(output, is_target);
}

Tensor& multilabel_margin_loss_backward_cuda_out(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    int64_t reduction,
    const Tensor& is_target,
    Tensor& grad_input) {
  multilabel_margin_loss_backward_cuda_out_template(
      grad_output, self, target, reduction, is_target, grad_input);
  return grad_input;
}

Tensor multilabel_margin_loss_backward_cuda(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    int64_t reduction,
    const Tensor& is_target) {
  auto grad_input = at::zeros_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  multilabel_margin_loss_backward_cuda_out_template(
      grad_output, self, target, reduction, is_target, grad_input);
  return grad_input;
}

} // namespace native
} // namespace at
