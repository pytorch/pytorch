#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/native/Resize.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDADeviceAssertion.h>
#include <c10/cuda/CUDAException.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/sum.h>
#include <ATen/ops/multi_margin_loss_native.h>
#include <ATen/ops/multi_margin_loss_backward_native.h>
#endif

namespace at::native {
namespace {
constexpr int MULTIMARGIN_THREADS = 128;

template <int P, typename scalar_t>
__global__ void MultiMarginLoss_forward_kernel(
    scalar_t *output, const scalar_t *input, const int64_t *target, const scalar_t *weights,
    int nframe, int dim, bool sizeAverage, scalar_t margin, TORCH_DSA_KERNEL_ARGS) {
  using acc_t = at::acc_type<scalar_t, true>;
  __shared__ acc_t buffer[MULTIMARGIN_THREADS];
  int k = blockIdx.x;
  const scalar_t *input_k = input + k*dim;
  scalar_t *output_k = output + k;
  int target_k = static_cast<int>(target[k]);
  CUDA_KERNEL_ASSERT2(target_k >= 0 && target_k < dim && "target index is out of bounds");
  scalar_t input_target_k = input_k[target_k];

  int i_start = threadIdx.x;
  int i_end = dim;
  int i_step = blockDim.x;

  buffer[threadIdx.x] = 0;
  for (int i = i_start; i < i_end; i += i_step) {
    scalar_t z = margin - input_target_k + input_k[i];
    if (i == target_k) {
      continue;
    }

    if (z > 0) {
      scalar_t h = (P==1) ? z : z*z;
      if (weights) {
        h *= weights[target_k];
      }
      buffer[threadIdx.x] += h;
    }
  }
  __syncthreads();

  // reduce
  if (threadIdx.x == 0) {
    acc_t sum = 0;
    for (int i=0; i < blockDim.x; i++)
      sum += buffer[i];

    const int denom = sizeAverage ? nframe * dim : dim;
    *output_k = static_cast<scalar_t>(sum / denom);
  }
}

template <int P, typename scalar_t>
__global__ void MultiMarginLoss_backward_kernel(
    scalar_t *gradInput, const scalar_t *gradOutput, const scalar_t *input, const int64_t *target,
    const scalar_t *weights, int nframe, int dim, bool sizeAverage, scalar_t margin,
    bool reduce) {
  using acc_t = at::acc_type<scalar_t, true>;
  __shared__ acc_t buffer[MULTIMARGIN_THREADS];
  int k = blockIdx.x;
  const scalar_t *input_k = input + k*dim;
  scalar_t *gradInput_k = gradInput + k*dim;
  int target_k = static_cast<int>(target[k]);
  scalar_t input_target_k = input_k[target_k];

  const scalar_t *gradOutput_k = gradOutput;
  if (!reduce) {
    gradOutput_k += k;
  }

  const int denom = sizeAverage && reduce ? nframe * dim : dim;
  const acc_t g = acc_t(1) / static_cast<acc_t>(denom);

  int i_start = threadIdx.x;
  int i_end = dim;
  int i_step = blockDim.x;

  buffer[threadIdx.x] = 0;
  for (int i=i_start; i<i_end; i+=i_step) {
    scalar_t z = margin - input_target_k + input_k[i];
    if (i == target_k) {
      continue;
    }

    if (z > 0) {
      acc_t h = (P == 1) ? g : 2*g*z;
      if (weights) {
        h *= weights[target_k];
      }

      buffer[threadIdx.x] -= static_cast<scalar_t>(h);
      gradInput_k[i] = static_cast<scalar_t>(h);
    } else {
      gradInput_k[i] = static_cast<scalar_t>(0);
    }
  }

  __syncthreads();

  // reduce
  if (threadIdx.x == 0) {
    acc_t gradInput_target_k = 0;
    for (int i=0; i<blockDim.x; i++) {
      gradInput_target_k += buffer[i];
    }
    gradInput_k[target_k] = static_cast<scalar_t>(gradInput_target_k);
  }

  for (int i=i_start; i<i_end; i+= i_step) {
    gradInput_k[i] *= * gradOutput_k;
  }
}

void multi_margin_loss_shape_check(
    int64_t& nframe,
    int64_t& dim,
    const int64_t& ndims,
    const Tensor& input,
    const Tensor& target,
    const c10::optional<Tensor>& weight) {
    TORCH_CHECK(
        (ndims == 2 && input.size(1) != 0) || (ndims == 1 && input.size(0) != 0) || ndims == 0,
        "Expected non-empty vector or matrix with optional 0-dim batch size, but got: ",
        input.sizes());

    if (ndims <= 1) {
      nframe = 1;
      dim = ndims == 0 ? 1 : input.size(0);
    } else {
      nframe = input.size(0);
      dim = input.size(1);
    }

    TORCH_CHECK(
        target.dim() <= 1 && target.numel() == nframe,
        "inconsistent target size, expected ", nframe, " but got ",
        target.sizes());
    if (weight && weight->defined()) {
      TORCH_CHECK(
          weight->dim() <= 1 && weight->numel() == dim,
          "inconsistent weight size, expected ", dim, " but got ",
          weight->sizes());
    }
}

}  // namespace (anonymous)

Tensor& multi_margin_loss_cuda_out(
    const Tensor &input_, const Tensor &target_, const Scalar &p_, const Scalar &margin_,
    const c10::optional<Tensor> &weights_, int64_t reduction, Tensor& out_) {
  auto p = p_.toLong();
  int64_t nframe, dim;
  const auto ndims = input_.dim();

  TORCH_CHECK(p == 1 || p == 2, "multi_margin_loss: Invalid p, expected 1 or 2 but got ", p);

  multi_margin_loss_shape_check(nframe, dim, ndims, input_, target_, weights_);

  // produce a scalar output for 1d input
  if (reduction == Reduction::None && target_.dim() > 0) {
    resize_output(out_, {nframe});
  } else {
    resize_output(out_, {});
  }
  if (input_.numel() == 0) {
    return out_;
  }

  auto input = input_.contiguous();
  auto target = target_.contiguous();
  Tensor weights;
  if (weights_ && weights_->defined()) {
    weights = weights_->contiguous();
  }
  auto out = (out_.is_contiguous() ? out_ :
              at::empty(out_.sizes(), input.options()));

  const auto stream = c10::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, input.scalar_type(), "multi_margin_loss_cuda", [&] {
    const scalar_t margin = margin_.to<scalar_t>();
    if (input.dim() <= 1) {
      TORCH_CHECK(target.dim() <= 1 && target.numel() == nframe, "inconsistent target size");
      dim3 blocks(1);
      dim3 threads(MULTIMARGIN_THREADS);
      if (p == 1) {
        TORCH_DSA_KERNEL_LAUNCH(
            (MultiMarginLoss_forward_kernel<1>),
            blocks,
            threads,
            0,
            stream,
            out.mutable_data_ptr<scalar_t>(),
            input.const_data_ptr<scalar_t>(),
            target.const_data_ptr<int64_t>(),
            weights.defined() ? weights.const_data_ptr<scalar_t>() : nullptr,
            1,
            input.dim() < 1 ? input.numel() : input.sizes()[0],
            reduction == at::Reduction::Mean,
            margin);
      } else if (p == 2) {
        TORCH_DSA_KERNEL_LAUNCH((MultiMarginLoss_forward_kernel<2>),
            blocks,
            threads,
            0,
            stream,
            out.mutable_data_ptr<scalar_t>(),
            input.const_data_ptr<scalar_t>(),
            target.const_data_ptr<int64_t>(),
            weights.defined() ? weights.const_data_ptr<scalar_t>() : nullptr,
            1,
            input.dim() < 1 ? input.numel() : input.sizes()[0],
            reduction == at::Reduction::Mean,
            margin);
      }
    } else {
      auto in_sizes = input.sizes();
      TORCH_INTERNAL_ASSERT(in_sizes.size() == 2);
      // allow zero-dim target for 2D input.
      TORCH_CHECK(in_sizes[1] != 0 && target.dim() <= 1 && target.numel() == nframe,
                "inconsistent target size");
      dim3 blocks(nframe);
      dim3 threads(MULTIMARGIN_THREADS);

      if (reduction == at::Reduction::None) {
        if (p == 1) {
          TORCH_DSA_KERNEL_LAUNCH(MultiMarginLoss_forward_kernel<1>,
              blocks,
              threads,
              0,
              stream,
              out.mutable_data_ptr<scalar_t>(),
              input.const_data_ptr<scalar_t>(),
              target.const_data_ptr<int64_t>(),
              weights.defined() ? weights.const_data_ptr<scalar_t>() : nullptr,
              nframe, in_sizes[1],
              false,
              margin);
        } else if (p == 2) {
          TORCH_DSA_KERNEL_LAUNCH(MultiMarginLoss_forward_kernel<2>,
              blocks,
              threads,
              0,
              stream,
              out.mutable_data_ptr<scalar_t>(),
              input.const_data_ptr<scalar_t>(),
              target.const_data_ptr<int64_t>(),
              weights.defined() ? weights.const_data_ptr<scalar_t>() : nullptr,
              nframe, in_sizes[1],
              false,
              margin);
        }
      } else {
        auto tmp_output = at::empty({nframe}, input.options());
        if (p == 1) {
          TORCH_DSA_KERNEL_LAUNCH(MultiMarginLoss_forward_kernel<1>,
              blocks,
              threads,
              0,
              stream,
              tmp_output.mutable_data_ptr<scalar_t>(),
              input.const_data_ptr<scalar_t>(),
              target.const_data_ptr<int64_t>(),
              weights.defined() ? weights.const_data_ptr<scalar_t>() : nullptr,
              nframe, in_sizes[1],
              reduction == Reduction::Mean,
              margin);
        } else if (p == 2) {
          TORCH_DSA_KERNEL_LAUNCH(MultiMarginLoss_forward_kernel<2>,
              blocks,
              threads,
              0,
              stream,
              tmp_output.mutable_data_ptr<scalar_t>(),
              input.const_data_ptr<scalar_t>(),
              target.const_data_ptr<int64_t>(),
              weights.defined() ? weights.const_data_ptr<scalar_t>() : nullptr,
              nframe, in_sizes[1],
              reduction == Reduction::Mean,
              margin);
        }
        at::sum_out(out, tmp_output, IntArrayRef{});
      }
    }
  });

  if (!out.is_alias_of(out_)) {
    out_.copy_(out);
  }
  return out_;
}

Tensor multi_margin_loss_cuda(
    const Tensor &input, const Tensor &target, const Scalar &p, const Scalar &margin,
    const c10::optional<Tensor> &weights, int64_t reduction) {
  auto out = at::empty({0}, input.options());
  multi_margin_loss_cuda_out(input, target, p, margin, weights, reduction, out);
  return out;
}

Tensor& multi_margin_loss_cuda_backward_out(
    const Tensor &grad_output_,const Tensor &input_, const Tensor &target_,
    const Scalar &p_, const Scalar &margin_, const c10::optional<Tensor> &weights_,
    int64_t reduction, Tensor &grad_input_) {
  auto p = p_.toLong();
  int64_t nframe, dim;
  const auto ndims = input_.dim();

  TORCH_CHECK(p == 1 || p == 2,
              "multi_margin_loss_backward: Invalid p, expected 1 or 2 but got ", p);

  multi_margin_loss_shape_check(nframe, dim, ndims, input_, target_, weights_);
  resize_output(grad_input_, input_.sizes());

  if (input_.numel() == 0) {
    return grad_input_;
  }

  auto input = input_.contiguous();
  auto grad_input = (grad_input_.is_contiguous() ? grad_input_ :
                     at::empty(grad_input_.sizes(), input.options()));
  auto grad_output = grad_output_.contiguous();
  auto target = target_.contiguous();
  Tensor weights;
  if (weights_ && weights_->defined()) {
    weights = weights_->contiguous();
  }

  const auto stream = c10::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, input.scalar_type(),
                                  "multi_margin_loss_backward_cuda", [&] {
    const scalar_t margin = margin_.to<scalar_t>();

    if (input.dim() <= 1) {
      dim3 blocks(1);
      dim3 threads(MULTIMARGIN_THREADS);

      if (p == 1) {
        MultiMarginLoss_backward_kernel<1> <<<blocks, threads, 0, stream>>>(
            grad_input.mutable_data_ptr<scalar_t>(),
            grad_output.const_data_ptr<scalar_t>(),
            input.const_data_ptr<scalar_t>(),
            target.const_data_ptr<int64_t>(),
            weights.defined() ? weights.const_data_ptr<scalar_t>() : nullptr,
            1,
            input.dim() == 0 ? 1 : input.sizes()[0],
            reduction == at::Reduction::Mean,
            margin,
            reduction != at::Reduction::None);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      } else if (p == 2) {
        MultiMarginLoss_backward_kernel<2> <<<blocks, threads, 0, stream>>>(
            grad_input.mutable_data_ptr<scalar_t>(),
            grad_output.const_data_ptr<scalar_t>(),
            input.const_data_ptr<scalar_t>(),
            target.const_data_ptr<int64_t>(),
            weights.defined() ? weights.const_data_ptr<scalar_t>() : nullptr,
            1,
            input.dim() == 0 ? 1 : input.sizes()[0],
            reduction == at::Reduction::Mean,
            margin,
            reduction != at::Reduction::None);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }
    } else {
      auto in_sizes = input.sizes();
      TORCH_INTERNAL_ASSERT(in_sizes.size() == 2);
      TORCH_CHECK((in_sizes[1] != 0) && (target.dim() <= 1) && (target.numel() == nframe),
                  "inconsistent target size");
      dim3 blocks(in_sizes[0]);
      dim3 threads(MULTIMARGIN_THREADS);

      if (p == 1) {
        MultiMarginLoss_backward_kernel<1> <<<blocks, threads, 0, stream>>>(
            grad_input.mutable_data_ptr<scalar_t>(),
            grad_output.const_data_ptr<scalar_t>(),
            input.const_data_ptr<scalar_t>(),
            target.const_data_ptr<int64_t>(),
            weights.defined() ? weights.const_data_ptr<scalar_t>() : nullptr,
            nframe, in_sizes[1],
            reduction == at::Reduction::Mean,
            margin,
            reduction != at::Reduction::None);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      } else if (p == 2) {
        MultiMarginLoss_backward_kernel<2> <<<blocks, threads, 0, stream>>>(
            grad_input.mutable_data_ptr<scalar_t>(),
            grad_output.const_data_ptr<scalar_t>(),
            input.const_data_ptr<scalar_t>(),
            target.const_data_ptr<int64_t>(),
            weights.defined() ? weights.const_data_ptr<scalar_t>() : nullptr,
            nframe, in_sizes[1],
            reduction == at::Reduction::Mean,
            margin,
            reduction != at::Reduction::None);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }
    }
  });

  if (!grad_input.is_alias_of(grad_input_)) {
    grad_input_.copy_(grad_input);
  }
  return grad_input_;
}

Tensor multi_margin_loss_cuda_backward(
    const Tensor &grad_output, const Tensor &input, const Tensor &target,
    const Scalar &p, const Scalar &margin, const c10::optional<Tensor> &weights,
    int64_t reduction) {
  auto grad_input = at::empty({0}, input.options());
  multi_margin_loss_cuda_backward_out(
      grad_output, input, target, p, margin, weights, reduction, grad_input);
  return grad_input;
}

}  // namespace at::native
