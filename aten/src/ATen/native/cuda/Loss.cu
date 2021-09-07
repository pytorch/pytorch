#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/native/TensorIterator.h>
#include <aten/src/ATen/TensorUtils.h>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/native/cuda/Loops.cuh>

constexpr float EPSILON = 1e-12;

namespace {

using namespace at;

void binary_cross_entropy_backward_out_kernel(Tensor& grad_input, const Tensor& grad, const Tensor& input, const Tensor& target) {
  at::TensorIterator iter = TensorIteratorConfig()
      .add_output(grad_input)
      .add_input(grad)
      .add_input(input)
      .add_input(target)
      .build();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.common_dtype(), "binary_cross_entropy_backward_out_cuda", [&]() {
    at::native::gpu_kernel(iter, [] GPU_LAMBDA (
        scalar_t grad_val,
        scalar_t input_val,
        scalar_t target_val
      ) -> scalar_t {
        const scalar_t one = 1;
        const scalar_t epsilon = EPSILON;

        scalar_t grad_input_denominator = max(
          (one - input_val) * input_val,
          epsilon
        );

        return grad_val * (input_val - target_val) / grad_input_denominator;
      }
    );
  });
}

} // namespace

namespace at { namespace native {

Tensor kl_div_backward_cuda(const Tensor& grad, const Tensor& input, const Tensor& target, int64_t reduction, bool log_target) {
  auto grad_input = at::empty_like(input);
  if (!log_target) {
    TensorIterator iter = TensorIteratorConfig()
        .add_output(grad_input)
        .add_input(target)
        .add_input(grad)
        .build();
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "kl_div_backward_cuda", [&]() {
      scalar_t inv = (reduction == at::Reduction::Mean) ? scalar_t(1.0 / input.numel()) : scalar_t(1.0);
      gpu_kernel(iter,
        [inv] GPU_LAMBDA (scalar_t target_val, scalar_t grad_val) {
          return (target_val > 0) ? scalar_t(-target_val * grad_val * inv) : scalar_t(0.0);
        });
    });
  }
  else {
    grad_input = -at::exp(target) * grad;
    if (reduction == at::Reduction::Mean) {
      grad_input /= input.numel();
    }
  }

  return grad_input;
}

Tensor binary_cross_entropy_cuda(const Tensor& input, const Tensor& target, const c10::optional<Tensor>& weight_opt, int64_t reduction) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;

    Tensor loss = at::empty_like(input);
    return at::native::binary_cross_entropy_out_cuda(
        input, target, weight, reduction, loss);
}

Tensor& binary_cross_entropy_out_cuda(const Tensor& input, const Tensor& target, const c10::optional<Tensor>& weight_opt, int64_t reduction, Tensor& loss) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;

  Tensor loss_squeezed = at::squeeze(loss);

  TensorIterator iter = TensorIteratorConfig()
      .add_output(loss_squeezed)
      .add_owned_input(at::squeeze(input))
      .add_owned_input(at::squeeze(target))
      .build();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.common_dtype(), "binary_cross_entropy_out_cuda", [&]() {
    gpu_kernel(iter,
      [] GPU_LAMBDA (scalar_t input_val, scalar_t target_val) -> scalar_t {
        const scalar_t zero = 0;
        const scalar_t one = 1;
        const scalar_t neg_100 = -100;

        CUDA_KERNEL_ASSERT(input_val >= zero && input_val <= one);

        scalar_t log_input_val = std::log(input_val);
        scalar_t log_1_minus_input_val = std::log(one - input_val);

        log_input_val = std::max(log_input_val, neg_100);
        log_1_minus_input_val = std::max(log_1_minus_input_val, neg_100);

        return ((target_val - one) * log_1_minus_input_val) - (target_val * log_input_val);
      }
    );
  });
  if (weight.defined()) {
    loss.mul_(weight);
  }

  if (reduction != at::Reduction::None) {
    Tensor loss_reduced;
    if (reduction == at::Reduction::Mean) {
      loss_reduced = loss.mean();
    } else if (reduction == at::Reduction::Sum) {
      loss_reduced = loss.sum();
    }
    loss.resize_as_(loss_reduced).copy_(loss_reduced);
  }

  return loss;
}

Tensor binary_cross_entropy_backward_cuda(const Tensor& grad, const Tensor& input, const Tensor& target, const c10::optional<Tensor>& weight_opt, int64_t reduction) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;

  Tensor grad_input = at::empty_like(input);
  return at::native::binary_cross_entropy_backward_out_cuda(
      grad, input, target, weight, reduction, grad_input);
}

Tensor& binary_cross_entropy_backward_out_cuda(const Tensor& grad, const Tensor& input, const Tensor& target, const c10::optional<Tensor>& weight_opt, int64_t reduction, Tensor& grad_input) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;

  Tensor grad_expand = grad.expand_as(input);
  binary_cross_entropy_backward_out_kernel(grad_input, grad_expand, input, target);

  if (weight.defined()) {
    grad_input.mul_(weight);
  }
  if (reduction == at::Reduction::Mean) {
    grad_input.div_(input.numel());
  }
  return grad_input;
}

// -----------------------------------
// nll_loss
// -----------------------------------
namespace {

constexpr int NLL_LOSS_THREADS = 32;

#define AT_DISPATCH_NLL_LOSS_INDEX_TYPES(TYPE, NAME, ...)                   \
  [&] {                                                                     \
    at::ScalarType _it = TYPE;                                              \
    RECORD_KERNEL_FUNCTION_DTYPE(NAME, _it)                                 \
    switch (_it) {                                                          \
      AT_PRIVATE_CASE_TYPE_USING_HINT(NAME, at::ScalarType::Byte, uint8_t, index_t, __VA_ARGS__) \
      AT_PRIVATE_CASE_TYPE_USING_HINT(NAME, at::ScalarType::Long, int64_t, index_t, __VA_ARGS__)\
      default:                                                              \
        AT_ERROR(#NAME, " not implemented for '", toString(_it), "'");      \
    }                                                                       \
  }()

template <typename scalar_t, typename index_t>
__global__ void nll_loss_forward_no_reduce_cuda_kernel(
    const int64_t batch_size,
    const PackedTensorAccessor64<scalar_t, 2> input,
    const index_t* const __restrict__ target,
    scalar_t* const __restrict__ output,
    const scalar_t* const __restrict__ weights,
    const int n_classes,
    const int ignore_index) {
  CUDA_KERNEL_LOOP(index, batch_size) {
    int cur_target = static_cast<int>(target[index]);
    if (cur_target == ignore_index) {
      output[index] = scalar_t{0};
      continue;
    }
    CUDA_KERNEL_ASSERT(cur_target >= 0 && cur_target < n_classes);
    if (weights != nullptr) {
      output[index] = -weights[cur_target] * input[index][cur_target];
    } else {
      output[index] = -input[index][cur_target];
    }
  }
}

template <typename scalar_t, typename index_t>
__global__ void nll_loss_forward_reduce_cuda_kernel_1d(
    scalar_t* const __restrict__ output,
    const scalar_t* const __restrict__ input,
    const scalar_t* const __restrict__ weight,
    const index_t* const __restrict__ target,
    const bool size_average,
    const int n_classes,
    const int ignore_index) {
  auto cur_target = static_cast<int>(*target);
  if (cur_target == ignore_index) {
    *output = scalar_t{0};
    return;
  } else {
    CUDA_KERNEL_ASSERT(cur_target >= 0 && cur_target < n_classes);
    if (!size_average && weight != nullptr) {
      *output = - input[cur_target] * weight[cur_target];
    } else {
      *output = - input[cur_target];
    }
  }
}

template <typename scalar_t, typename accscalar_t, typename index_t>
__global__ void nll_loss_forward_reduce_cuda_kernel_2d(
    scalar_t* const __restrict__ output,
    const scalar_t* const __restrict__ input,
    const index_t* const __restrict__ target,
    const scalar_t* const __restrict__ weights,
    scalar_t* const __restrict__ total_weight,
    const bool size_average,
    const int nframe,
    const int ndim,
    const int n_classes,
    const int ignore_index) {

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  __shared__ accscalar_t sh_inputs[NLL_LOSS_THREADS],
      acc_weight[NLL_LOSS_THREADS];

  sh_inputs[threadIdx.x] = accscalar_t{0};
  acc_weight[threadIdx.x] = accscalar_t{0};
  for (int i = threadIdx.x; i < nframe; i += NLL_LOSS_THREADS) {
    int t = target[i];
    if (t == ignore_index) {
      continue;
    }
    CUDA_KERNEL_ASSERT(t >= 0 && t < n_classes);
    if (weights != nullptr) {
      sh_inputs[threadIdx.x] -= input[i * ndim + t] * weights[t];
      if (size_average) {
        acc_weight[threadIdx.x] += weights[t];
      }
    } else {
      sh_inputs[threadIdx.x] -= input[i * ndim + t];
    }
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    accscalar_t output_acc = 0;
    accscalar_t total_weight_acc = 0;
    for (int i = 0; i < NLL_LOSS_THREADS; ++i) {
      output_acc += sh_inputs[i];
      if (weights != nullptr && size_average) {
        total_weight_acc += acc_weight[i];
      }
    }
    if (weights != nullptr && size_average) {
      *total_weight = static_cast<scalar_t>(total_weight_acc);
      *output = static_cast<scalar_t>(output_acc / total_weight_acc);
    } else {
      *output = static_cast<scalar_t>(output_acc);
    }
  }
}

template <typename scalar_t, typename index_t>
__global__ void nll_loss_backward_no_reduce_cuda_kernel(
  const int batch_size,
  const index_t * const __restrict__ target,
  const PackedTensorAccessor64<scalar_t, 1> grad_output,
  PackedTensorAccessor64<scalar_t, 2> grad_input,
  const scalar_t * const __restrict__ weights,
  const int n_classes,
  const int ignore_index) {

  CUDA_KERNEL_LOOP(index, batch_size) {
    const auto cur_target = static_cast<int>(target[index]);
    if (cur_target == ignore_index) {
      continue;
    }
    CUDA_KERNEL_ASSERT(cur_target >= 0 && cur_target < n_classes);
    auto grad = -grad_output[index];
    if (weights != nullptr) {
      grad *= weights[cur_target];
    }
    grad_input[index][cur_target] = grad;
  }
}

template <typename scalar_t, typename index_t>
__global__ void nll_loss_backward_reduce_cuda_kernel_1d(
  scalar_t*  const __restrict__ grad_input,
  const scalar_t*  const __restrict__ grad_output,
  const scalar_t*  const __restrict__ weight,
  const index_t*  const __restrict__ target,
  const bool size_average,
  const int n_classes,
  const int ignore_index) {
  auto cur_target = static_cast<int>(*target);
  if (cur_target == ignore_index) {
    return;
  }
  CUDA_KERNEL_ASSERT(cur_target >= 0 && cur_target < n_classes);
  if (!size_average && weight != nullptr) {
    grad_input[cur_target] = - (*grad_output) * weight[cur_target];
  } else {
    grad_input[cur_target] = - (*grad_output);
  }
}

template <typename scalar_t, typename index_t>
__global__ void nll_loss_backward_reduce_cuda_kernel_2d(
    scalar_t* const __restrict__ grad_input,
    const scalar_t* const __restrict__ grad_output,
    const index_t* const __restrict__ target,
    const scalar_t* const __restrict__ weights,
    const scalar_t* const __restrict__ total_weight,
    const bool size_average,
    const int nframe,
    const int ndim,
    const int n_classes,
    const int ignore_index) {
  scalar_t grad_normalized = -*grad_output;
  if (weights != nullptr && size_average) {
    grad_normalized /= *total_weight;
  }
  for (int i = threadIdx.x; i < nframe; i += NLL_LOSS_THREADS) {
    const int cur_target = target[i];
    if (cur_target == ignore_index) {
      continue;
    }
    CUDA_KERNEL_ASSERT(cur_target >= 0 && cur_target < n_classes);
    if (weights != nullptr) {
      grad_input[i * ndim + cur_target] = grad_normalized * weights[cur_target];
    } else {
      grad_input[i * ndim + cur_target] = grad_normalized;
    }
  }
}
} // namespace

TORCH_IMPL_FUNC(nll_loss_forward_out_cuda)
(const Tensor& self,
 const Tensor& target,
 const OptionalTensorRef weight_opt,
 int64_t reduction,
 int64_t ignore_index,
 const Tensor& output,
 const Tensor& total_weight) {
  const Tensor& weight = weight_opt.getTensorRef();
  const auto weight_ = weight.defined() ? weight.contiguous() : weight;
  const int64_t n_dims = self.dim();
  const int64_t n_classes = self.size(-1);

  if (n_dims == 1) {
    total_weight.resize_({});
    output.resize_({});
    auto self_ = self.contiguous();
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        self_.scalar_type(),
        "nll_loss_forward_reduce_cuda_kernel_1d",
        [&] {
          AT_DISPATCH_NLL_LOSS_INDEX_TYPES(
              target.scalar_type(),
              "nll_loss_forward_reduce_cuda_kernel_1d_index",
              [&] {
                nll_loss_forward_reduce_cuda_kernel_1d<scalar_t, index_t>
                    <<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(
                        output.data_ptr<scalar_t>(),
                        self_.data_ptr<scalar_t>(),
                        weight_.defined() ? weight_.data_ptr<scalar_t>() : nullptr,
                        target.data_ptr<index_t>(),
                        reduction == at::Reduction::Mean,
                        n_classes,
                        static_cast<int>(ignore_index));
                C10_CUDA_KERNEL_LAUNCH_CHECK();
              });
        });
  } else { // n_dims == 2
    const int64_t batch_size = self.sizes().front();
    // This guards from unnecessary operations and launching CUDA kernel with 0 blocks.
    if (batch_size == 0) {
      return;
    }
    const auto target_ = target.contiguous();
    if (reduction == Reduction::None) {
      output.resize_({batch_size});

      AT_DISPATCH_FLOATING_TYPES_AND2(
          at::ScalarType::Half,
          at::ScalarType::BFloat16,
          self.scalar_type(),
          "nll_loss_forward_no_reduce_cuda_kernel",
          [&] {
            AT_DISPATCH_NLL_LOSS_INDEX_TYPES(
                target.scalar_type(),
                "nll_loss_forward_no_reduce_cuda_kernel_index",
                [&] {
                  nll_loss_forward_no_reduce_cuda_kernel<scalar_t, index_t>
                      <<<at::cuda::detail::GET_BLOCKS(batch_size),
                         at::cuda::detail::CUDA_NUM_THREADS,
                         0,
                         at::cuda::getCurrentCUDAStream()>>>(
                          batch_size,
                          self.packed_accessor64<scalar_t, 2>(),
                          target.data_ptr<index_t>(),
                          output.data_ptr<scalar_t>(),
                          weight_.defined() ? weight_.data_ptr<scalar_t>() : nullptr,
                          n_classes,
                          static_cast<int>(ignore_index));
                  C10_CUDA_KERNEL_LAUNCH_CHECK();
                });
          });
    } else {
      total_weight.resize_({});
      output.resize_({});
      auto self_ = self.contiguous();
      AT_DISPATCH_FLOATING_TYPES_AND2(
          at::ScalarType::Half,
          at::ScalarType::BFloat16,
          self.scalar_type(),
          "nll_loss_forward_reduce_cuda_kernel_2d",
          [&] {
            AT_DISPATCH_NLL_LOSS_INDEX_TYPES(
                target.scalar_type(),
                "nll_loss_forward_reduce_cuda_kernel_2d_index",
                [&] {
                  using accscalar_t = at::acc_type<scalar_t, /*is_cuda*/true>;
                  nll_loss_forward_reduce_cuda_kernel_2d<scalar_t, accscalar_t, index_t>
                      <<<1,
                         NLL_LOSS_THREADS,
                         0,
                         at::cuda::getCurrentCUDAStream()>>>(
                          output.data_ptr<scalar_t>(),
                          self_.data_ptr<scalar_t>(),
                          target_.data_ptr<index_t>(),
                          weight_.defined() ? weight_.data_ptr<scalar_t>() : nullptr,
                          total_weight.data_ptr<scalar_t>(),
                          reduction == at::Reduction::Mean,
                          self.size(0),
                          self.size(1),
                          n_classes,
                          static_cast<int>(ignore_index));
                  C10_CUDA_KERNEL_LAUNCH_CHECK();
                });
          });
    }
  }
}

TORCH_IMPL_FUNC(nll_loss_backward_out_cuda)
(const Tensor& grad_output,
 const Tensor& input,
 const Tensor& target,
 OptionalTensorRef weight_opt,
 int64_t reduction,
 int64_t ignore_index,
 const Tensor& total_weight,
 const Tensor& grad_input) {
  grad_input.zero_();
  const Tensor& weight = weight_opt.getTensorRef();
  const auto weight_ = weight.defined() ? weight.contiguous() : weight;
  const int64_t n_dims = input.dim();
  const int64_t n_classes = input.size(-1);

  if (n_dims == 1) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        input.scalar_type(),
        "nll_loss_backward_reduce_cuda_kernel_1d",
        [&] {
          AT_DISPATCH_NLL_LOSS_INDEX_TYPES(
              target.scalar_type(),
              "nll_loss_backward_reduce_cuda_kernel_1d_index",
              [&] {
                nll_loss_backward_reduce_cuda_kernel_1d<scalar_t, index_t>
                    <<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(
                        grad_input.data_ptr<scalar_t>(),
                        grad_output.data_ptr<scalar_t>(),
                        weight_.defined() ? weight_.data_ptr<scalar_t>() : nullptr,
                        target.data_ptr<index_t>(),
                        reduction == at::Reduction::Mean,
                        n_classes,
                        static_cast<int>(ignore_index));
                C10_CUDA_KERNEL_LAUNCH_CHECK();
              });
        });
  } else { // n_dims == 2
    const int64_t batch_size = input.sizes().front();
    // This guards from unnecessary operations and launching CUDA kernel with 0 blocks.
    if (batch_size == 0) {
      return;
    }
    const auto target_ = target.contiguous();
    if (reduction == Reduction::None) {
      AT_DISPATCH_FLOATING_TYPES_AND2(
          at::ScalarType::Half,
          at::ScalarType::BFloat16,
          input.scalar_type(),
          "nll_loss_backward_no_reduce_cuda_kernel",
          [&] {
            AT_DISPATCH_NLL_LOSS_INDEX_TYPES(
                target_.scalar_type(),
                "nll_loss_backward_no_reduce_cuda_kernel_index",
                [&] {
                  nll_loss_backward_no_reduce_cuda_kernel<scalar_t, index_t>
                      <<<at::cuda::detail::GET_BLOCKS(batch_size),
                         at::cuda::detail::CUDA_NUM_THREADS,
                         0,
                         at::cuda::getCurrentCUDAStream()>>>(
                          batch_size,
                          target_.data_ptr<index_t>(),
                          grad_output.packed_accessor64<scalar_t, 1>(),
                          grad_input.packed_accessor64<scalar_t, 2>(),
                          weight_.defined() ? weight_.data_ptr<scalar_t>() : nullptr,
                          n_classes,
                          static_cast<int>(ignore_index));
                  C10_CUDA_KERNEL_LAUNCH_CHECK();
                });
          });
    } else {
      AT_DISPATCH_FLOATING_TYPES_AND2(
          at::ScalarType::Half,
          at::ScalarType::BFloat16,
          input.scalar_type(),
          "nll_loss_backward_reduce_cuda_kernel_2d",
          [&] {
            AT_DISPATCH_NLL_LOSS_INDEX_TYPES(
                target_.scalar_type(),
                "nll_loss_backward_reduce_cuda_kernel_2d_index",
                [&] {
              nll_loss_backward_reduce_cuda_kernel_2d<scalar_t, index_t>
                  <<<1, NLL_LOSS_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
                      grad_input.data_ptr<scalar_t>(),
                      grad_output.data_ptr<scalar_t>(),
                      target_.data_ptr<index_t>(),
                      weight_.defined() ? weight_.data_ptr<scalar_t>() : nullptr,
                      total_weight.data_ptr<scalar_t>(),
                      reduction == at::Reduction::Mean,
                      input.size(0),
                      input.size(1),
                      n_classes,
                      static_cast<int>(ignore_index));
              C10_CUDA_KERNEL_LAUNCH_CHECK();
            });
          });
    }
  }
}
#undef AT_DISPATCH_NLL_LOSS_INDEX_TYPES
}}  // namespace at::native
