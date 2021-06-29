#include <ATen/ATen.h>
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

const int NLL_LOSS_THREADS = 32;

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
    int64_t batch_size,
    PackedTensorAccessor64<scalar_t, 2> input,
    index_t* target,
    scalar_t* output,
    scalar_t* weights,
    int n_classes,
    int ignore_index) {
  CUDA_KERNEL_LOOP(index, batch_size) {
    int cur_target = target[index];
    if (cur_target == ignore_index) {
      output[index] = static_cast<scalar_t>(0);
      continue;
    }
    CUDA_KERNEL_ASSERT(cur_target >= 0 && cur_target < n_classes);
    auto cur_weight =
        weights != nullptr ? weights[cur_target] : static_cast<scalar_t>(1);
    output[index] = -cur_weight * input[index][cur_target];
  }
}

template <typename scalar_t, typename index_t>
__global__ void nll_loss_forward_reduce_cuda_kernel_1d(
    scalar_t* output,
    scalar_t* total_weight,
    scalar_t* input,
    index_t* target,
    scalar_t* weights,
    bool size_average,
    int n_classes,
    int64_t ignore_index) {
  CUDA_KERNEL_ASSERT(threadIdx.x == 0 && threadIdx.y == 0 & threadIdx.z == 0);

  int t = static_cast<int>(*target);
  if (t != static_cast<int>(ignore_index)) {
    CUDA_KERNEL_ASSERT(t >= 0 && t < n_classes);
    scalar_t cur_weight =
        weights != nullptr ? weights[t] : static_cast<scalar_t>(1);
    *output = -cur_weight * input[t];
    *total_weight = cur_weight;
    if (size_average && *total_weight > 0) {
      *output /= *total_weight;
    }
  }
}

template <typename scalar_t, typename accscalar_t, typename index_t>
__global__ void nll_loss_forward_reduce_cuda_kernel_2d(
    scalar_t* output,
    scalar_t* total_weight,
    scalar_t* input,
    index_t* target,
    scalar_t* weights,
    bool size_average,
    int nframe,
    int ndim,
    int n_classes,
    int64_t ignore_index) {
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  __shared__ accscalar_t sh_inputs[NLL_LOSS_THREADS],
      acc_weight[NLL_LOSS_THREADS];

  sh_inputs[threadIdx.x] = static_cast<accscalar_t>(0);
  acc_weight[threadIdx.x] = static_cast<accscalar_t>(0);
  for (int i = threadIdx.x; i < nframe; i += NLL_LOSS_THREADS) {
    int t = target[i];
    if (t != static_cast<int>(ignore_index)) {
      CUDA_KERNEL_ASSERT(t >= 0 && t < n_classes);
      scalar_t cur_weight =
          weights != nullptr ? weights[t] : static_cast<scalar_t>(1);
      sh_inputs[threadIdx.x] -= input[i * ndim + t] * cur_weight;
      acc_weight[threadIdx.x] += cur_weight;
    }
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    accscalar_t output_acc = 0;
    accscalar_t total_weight_acc = 0;
    for (int i = 0; i < NLL_LOSS_THREADS; ++i) {
      output_acc += sh_inputs[i];
      total_weight_acc += acc_weight[i];
    }
    *total_weight = static_cast<scalar_t>(total_weight_acc);
    if (size_average && nframe == 0) {
      // Mean reduction on empty tensors produces NaN
      *output = std::numeric_limits<double>::quiet_NaN();
    } else if (size_average && total_weight_acc != 0) {
      *output = static_cast<scalar_t>(output_acc / total_weight_acc);
    } else {
      *output = static_cast<scalar_t>(output_acc);
    }
  }
}

void nll_loss_forward_out_cuda_template(
    Tensor& output,
    Tensor& total_weight,
    const Tensor& input,
    const Tensor& target,
    const c10::optional<Tensor>& weight_opt,
    int64_t reduction,
    int64_t ignore_index) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weight_maybe_owned =
      at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;

  TORCH_CHECK(
      target.dim() == 1,
      "1D target tensor expected, multi-target not supported");

  int64_t n_classes = input.size(-1);
  int64_t n_dims = input.dim();

  TORCH_CHECK(n_dims > 0 && n_dims <= 2, "input tensor should be 1D or 2D");
  int64_t batch_size = n_dims == 1 ? 1 : input.size(0);
  int64_t num_targets = target.size(0);
  TORCH_CHECK(
      batch_size == num_targets,
      "size mismatch (got input: ",
      input.sizes(),
      ", target: ",
      target.sizes(),
      ")")

  TORCH_CHECK(
      !weight.defined() || (weight.dim() == 1 && weight.numel() == n_classes),
      "weight tensor should be defined either for all ",
      n_classes,
      " classes or no classes"
      " but got weight tensor of shape: ",
      weight.sizes());

  auto weight_ = weight.defined() ? weight.contiguous() : weight;

  if (reduction == Reduction::None & n_dims == 2) {
    output.resize_({batch_size});
    if (batch_size == 0) {
      // This guards from unnecessary operations and launching CUDA kernel with
      // 0 blocks.
      return;
    }

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        input.scalar_type(),
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
                        input.packed_accessor64<scalar_t, 2>(),
                        target.data_ptr<index_t>(),
                        output.data_ptr<scalar_t>(),
                        weight_.defined() ? weight_.data_ptr<scalar_t>()
                                          : nullptr,
                        n_classes,
                        ignore_index);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
              });
        });
    return;
  }

  output.resize_({});
  total_weight.resize_({});

  auto input_ = input.contiguous();
  auto target_ = target.contiguous();

  if (n_dims == 1) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        input.scalar_type(),
        "nll_loss_forward_reduce_cuda_kernel_1d",
        [&] {
          AT_DISPATCH_NLL_LOSS_INDEX_TYPES(
              target.scalar_type(),
              "nll_loss_forward_reduce_cuda_kernel_1d_index",
              [&] {
                nll_loss_forward_reduce_cuda_kernel_1d<scalar_t, index_t>
                    <<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(
                        output.data_ptr<scalar_t>(),
                        total_weight.data_ptr<scalar_t>(),
                        input_.data_ptr<scalar_t>(),
                        target_.data_ptr<index_t>(),
                        weight_.defined() ? weight_.data_ptr<scalar_t>()
                                          : nullptr,
                        reduction == at::Reduction::Mean,
                        n_classes,
                        ignore_index);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
              });
        });
  } else if (n_dims == 2) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        input.scalar_type(),
        "nll_loss_forward_reduce_cuda_kernel_2d",
        [&] {
          AT_DISPATCH_NLL_LOSS_INDEX_TYPES(
              target.scalar_type(),
              "nll_loss_forward_reduce_cuda_kernel_2d_index",
              [&] {
                nll_loss_forward_reduce_cuda_kernel_2d<scalar_t, float, index_t>
                    <<<1,
                       NLL_LOSS_THREADS,
                       0,
                       at::cuda::getCurrentCUDAStream()>>>(
                        output.data_ptr<scalar_t>(),
                        total_weight.data_ptr<scalar_t>(),
                        input_.data_ptr<scalar_t>(),
                        target_.data_ptr<index_t>(),
                        weight_.defined() ? weight_.data_ptr<scalar_t>()
                                          : nullptr,
                        reduction == at::Reduction::Mean,
                        input.size(0),
                        input.size(1),
                        n_classes,
                        ignore_index);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
              });
        });
  }
}

template <typename scalar_t, typename index_t>
__global__ void nll_loss_backward_no_reduce_cuda_kernel(
  int batch_size,
  index_t *target,
  PackedTensorAccessor64<scalar_t, 1> grad_output,
  PackedTensorAccessor64<scalar_t, 2> grad_input,
  scalar_t *weights,
  int n_classes,
  int ignore_index) {

  CUDA_KERNEL_LOOP(index, batch_size) {
    int cur_target = target[index];
    if (cur_target == ignore_index) {
      continue;
    }
    CUDA_KERNEL_ASSERT(cur_target >= 0 && cur_target < n_classes);
    scalar_t weight = weights != nullptr ? weights[cur_target] : static_cast<scalar_t>(1);
    grad_input[index][cur_target] = -weight * grad_output[index];
  }
};

template <typename scalar_t, typename index_t>
__global__ void nll_loss_backward_reduce_cuda_kernel_1d(
  scalar_t *grad_input,
  scalar_t *grad_output,
  scalar_t *weights,
  index_t *target,
  scalar_t *total_weight,
  bool size_average,
  int n_classes,
  int64_t ignore_index
) {
  if (*total_weight <= 0) {
    return;
  }
  scalar_t norm = size_average ? (static_cast<scalar_t>(1) / *total_weight) : static_cast<scalar_t>(1);
  int t = static_cast<int>(*target);
  if (t != static_cast<int>(ignore_index)) {
    CUDA_KERNEL_ASSERT(t >= 0 && t < n_classes);
    grad_input[t] = -(weights != nullptr ? weights[t] : static_cast<scalar_t>(1)) * norm * grad_output[0];
  }
};

template <typename scalar_t, typename index_t>
__global__ void nll_loss_backward_reduce_cuda_kernel_2d(
    scalar_t* grad_input,
    scalar_t* grad_output,
    index_t* target,
    scalar_t* weights,
    scalar_t* total_weight,
    bool size_average,
    int nframe,
    int ndim,
    int n_classes,
    int64_t ignore_index) {
  if (*total_weight <= 0) {
    return;
  }
  scalar_t norm = size_average ? (static_cast<scalar_t>(1) / *total_weight) : static_cast<scalar_t>(1);

  for (int i = threadIdx.x; i < nframe; i += NLL_LOSS_THREADS) {
    int t = target[i];
    if (t != static_cast<int>(ignore_index)) {
      CUDA_KERNEL_ASSERT(t >= 0 && t < n_classes);
      grad_input[i * ndim + t] = -(weights != nullptr ? weights[t] : static_cast<scalar_t>(1)) * norm * grad_output[0];
    }
  }
};

void nll_loss_backward_out_cuda_template(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& target,
    const Tensor& total_weight,
    const c10::optional<Tensor>& weight_opt,
    int64_t reduction,
    int64_t ignore_index) {
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;

  TORCH_CHECK(
      target.dim() == 1,
      "1D target tensor expected, multi-target not supported");
  int64_t n_dims = input.dim();
  TORCH_CHECK(
      n_dims > 0 && n_dims <= 2, "input tensor should be 1D or 2D");

  int64_t n_classes = input.size(-1);
  int64_t batch_size = n_dims == 1 ? 1 : input.size(0);
  int64_t num_targets = target.size(0);

  TORCH_CHECK(
      batch_size == num_targets,
      "size mismatch (got input: ",
      input.sizes(),
      ", target: ",
      target.sizes(),
      ")")
  TORCH_CHECK(
      !weight.defined() || (weight.dim() == 1 && weight.numel() == n_classes),
      "weight tensor should be defined either for all or no classes");

  TORCH_CHECK(grad_input.is_contiguous(), "grad_input must be contiguous");
  auto weight_ = weight.defined() ? weight.contiguous() : weight;

  if (reduction == at::Reduction::None && n_dims == 2) {
    check_dim_size(grad_output, 1, 0, batch_size);
    if (batch_size == 0) {
      // This guards from unnecessary operations and launching CUDA kernel with 0 blocks.
      return;
    }
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        input.scalar_type(),
        "nll_loss_backward_no_reduce_cuda_kernel",
        [&] {
          AT_DISPATCH_NLL_LOSS_INDEX_TYPES(
              target.scalar_type(),
              "nll_loss_backward_no_reduce_cuda_kernel_index",
              [&] {
                nll_loss_backward_no_reduce_cuda_kernel<scalar_t, index_t>
                    <<<at::cuda::detail::GET_BLOCKS(batch_size),
                       at::cuda::detail::CUDA_NUM_THREADS,
                       0,
                       at::cuda::getCurrentCUDAStream()>>>(
                        batch_size,
                        target.data_ptr<index_t>(),
                        grad_output.packed_accessor64<scalar_t, 1>(),
                        grad_input.packed_accessor64<scalar_t, 2>(),
                        weight.defined() ? weight_.data_ptr<scalar_t>()
                                         : nullptr,
                        n_classes,
                        ignore_index);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
              });
        });
    return;
  }

  auto target_ = target.contiguous();
  TORCH_CHECK(grad_output.numel() == 1);

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
                        weight.defined() ? weight_.data_ptr<scalar_t>()
                                         : nullptr,
                        target.data_ptr<index_t>(),
                        total_weight.data_ptr<scalar_t>(),
                        reduction == at::Reduction::Mean,
                        n_classes,
                        ignore_index);
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
              target.scalar_type(),
              "nll_loss_backward_reduce_cuda_kernel_2d_index",
              [&] {
            nll_loss_backward_reduce_cuda_kernel_2d<scalar_t, index_t>
                <<<1, NLL_LOSS_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
                    grad_input.data_ptr<scalar_t>(),
                    grad_output.data_ptr<scalar_t>(),
                    target.data_ptr<index_t>(),
                    weight.defined() ? weight_.data_ptr<scalar_t>() : nullptr,
                    total_weight.data_ptr<scalar_t>(),
                    reduction == at::Reduction::Mean,
                    input.size(0),
                    input.size(1),
                    n_classes,
                    ignore_index);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
          });
        });
  }
}

#undef AT_DISPATCH_NLL_LOSS_INDEX_TYPES

} // namespace

std::tuple<Tensor&, Tensor&> nll_loss_forward_out_cuda(
    const Tensor& self,
    const Tensor& target,
    const c10::optional<Tensor>& weight_opt,
    int64_t reduction,
    int64_t ignore_index,
    Tensor& output,
    Tensor& total_weight) {
  nll_loss_forward_out_cuda_template(
      output, total_weight, self, target, weight_opt, reduction, ignore_index);
  return std::tuple<Tensor&, Tensor&>(output, total_weight);
}

std::tuple<Tensor, Tensor> nll_loss_forward_cuda(
    const Tensor& self,
    const Tensor& target,
    const c10::optional<Tensor>& weight_opt,
    int64_t reduction,
    int64_t ignore_index) {
  auto output = at::empty({0}, self.options());
  auto total_weight = at::empty({0}, self.options());
  nll_loss_forward_out_cuda_template(
      output, total_weight, self, target, weight_opt, reduction, ignore_index);
  return std::make_tuple(output, total_weight);
}

Tensor& nll_loss_backward_out_cuda(const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    const c10::optional<Tensor>& weight_opt,
    int64_t reduction,
    int64_t ignore_index,
    const Tensor& total_weight,
    Tensor& grad_input) {

  grad_input.resize_as_(self);
  grad_input.zero_();
  nll_loss_backward_out_cuda_template(
      grad_input,
      grad_output,
      self,
      target,
      total_weight,
      weight_opt,
      reduction,
      ignore_index);
  return grad_input;
}

Tensor nll_loss_backward_cuda(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target, const c10::optional<Tensor>& weight_opt,
    int64_t reduction,
    int64_t ignore_index,
    const Tensor& total_weight) {

  auto grad_input = at::zeros_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  nll_loss_backward_out_cuda_template(
      grad_input,
      grad_output,
      self,
      target,
      total_weight,
      weight_opt,
      reduction,
      ignore_index);
  return grad_input;
}

}}  // namespace at::native
