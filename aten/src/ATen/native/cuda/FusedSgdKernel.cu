#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/ForeachUtils.h>
#include <c10/util/Exception.h>
#include <ATen/native/cuda/ForeachFunctors.cuh>
#include <ATen/native/cuda/MultiTensorApply.cuh>

namespace at::native {

namespace {

template <typename scalar_t, int depth>
C10_DEVICE __forceinline__ void sgd_math(
    scalar_t r_args[depth][kILP],
    const double weight_decay,
    const double momentum,
    const float* lr_ptr,
    const double lr,
    const double dampening,
    const bool nesterov,
    const bool maximize,
    const bool is_first_step,
    const float* grad_scale_ptr) {
  using opmath_t = at::opmath_type<scalar_t>;
  const double double_lr = lr_ptr != nullptr ? *lr_ptr : lr;
#pragma unroll
  for (int ii = 0; ii < kILP; ii++) {
    auto p = static_cast<opmath_t>(r_args[0][ii]);
    auto g = static_cast<opmath_t>(r_args[1][ii]);
    if (grad_scale_ptr) {
      g /= static_cast<double>(*grad_scale_ptr);
      r_args[1][ii] = g;
    }
    if (maximize) {
      g *= -1.0;
    }
    if (weight_decay != 0) {
      g += weight_decay * p;
    }
    if (depth > 2) {
      const auto momentum_buffer = is_first_step
          ? g
          : (momentum * static_cast<opmath_t>(r_args[2][ii]) +
             (1 - dampening) * g);
      r_args[2][ii] = momentum_buffer;

      if (nesterov) {
        g = g + momentum * momentum_buffer;
      } else {
        g = momentum_buffer;
      }
    }
    p -= double_lr * g;
    r_args[0][ii] = p;
  }
}

template <typename scalar_t, int depth>
struct FusedSgdMathFunctor {
  static_assert(
      depth == 2 || depth == 3,
      "depth of 2 for SGD w/ momentum == 0, 3 for SGD w/ momentum != 0");
  C10_DEVICE __forceinline__ void operator()(
      const int chunk_size,
      TensorListMetadata<depth>& tl,
      const double weight_decay,
      const double momentum,
      const float* lr_ptr,
      const double lr,
      const double dampening,
      const bool nesterov,
      const bool maximize,
      const bool is_first_step,
      const float* grad_scale_ptr,
      const float* found_inf_ptr) {
    if (found_inf_ptr && *found_inf_ptr == 1) {
      return;
    }
    const auto tensor_loc = tl.block_to_tensor[blockIdx.x];
    const auto chunk_idx = tl.block_to_chunk[blockIdx.x];

    scalar_t* args[depth];
    scalar_t r_args[depth][kILP];
    const auto all_aligned{
        init_args<depth>(args, tl, chunk_idx, chunk_size, tensor_loc)};
    const auto n = tl.numel_for_tensor[tensor_loc] - chunk_idx * chunk_size;

    const auto use_faster_load_store =
        (n % kILP == 0) && (chunk_size % kILP == 0) && all_aligned;
    if (use_faster_load_store) {
      for (auto i_start = threadIdx.x;
           i_start * kILP < n && i_start * kILP < chunk_size;
           i_start += blockDim.x) {
#pragma unroll
        for (auto i = 0; i < depth; i++) {
          load_store(r_args[i], args[i], 0, i_start);
        }
        sgd_math<scalar_t, depth>(
            r_args,
            weight_decay,
            momentum,
            lr_ptr,
            lr,
            dampening,
            nesterov,
            maximize,
            is_first_step,
            grad_scale_ptr);
        load_store(args[0], r_args[0], i_start, 0);
        if (grad_scale_ptr) {
          load_store(args[1], r_args[1], i_start, 0);
        }
        if (depth > 2) {
          load_store(args[2], r_args[2], i_start, 0);
        }
      }
    } else {
      for (auto i_start = 0; i_start < n && i_start < chunk_size;
           i_start += blockDim.x * kILP) {
        load_args<depth>(r_args, args, i_start, chunk_size, n);
        sgd_math<scalar_t, depth>(
            r_args,
            weight_decay,
            momentum,
            lr_ptr,
            lr,
            dampening,
            nesterov,
            maximize,
            is_first_step,
            grad_scale_ptr);
        store_args(args[0], r_args[0], i_start, chunk_size, n);
        if (grad_scale_ptr) {
          store_args(args[1], r_args[1], i_start, chunk_size, n);
        }
        if (depth > 2) {
          store_args(args[2], r_args[2], i_start, chunk_size, n);
        }
      }
    }
  }
};

void _fused_sgd_with_momentum_kernel_cuda_(
    at::TensorList params,
    at::TensorList grads,
    at::TensorList momentum_buffer_list,
    const double weight_decay,
    const double momentum,
    const double lr,
    const double dampening,
    const bool nesterov,
    const bool maximize,
    const bool is_first_step,
    const std::optional<at::Tensor>& grad_scale,
    const std::optional<at::Tensor>& found_inf) {
  TORCH_CHECK_GT(momentum, 0);
  TORCH_CHECK(at::native::check_fast_path_restrictions(
      {params, grads, momentum_buffer_list}));
  float* grad_scale_ptr =
      grad_scale.has_value() ? grad_scale->data_ptr<float>() : nullptr;
  float* found_inf_ptr =
      found_inf.has_value() ? found_inf->data_ptr<float>() : nullptr;
  float* lr_ptr = nullptr;

  std::vector<std::vector<at::Tensor>> tensor_lists{
      params.vec(), grads.vec(), momentum_buffer_list.vec()};
  AT_DISPATCH_FLOATING_TYPES_AND2(
      kHalf,
      kBFloat16,
      params[0].scalar_type(),
      "fused_sgd_with_momentum_kernel_cuda",
      [&]() {
        multi_tensor_apply<3>(
            tensor_lists,
            FusedSgdMathFunctor<scalar_t, 3>(),
            weight_decay,
            momentum,
            lr_ptr,
            lr,
            dampening,
            nesterov,
            maximize,
            is_first_step,
            grad_scale_ptr,
            found_inf_ptr);
      });
}

void _fused_sgd_with_momentum_kernel_cuda_(
    at::TensorList params,
    at::TensorList grads,
    at::TensorList momentum_buffer_list,
    const double weight_decay,
    const double momentum,
    const at::Tensor& lr,
    const double dampening,
    const bool nesterov,
    const bool maximize,
    const bool is_first_step,
    const std::optional<at::Tensor>& grad_scale,
    const std::optional<at::Tensor>& found_inf) {
  if (lr.is_cpu()) {
    _fused_sgd_with_momentum_kernel_cuda_(
        params,
        grads,
        momentum_buffer_list,
        weight_decay,
        momentum,
        lr.item<double>(),
        dampening,
        nesterov,
        maximize,
        is_first_step,
        grad_scale,
        found_inf);
    return;
  }
  TORCH_CHECK_GT(momentum, 0);
  TORCH_CHECK(at::native::check_fast_path_restrictions(
      {params, grads, momentum_buffer_list}));
  if (grad_scale != c10::nullopt) {
    TORCH_CHECK(
        grad_scale->device() == params[0].device(),
        "grad_scale must be on the same GPU device as the params");
  }
  if (found_inf != c10::nullopt) {
    TORCH_CHECK(
        found_inf->device() == params[0].device(),
        "found_inf must be on the same GPU device as the params");
  }
  TORCH_CHECK(
      lr.device() == params[0].device(),
      "found_inf must be on the same GPU device as the params");
  float* grad_scale_ptr =
      grad_scale.has_value() ? grad_scale->data_ptr<float>() : nullptr;
  float* found_inf_ptr =
      found_inf.has_value() ? found_inf->data_ptr<float>() : nullptr;

  std::vector<std::vector<at::Tensor>> tensor_lists{
      params.vec(), grads.vec(), momentum_buffer_list.vec()};
  AT_DISPATCH_FLOATING_TYPES_AND2(
      kHalf,
      kBFloat16,
      params[0].scalar_type(),
      "fused_sgd_with_momentum_kernel_cuda",
      [&]() {
        multi_tensor_apply<3>(
            tensor_lists,
            FusedSgdMathFunctor<scalar_t, 3>(),
            weight_decay,
            momentum,
            lr.data_ptr<float>(),
            1.0,
            dampening,
            nesterov,
            maximize,
            is_first_step,
            grad_scale_ptr,
            found_inf_ptr);
      });
}

} // namespace

void _fused_sgd_kernel_cuda_(
    at::TensorList params,
    at::TensorList grads,
    at::TensorList momentum_buffer_list,
    const double weight_decay,
    const double momentum,
    const double lr,
    const double dampening,
    const bool nesterov,
    const bool maximize,
    const bool is_first_step,
    const std::optional<at::Tensor>& grad_scale,
    const std::optional<at::Tensor>& found_inf) {
  if (!momentum_buffer_list.empty()) {
    _fused_sgd_with_momentum_kernel_cuda_(
        params,
        grads,
        momentum_buffer_list,
        weight_decay,
        momentum,
        lr,
        dampening,
        nesterov,
        maximize,
        is_first_step,
        grad_scale,
        found_inf);
    return;
  }
  TORCH_CHECK_EQ(momentum, 0);
  TORCH_CHECK(at::native::check_fast_path_restrictions({params, grads}));
  if (is_first_step) {
    TORCH_WARN_ONCE(
        "`is_first_step` argument has no effect when `momentum_buffer_list` is empty");
  }
  float* grad_scale_ptr =
      grad_scale.has_value() ? grad_scale->data_ptr<float>() : nullptr;
  float* found_inf_ptr =
      found_inf.has_value() ? found_inf->data_ptr<float>() : nullptr;
  float* lr_ptr = nullptr;

  std::vector<std::vector<at::Tensor>> tensor_lists{params.vec(), grads.vec()};
  AT_DISPATCH_FLOATING_TYPES_AND2(
      kHalf,
      kBFloat16,
      params[0].scalar_type(),
      "fused_sgd_kernel_cuda",
      [&]() {
        multi_tensor_apply<2>(
            tensor_lists,
            FusedSgdMathFunctor<scalar_t, 2>(),
            weight_decay,
            momentum,
            lr_ptr,
            lr,
            dampening,
            nesterov,
            maximize,
            /* is_first_step */ false,
            grad_scale_ptr,
            found_inf_ptr);
      });
}

void _fused_sgd_kernel_cuda_(
    at::TensorList params,
    at::TensorList grads,
    at::TensorList momentum_buffer_list,
    const double weight_decay,
    const double momentum,
    const at::Tensor& lr,
    const double dampening,
    const bool nesterov,
    const bool maximize,
    const bool is_first_step,
    const std::optional<at::Tensor>& grad_scale,
    const std::optional<at::Tensor>& found_inf) {
  if (!momentum_buffer_list.empty()) {
    _fused_sgd_with_momentum_kernel_cuda_(
        params,
        grads,
        momentum_buffer_list,
        weight_decay,
        momentum,
        lr,
        dampening,
        nesterov,
        maximize,
        is_first_step,
        grad_scale,
        found_inf);
    return;
  }
  if (lr.is_cpu()) {
    _fused_sgd_kernel_cuda_(
        params,
        grads,
        momentum_buffer_list,
        weight_decay,
        momentum,
        lr.item<double>(),
        dampening,
        nesterov,
        maximize,
        is_first_step,
        grad_scale,
        found_inf);
    return;
  }
  TORCH_CHECK_EQ(momentum, 0);
  TORCH_CHECK(at::native::check_fast_path_restrictions({params, grads}));
  if (is_first_step) {
    TORCH_WARN_ONCE(
        "`is_first_step` argument has no effect when `momentum_buffer_list` is empty");
  }
  if (grad_scale.has_value()) {
    TORCH_CHECK(
        grad_scale->device() == params[0].device(),
        "grad_scale must be on the same GPU device as the params");
  }
  if (found_inf.has_value()) {
    TORCH_CHECK(
        found_inf->device() == params[0].device(),
        "found_inf must be on the same GPU device as the params");
  }
  TORCH_CHECK(
      lr.device() == params[0].device(),
      "lr must be on the same GPU device as the params");
  float* grad_scale_ptr =
      grad_scale.has_value() ? grad_scale->data_ptr<float>() : nullptr;
  float* found_inf_ptr =
      found_inf.has_value() ? found_inf->data_ptr<float>() : nullptr;

  std::vector<std::vector<at::Tensor>> tensor_lists{params.vec(), grads.vec()};
  AT_DISPATCH_FLOATING_TYPES_AND2(
      kHalf,
      kBFloat16,
      params[0].scalar_type(),
      "fused_sgd_kernel_cuda",
      [&]() {
        multi_tensor_apply<2>(
            tensor_lists,
            FusedSgdMathFunctor<scalar_t, 2>(),
            weight_decay,
            momentum,
            lr.data_ptr<float>(),
            1.0,
            dampening,
            nesterov,
            maximize,
            /* is_first_step */ false,
            grad_scale_ptr,
            found_inf_ptr);
      });
}

} // namespace at::native
