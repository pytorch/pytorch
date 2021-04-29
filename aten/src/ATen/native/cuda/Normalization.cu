#include <ATen/native/TensorIterator.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/native/Resize.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/Reduce.cuh>
#include <ATen/native/cuda/Normalization.cuh>
#include <c10/cuda/CUDAMathCompat.h>

inline bool batch_norm_use_channels_last_kernels(const at::Tensor& self) {
  return self.is_contiguous(at::MemoryFormat::ChannelsLast) || self.ndimension() == 2;
}

namespace at { namespace native {

namespace {
void batch_norm_elementwise(
    const Tensor& out, const Tensor& self, const c10::optional<Tensor>& weight_opt,
    const c10::optional<Tensor>& bias_opt, const Tensor& mean_, const Tensor& invstd_) {
  const int64_t ndim = self.dim();
  DimVector sizes(ndim, 1), strides(ndim, 0);
  // Helper to convert 1d tensors to an nd tensor that broadcasts with input
  // All elements go into the channel dimension
  auto as_nd = [&](const Tensor& t) {
    TORCH_INTERNAL_ASSERT(t.defined() && t.dim() == 1);
    sizes[1] = t.sizes()[0];
    strides[1] = t.strides()[0];
    return t.as_strided(sizes, strides);
  };

  auto weight = weight_opt.has_value() && weight_opt->defined() ?
      as_nd(*weight_opt) : at::scalar_tensor(1, mean_.options());
  auto bias = bias_opt.has_value() && bias_opt->defined() ?
      as_nd(*bias_opt) : at::scalar_tensor(0, mean_.options());
  auto mean = as_nd(mean_);
  auto invstd = as_nd(invstd_);

  auto iter = TensorIteratorConfig()
      .add_output(out)
      .add_input(self)
      .add_input(weight)
      .add_input(bias)
      .add_input(mean)
      .add_input(invstd)
      .check_all_same_dtype(false)
      .promote_inputs_to_common_dtype(false)
      .build();

  AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, self.scalar_type(),
                                  "batch_norm_elementwise_cuda", [&] {
    using acc_t = at::acc_type<scalar_t, true>;
    gpu_kernel(iter, [] GPU_LAMBDA (scalar_t input, acc_t weight, acc_t bias,
                                    acc_t mean, acc_t invstd) -> scalar_t {
      return ((input - mean) * invstd) * weight + bias;
    });
  });
}

void batch_norm_var_mean(const Tensor& self, Tensor& save_var, Tensor& save_mean) {
  const int64_t ndim = self.dim();
  DimVector reduce_dims(ndim - 1);
  reduce_dims[0] = 0;
  for (int64_t i = 2; i < ndim; ++i) {
    reduce_dims[i - 1] = i;
  }

  // For some reason this isn't an actual operator but it exists anyway...
  at::native::var_mean_out(save_var, save_mean, self, /*dims=*/reduce_dims,
                           /*unbiased=*/false, /*keepdim=*/false);
}

void batch_norm_update_stats(
    const Tensor& save_mean, const Tensor& save_var,
    const Tensor& running_mean, const Tensor& running_var,
    double momentum, int64_t N) {

  auto iter = TensorIteratorConfig()
      .add_output(running_mean)
      .add_output(running_var)
      .add_input(save_mean)
      .add_input(save_var)
      .add_input(running_mean)
      .add_input(running_var)
      .check_all_same_dtype(false)
      .promote_inputs_to_common_dtype(false)
      .build();

  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, iter.common_dtype(),
                                  "batch_norm_update_stats_cuda", [&] {
      using acc_t = at::acc_type<scalar_t, true>;
      const auto bessel_correction_factor = static_cast<acc_t>(
          static_cast<double>(N) / static_cast<double>(N - 1));
      gpu_kernel_multiple_outputs(
          iter, [=] GPU_LAMBDA (acc_t mean, acc_t var, scalar_t running_mean, scalar_t running_var)
               -> thrust::tuple<scalar_t, scalar_t> {
        const auto unbiased_var = var * bessel_correction_factor;
        return thrust::tuple<scalar_t, scalar_t>{
          mean * momentum + (1 - momentum) * running_mean,
          unbiased_var * momentum + (1 - momentum) * running_var,
        };
      });
  });
}

void batch_norm_update_stats_and_invert(
    const Tensor& save_mean, const Tensor& save_var,
    const Tensor& running_mean, const Tensor& running_var,
    double momentum, double epsilon, int64_t N) {

  auto iter = TensorIteratorConfig()
      .add_output(running_mean)
      .add_output(running_var)
      .add_output(save_var)
      .add_input(save_mean)
      .add_input(save_var)
      .add_input(running_mean)
      .add_input(running_var)
      .check_all_same_dtype(false)
      .promote_inputs_to_common_dtype(false)
      .build();

  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, iter.common_dtype(),
                                  "batch_norm_update_stats_cuda", [&] {
      using acc_t = at::acc_type<scalar_t, true>;
      const auto bessel_correction_factor = static_cast<acc_t>(
          static_cast<double>(N) / static_cast<double>(N - 1));
      auto eps = static_cast<acc_t>(epsilon);
      gpu_kernel_multiple_outputs(
          iter, [=] GPU_LAMBDA (acc_t mean, acc_t var, scalar_t running_mean, scalar_t running_var)
               -> thrust::tuple<scalar_t, scalar_t, acc_t> {
        const auto unbiased_var = var * bessel_correction_factor;
        return thrust::tuple<scalar_t, scalar_t, acc_t>{
          mean * momentum + (1 - momentum) * running_mean,
          unbiased_var * momentum + (1 - momentum) * running_var,
          c10::cuda::compat::rsqrt(var + eps)
        };
      });
  });
}

void batch_norm_calc_invstd(const Tensor& out_invstd, const Tensor& running_var, double epsilon) {
  auto iter = TensorIteratorConfig()
      .add_output(out_invstd)
      .add_input(running_var)
      .check_all_same_dtype(false)
      .build();

  AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, iter.dtype(1),
                                  "batch_norm_invert_std_cuda", [&] {
    using acc_t = at::acc_type<scalar_t, true>;
    auto eps = static_cast<acc_t>(epsilon);
    gpu_kernel(iter, [eps] GPU_LAMBDA (scalar_t var) -> acc_t {
      return c10::cuda::compat::rsqrt(var + eps);
    });
  });
}
}

std::tuple<Tensor&, Tensor&, Tensor&> batch_norm_cuda_out(const Tensor& self, const c10::optional<Tensor>& weight_opt, const c10::optional<Tensor>& bias_opt, const c10::optional<Tensor>& running_mean_opt, const c10::optional<Tensor>& running_var_opt, bool train, double momentum, double epsilon, Tensor& output, Tensor& save_mean, Tensor& save_invstd) {
  const bool has_running_mean = (running_mean_opt.has_value() && running_mean_opt->defined());
  const bool has_running_var = (running_mean_opt.has_value() && running_mean_opt->defined());
  TORCH_CHECK(has_running_mean == has_running_var);

  if (train) {
    batch_norm_var_mean(self, save_invstd, save_mean);
    if (has_running_mean) {
      const int64_t N = self.numel() / save_mean.numel();
      batch_norm_update_stats_and_invert(
          save_mean, save_invstd, *running_mean_opt, *running_var_opt,
          momentum, epsilon, N);
    } else {
      batch_norm_calc_invstd(save_invstd, save_invstd, epsilon);
    }
  } else {
    at::native::resize_output(save_mean, running_mean_opt->sizes());
    save_mean.copy_(*running_mean_opt);
    batch_norm_calc_invstd(save_invstd, running_var_opt.value(), epsilon);
  }

  batch_norm_elementwise(output, self, weight_opt, bias_opt, save_mean, save_invstd);
  return std::tuple<Tensor&, Tensor&, Tensor&>(output, save_mean, save_invstd);
}

std::tuple<Tensor, Tensor, Tensor> batch_norm_cuda(const Tensor& self, const c10::optional<Tensor>& weight_opt, const c10::optional<Tensor>& bias_opt, const c10::optional<Tensor>& running_mean_opt, const c10::optional<Tensor>& running_var_opt, bool train, double momentum, double epsilon) {
  auto output = at::empty_like(self, at::MemoryFormat::Contiguous);
  int64_t n_input = self.size(1);
  auto options = self.options().dtype(
      at::toAccumulateType(self.scalar_type(), /*is_cuda=*/true));
  auto save_mean = at::empty({n_input}, options);
  auto save_invstd = at::empty({n_input}, options);

  at::native::batch_norm_cuda_out(
      self,
      weight_opt,
      bias_opt,
      running_mean_opt,
      running_var_opt,
      train,
      momentum,
      epsilon,
      output,
      save_mean,
      save_invstd);
  return std::make_tuple(output, save_mean, save_invstd);
}

std::tuple<Tensor, Tensor, Tensor> batch_norm_backward_cuda(const Tensor& grad_out, const Tensor& self, const c10::optional<Tensor>& weight_opt, const c10::optional<Tensor>& running_mean_opt, const c10::optional<Tensor>& running_var_opt, const c10::optional<Tensor>& save_mean_opt, const c10::optional<Tensor>& save_invstd_opt, bool train, double epsilon, std::array<bool,3> grad_input_mask) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;
  const Tensor& running_mean = c10::value_or_else(running_mean_opt, [] {return Tensor();});
  const Tensor& running_var = c10::value_or_else(running_var_opt, [] {return Tensor();});
  const Tensor& save_mean = c10::value_or_else(save_mean_opt, [] {return Tensor();});
  const Tensor& save_invstd = c10::value_or_else(save_invstd_opt, [] {return Tensor();});

  return AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, self.scalar_type(), "batch_norm_backward_cuda", [&] {
    auto mean_st = running_mean.dtype();
    auto var_st = running_var.dtype();
    TORCH_CHECK(mean_st == var_st, "running_mean and running_var need to have the same data types");
    bool is_half_float = std::is_same<scalar_t, at::Half>::value && mean_st == at::kFloat;
    bool is_bfloat16_float = std::is_same<scalar_t, at::BFloat16>::value && mean_st == at::kFloat;
    if (cuda::detail::canUse32BitIndexMath(self)) {
      if (is_half_float || is_bfloat16_float) {
        return batch_norm_backward_cuda_template<scalar_t, float, int32_t>(grad_out, self, weight, running_mean, running_var, save_mean, save_invstd, train, epsilon, grad_input_mask);
      } else {
        return batch_norm_backward_cuda_template<scalar_t, scalar_t, int32_t>(grad_out, self, weight, running_mean, running_var, save_mean, save_invstd, train, epsilon, grad_input_mask);
      }
    } else {
      if (is_half_float || is_bfloat16_float) {
        return batch_norm_backward_cuda_template<scalar_t, float, int64_t>(grad_out, self, weight, running_mean, running_var, save_mean, save_invstd, train, epsilon, grad_input_mask);
      } else {
        return batch_norm_backward_cuda_template<scalar_t, scalar_t, int64_t>(grad_out, self, weight, running_mean, running_var, save_mean, save_invstd, train, epsilon, grad_input_mask);
      }
    }
  });
}

std::tuple<Tensor, Tensor> batch_norm_stats_cuda(const Tensor& self, double epsilon) {
  const int64_t n_input = self.size(1);
  auto options = self.options().dtype(
      at::toAccumulateType(self.scalar_type(), /*is_cuda=*/true));
  auto save_mean = at::empty({n_input}, options);
  auto save_var = at::empty({n_input}, options);

  // FIXME: Epsilon parameter isn't required, we don't take the reciprocal
  batch_norm_var_mean(self, save_var, save_mean);
  return std::tuple<Tensor, Tensor>(save_mean, save_var);
}

Tensor batch_norm_elemt_cuda(const Tensor& self, const c10::optional<Tensor>& weight_opt, const c10::optional<Tensor>& bias_opt,
                             const Tensor& mean, const Tensor& invstd, double epsilon) {
  auto output = at::empty_like(self, self.suggest_memory_format());
  // FIXME: Epsilon parameter isn't required, we don't take the reciprocal
  batch_norm_elementwise(output, self, weight_opt, bias_opt, mean, invstd);
  return output;
}

Tensor& batch_norm_elemt_cuda_out(const Tensor& self, const c10::optional<Tensor>& weight_opt, const c10::optional<Tensor>& bias_opt,
                                  const Tensor& mean, const Tensor& invstd, double epsilon, Tensor& output) {
  // FIXME: Epsilon parameter isn't required, we don't take the reciprocal
  batch_norm_elementwise(output, self, weight_opt, bias_opt, mean, invstd);
  return output;
}

// accepting input(self) here to determine template data types, since running_mean/running_var are optional
std::tuple<Tensor, Tensor> batch_norm_gather_stats_cuda(const Tensor& self, const Tensor& mean, const Tensor& invstd, const c10::optional<Tensor>& running_mean_opt, const c10::optional<Tensor>& running_var_opt, double momentum, double epsilon, int64_t count) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> running_mean_maybe_owned = at::borrow_from_optional_tensor(running_mean_opt);
  const Tensor& running_mean = *running_mean_maybe_owned;
  const Tensor& running_var = c10::value_or_else(running_var_opt, [] {return Tensor();});

  std::vector<int64_t> counts(mean.size(0), count);
  Tensor counts_ = at::from_blob((void*)counts.data(), {(int64_t)counts.size()}, self.options().dtype(at::kLong).device(at::kCPU));
  counts_ = counts_.to(self.device()).to(running_mean.defined() ? running_mean.dtype() : self.dtype());
  return batch_norm_gather_stats_with_counts_cuda(self, mean, invstd, running_mean, running_var, momentum, epsilon, counts_);
}


std::tuple<Tensor, Tensor> batch_norm_gather_stats_with_counts_cuda(
    const Tensor& self, const Tensor& mean, const Tensor& invstd, const c10::optional<Tensor>& running_mean_opt /* optional */, const c10::optional<Tensor>& running_var_opt /* optional */, double momentum, double epsilon, const Tensor& counts) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> running_mean_maybe_owned = at::borrow_from_optional_tensor(running_mean_opt);
  const Tensor& running_mean = *running_mean_maybe_owned;
  const Tensor& running_var = c10::value_or_else(running_var_opt, [] {return Tensor();});


  auto scalar_type = running_mean.defined() ? running_mean.scalar_type() : self.scalar_type();
  return AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, scalar_type, "batch_norm_update_stats_cuda", [&] {
    using accscalar_t = at::acc_type<scalar_t, true>;
    if (cuda::detail::canUse32BitIndexMath(self)) {
      return batch_norm_gather_stats_cuda_template<scalar_t, accscalar_t, int32_t>(mean, invstd, running_mean, running_var, momentum, epsilon, counts);
    } else {
      return batch_norm_gather_stats_cuda_template<scalar_t, accscalar_t, int64_t>(mean, invstd, running_mean, running_var, momentum, epsilon, counts);
    }
  });
}

std::tuple<Tensor, Tensor, Tensor, Tensor> batch_norm_backward_reduce_cuda(const Tensor& self, const Tensor& input, const Tensor& mean, const Tensor& invstd, const c10::optional<Tensor>& weight_opt, bool input_g, bool weight_g, bool bias_g) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;

  // self is grad_output
  if (at::cuda::detail::canUse32BitIndexMath(self) && batch_norm_use_channels_last_kernels(self)){
    return batch_norm_backward_reduce_cuda_channels_last_template(self, input, mean, invstd, weight, input_g, weight_g, bias_g);
  }

  return AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, self.scalar_type(), "batch_norm_backward_reduce", [&] {
    auto mean_st = mean.dtype();
    auto invstd_st = invstd.dtype();
    TORCH_CHECK(mean_st == invstd_st, "mean and invstd need to have the same data types");
    bool is_half_float = std::is_same<scalar_t, at::Half>::value && mean_st == at::kFloat;
    bool is_bfloat16_float = std::is_same<scalar_t, at::BFloat16>::value && mean_st == at::kFloat;
    if (cuda::detail::canUse32BitIndexMath(self)) {
      if (is_half_float || is_bfloat16_float) {
        return batch_norm_backward_reduce_cuda_template<scalar_t, float, int32_t>(self, input, mean, invstd, weight, input_g, weight_g, bias_g);
      } else {
        return batch_norm_backward_reduce_cuda_template<scalar_t, scalar_t, int32_t>(self, input, mean, invstd, weight, input_g, weight_g, bias_g);
      }
    } else {
      if (is_half_float || is_bfloat16_float) {
        return batch_norm_backward_reduce_cuda_template<scalar_t, float, int64_t>(self, input, mean, invstd, weight, input_g, weight_g, bias_g);
      } else {
        return batch_norm_backward_reduce_cuda_template<scalar_t, scalar_t, int64_t>(self, input, mean, invstd, weight, input_g, weight_g, bias_g);
      }
    }
  });
}

Tensor batch_norm_backward_elemt_cuda(const Tensor& self, const Tensor& input, const Tensor& mean, const Tensor& invstd, const c10::optional<Tensor>& weight_opt, const Tensor& sum_dy, const Tensor& sum_dy_xmu, const Tensor& count) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;

  if (at::cuda::detail::canUse32BitIndexMath(self) && batch_norm_use_channels_last_kernels(self)){
    return batch_norm_backward_elemt_channels_last_cuda_template(self, input, mean, invstd, weight, sum_dy, sum_dy_xmu, count);
  }

  return AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, self.scalar_type(), "batch_norm_backward_elemt", [&] {
    auto mean_st = mean.dtype();
    auto invstd_st = invstd.dtype();
    TORCH_CHECK(mean_st == invstd_st, "mean and invstd need to have the same data types");
    bool is_half_float = std::is_same<scalar_t, at::Half>::value && mean_st == at::kFloat;
    bool is_bfloat16_float = std::is_same<scalar_t, at::BFloat16>::value && mean_st == at::kFloat;
    if (cuda::detail::canUse32BitIndexMath(self)) {
      if (is_half_float || is_bfloat16_float) {
        return batch_norm_backward_elemt_cuda_template<scalar_t, float, int32_t>(self, input, mean, invstd, weight, sum_dy, sum_dy_xmu, count);
      } else {
        return batch_norm_backward_elemt_cuda_template<scalar_t, scalar_t, int32_t>(self, input, mean, invstd, weight, sum_dy, sum_dy_xmu, count);
      }
    } else {
      if (is_half_float || is_bfloat16_float) {
        return batch_norm_backward_elemt_cuda_template<scalar_t, float, int64_t>(self, input, mean, invstd, weight, sum_dy, sum_dy_xmu, count);
      } else {
        return batch_norm_backward_elemt_cuda_template<scalar_t, scalar_t, int64_t>(self, input, mean, invstd, weight, sum_dy, sum_dy_xmu, count);
      }
    }
  });
}

std::tuple<Tensor, Tensor> batch_norm_update_stats_cuda(
        const Tensor& self, const c10::optional<Tensor>& running_mean_opt, const c10::optional<Tensor>& running_var_opt, double momentum) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> running_mean_maybe_owned = at::borrow_from_optional_tensor(running_mean_opt);
  const Tensor& running_mean = *running_mean_maybe_owned;
  const Tensor& running_var = c10::value_or_else(running_var_opt, [] {return Tensor();});

  const int64_t n_input = self.size(1);
  auto options = self.options().dtype(
      at::toAccumulateType(self.scalar_type(), /*is_cuda=*/true));
  auto save_mean = at::empty({n_input}, options);
  auto save_var = at::empty({n_input}, options);

  batch_norm_var_mean(self, save_var, save_mean);
  TORCH_CHECK(running_mean.defined() == running_var.defined());
  if (running_mean.defined()) {
    const int64_t N = self.numel() / save_mean.numel();
    batch_norm_update_stats(save_mean, save_var, running_mean, running_var, momentum, N);
  }

  return std::tuple<Tensor, Tensor>(save_mean, save_var);
}

} } // namespace at::native
