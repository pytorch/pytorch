#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/AccumulateType.h>
#include <ATen/Config.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/ScalarOps.h>
#include <ATen/TensorIterator.h>
#include <ATen/TensorMeta.h>
#include <ATen/TensorOperators.h>
#include <ATen/TensorUtils.h>

#include <ATen/detail/CUDAHooksInterface.h>
#include <ATen/native/cpu/Loops.h>
#include <ATen/native/batch_norm.h>
#include <ATen/native/Normalization.h>
#include <ATen/native/Resize.h>
#include <ATen/native/cpu/mixed_data_type.h>
#include <c10/util/irange.h>
#include <ATen/OpMathType.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_batch_norm_impl_index.h>
#include <ATen/ops/_batch_norm_impl_index_backward_native.h>
#include <ATen/ops/_batch_norm_impl_index_native.h>
#include <ATen/ops/_native_batch_norm_legit_native.h>
#include <ATen/ops/_native_batch_norm_legit_no_training.h>
#include <ATen/ops/_native_batch_norm_legit_no_training_native.h>
#include <ATen/ops/_batch_norm_with_update.h>
#include <ATen/ops/_batch_norm_with_update_native.h>
#include <ATen/ops/_batch_norm_no_update.h>
#include <ATen/ops/_batch_norm_no_update_native.h>
#include <ATen/ops/batch_norm_backward_native.h>
#include <ATen/ops/alias.h>
#include <ATen/ops/batch_norm.h>
#include <ATen/ops/batch_norm_native.h>
#include <ATen/ops/batch_norm_update_stats_native.h>
#include <ATen/ops/cudnn_batch_norm.h>
#include <ATen/ops/cudnn_batch_norm_backward.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/instance_norm_native.h>
#include <ATen/ops/linalg_vector_norm.h>
#include <ATen/ops/mean.h>
#include <ATen/ops/miopen_batch_norm.h>
#include <ATen/ops/miopen_batch_norm_backward.h>
#include <ATen/ops/mul.h>
#include <ATen/ops/native_batch_norm.h>
#include <ATen/ops/native_batch_norm_backward.h>
#include <ATen/ops/native_batch_norm_backward_native.h>
#include <ATen/ops/native_batch_norm_native.h>
#include <ATen/ops/_native_batch_norm_legit.h>
#include <ATen/ops/renorm_native.h>
#include <ATen/ops/sum.h>
#include <ATen/ops/sqrt.h>
#endif

#include <c10/core/SymIntArrayRef.h>
#include <utility>
#include <vector>

static const int MIOPEN_DIM_MAX = 5;

namespace at::meta {

TORCH_META_FUNC(renorm)(const Tensor& self, const Scalar& p, int64_t dim, const Scalar& maxnorm) {
  TORCH_CHECK(!p.isComplex(), "renorm: p must be real-valued");
  TORCH_CHECK(p.toDouble() > 0.0, "renorm: non-positive-norm not supported");
  TORCH_CHECK(!maxnorm.isComplex(), "renorm: maxnorm must be real-valued");
  TORCH_CHECK(maxnorm.toDouble() >= 0.0,
              "renorm: expected maxnorm to be >= 0 but got ", maxnorm.toDouble());
  const auto ndim = self.dim();
  TORCH_CHECK(ndim > 1, "renorm: input needs at least 2 dimensions, got ", ndim, " dimensions");
  set_output_raw_strided(0, self.sizes(), {}, self.options());
}

}  // namespace at::meta

namespace at::native {

DEFINE_DISPATCH(batch_norm_cpu_stub);
DEFINE_DISPATCH(batch_norm_cpu_collect_stats_stub);
DEFINE_DISPATCH(batch_norm_cpu_backward_stub);
DEFINE_DISPATCH(renorm_scale_factor_stub);

namespace {
  void check_dims_match_num_input_features(const char* arg_name, SymInt expected, SymInt actual){
    TORCH_CHECK(actual == expected,
             arg_name, " should contain ", expected, " elements not ", actual);
  }

  static inline Tensor repeat_if_defined(const Tensor& t, SymInt repeat) {
    if (t.defined()) {
      return t.repeat_symint(repeat);
    }
    return t;
  }
}

template<typename T>
struct InvStd {
  T operator()(T var, double epsilon) const {
    T invstd = 0;
    if (var != static_cast<T>(0) || epsilon != static_cast<T>(0)) {
      invstd = static_cast<T>(1) / std::sqrt(var + epsilon);
    }
    return invstd;
  }
};

template<typename T>
struct Var {
  T operator()(T var, double epsilon) const {
    return var;
  }
};

static inline bool is_contiguous(const Tensor& t) {
  return t.is_contiguous() || t.is_contiguous(at::MemoryFormat::ChannelsLast) || t.is_contiguous(at::MemoryFormat::ChannelsLast3d);
}

// For some ambiguous cases, it is possible a channels last contiguous Tensor has
//   `suggest_memory_format` of Contiguous.
// See https://github.com/pytorch/pytorch/issues/63224 for details.
static inline MemoryFormat suggest_memory_format_contig(const Tensor& t) {
  return t.is_contiguous() ?
    at::MemoryFormat::Contiguous : (t.is_contiguous(at::MemoryFormat::ChannelsLast3d) ?
    at::MemoryFormat::ChannelsLast3d : at::MemoryFormat::ChannelsLast);
}

template<typename scalar_t, typename param_t>
std::tuple<Tensor,Tensor,Tensor> batch_norm_cpu_transform_input_template(
    const Tensor& input, const Tensor& weight, const Tensor& bias,
    const Tensor& save_mean /* optional */, const Tensor& save_invstd /* optional */,
    const Tensor& running_mean /* optional */, const Tensor& running_var /* optional */,
    bool train, double eps, Tensor& output) {

  bool all_contiguous = is_contiguous(input)
    && is_contiguous(output)
    && (!weight.defined() || weight.is_contiguous())
    && (!bias.defined() || bias.is_contiguous())
    && running_mean.is_contiguous()
    && running_var.is_contiguous();

  // inference contiguous path
  if (all_contiguous) {
    if (input.numel() != 0) {
      batch_norm_cpu_stub(kCPU, output, input, weight, bias,
          save_mean, save_invstd, running_mean, running_var, train, eps);
    }
    return std::make_tuple(output, save_mean, save_invstd);
  }

  const int64_t ndim = input.dim();
  // Helper to convert 1d tensors to an nd tensor that broadcasts with input
  // All elements go into the channel dimension
  DimVector sizes(ndim, 1), strides(ndim, 0);
  auto as_nd = [&](const Tensor& t) {
    TORCH_INTERNAL_ASSERT(t.defined() && t.dim() == 1);
    sizes[1] = t.sizes()[0];
    strides[1] = t.strides()[0];
    return t.as_strided(sizes, strides);
  };

  auto mean = as_nd(train ? save_mean : running_mean);
  auto invstd = as_nd([&]{
    if (train) {
      return save_invstd;
    } else {
      return 1 / at::sqrt(running_var + eps);
    }
  }());
  constexpr bool mixed_type = !std::is_same<scalar_t, param_t>::value;
  const auto dtype = mixed_type ? kFloat : input.scalar_type();
  auto w = weight.defined() ? as_nd(weight) :
      at::detail::scalar_tensor_static(1, dtype, kCPU);
  auto b = bias.defined() ? as_nd(bias) :
      at::detail::scalar_tensor_static(0, dtype, kCPU);

  auto iter = TensorIteratorConfig()
    .add_output(output)
    .add_input(input)
    .add_input(mean)
    .add_input(invstd)
    .add_input(w)
    .add_input(b)
    .check_all_same_dtype(false)
    .promote_inputs_to_common_dtype(false)
    .build();
  cpu_kernel(iter, [=](scalar_t input, param_t mean, param_t invstd, param_t weight, param_t bias) -> scalar_t {
    return ((input - mean) * invstd) * weight + bias;
  });
  return std::make_tuple(output, save_mean, save_invstd);
}

template<typename scalar_t, typename param_t, template<typename T> class VarTransform>
std::tuple<Tensor,Tensor> batch_norm_cpu_update_stats_template(
    const Tensor& input, const Tensor& running_mean, const Tensor& running_var,
    double momentum, double eps, Tensor& save_mean, Tensor& save_var_transform) {

  using accscalar_t = at::acc_type<scalar_t, false>;

  int64_t n_input = input.size(1);
  TORCH_CHECK(input.numel() != 0, "input tensor must have at least one element, but got input_sizes = ", input.sizes());
  int64_t n = input.numel() / n_input;

  bool all_contiguous = is_contiguous(input);
  constexpr bool mixed_type = !std::is_same<scalar_t, param_t>::value;
  const auto dtype = mixed_type ? kFloat : input.scalar_type();

  auto save_mean_a = save_mean.accessor<param_t, 1>();
  auto save_var_transform_a = save_var_transform.accessor<param_t, 1>();

  auto running_mean_a = conditional_accessor_1d<param_t>(running_mean);
  auto running_var_a = conditional_accessor_1d<param_t>(running_var);

  if (all_contiguous) {
    auto _mean = at::empty({n_input}, input.options().dtype(dtype));
    auto _var_sum = at::empty({n_input}, input.options().dtype(dtype));
    auto _mean_a = _mean.accessor<param_t, 1>();
    auto _var_sum_a = _var_sum.accessor<param_t, 1>();
    auto momentum_ = static_cast<param_t>(momentum);

    batch_norm_cpu_collect_stats_stub(kCPU, _mean, _var_sum, input);

    parallel_for(0, n_input, 1, [&](int64_t b_begin, int64_t b_end) {
      for (const auto f : c10::irange(b_begin, b_end)) {
        save_mean_a[f] = _mean_a[f];
        save_var_transform_a[f] = VarTransform<accscalar_t>{}(_var_sum_a[f] / n, eps);

        if (running_mean.defined()) {
          running_mean_a[f] = momentum_ * _mean_a[f] + (1 - momentum_) * running_mean_a[f];
        }
        if (running_var.defined()) {
          accscalar_t unbiased_var = _var_sum_a[f] / (n - 1);
          running_var_a[f] = momentum_ * unbiased_var + (1 - momentum_) * running_var_a[f];
        }
      }
    });

    return std::make_tuple(save_mean, save_var_transform);
  }

  // non-contiguous path
  auto channel_stride = input.strides()[1];
  auto in_data = input.data_ptr<scalar_t>();
  auto reduce_iter = TensorIteratorConfig()
      .add_input(input)
      .resize_outputs(false)
      .declare_static_shape(input.sizes(), /*squash_dims=*/1)
      .check_all_same_dtype(false)
      .promote_inputs_to_common_dtype(false)
      .build();

  parallel_for(0, n_input, 1, [&](int64_t b_begin, int64_t b_end) {
    TensorIterator iter(reduce_iter);
    for (const auto f : c10::irange(b_begin, b_end)) {
      // compute variance per input
      iter.unsafe_replace_operand(0, in_data + channel_stride * f);
      accscalar_t var_sum = 0;
      auto mean = static_cast<accscalar_t>(save_mean_a[f]);
      cpu_serial_kernel(iter, [&](const scalar_t i) -> void {
        var_sum += (i - mean) * (i - mean);
      });
      save_var_transform_a[f] = VarTransform<accscalar_t>{}(var_sum / n, eps);

      // update running averages
      if (running_mean.defined()) {
        running_mean_a[f] = momentum * mean + (1 - momentum) * running_mean_a[f];
      }
      if (running_var.defined()) {
        accscalar_t unbiased_var = var_sum / (n - 1);
        running_var_a[f] = momentum * unbiased_var + (1 - momentum) * running_var_a[f];
      }
    }
  });
  return std::make_tuple(save_mean, save_var_transform);
}

template<typename scalar_t, typename param_t, template<typename T> class VarTransform>
std::tuple<Tensor,Tensor> batch_norm_cpu_update_stats_template(
    const Tensor& input, const Tensor& running_mean, const Tensor& running_var,
    double momentum, double eps) {
  int64_t n_input = input.size(1);
  const int64_t ndim = input.dim();
  DimVector reduce_dims(ndim - 1);
  reduce_dims[0] = 0;
  for (const auto i : c10::irange(2, ndim)) {
    reduce_dims[i - 1] = i;
  }

  constexpr bool mixed_type = !std::is_same<scalar_t, param_t>::value;
  const auto dtype = mixed_type ? kFloat : input.scalar_type();
  Tensor save_mean = is_contiguous(input) ? at::empty({n_input}, input.options().dtype(dtype)) : at::mean(input, /*dim=*/reduce_dims, /*keepdim=*/false, dtype);
  Tensor save_var_transform = at::empty({n_input}, input.options().dtype(dtype));
  return batch_norm_cpu_update_stats_template<scalar_t, param_t, VarTransform>(input, running_mean, running_var, momentum, eps, save_mean, save_var_transform);
}

template<typename scalar_t, typename param_t>
std::tuple<Tensor, Tensor, Tensor> batch_norm_backward_cpu_template(
    const Tensor& grad_out_, const Tensor& input, const Tensor& weight,
    const Tensor& running_mean, const Tensor& running_var, const Tensor& save_mean, const Tensor& save_invstd,
    bool train, double eps, std::array<bool,3> grad_input_mask) {

  using accscalar_t = at::acc_type<scalar_t, false>;

  constexpr bool mixed_type = !std::is_same<scalar_t, param_t>::value;
  const auto dtype = mixed_type ? kFloat : input.scalar_type();

  Tensor grad_input;
  Tensor grad_weight;
  Tensor grad_bias;
  if (grad_input_mask[0]) {
    grad_input = at::empty_like(input, input.suggest_memory_format());
  }
  if (grad_input_mask[1]) {
    grad_weight = at::empty({input.size(1)}, input.options().dtype(dtype));
  }
  if (grad_input_mask[2]) {
    grad_bias = at::empty({input.size(1)}, input.options().dtype(dtype));
  }

  // since we are directly manipulating pointers in contiguous path,
  // need to make sure input and grad_out have the same memory format.
  bool all_contiguous = is_contiguous(input)
      && is_contiguous(grad_out_)
      && input.suggest_memory_format() == grad_out_.suggest_memory_format();

  if (all_contiguous) {
    if (grad_input_mask[0]) {
      grad_input = at::empty_like(input, suggest_memory_format_contig(input));
    }
    batch_norm_cpu_backward_stub(kCPU, grad_input, grad_weight, grad_bias,
        grad_out_, input, weight, running_mean, running_var, save_mean, save_invstd, train, eps);
    return std::make_tuple(grad_input, grad_weight, grad_bias);
  }

  auto weight_a = conditional_accessor_1d<const param_t>(weight);
  auto grad_weight_a = conditional_accessor_1d<param_t>(grad_weight);
  auto grad_bias_a = conditional_accessor_1d<param_t>(grad_bias);

  int64_t n_input = input.size(1);
  int64_t n = input.numel() / n_input;

  auto save_mean_a = conditional_accessor_1d<const param_t>(save_mean);
  auto save_invstd_a = conditional_accessor_1d<const param_t>(save_invstd);

  auto running_mean_a = conditional_accessor_1d<const param_t>(running_mean);
  auto running_var_a = conditional_accessor_1d<const param_t>(running_var);

  const int64_t ndim = input.dim();

  // Reduce all dimensions except dim=1
  DimVector reduce_dims(ndim - 1);
  reduce_dims[0] = 0;
  for (const auto i : c10::irange(2, ndim)) {
    reduce_dims[i - 1] = i;
  }

  auto sum = at::sum(grad_out_, /*dims=*/reduce_dims);
  auto sum_a = sum.accessor<scalar_t, 1>();

  auto reduce_iter = TensorIteratorConfig()
      .add_const_input(input)
      .add_const_input(grad_out_)
      .resize_outputs(false)
      .declare_static_shape(input.sizes(), /*squash_dims=*/1)
      .build();

  TensorIterator unary_iter;
  TensorIterator binary_iter;
  if (grad_input_mask[0]) {
    unary_iter.build(
        TensorIteratorConfig()
        .add_output(grad_input)
        .add_const_input(train ? input : grad_out_)
        .resize_outputs(false)
        .declare_static_shape(input.sizes(), /*squash_dims=*/1));

    if (train) {
      binary_iter.build(
          TensorIteratorConfig()
          .add_output(grad_input)
          .add_input(grad_input)
          .add_const_input(grad_out_)
          .resize_outputs(false)
          .declare_static_shape(input.sizes(), /*squash_dims=*/1));
    }
  }

  auto in_channel_stride = input.strides()[1];
  auto in_data = input.const_data_ptr<scalar_t>();
  auto grad_in_channel_stride = grad_input_mask[0] ? grad_input.strides()[1] : 0;
  auto grad_in_data = grad_input_mask[0] ? grad_input.mutable_data_ptr<scalar_t>() : nullptr;
  auto grad_out_channel_stride = grad_out_.strides()[1];
  auto grad_out_data = grad_out_.const_data_ptr<scalar_t>();

  parallel_for(0, n_input, 1, [&](int64_t b_begin, int64_t b_end) {
      TensorIterator reduce_iter_local(reduce_iter);
      TensorIterator unary_iter_local(unary_iter);
      TensorIterator binary_iter_local(binary_iter);

      for (const auto f : c10::irange(b_begin, b_end)) {
        param_t w = weight.defined() ? weight_a[f] : param_t(1);

        param_t mean, invstd;
        if (train) {
          mean = save_mean_a[f];
          invstd = save_invstd_a[f];
        } else {
          mean = running_mean_a[f];
          invstd = 1 / std::sqrt(running_var_a[f] + eps);
        }

        // dot product of the Q(X) and gradOutput
        accscalar_t dotp = 0;
        reduce_iter_local.unsafe_replace_operand(
            0, const_cast<scalar_t*>(in_data + f * in_channel_stride));
        reduce_iter_local.unsafe_replace_operand(
            1, const_cast<scalar_t*>(grad_out_data + f * grad_out_channel_stride));

        cpu_serial_kernel(reduce_iter_local, [&](const scalar_t i, const scalar_t go) -> void {
          dotp += (i - mean) * go;
        });

        if (grad_input_mask[0]) {
          if (train) {
            // when in training mode
            // Q(X) = X - E[x] ; i.e. input centered to zero mean
            // Y = Q(X) / sigma    ; i.e. BN output before weight and bias
            // dL/dX = (Q(dL/dY) - dot(Y, dL/dY) * Y) / sigma * w

            // projection of gradOutput on to output scaled by std
            scalar_t k = (scalar_t) dotp * invstd * invstd / n;
            {
              unary_iter_local.unsafe_replace_operand(
                  0, grad_in_data + f * grad_in_channel_stride);
              unary_iter_local.unsafe_replace_operand(
                  1, const_cast<scalar_t*>(in_data + f * in_channel_stride));
              cpu_serial_kernel(unary_iter_local, [&](const scalar_t i) -> scalar_t {
                return (i - mean) * k;
              });
            }

            scalar_t grad_mean = sum_a[f] / n;
            {
              auto gI_data = grad_in_data + f * grad_in_channel_stride;
              binary_iter_local.unsafe_replace_operand(0, gI_data);
              binary_iter_local.unsafe_replace_operand(1, gI_data);
              binary_iter_local.unsafe_replace_operand(
                  2, const_cast<scalar_t*>(grad_out_data + f * grad_out_channel_stride));
              cpu_serial_kernel(binary_iter_local, [&](scalar_t gi, scalar_t go) -> scalar_t {
                return (go - grad_mean - gi) * invstd * w;
              });
            }
          } else {
            // when in evaluation mode
            // Q(X) = X - running_mean  ; i.e. input centered to zero mean
            // Y = Q(X) / running_std    ; i.e. BN output before weight and bias
            // dL/dX = w / running_std
            {
              unary_iter_local.unsafe_replace_operand(
                  0, grad_in_data + f * grad_in_channel_stride);
              unary_iter_local.unsafe_replace_operand(
                  1, const_cast<scalar_t*>(grad_out_data + f * grad_out_channel_stride));
              cpu_serial_kernel(unary_iter_local, [&](const scalar_t i) -> scalar_t {
                return i * invstd * w;
              });
            }
          }
        }
        if (grad_input_mask[1]) {
          grad_weight_a[f] = dotp * invstd;
        }

        if (grad_input_mask[2]) {
          grad_bias_a[f] = sum_a[f];
        }
      }
    });
  return std::make_tuple(grad_input, grad_weight, grad_bias);
}

BatchNormBackend _select_batch_norm_backend(
    const Tensor& input, const Tensor& weight, const Tensor& bias, const Tensor& running_mean,
    const Tensor& running_var, bool training, double eps) {

  auto& ctx = at::globalContext();
  bool cudnn_enabled = ctx.userEnabledCuDNN();

  if (
      input.is_cuda()
      && input.scalar_type() != at::kBFloat16 && weight.scalar_type() != at::kBFloat16
      && (input.scalar_type() != at::kHalf
        || weight.scalar_type() == at::kFloat)
      && weight.defined() && bias.defined()
      && ((running_mean.defined() && running_var.defined())
        || (!running_mean.defined() && !running_var.defined() && training))
      && (input.dim() >= 3)
      && ((input.sym_size(0) <= 880801 && training) // spatial, training
          ||(input.sym_size(0) <= 65535 && !training)) //spatial, eval
      && detail::getCUDAHooks().compiledWithCuDNN()
      && eps >= detail::getCUDAHooks().batchnormMinEpsilonCuDNN()
      && cudnn_enabled && detail::getCUDAHooks().versionCuDNN() >= 5110L
      && input.sym_numel() < std::numeric_limits<std::int32_t>::max() // some cuDNN kernels have 32-bit indexing limitations
  ) {
    return BatchNormBackend::Cudnn;
  }

  if (
      input.is_cuda()
      && input.dim() <= MIOPEN_DIM_MAX
      && input.scalar_type() != at::kDouble
      && input.scalar_type() != at::kBFloat16
      && (weight.scalar_type() != at::kHalf)
      && weight.defined() && bias.defined()
      && ((running_mean.defined() && running_var.defined())
        || (!running_mean.defined() && !running_var.defined() && training))
      && (input.dim() >= 3)
      && detail::getCUDAHooks().compiledWithMIOpen()
      && cudnn_enabled
      && input.suggest_memory_format() != MemoryFormat::ChannelsLast
      && input.suggest_memory_format() != MemoryFormat::ChannelsLast3d
  ) {
    return BatchNormBackend::Miopen;
  }

  return BatchNormBackend::Native;
}


// _batch_norm_impl_index(_backward) are used in the JIT be able to keep the run-time selection
// of backends, while enabling it to keep the information about the used backend, so that it can
// use its corresponding backward implementation.
// XXX: The indices of backends need to be kept synchronized between this function and its _backward.
// TODO: remove cudnn_enabled arg
std::tuple<Tensor, Tensor, Tensor, Tensor, int64_t> _batch_norm_impl_index(
    const Tensor& input, const std::optional<Tensor>& weight_opt /* optional */, const std::optional<Tensor>& bias_opt /* optional */, const std::optional<Tensor>& running_mean_opt /* optional */, const std::optional<Tensor>& running_var_opt /* optional */,
    bool training, double momentum, double eps, bool cudnn_enabled) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;
  const Tensor& bias = c10::value_or_else(bias_opt, [] {return Tensor();});
  const Tensor& running_mean = c10::value_or_else(running_mean_opt, [] {return Tensor();});
  const Tensor& running_var = c10::value_or_else(running_var_opt, [] {return Tensor();});

  auto num_features = input.sym_sizes()[1];

  if (input.sym_numel() == 0) {
    Tensor reserve = at::empty({0}, input.options().dtype(kByte));
    auto options = input.options().dtype(
        at::toAccumulateType(input.scalar_type(), /*is_cuda=*/input.is_cuda()));
    auto save_mean = at::empty_symint(c10::SymIntArrayRef({num_features}), options);
    auto save_invstd = at::empty_symint(c10::SymIntArrayRef({std::move(num_features)}), options);

    // don't return view of input, don't return empty tensor because it will break gradient chain
    auto out = input.clone();
    if (weight.defined()) out = out * weight[0];
    if (bias.defined()) out = out + bias[0];
    return std::tuple<Tensor, Tensor, Tensor, Tensor, int64_t>(
        out, save_mean, save_invstd, reserve, 0);
  }

  if (running_mean.defined()) {
    check_dims_match_num_input_features("running_mean", num_features, running_mean.sym_numel());
  } else if (!training) {
    AT_ERROR("running_mean must be defined in evaluation mode");
  }
  if (running_var.defined()) {
    check_dims_match_num_input_features("running_var", num_features, running_var.sym_numel());
  } else if (!training) {
    AT_ERROR("running_var must be defined in evaluation mode");
  }
  if (weight.defined()) {
    check_dims_match_num_input_features("weight", num_features, weight.sym_numel());
  }
  if (bias.defined()) {
    check_dims_match_num_input_features("bias", std::move(num_features), bias.sym_numel());
  }

  BatchNormBackend backend = _select_batch_norm_backend(input, weight, bias, running_mean, running_var, training, eps);

  if (backend == BatchNormBackend::Cudnn) {
    auto input_c = input.contiguous(input.suggest_memory_format());
    auto weight_c = weight.contiguous();
    auto bias_c = bias.contiguous();
    auto rmean_c = running_mean.defined() ? running_mean.contiguous() : running_mean;
    auto rvar_c = running_var.defined() ? running_var.contiguous() : running_var;

    auto [output, save_mean, save_var, reserve] =
        at::cudnn_batch_norm(input_c, weight_c, bias_c, rmean_c, rvar_c,
                             training, momentum, eps);

    return std::tuple<Tensor, Tensor, Tensor, Tensor, int64_t>(
        output, save_mean, save_var, reserve, 1);
  }

  Tensor reserve = at::empty({0}, input.options().dtype(kByte));

  if (backend == BatchNormBackend::Miopen) {
    return std::tuple_cat(
             at::miopen_batch_norm(
               input.contiguous(), weight.contiguous(), bias.contiguous(),
               running_mean.defined() ? running_mean.contiguous() : running_mean,
               running_var.defined() ? running_var.contiguous() : running_var,
               training, momentum, eps),
             std::tuple<Tensor>(reserve),
             std::make_tuple(2));
  }

  return std::tuple_cat(
           at::native_batch_norm(
             input, weight, bias, running_mean, running_var, training, momentum, eps),
           std::tuple<Tensor>(reserve),
           std::make_tuple(0));
}

std::tuple<Tensor, Tensor, Tensor> _batch_norm_impl_index_backward(
    int64_t impl_index,
    const Tensor& input, const Tensor& grad_output, const std::optional<Tensor>& weight_opt /* optional */, const std::optional<Tensor>& running_mean_opt /* optional */, const std::optional<Tensor>& running_var_opt /* optional */, const std::optional<Tensor>& save_mean_opt /* optional */, const std::optional<Tensor>& save_var_transform_opt /* optional */,
    bool train, double epsilon, std::array<bool, 3> output_mask, const Tensor &reservedSpace) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;
  const Tensor& running_mean = c10::value_or_else(running_mean_opt, [] {return Tensor();});
  const Tensor& running_var = c10::value_or_else(running_var_opt, [] {return Tensor();});
  const Tensor& save_mean = c10::value_or_else(save_mean_opt, [] {return Tensor();});
  const Tensor& save_var_transform = c10::value_or_else(save_var_transform_opt, [] {return Tensor();});

  if (input.numel() == 0) {
    std::vector<int64_t> dims(input.dim() - 1);
    dims[0] = 0;
    std::iota(dims.begin() + 1, dims.end(), 2);

    // don't return empty tensor because it will break gradient chain
    Tensor grad_input;
    Tensor grad_weight;
    Tensor grad_bias;
    if (output_mask[2]) {
      grad_bias = grad_output.sum(dims);
    }
    if (output_mask[1]) {
      grad_weight = (grad_output * input).sum(dims);
    }
    if (output_mask[0] && weight.defined()) {
      grad_input = grad_output * weight[0];
    }
    return std::make_tuple(grad_input, grad_weight, grad_bias);
  }

  // backward in inference mode is not supported in cudnn, fallback to native
  if (impl_index == 0 || (!train)) {
    return at::native_batch_norm_backward(grad_output, input, weight, running_mean, running_var, save_mean, save_var_transform, train, epsilon, output_mask);
  } else if (impl_index == 1) {
    // TODO: _batch_norm_impl_index_backward is only used in JIT. cudnn NHWC
    // format conversion is done inside cudnn_batch_norm_backward instead
    return at::cudnn_batch_norm_backward(input, grad_output, weight, running_mean, running_var, save_mean, save_var_transform, epsilon, reservedSpace);
  } else if (impl_index == 2) {
    return at::miopen_batch_norm_backward(input, grad_output, weight, running_mean, running_var, save_mean, save_var_transform, epsilon);
  }
  TORCH_INTERNAL_ASSERT(false, "Unsupported impl_index in _batch_norm_impl_index_backward: ", impl_index);
}

// TODO: remove cudnn_enabled arg
Tensor batch_norm(
    const Tensor& input, const std::optional<Tensor>& weight_opt, const std::optional<Tensor>& bias_opt,
    const std::optional<Tensor>& running_mean_opt, const std::optional<Tensor>& running_var_opt,
    bool training, double momentum, double eps, bool cudnn_enabled) {
  const Tensor& weight = c10::value_or_else(weight_opt, [] {return Tensor();});
  const Tensor& bias = c10::value_or_else(bias_opt, [] {return Tensor();});
  const Tensor& running_mean = c10::value_or_else(running_mean_opt, [] {return Tensor();});
  const Tensor& running_var = c10::value_or_else(running_var_opt, [] {return Tensor();});
  return std::get<0>(at::_batch_norm_impl_index(input, weight, bias, running_mean, running_var,
                                                training, momentum, eps, cudnn_enabled));
  // TODO: switch to the new stack after the 2 week FC window
  // if (training) {
  //   BatchNormBackend backend = _select_batch_norm_backend(input, weight, bias, running_mean, running_var, training, eps);
  //   if (backend == BatchNormBackend::Cudnn || backend == BatchNormBackend::Miopen) {
  //     auto input_c = input;
  //     if (backend == BatchNormBackend::Cudnn) {
  //         input_c = input.contiguous(input.suggest_memory_format());
  //     } else {
  //         input_c = input.contiguous();
  //     }
  //     auto weight_c = weight.contiguous();
  //     auto bias_c = bias.contiguous();
  //     auto rmean_c = running_mean.defined() ? running_mean.contiguous() : running_mean;
  //     auto rvar_c = running_var.defined() ? running_var.contiguous() : running_var;
  //     return std::get<0>(at::_batch_norm_with_update(input_c, weight_c, bias_c, const_cast<Tensor&>(rmean_c),
  //                                                   const_cast<Tensor&>(rvar_c), momentum, eps));
  //   } else {
  //     return std::get<0>(at::_batch_norm_with_update(input, weight, bias, const_cast<Tensor&>(running_mean),
  //                                                   const_cast<Tensor&>(running_var), momentum, eps));
  //   }
  // } else {
  //   return std::get<0>(at::_batch_norm_no_update(input, weight, bias, running_mean, running_var,
  //                                               momentum, eps));
  // }
}

Tensor instance_norm(
    const Tensor& input, const std::optional<Tensor>& weight_opt /* optional */, const std::optional<Tensor>& bias_opt /* optional */, const std::optional<Tensor>& running_mean_opt /* optional */, const std::optional<Tensor>& running_var_opt /* optional */,
    bool use_input_stats, double momentum, double eps, bool cudnn_enabled) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;
  const Tensor& bias = c10::value_or_else(bias_opt, [] {return Tensor();});
  const Tensor& running_mean = c10::value_or_else(running_mean_opt, [] {return Tensor();});
  const Tensor& running_var = c10::value_or_else(running_var_opt, [] {return Tensor();});

 TORCH_CHECK(use_input_stats || (running_mean.defined() && running_var.defined()),
           "Expected running_mean and running_var to be defined when use_input_stats is false");
  std::vector<SymInt> shape = input.sym_sizes().vec();
  SymInt b = input.sym_size(0);
  SymInt c = input.sym_size(1);
  shape[1] = b * c;
  shape[0] = SymInt(1);

  Tensor weight_ = repeat_if_defined(weight, b);
  Tensor bias_ = repeat_if_defined(bias, b);
  Tensor running_mean_ = repeat_if_defined(running_mean, b);
  Tensor running_var_ = repeat_if_defined(running_var, b);

  auto input_reshaped = input.contiguous().view_symint(shape);
  auto out = at::batch_norm(input_reshaped, weight_, bias_, running_mean_, running_var_,
                            use_input_stats, momentum, eps, cudnn_enabled);

  // we alias running_mean and running_var because they are const but we want to modify their data
  if (running_mean.defined()) {
    at::alias(running_mean).copy_(running_mean_.view_symint({ b, c }).mean(0, false));
  }
  if (running_var.defined()) {
    at::alias(running_var).copy_(running_var_.view_symint({ std::move(b), std::move(c) }).mean(0, false));
  }

  return out.view_symint(input.sym_sizes());
}

std::tuple<Tensor, Tensor> batch_norm_update_stats_cpu(
        const Tensor& self, const std::optional<Tensor>& running_mean_opt, const std::optional<Tensor>& running_var_opt, double momentum) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> running_mean_maybe_owned = at::borrow_from_optional_tensor(running_mean_opt);
  const Tensor& running_mean = *running_mean_maybe_owned;
  const Tensor& running_var = c10::value_or_else(running_var_opt, [] {return Tensor();});

  const bool mixed_type = is_mixed_type(self, running_mean, running_var);
  return AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::BFloat16, ScalarType::Half, self.scalar_type(), "batch_norm_update_stats_cpu", [&] {
    using opmath_t = at::opmath_type<scalar_t>;
    if (mixed_type) {
      check_mixed_data_type(self, running_mean, running_var);
      return batch_norm_cpu_update_stats_template<scalar_t, opmath_t, Var>(self, running_mean, running_var, momentum, 0);
    } else {
      return batch_norm_cpu_update_stats_template<scalar_t, scalar_t, Var>(self, running_mean, running_var, momentum, 0);
    }
  });
}

std::tuple<Tensor&, Tensor&, Tensor&> batch_norm_cpu_out(const Tensor& self, const std::optional<Tensor>& weight_opt, const std::optional<Tensor>& bias_opt, const std::optional<Tensor>& running_mean_opt, const std::optional<Tensor>& running_var_opt,
                                                  bool train, double momentum, double eps, Tensor& out, Tensor& save_mean, Tensor& save_var) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;
  const Tensor& bias = c10::value_or_else(bias_opt, [] {return Tensor();});
  const Tensor& running_mean = c10::value_or_else(running_mean_opt, [] {return Tensor();});
  const Tensor& running_var = c10::value_or_else(running_var_opt, [] {return Tensor();});

  checkBackend("batch_norm_cpu_out", {self, weight, bias, running_mean, running_var}, Backend::CPU);
  // Resize out
  at::native::resize_output(out, self.sizes());

  const bool mixed_type = is_mixed_type(self, weight, bias, running_mean, running_var);
  AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::BFloat16, ScalarType::Half, self.scalar_type(), "batch_norm", [&] {
    using opmath_t = at::opmath_type<scalar_t>;
    if (mixed_type) {
      check_mixed_data_type(self, weight, bias, running_mean, running_var);
      if (!train) {
        return batch_norm_cpu_transform_input_template<scalar_t, opmath_t>(self, weight, bias, save_mean, save_var, running_mean, running_var, train, eps, out);
      } else {
        // Resize save_mean and save_var
        at::native::resize_output(save_mean, {self.size(1)});
        at::native::resize_output(save_var, {self.size(1)});
        auto save_stats = batch_norm_cpu_update_stats_template<scalar_t, opmath_t, InvStd>(self, running_mean, running_var, momentum, eps, save_mean, save_var);
        return batch_norm_cpu_transform_input_template<scalar_t, opmath_t>(self, weight, bias, std::get<0>(save_stats), std::get<1>(save_stats), running_mean, running_var, train, eps, out);
      }
    } else {
      if (!train) {
        return batch_norm_cpu_transform_input_template<scalar_t, scalar_t>(self, weight, bias, save_mean, save_var, running_mean, running_var, train, eps, out);
      } else {
        // Resize save_mean and save_var
        at::native::resize_output(save_mean, {self.size(1)});
        at::native::resize_output(save_var, {self.size(1)});
        auto save_stats = batch_norm_cpu_update_stats_template<scalar_t, scalar_t, InvStd>(self, running_mean, running_var, momentum, eps, save_mean, save_var);
        return batch_norm_cpu_transform_input_template<scalar_t, scalar_t>(self, weight, bias, std::get<0>(save_stats), std::get<1>(save_stats), running_mean, running_var, train, eps, out);
      }
    }
  });

  return std::tuple<Tensor& ,Tensor&, Tensor&>(out, save_mean, save_var);
}

std::tuple<Tensor, Tensor, Tensor> batch_norm_cpu(const Tensor& self, const std::optional<Tensor>& weight_opt, const std::optional<Tensor>& bias_opt, const std::optional<Tensor>& running_mean_opt, const std::optional<Tensor>& running_var_opt,
                                                  bool train, double momentum, double eps) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;
  const Tensor& bias = c10::value_or_else(bias_opt, [] {return Tensor();});
  const Tensor& running_mean = c10::value_or_else(running_mean_opt, [] {return Tensor();});
  const Tensor& running_var = c10::value_or_else(running_var_opt, [] {return Tensor();});

  checkBackend("batch_norm_cpu", {self, weight, bias, running_mean, running_var}, Backend::CPU);

  // Prepare output tensor
  const bool all_contiguous = is_contiguous(self)
    && (!weight.defined() || weight.is_contiguous())
    && (!bias.defined() || bias.is_contiguous())
    && running_mean.is_contiguous()
    && running_var.is_contiguous();
  Tensor output = at::empty_like(self, all_contiguous ? suggest_memory_format_contig(self) : self.suggest_memory_format());

  // Prepare save_mean and save_var
  Tensor save_var;
  Tensor save_mean;
  const bool mixed_type = is_mixed_type(self, weight, bias, running_mean, running_var);
  const int64_t ndim = self.dim();
  DimVector reduce_dims(ndim - 1);
  reduce_dims[0] = 0;
  for (const auto i : c10::irange(2, ndim)) {
    reduce_dims[i - 1] = i;
  }
  if (mixed_type) {
    if (!train) {
      save_mean = at::empty({0}, self.options().dtype(kFloat));
      save_var = at::empty({0}, self.options().dtype(kFloat));
    } else {
      save_mean = is_contiguous(self) ? at::empty({self.size(1)}, self.options().dtype(kFloat)) : at::mean(self, /*dim=*/reduce_dims, /*keepdim=*/false, kFloat);
      save_var = at::empty({self.size(1)}, self.options().dtype(kFloat));
    }
  } else {
    if (!train) {
      save_mean = at::empty({0}, self.options());
      save_var = at::empty({0}, self.options());
    } else {
      save_mean = is_contiguous(self) ? at::empty({self.size(1)}, self.options()) : at::mean(self, /*dim=*/reduce_dims, /*keepdim=*/false);
      save_var = at::empty({self.size(1)}, self.options());
    }
  }
  return batch_norm_cpu_out(self, weight_opt, bias_opt, running_mean_opt, running_var_opt, train, momentum, eps, output, save_mean, save_var);
}

std::tuple<Tensor, Tensor, Tensor, Tensor> _batch_norm_with_update_cpu(
    const Tensor& input, const std::optional<Tensor>& weight_opt, const std::optional<Tensor>& bias_opt,
    Tensor& running_mean, Tensor& running_var, double momentum, double eps) {
  auto [output, save_mean, save_var] =
    batch_norm_cpu(input, weight_opt, bias_opt, running_mean, running_var, /*update*/true, momentum, eps);
  Tensor reserve = at::empty({0}, input.options().dtype(kByte));
  return std::tuple<Tensor, Tensor, Tensor, Tensor>(output, save_mean, save_var, reserve);
}

std::tuple<Tensor&, Tensor&, Tensor&, Tensor&> _batch_norm_with_update_cpu_out(
    const Tensor& input, const std::optional<Tensor>& weight_opt, const std::optional<Tensor>& bias_opt,
    Tensor& running_mean, Tensor& running_var, double momentum, double eps,
    Tensor& out, Tensor& save_mean, Tensor& save_var, Tensor& reserve) {
  std::tie(out, save_mean, save_var) =
    batch_norm_cpu_out(input, weight_opt, bias_opt, running_mean, running_var, /*update*/true, momentum, eps, out, save_mean, save_var);
  return std::tuple<Tensor&, Tensor&, Tensor&, Tensor&>(out, save_mean, save_var, reserve);
}


std::tuple<Tensor, Tensor, Tensor, Tensor> _batch_norm_no_update(
    const Tensor& input, const std::optional<Tensor>& weight_opt, const std::optional<Tensor>& bias_opt,
    const std::optional<Tensor>& running_mean_opt, const std::optional<Tensor>& running_var_opt,
    double momentum, double eps) {
  const Tensor& running_mean = c10::value_or_else(running_mean_opt, [] {return Tensor();});
  const Tensor& running_var = c10::value_or_else(running_var_opt, [] {return Tensor();});
  auto [output, save_mean, save_var] =
    batch_norm_cpu(input, weight_opt, bias_opt, const_cast<Tensor&>(running_mean), const_cast<Tensor&>(running_var), /*update*/false, momentum, eps);
  Tensor reserve = at::empty({0}, input.options().dtype(kByte));
  return std::tuple<Tensor, Tensor, Tensor, Tensor>(output, save_mean, save_var, reserve);
}

std::tuple<Tensor, Tensor, Tensor> _batch_norm_legit_cpu(
    const Tensor& self, const std::optional<Tensor>& weight_opt, const std::optional<Tensor>& bias_opt,
    Tensor& running_mean, Tensor& running_var, bool train, double momentum, double eps) {
  return batch_norm_cpu(self, weight_opt, bias_opt, running_mean, running_var, train, momentum, eps);
}

std::tuple<Tensor, Tensor, Tensor> _batch_norm_legit_no_stats_cpu(
    const Tensor& self, const std::optional<Tensor>& weight_opt, const std::optional<Tensor>& bias_opt,
    bool train, double momentum, double eps) {
  return batch_norm_cpu(self, weight_opt, bias_opt, Tensor(), Tensor(), train, momentum, eps);
}
std::tuple<Tensor, Tensor, Tensor> _batch_norm_legit_no_training(
    const Tensor& self, const std::optional<Tensor>& weight_opt, const std::optional<Tensor>& bias_opt,
    const Tensor& running_mean, const Tensor& running_var, double momentum, double eps) {
  return at::_native_batch_norm_legit(self, weight_opt, bias_opt, const_cast<Tensor&>(running_mean), const_cast<Tensor&>(running_var), /*train=*/false, momentum, eps);
}


std::tuple<Tensor&, Tensor&, Tensor&> _batch_norm_legit_cpu_out(const Tensor& self, const std::optional<Tensor>& weight_opt, const std::optional<Tensor>& bias_opt, Tensor& running_mean, Tensor& running_var, bool train, double momentum, double eps, Tensor& out, Tensor& save_mean, Tensor& save_var) {
  return batch_norm_cpu_out(self, weight_opt, bias_opt, running_mean, running_var, train, momentum, eps, out, save_mean, save_var);
}


std::tuple<Tensor&, Tensor&, Tensor&> _batch_norm_legit_no_stats_cpu_out(const Tensor& self, const std::optional<Tensor>& weight_opt, const std::optional<Tensor>& bias_opt, bool train, double momentum, double eps, Tensor& out, Tensor& save_mean, Tensor& save_var) {
  return batch_norm_cpu_out(self, weight_opt, bias_opt, Tensor(), Tensor(), train, momentum, eps, out, save_mean, save_var);
}

std::tuple<Tensor, Tensor, Tensor> _new_batch_norm_backward_cpu(
    const Tensor& grad_output, const Tensor& input, const Tensor& weight,
    const std::optional<Tensor>& running_mean_opt, const std::optional<Tensor>& running_var_opt,
    const std::optional<Tensor>& save_mean_opt, const std::optional<Tensor>& save_var_opt,
    bool update, double eps, std::array<bool,3> grad_input_mask, const Tensor& reserve) {
  return batch_norm_backward_cpu(grad_output, input, weight, running_mean_opt, running_var_opt, save_mean_opt, save_var_opt, update, eps, grad_input_mask);
}

std::tuple<Tensor, Tensor, Tensor> batch_norm_backward_cpu(const Tensor& grad_out, const Tensor& self, const std::optional<Tensor>& weight_opt, const std::optional<Tensor>& running_mean_opt, const std::optional<Tensor>& running_var_opt, const std::optional<Tensor>& save_mean_opt, const std::optional<Tensor>& save_invstd_opt,
                                                           bool train, double eps, std::array<bool,3> grad_input_mask) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> weight_maybe_owned = at::borrow_from_optional_tensor(weight_opt);
  const Tensor& weight = *weight_maybe_owned;
  const Tensor& running_mean = c10::value_or_else(running_mean_opt, [] {return Tensor();});
  const Tensor& running_var = c10::value_or_else(running_var_opt, [] {return Tensor();});
  const Tensor& save_mean = c10::value_or_else(save_mean_opt, [] {return Tensor();});
  const Tensor& save_invstd = c10::value_or_else(save_invstd_opt, [] {return Tensor();});

  const bool mixed_type = is_mixed_type(self, weight, running_mean, running_var, save_mean, save_invstd);
  return AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::BFloat16, ScalarType::Half, self.scalar_type(), "batch_norm_backward_cpu", [&] {
    using opmath_t = at::opmath_type<scalar_t>;
    if (mixed_type) {
      check_mixed_data_type(self, weight, running_mean, running_var, save_mean, save_invstd);
      return batch_norm_backward_cpu_template<scalar_t, opmath_t>(grad_out, self, weight, running_mean, running_var, save_mean, save_invstd, train, eps, grad_input_mask);
    } else {
      return batch_norm_backward_cpu_template<scalar_t, scalar_t>(grad_out, self, weight, running_mean, running_var, save_mean, save_invstd, train, eps, grad_input_mask);
    }
  });
}

TORCH_IMPL_FUNC(renorm_out)(const Tensor& self, const Scalar& p, int64_t dim,
                            const Scalar& maxnorm, const Tensor& out) {
  auto self_sizes = self.sizes();
  dim = c10::maybe_wrap_dim(dim, self_sizes.size());

  DimVector reduce_dims(self_sizes.size());
  std::iota(reduce_dims.begin(), reduce_dims.end(), 0);
  reduce_dims.erase(reduce_dims.begin() + dim);

  // For cuda half, calculate norm in float precision then cast
  // normalization factor to half
  auto dtype = self.scalar_type();
  auto acc_type = at::toAccumulateType(dtype, /*is_cuda=*/true);
  Tensor norm;
  if (acc_type != dtype) {
    norm = at::linalg_vector_norm(self, p.toDouble(), reduce_dims,
                                  /*keepdim=*/true, /*dtype=*/acc_type);
  } else {
    norm = at::linalg_vector_norm(self, p.toDouble(), reduce_dims,
                                  /*keepdim=*/true);
  }

  auto factor = (acc_type == c10::toRealValueType(dtype)) ?
      norm : at::empty(norm.sizes(), self.options());
  auto iter = TensorIteratorConfig()
      .add_output(factor)
      .add_input(norm)
      .set_check_mem_overlap(false)
      .cast_common_dtype_to_outputs(true)
      .build();

  renorm_scale_factor_stub(iter.device_type(), iter, maxnorm.toDouble());
  at::mul_outf(self, factor, const_cast<Tensor&>(out));
}

} // at::native
