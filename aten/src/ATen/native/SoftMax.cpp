#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>
#include <ATen/TensorMeta.h>
#include <ATen/TensorUtils.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/native/cpu/SoftmaxKernel.h>
#include <ATen/NamedTensorUtils.h>

#include <c10/core/TensorOptions.h>
#include <c10/macros/Macros.h>
#include <c10/util/irange.h>

namespace at {
namespace meta {
TORCH_META_FUNC(_softmax)
(const Tensor& input, const int64_t dim, const bool half_to_float) {
  int64_t dim_ = maybe_wrap_dim(dim, input.dim());

  auto output_options =
      input.options().memory_format(LEGACY_CONTIGUOUS_MEMORY_FORMAT);

  if (half_to_float) {
    output_options = output_options.dtype(ScalarType::Float);
  }

  int64_t input_dim = input.dim() > 0 ? input.dim() : 1;
  TORCH_CHECK(
      dim_ >= 0 && dim_ < input_dim,
      "dim must be non-negative and less than input dimensions");

  set_output(input.sizes(), output_options);
}

TORCH_META_FUNC2(log_softmax, int) (
  const Tensor& input,
  const int64_t dim,
  c10::optional<ScalarType> dtype) {
  int64_t dim_ = maybe_wrap_dim(dim, input.dim());

  auto output_options =
      input.options().memory_format(LEGACY_CONTIGUOUS_MEMORY_FORMAT);

  if (dtype.has_value()) {
    output_options = output_options.dtype(dtype.value());
  }

  int64_t input_dim = input.dim() > 0 ? input.dim() : 1;
  TORCH_CHECK(
      dim_ >= 0 && dim_ < input_dim,
      "dim must be non-negative and less than input dimensions");
  set_output(0, input.sizes(), {}, output_options, input.has_names() ? input.names() : ArrayRef<Dimname>());
}

TORCH_META_FUNC(_softmax_backward_data)
(const Tensor& grad,
 const Tensor& output,
 int64_t dim,
 ScalarType input_dtype) {
  TensorArg grad_arg{grad, "grad", 1}, output_arg{output, "output", 2};
  checkSameSize("softmax_backward", grad_arg, output_arg);

  int64_t dim_ = maybe_wrap_dim(dim, grad.dim());

  auto grad_input_options =
      grad.options().memory_format(LEGACY_CONTIGUOUS_MEMORY_FORMAT);

  bool half_to_float = grad.scalar_type() != input_dtype;
  if (half_to_float) {
    // The code below is only valid for the CUDA implementation. It's "okay"
    // to put it here because half-to-float conversion is not supported by
    // the CPU implementation of _softmax. There is a TORCH_CHECK in the CUDA
    // implementation that should ideally go here as well, but there is at least
    // one test in which the grad and input dtypes do not match for the CPU
    // implementation of this kernel and it is not true that the grad type is
    // float and the input dtype is half (see #63057).
    if (grad.scalar_type() == ScalarType::Float &&
        input_dtype == ScalarType::Half) {
      grad_input_options = grad_input_options.dtype(ScalarType::Half);
    }
  }

  int64_t grad_dim = grad.dim() > 0 ? grad.dim() : 1;
  TORCH_CHECK(
      dim_ >= 0 && dim_ < grad_dim,
      "dim must be non-negative and less than input dimensions");

  set_output(grad.sizes(), grad_input_options);
}

TORCH_META_FUNC(_log_softmax_backward_data)
(const Tensor& grad,
 const Tensor& output,
 int64_t dim,
 ScalarType input_dtype){
  int64_t dim_ = maybe_wrap_dim(dim, grad.dim());
  TensorOptions grad_input_options(
      grad.options().memory_format(LEGACY_CONTIGUOUS_MEMORY_FORMAT));

  bool half_to_float = grad.scalar_type() != input_dtype;
  if (half_to_float) {
    // The code below is only valid for the CUDA implementation. It's "okay"
    // to put it here because half-to-float conversion is not supported by
    // the CPU implementation of _softmax. There is a TORCH_CHECK in the CUDA
    // implementation that should ideally go here as well, but there is at least
    // one test in which the grad and input dtypes do not match for the CPU
    // implementation of this kernel and it is not true that the grad type is
    // float and the input dtype is half (see #63057).
    if (grad.scalar_type() == ScalarType::Float &&
        input_dtype == ScalarType::Half) {
      grad_input_options = grad_input_options.dtype(ScalarType::Half);
    }
  }

  int64_t grad_dim = grad.dim() > 0 ? grad.dim() : 1;
  TORCH_CHECK(
      dim_ >= 0 && dim_ < grad_dim,
      "dim must be non-negative and less than input dimensions");

  set_output(grad.sizes(), grad_input_options);
}
}

namespace native {
namespace {

template <typename scalar_t, bool LogSoftMax, bool MaskedSoftMax = false>
void host_softmax(
    Tensor output,
    const Tensor& input,
    const int64_t dim,
    bool* mask = nullptr) {
  int64_t outer_size = 1;
  int64_t dim_size = input.size(dim);
  int64_t inner_size = 1;
  for (const auto i : c10::irange(dim)) {
    outer_size *= input.size(i);
  }
  for (int64_t i = dim + 1; i < input.dim(); ++i) {
    inner_size *= input.size(i);
  }
  int64_t dim_stride = inner_size;
  int64_t outer_stride = dim_size * dim_stride;
  scalar_t* input_data_base = input.data_ptr<scalar_t>();
  scalar_t* output_data_base = output.data_ptr<scalar_t>();
  bool* mask_data_base = mask;
  int64_t grain_size = std::min(internal::GRAIN_SIZE / dim_size, (int64_t)1);
  parallel_for(
      0, outer_size * inner_size, grain_size,
      [&](int64_t begin, int64_t end) __ubsan_ignore_float_divide_by_zero__ {
        for (const auto i : c10::irange(begin, end)) {
          int64_t outer_idx = i / inner_size;
          int64_t inner_idx = i % inner_size;
          scalar_t* input_data =
              input_data_base + outer_idx * outer_stride + inner_idx;
          scalar_t* output_data =
              output_data_base + outer_idx * outer_stride + inner_idx;
          bool* mask_data = nullptr;
          if (MaskedSoftMax) {
            mask_data = mask_data_base + outer_idx * outer_stride + inner_idx;
          }

          // Calc max in softmax dim
          bool is_meaningful_max = false;
          scalar_t max_input = input_data[0];
          if (!MaskedSoftMax) {
            for (const auto d : c10::irange(1, dim_size)) {
              max_input = std::max(max_input, input_data[d * dim_stride]);
            }
          } else {
            for (const auto d : c10::irange(0, dim_size)) {
              if (!mask_data[d * dim_stride]) {
                max_input = is_meaningful_max
                    ? std::max(max_input, input_data[d * dim_stride])
                    : input_data[d * dim_stride];
                is_meaningful_max = true;
              }
            }
          }

          // Calc sum in softmax dim
          acc_type<scalar_t, false> tmpsum = 0;
          for (const auto d : c10::irange(dim_size)) {
            scalar_t z{};
            if (!MaskedSoftMax || !mask_data[d * dim_stride]) {
              z = std::exp(input_data[d * dim_stride] - max_input);
            } else {
              z = 0;
            }
            if (!LogSoftMax) {
              output_data[d * dim_stride] = z;
            }
            tmpsum += z;
          }

          if (LogSoftMax) {
            tmpsum = std::log(tmpsum);
          } else if (tmpsum == 0) {
            tmpsum = std::numeric_limits<scalar_t>::quiet_NaN();
          } else {
            tmpsum = 1 / tmpsum;
          }

          // update output
          for (const auto d : c10::irange(dim_size)) {
            // LogSoftMax and MaskedSoftMax should not both be true
            if (LogSoftMax) {
              output_data[d * dim_stride] =
                  input_data[d * dim_stride] - max_input - tmpsum;
            } else {
              output_data[d * dim_stride] *= tmpsum;
            }
          }
        }
      });
}

template <typename scalar_t, bool LogSoftMax, bool MaskedSoftMax = false>
void host_softmax_backward(
    const Tensor& gI,
    const Tensor& grad,
    const Tensor& output,
    int64_t dim,
    bool* mask = nullptr) {

  int64_t outer_size = 1;
  int64_t dim_size = grad.size(dim);
  int64_t inner_size = 1;
  for (const auto i : c10::irange(dim)) {
    outer_size *= grad.size(i);
  }
  for (int64_t i = dim + 1; i < grad.dim(); ++i) {
    inner_size *= grad.size(i);
  }
  int64_t dim_stride = inner_size;
  int64_t outer_stride = dim_size * dim_stride;
  scalar_t* gradInput_data_base = gI.data_ptr<scalar_t>();
  scalar_t* output_data_base = output.data_ptr<scalar_t>();
  scalar_t* gradOutput_data_base = grad.data_ptr<scalar_t>();
  bool* mask_data_base = mask;
  int64_t grain_size = std::min(internal::GRAIN_SIZE / dim_size, (int64_t)1);
  parallel_for(
      0, outer_size * inner_size, grain_size, [&](int64_t begin, int64_t end) {
        for (const auto i : c10::irange(begin, end)) {
          int64_t outer_idx = i / inner_size;
          int64_t inner_idx = i % inner_size;
          scalar_t* gradInput_data =
              gradInput_data_base + outer_idx * outer_stride + inner_idx;
          scalar_t* output_data =
              output_data_base + outer_idx * outer_stride + inner_idx;
          const scalar_t* gradOutput_data =
              gradOutput_data_base + outer_idx * outer_stride + inner_idx;
          bool* mask_data = nullptr;
          if (MaskedSoftMax) {
            mask_data = mask_data_base + outer_idx * outer_stride + inner_idx;
          }

          acc_type<scalar_t, false> sum = 0;
          for (const auto d : c10::irange(dim_size)) {
            if (!MaskedSoftMax || !mask_data[d * dim_stride]) {
              if (LogSoftMax) {
                sum += gradOutput_data[d * dim_stride];
              } else {
                sum +=
                    gradOutput_data[d * dim_stride] * output_data[d * dim_stride];
              }
            }
          }

          for (const auto d : c10::irange(dim_size)) {
            if (MaskedSoftMax && mask_data[d * dim_stride]) {
              gradInput_data[d * dim_stride] = 0;
            }
            else if (LogSoftMax) {
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

TORCH_IMPL_FUNC(softmax_cpu_out)
(const Tensor& input,
 const int64_t dim,
 const bool half_to_float,
 const Tensor& output) {
  TORCH_CHECK(!half_to_float, "softmax with half to float conversion is not supported on CPU");

  if (input.numel() == 0) {
    return;
  }

  auto input_ = input.contiguous();
  int64_t dim_ = maybe_wrap_dim(dim, input_.dim());

  if (input_.dim() == 0) {
    input_ = input_.view(1);
  }

  TORCH_CHECK(
      dim_ >= 0 && dim_ < input_.dim(),
      "dim must be non-negative and less than input dimensions");
  if (input_.ndimension() > 0 && dim_ == input_.ndimension() - 1) {
    softmax_lastdim_kernel(kCPU, output, input_);
  } else {
    softmax_kernel(kCPU, output, input_, dim_);
  }
}

void _log_softmax_cpu_out
(const Tensor& input,
 const int64_t dim,
 const Tensor& output) {

  if (input.numel() == 0) {
    return;
  }

  auto input_ = input.contiguous();
  int64_t dim_ = maybe_wrap_dim(dim, input_.dim());

  if (input_.dim() == 0) {
    input_ = input_.view(1);
  }

  if (input_.ndimension() > 0 && dim_ == input_.ndimension() - 1) {
    log_softmax_lastdim_kernel(kCPU, output, input_);
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND(
        at::ScalarType::BFloat16, input_.scalar_type(), "log_softmax", [&] {
          host_softmax<
              scalar_t,
              true /* LogSoftMax */,
              false /* MaskedSoftMax */>(output, input_, dim_);
        });
  }
}

TORCH_IMPL_FUNC(log_softmax_cpu_out)
(const Tensor& input,
 int64_t dim,
 c10::optional<ScalarType> dtype,
 const Tensor& output) {
  Tensor converted = dtype.has_value()? input.toType(dtype.value()) : input;
  auto output_temp = output.is_contiguous() ? output : at::empty_like(output, MemoryFormat::Contiguous) ;

  _log_softmax_cpu_out(converted.contiguous(), dim, output_temp);
  if (!output.is_contiguous()) {
    output.copy_(output_temp);
  }
}


TORCH_IMPL_FUNC(softmax_backward_cpu_out)
(const Tensor& grad,
 const Tensor& output,
 int64_t dim,
 ScalarType input_dtype,
 const Tensor& grad_input) {
  int64_t dim_ = maybe_wrap_dim(dim, grad.dim());
  auto grad_ = grad.contiguous();
  auto output_ = output.contiguous();

  if (output.numel() == 0) {
    return;
  }

  if (grad_.dim() == 0) {
    grad_ = grad_.view(1);
  }

  if (output_.dim() == 0) {
    output_ = output_.view(1);
  }

  if (grad_.ndimension() > 0 && dim_ == grad_.ndimension() - 1) {
    softmax_backward_lastdim_kernel(kCPU, grad_input, grad_, output);
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND(
        at::ScalarType::BFloat16, grad.scalar_type(), "softmax_backward", [&] {
          host_softmax_backward<
              scalar_t,
              false /* LogSoftMax */,
              false /* MaskedSoftmax */>(grad_input, grad_, output, dim_);
        });
  }
}

TORCH_IMPL_FUNC(log_softmax_backward_cpu_out) (
    const Tensor& grad,
    const Tensor& output,
    int64_t dim,
    ScalarType input_dtype,
    const Tensor& grad_input) {
  int64_t dim_ = maybe_wrap_dim(dim, grad.dim());
  auto grad_ = grad.contiguous();
  auto output_ = output.contiguous();

  if (output.numel() != 0) {
    if (grad_.dim() == 0)
      grad_ = grad_.view(1);
    if (output_.dim() == 0) {
      output_ = output_.view(1);
    }
    if (grad_.ndimension() > 0 && dim_ == grad_.ndimension() - 1) {
      log_softmax_backward_lastdim_kernel(kCPU, grad_input, grad_, output_);
    } else {
      AT_DISPATCH_FLOATING_TYPES_AND(
          at::ScalarType::BFloat16,
          grad.scalar_type(),
          "log_softmax_backward",
          [&] {
            host_softmax_backward<
                scalar_t,
                true /* LogSoftMax */,
                false /* MaskedSoftMax */>(grad_input, grad_, output_, dim_);
          });
    }
  }
}

Tensor softmax(const Tensor& input_, const int64_t dim_) {
  auto result = [&]() {
    NoNamesGuard guard;
    return at::_softmax(input_, dim_, false);
  }();
  namedinference::propagate_names(result, input_);
  return result;
}

Tensor softmax(const Tensor& input_, const int64_t dim_, c10::optional<ScalarType> dtype) {
  auto result = [&]() {
    NoNamesGuard guard;
    if (input_.is_cuda() && input_.scalar_type() == ScalarType::Half && dtype == ScalarType::Float){
        return at::_softmax(input_, dim_, true);
    } else {
        Tensor converted = dtype.has_value() ? input_.toType(dtype.value()) : input_;
        return at::_softmax(converted, dim_, false);
    }
  }();
  namedinference::propagate_names(result, input_);
  return result;
}

Tensor& softmax_out(
    const Tensor& input_,
    const int64_t dim_,
    c10::optional<ScalarType> dtype,
    Tensor& output_) {
  Tensor output_temp;
  if (input_.is_cuda() && input_.scalar_type() == ScalarType::Half &&
      dtype == ScalarType::Float) {
    if (!output_.is_contiguous()) {
      auto options =
          TensorOptions().dtype(output_.dtype()).device(output_.device());
      output_temp = at::empty(output_.sizes(), options);
      at::_softmax_out(output_temp, input_, dim_, true);
    } else {
      at::_softmax_out(output_, input_, dim_, true);
    }
  } else {
    Tensor converted =
        dtype.has_value() ? input_.toType(dtype.value()) : input_;
    if (!output_.is_contiguous()) {
      auto options =
          TensorOptions().dtype(output_.dtype()).device(output_.device());
      output_temp = at::empty(output_.sizes(), options);
      at::_softmax_out(output_temp, converted, dim_, false);
    } else {
      at::_softmax_out(output_, converted, dim_, false);
    }
  }

  if (!output_.is_contiguous()) {
    output_.resize_(output_temp.sizes());
    output_.copy_(output_temp);
  }

  return output_;
}

// special_softmax, alias for softmax
Tensor special_softmax(const Tensor& input_, const int64_t dim_, c10::optional<ScalarType> dtype) {
  return at::softmax(input_, dim_, dtype);
}

Tensor log_softmax(const Tensor& input_, const int64_t dim_) {
  return at::log_softmax(input_, dim_, c10::nullopt);
}

Tensor _log_softmax(const Tensor& input_, const int64_t dim_, bool half_to_float) {
  TORCH_INTERNAL_ASSERT("This operator is being removed. This is currently left only as a placeholder"
                        "while updating other libraries that depend on - please don't use it.");
  return input_;
}

Tensor special_log_softmax(const Tensor& input, const int64_t dim, c10::optional<ScalarType> dtype) {
  return at::log_softmax(input, dim, dtype);
}

DEFINE_DISPATCH(softmax_lastdim_kernel);
DEFINE_DISPATCH(log_softmax_lastdim_kernel);
DEFINE_DISPATCH(softmax_backward_lastdim_kernel);
DEFINE_DISPATCH(log_softmax_backward_lastdim_kernel);

DEFINE_DISPATCH(softmax_kernel);
DEFINE_DISPATCH(log_softmax_kernel);

Tensor softmax(const Tensor& self, Dimname dim, optional<ScalarType> dtype) {
  return at::softmax(self, dimname_to_position(self, dim), dtype);
}

Tensor log_softmax(const Tensor& self, Dimname dim, optional<ScalarType> dtype) {
  return at::log_softmax(self, dimname_to_position(self, dim), dtype);
}

Tensor masked_softmax_cpu(const Tensor& input_, const Tensor& mask_, const c10::optional<int64_t> dim_) {
  TORCH_CHECK(
      input_.sizes() == mask_.sizes(), "Mask shape should match input shape");
  TORCH_CHECK(
      mask_.scalar_type() == ScalarType::Bool,
      "Mask should be a boolean tensor");

  Tensor output = at::empty_like(input_, input_.options());
  auto input = input_.contiguous();
  auto mask = mask_.contiguous();
  int64_t dim = dim_.has_value() ? dim_.value() : input.dim() - 1;
  dim = maybe_wrap_dim(dim, input_.dim());

  if (input.dim() == 0) {
    input = input.view(1);
  }

  AT_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16, input.scalar_type(), "masked_softmax", [&] {
        host_softmax<
            scalar_t,
            false /* LogSoftMax */,
            true /* MaskedSoftMax */>(
            output, input, dim, mask.data_ptr<bool>());
      });
  return output;
}

Tensor masked_softmax_backward_cpu(
    const Tensor& grad_,
    const Tensor& output_,
    const Tensor& mask_,
    const c10::optional<int64_t> dim_) {
  TORCH_CHECK(
      grad_.sizes() == mask_.sizes(), "Mask shape should match grad shape");
  TORCH_CHECK(
      mask_.scalar_type() == ScalarType::Bool,
      "Mask should be a boolean tensor");
  auto grad = grad_.contiguous();
  auto output = output_.contiguous();
  auto mask = mask_.contiguous();

  int64_t dim = dim_.has_value() ? dim_.value() : output.dim() - 1;
  dim = maybe_wrap_dim(dim, grad.dim());

  grad = grad.dim() == 0 ? grad.view(1) : grad;
  output = output.dim() == 0 ? output.view(1) : output;
  mask = mask.dim() == 0 ? mask.view(1) : mask;

  Tensor grad_input = at::empty_like(grad, grad.options());
  AT_DISPATCH_FLOATING_TYPES_AND(
      at::ScalarType::BFloat16, grad.scalar_type(), "masked_softmax_backward", [&] {
        host_softmax_backward<
            scalar_t,
            false /* LogSoftMax */,
            true /* MaskedSoftmax */>(grad_input, grad, output, dim, mask.data_ptr<bool>());
      });
  return grad_input;
}
}
}
