#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>
#include <ATen/TensorUtils.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/native/cpu/SoftmaxKernel.h>
#include <ATen/NamedTensorUtils.h>

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
  scalar_t* input_data_base = input.data_ptr<scalar_t>();
  scalar_t* output_data_base = output.data_ptr<scalar_t>();
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

          acc_type<scalar_t, false> tmpsum = 0;
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
  scalar_t* gradInput_data_base = gI.data_ptr<scalar_t>();
  scalar_t* output_data_base = output.data_ptr<scalar_t>();
  scalar_t* gradOutput_data_base = grad.data_ptr<scalar_t>();
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

          acc_type<scalar_t, false> sum = 0;
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

template <typename scalar_t>
void host_masked_softmax(Tensor output, const Tensor& input, const Tensor& mask, const int64_t dim) {
  int64_t outer_size = 1;
  int64_t dim_size = input.size(dim);
  int64_t inner_size = 1;
  for (int64_t i = 0; i < dim; ++i)
    outer_size *= input.size(i);
  for (int64_t i = dim + 1; i < input.dim(); ++i)
    inner_size *= input.size(i);
  int64_t dim_stride = inner_size;
  int64_t outer_stride = dim_size * dim_stride;
  bool* mask_data_base = mask.data_ptr<bool>();
  scalar_t* input_data_base = input.data_ptr<scalar_t>();
  scalar_t* output_data_base = output.data_ptr<scalar_t>();
  int64_t grain_size = std::min(internal::GRAIN_SIZE / dim_size, (int64_t)1);
  parallel_for(
      0, outer_size * inner_size, grain_size,
      [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; i++) {
          int64_t outer_idx = i / inner_size;
          int64_t inner_idx = i % inner_size;
          bool* mask_data =
              mask_data_base + outer_idx * outer_stride + inner_idx;
          scalar_t* input_data =
              input_data_base + outer_idx * outer_stride + inner_idx;
          scalar_t* output_data =
              output_data_base + outer_idx * outer_stride + inner_idx;
          scalar_t max_input = input_data[0];

          for (int64_t d = 0; d < dim_size; d++){
            if (mask_data[d * dim_stride] == false) {
              max_input = std::max(max_input, input_data[d * dim_stride]);
            }
            else {
              output_data[d * dim_stride] = 0;
            }
          }

          acc_type<scalar_t, false> tmpsum = 0;
          for (int64_t d = 0; d < dim_size; d++){
            if (mask_data[d * dim_stride] == false) {
              scalar_t z = std::exp(input_data[d * dim_stride] - max_input);
              output_data[d * dim_stride] = z;
              tmpsum += z;
            }
          }
          for (int64_t d = 0; d < dim_size; d++) {
            // this is intentionally doing 0/0 = nan when an entire row is masked
            output_data[d * dim_stride] /= tmpsum;
          }
        }
      });
}
} // namespace

Tensor softmax_cpu(const Tensor& input_, const int64_t dim_, const bool half_to_float) {
  AT_ASSERTM(!half_to_float, "softmax with half to float conversion is not supported on CPU");
  auto input = input_.contiguous();
  Tensor output = at::native::empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
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

Tensor masked_softmax_cpu(const Tensor& input_, const Tensor& mask_, const int64_t dim_, const bool half_to_float) {
  // check both on cpu
  if (!(input_.device().type() == at::Device::Type::CPU)) {
    AT_ERROR("masked_softmax only supports for CPU tensor inputs");
  }
  if (!(mask_.device().type() == at::Device::Type::CPU)) {
    AT_ERROR("masked_softmax only supports for CPU tensor inputs");
  }

  // check input dype
  if (!(input_.scalar_type() == ScalarType::Float)) {
      AT_ERROR("masked_softmax requires input with dtype torch.float32.");
  }
  // check mask dtype + byte to bool conversion
  if (!(mask_.dtype() == ScalarType::Bool)) {
    if (mask_.dtype() == ScalarType::Byte) {
      TORCH_WARN("masked_softmax received a mask with dtype torch.uint8, this behavior is now deprecated,"
        "please use a mask with dtype torch.bool instead.");
      auto mask = mask_.to(at::kBool);
    }
    else {
      AT_ERROR("masked_softmax requires mask with dtype torch.bool.");
    }
  }

  AT_ASSERTM(!half_to_float, "masked_softmax with half to float conversion is not supported on CPU");
  auto input = input_.contiguous();
  auto mask = mask_.contiguous();
  Tensor output = at::native::empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  int64_t dim = maybe_wrap_dim(dim_, input.dim());

  if (input.numel() == 0) {
    return output;
  }
 if (input.dim() == 0)
    input = input.view(1);
  TORCH_CHECK(
      dim >= 0 && dim < input.dim(),
      "dim must be non-negative and less than input dimensions");
  TORCH_CHECK(
      input.sizes() == mask.sizes(),
      "input and mask must have the same size.");
  host_masked_softmax<float>(output, input, mask, dim);
  return output;
}

Tensor log_softmax_cpu(const Tensor& input_, const int64_t dim_, const bool half_to_float) {
  AT_ASSERTM(!half_to_float, "softmax with half to float conversion is not supported on CPU");
  auto input = input_.contiguous();
  Tensor output = at::native::empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
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
    AT_DISPATCH_FLOATING_TYPES_AND(
        at::ScalarType::BFloat16, input.scalar_type(), "log_softmax",
        [&] { host_softmax<scalar_t, true>(output, input, dim); });
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
  Tensor grad_input = at::native::empty_like(grad, LEGACY_CONTIGUOUS_MEMORY_FORMAT);

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
  Tensor grad_input = at::native::empty_like(grad, LEGACY_CONTIGUOUS_MEMORY_FORMAT);

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
    AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::BFloat16, grad.scalar_type(),
                                   "log_softmax_backward", [&] {
                                     host_softmax_backward<scalar_t, true>(
                                         grad_input, grad, output, dim);
                                   });
  }
  return grad_input;
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

Tensor masked_softmax(const Tensor& input_, const Tensor& mask_, const int64_t dim_) {
  auto result = [&]() {
    NoNamesGuard guard;
    // Tensor masked_input;

    if (!(input_.scalar_type() == ScalarType::Float)) {
      AT_ERROR("masked_softmax requires input with dtype torch.float32.");
    }

    if (mask_.dtype() == ScalarType::Byte) {
      TORCH_WARN("masked_softmax received a mask with dtype torch.uint8, this behavior is now deprecated,"
        "please use a mask with dtype torch.bool instead.");
      return at::_masked_softmax(input_, mask_.to(at::kBool), dim_, false);

    }
    else if (mask_.dtype() == ScalarType::Bool){
      return at::_masked_softmax(input_, mask_, dim_, false);
    }
    else {
      AT_ERROR("masked_softmax requires mask with dtype torch.bool.");
    }
  }();
  namedinference::propagate_names(result, input_);
  return result;
}

Tensor masked_softmax(const Tensor& input_, const Tensor& mask_, const int64_t dim_, c10::optional<ScalarType> dtype) {
  auto result = [&]() {
    NoNamesGuard guard;

    if (!(input_.scalar_type() == ScalarType::Float)) {
      AT_ERROR("masked_softmax requires input with dtype torch.float32.");
    }

    if (mask_.dtype() == ScalarType::Byte) {
      TORCH_WARN("masked_softmax received a mask with dtype torch.uint8, this behavior is now deprecated," \
        "please use a mask with dtype torch.bool instead.");
      Tensor res = at::_masked_softmax(input_, mask_.to(at::kBool), dim_, false);
      Tensor converted = dtype.has_value() ? res.toType(dtype.value()) : res;
      return converted;
    }
    else if (mask_.dtype() == ScalarType::Bool){
      Tensor res = at::_masked_softmax(input_, mask_, dim_, false);
      Tensor converted = dtype.has_value() ? res.toType(dtype.value()) : res;
      return converted;
    }
    else {
      AT_ERROR("masked_softmax requires mask with dtype torch.bool.");
    }
  }();
  namedinference::propagate_names(result, input_);
  return result;
}

Tensor log_softmax(const Tensor& input_, const int64_t dim_) {
  auto result = [&]() {
    NoNamesGuard guard;
    return at::_log_softmax(input_, dim_, false);
  }();
  namedinference::propagate_names(result, input_);
  return result;
}

Tensor log_softmax(const Tensor& input_, const int64_t dim_, c10::optional<ScalarType> dtype) {
  auto result = [&]() {
    NoNamesGuard guard;
    if (input_.is_cuda() && input_.scalar_type() == ScalarType::Half && dtype == ScalarType::Float){
        return at::_log_softmax(input_, dim_, true);
    } else {
        Tensor converted = dtype.has_value()? input_.toType(dtype.value()) : input_;
        return at::_log_softmax(converted, dim_, false);
    }
  }();
  namedinference::propagate_names(result, input_);
  return result;
}

DEFINE_DISPATCH(softmax_lastdim_kernel);
DEFINE_DISPATCH(log_softmax_lastdim_kernel);
DEFINE_DISPATCH(softmax_backward_lastdim_kernel);
DEFINE_DISPATCH(log_softmax_backward_lastdim_kernel);

Tensor softmax(const Tensor& self, Dimname dim, optional<ScalarType> dtype) {
  return at::softmax(self, dimname_to_position(self, dim), dtype);
}

Tensor masked_softmax(const Tensor& self, const Tensor& mask, Dimname dim, optional<ScalarType> dtype) {
  return at::masked_softmax(self, mask, dimname_to_position(self, dim), dtype);
}

Tensor log_softmax(const Tensor& self, Dimname dim, optional<ScalarType> dtype) {
  return at::log_softmax(self, dimname_to_position(self, dim), dtype);
}

}
}
