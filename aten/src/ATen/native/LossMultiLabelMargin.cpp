#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorUtils.h>
#include <ATen/native/LossMulti.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/multilabel_margin_loss_backward_native.h>
#include <ATen/ops/multilabel_margin_loss_forward.h>
#include <ATen/ops/multilabel_margin_loss_forward_native.h>
#include <ATen/ops/multilabel_margin_loss_native.h>
#include <ATen/ops/zeros_like.h>
#endif

namespace at::native {

namespace {

template <typename scalar_t>
inline scalar_t multilabel_margin_loss_forward_inner_sum_cpu(
    const scalar_t* input_data,
    const int64_t* target_data,
    scalar_t* is_target_data,
    int64_t dim) {
  using accscalar_t = at::acc_type<scalar_t, false>;
  accscalar_t sum = 0;
  for (const auto ddt : c10::irange(dim)) {
    int64_t target_idx = target_data[ddt];
    if (target_idx < 0) {
      break;
    }
    is_target_data[target_idx] = 1;
  }
  for (const auto dt : c10::irange(dim)) {
    int64_t target_idx = target_data[dt];
    if (target_idx < 0) {
      break;
    }

    scalar_t input_target = input_data[target_idx];
    for (const auto d : c10::irange(dim)) {
      if (!is_target_data[d]) {
        scalar_t z = 1 - input_target + input_data[d];
        if (z > 0) {
          sum += z;
        }
      }
    }
  }

  return sum;
}

template <typename scalar_t>
static void multilabel_margin_loss_forward_out_frame(
    const Tensor& input_contiguous,
    const Tensor& target_contiguous,
    Tensor& output,
    Tensor& is_target,
    int64_t reduction,
    int64_t nframe,
    int64_t dim) {
  using accscalar_t = at::acc_type<scalar_t, false>;
  const scalar_t* input_data = input_contiguous.const_data_ptr<scalar_t>();
  const int64_t* target_data = target_contiguous.const_data_ptr<int64_t>();
  scalar_t* is_target_data = is_target.data_ptr<scalar_t>();

  if (reduction != Reduction::None || output.dim() == 0) {
    scalar_t* output_data = output.data_ptr<scalar_t>();

    accscalar_t sum = 0;

    for ([[maybe_unused]] const auto t : c10::irange(nframe)) {
      sum += multilabel_margin_loss_forward_inner_sum_cpu(
          input_data, target_data, is_target_data, dim);

      input_data += dim;
      target_data += dim;
      is_target_data += dim;
    }

    sum /= dim;
    if (reduction == Reduction::Mean) {
      sum /= nframe;
    }

    *output_data = sum; // write scalar output value
  } else {
    auto output_acc = output.accessor<scalar_t, 1>();

    for (const auto t : c10::irange(nframe)) {
      scalar_t sum = multilabel_margin_loss_forward_inner_sum_cpu(
          input_data, target_data, is_target_data, dim);

      sum /= dim;
      output_acc[t] = sum;

      input_data += dim;
      target_data += dim;
      is_target_data += dim;
    }
  }
}

static void multilabel_margin_loss_forward_out_cpu_template(
    const Tensor& input,
    const Tensor& target,
    Tensor& output,
    Tensor& is_target,
    int64_t reduction) {
#ifndef STRIP_ERROR_MESSAGES
  auto target_arg = TensorArg(target, "target", 2);
#endif
  int64_t nframe = 0, dim = 0;
  const int64_t ndims = input.dim();
  multilabel_margin_loss_shape_check(nframe, dim, ndims, input, target);

  // special case target.dim() <= 1: produce scalar output for scalar inputs
  // even if reduction == Reduction::None
  if (reduction != Reduction::None || target.dim() <= 1) {
    output.resize_({});
  } else {
    output.resize_({nframe});
  }

  is_target.resize_as_(target);
  TORCH_CHECK(is_target.is_contiguous(), "is_target must be contiguous");
  is_target.zero_();

  if (input.numel() == 0) {
    return;
  }

  TORCH_CHECK(
      target.min().item<int64_t>() >= -1, target_arg, " is out of range");
  TORCH_CHECK(
      target.max().item<int64_t>() < dim, target_arg, " is out of range");

  auto input_contiguous = input.contiguous();
  auto target_contiguous = target.contiguous();

  AT_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "multilabel_margin_loss_forward_out_frame", [&] {
        multilabel_margin_loss_forward_out_frame<scalar_t>(
            input_contiguous, target_contiguous, output, is_target, reduction, nframe, dim);
      });
}

template <typename scalar_t>
static void multilabel_margin_loss_backward_out_frame(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& input_contiguous,
    const Tensor& target_contiguous,
    int64_t reduction,
    const Tensor& is_target_contiguous,
    int64_t nframe,
    int64_t dim) {
#ifndef STRIP_ERROR_MESSAGES
  auto is_target_arg = TensorArg(is_target_contiguous, "is_target", 5);
#endif

  TORCH_CHECK(
      is_target_contiguous.min().item<scalar_t>() >= 0, is_target_arg, " is out of range");
  TORCH_CHECK(
      is_target_contiguous.max().item<scalar_t>() <= 1, is_target_arg, " is out of range");

  const scalar_t* input_data = input_contiguous.const_data_ptr<scalar_t>();
  const int64_t* target_data = target_contiguous.const_data_ptr<int64_t>();
  const scalar_t* is_target_data = is_target_contiguous.const_data_ptr<scalar_t>();
  scalar_t g = static_cast<scalar_t>(
      // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)
      reduction == Reduction::Mean ? 1. / (nframe * dim) : 1. / dim);

  scalar_t* grad_input_row_data = grad_input.mutable_data_ptr<scalar_t>();
  for ([[maybe_unused]] const auto t : c10::irange(nframe)) {
    for (const auto dt : c10::irange(dim)) {
      int64_t target_idx = target_data[dt];
      if (target_idx < 0) {
        break;
      }

      scalar_t input_target = input_data[target_idx];
      for (const auto d : c10::irange(dim)) {
        if (!is_target_data[d]) {
          scalar_t z = 1 - input_target + input_data[d];
          if (z > 0) {
            grad_input_row_data[target_idx] -= g;
            grad_input_row_data[d] += g;
          }
        }
      }
    }
    input_data += dim;
    target_data += dim;
    is_target_data += dim;
    grad_input_row_data += dim;
  }

  scalar_t* grad_input_data = grad_input.mutable_data_ptr<scalar_t>();
  if (reduction != Reduction::None || grad_output.dim() == 0) {
    assert(
        reduction != Reduction::None || grad_output.dim() > 0 || nframe == 1);
    const auto d = *grad_output.const_data_ptr<scalar_t>();
    for (int64_t t = 0; t < nframe * dim; t++) {
      grad_input_data[t] *= d;
    }
  } else {
    check_dim_size(grad_output, 1, 0, nframe);
    auto grad_output_acc = grad_output.accessor<const scalar_t, 1>();
    for (const auto t : c10::irange(nframe)) {
      for (const auto d : c10::irange(dim)) {
        grad_input_data[t * dim + d] *= grad_output_acc[t];
      }
    }
  }
}

static void multilabel_margin_loss_backward_out_cpu_template(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& target,
    int64_t reduction,
    const Tensor& is_target) {
  int64_t nframe = 0, dim = 0;
  CheckedFrom c = "multilabel_margin_loss_backward_cpu_template";
  auto target_arg = TensorArg(target, "target", 3);
  auto is_target_arg = TensorArg(is_target, "is_target", 5);
  const int64_t ndims = input.dim();

  multilabel_margin_loss_shape_check(nframe, dim, ndims, input, target);
  checkSameSize(c, target_arg, is_target_arg);

  grad_input.resize_as_(input);
  if (grad_input.numel() == 0) {
    return;
  }

  TORCH_CHECK(grad_input.is_contiguous(), "grad_input must be contiguous");
  grad_input.zero_();

  TORCH_CHECK(
      target.min().item<int64_t>() >= -1, target_arg, " is out of range");
  TORCH_CHECK(
      target.max().item<int64_t>() < dim, target_arg, " is out of range");

  auto input_contiguous = input.contiguous();
  auto target_contiguous = target.contiguous();
  auto is_target_contiguous = is_target.contiguous();

  AT_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "multilabel_margin_loss_backward_out_frame", [&] {
        multilabel_margin_loss_backward_out_frame<scalar_t>(
            grad_input,
            grad_output,
            input_contiguous,
            target_contiguous,
            reduction,
            is_target_contiguous,
            nframe,
            dim);
      });
}

} // namespace

std::tuple<Tensor&, Tensor&> multilabel_margin_loss_forward_out_cpu(const Tensor& self,
    const Tensor& target,
    int64_t reduction,
    Tensor& output,
    Tensor& is_target) {
  multilabel_margin_loss_forward_out_cpu_template(
      self, target, output, is_target, reduction);
  return std::tuple<Tensor&, Tensor&>(output, is_target);
}

std::tuple<Tensor, Tensor> multilabel_margin_loss_forward_cpu(
    const Tensor& self,
    const Tensor& target,
    int64_t reduction) {
  auto output = at::empty({0}, self.options());
  auto is_target = at::empty({0}, self.options());
  at::native::multilabel_margin_loss_forward_out_cpu(
      self, target, reduction, output, is_target);
  return std::make_tuple(output, is_target);
}

Tensor& multilabel_margin_loss_backward_cpu_out(const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    int64_t reduction,
    const Tensor& is_target,
    Tensor& grad_input) {
  multilabel_margin_loss_backward_out_cpu_template(
      grad_input, grad_output, self, target, reduction, is_target);
  return grad_input;
}

Tensor multilabel_margin_loss_backward_cpu(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& target,
    int64_t reduction,
    const Tensor& is_target) {
  auto grad_input = at::zeros_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  at::native::multilabel_margin_loss_backward_cpu_out(
      grad_output, self, target, reduction, is_target, grad_input);
  return grad_input;
}

Tensor & multilabel_margin_loss_out(const Tensor & self, const Tensor & target, int64_t reduction, Tensor & output) {
  Tensor is_target = at::empty({0}, self.options());
  return std::get<0>(at::multilabel_margin_loss_forward_out(output, is_target, self, target, reduction));
}

Tensor multilabel_margin_loss(const Tensor & self, const Tensor & target, int64_t reduction) {
  return std::get<0>(at::multilabel_margin_loss_forward(self, target, reduction));
}

} // namespace at::native
