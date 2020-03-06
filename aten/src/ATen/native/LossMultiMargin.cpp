#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/AccumulateType.h>

namespace at {
namespace native {

namespace {

template <typename scalar_t>
inline scalar_t multi_margin_inner_sum_cpu(
    const scalar_t* input_data,
    const scalar_t* weight_data,
    const int p,
    const scalar_t margin,
    const int64_t dim,
    const int64_t target_idx) {
  const scalar_t input_target = input_data[target_idx];
  scalar_t sum = 0;
  for (int64_t d = 0; d < dim; d++) {
    if (d == target_idx) {
      continue;
    }

    const scalar_t z = margin - input_target + input_data[d];
    if (z > 0) {
      scalar_t h = (p == 1) ? z : z * z;
      if (weight_data != nullptr) {
        h *= weight_data[target_idx];
      }
      sum += h;
    }
  }

  sum /= dim;
  return sum;
}

inline int64_t target_index_checked(
    const int64_t* target_data,
    const int64_t index,
    const int64_t dim) {
  const int64_t idx = target_data[index];
  TORCH_CHECK(idx >= 0 && idx < dim, "target out of range");
  return idx;
}

template <typename scalar_t>
static inline void multi_margin_loss_cpu_kernel(
    Tensor& output,
    scalar_t* input_data,
    int64_t* target_data,
    const int p,
    scalar_t margin,
    scalar_t* weight_data,
    const int64_t nframe,
    const int64_t dim,
    const int64_t reduction) {
  using accscalar_t = at::acc_type<scalar_t, false>;

  // dim() != 0 check is for 1d input which produces a scalar output (that
  // cannot be handled by TensorAccessor)
  if (reduction == Reduction::None && output.dim() > 0) {
    auto output_acc = output.accessor<scalar_t, 1>();
    for (int64_t t = 0; t < nframe; t++) {
      const auto idx = target_index_checked(target_data, t, dim);
      auto sum = multi_margin_inner_sum_cpu(
          input_data, weight_data, p, margin, dim, idx);
      output_acc[t] = sum;
      input_data += dim;
    }
  } else {
    accscalar_t sum = 0;
    auto output_acc = output.data_ptr<scalar_t>();
    for (int64_t t = 0; t < nframe; t++) {
      const auto idx = target_index_checked(target_data, t, dim);
      sum += multi_margin_inner_sum_cpu(
          input_data, weight_data, p, margin, dim, idx);
      input_data += dim;
    }
    if (reduction == Reduction::Mean) {
      sum /= nframe;
    }
    output_acc[0] = sum;
  }
}

void multi_margin_loss_out_cpu_template(
    Tensor& output,
    const Tensor& input,
    const Tensor& target,
    int p,
    Scalar margin,
    const Tensor& weight,
    int64_t reduction) {
  const auto ndims = input.dim();
  TORCH_CHECK(
      input.numel() > 0 && ndims <= 2,
      "non-empty vector or matrix expected, got size: ",
      input.sizes());

  TORCH_CHECK(p == 1 || p == 2, "only p == 1 and p == 2 supported");

  int64_t nframe, dim;
  if (ndims <= 1) {
    nframe = 1;
    dim = ndims == 0 ? 1 : input.size(0);
  } else {
    nframe = input.size(0);
    dim = input.size(1);
  }

  TORCH_CHECK(
      target.numel() > 0 && target.dim() <= 1 && target.numel() == nframe,
      "inconsistent target size, got: ",
      target.sizes());

  // produce a scalar output for 1d input
  if (reduction == Reduction::None && target.dim() > 0) {
    output.resize_({nframe});
  } else {
    output.resize_({});
  }

  auto input_contiguous = input.contiguous();
  auto target_contiguous = target.contiguous();

  AT_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "multi_margin_loss_cpu_kernel", [&] {
        auto input_data = input_contiguous.data_ptr<scalar_t>();
        auto target_data = target_contiguous.data_ptr<int64_t>();
        auto weight_data =
            weight.defined() ? weight.data_ptr<scalar_t>() : nullptr;
        multi_margin_loss_cpu_kernel<scalar_t>(
            output,
            input_data,
            target_data,
            p,
            margin.to<scalar_t>(),
            weight_data,
            nframe,
            dim,
            reduction);
      });
}

template <typename scalar_t>
static void multi_margin_loss_backward_cpu_kernel(
    scalar_t* grad_input_data,
    const Tensor& grad_output,
    scalar_t* input_data,
    int64_t* target_data,
    int p,
    scalar_t margin,
    scalar_t g,
    scalar_t* weight_data,
    int64_t nframe,
    int64_t dim,
    int64_t reduction) {
  scalar_t* grad_input_row_data = grad_input_data;
  for (int64_t t = 0; t < nframe; t++) {
    int64_t target_idx = target_index_checked(target_data, t, dim);
    scalar_t input_target = input_data[target_idx];
    scalar_t grad_input_target = 0;
    for (int64_t d = 0; d < dim; d++) {
      scalar_t z = margin - input_target + input_data[d];
      if (d == target_idx) {
        continue;
      }

      if (z > 0) {
        scalar_t h = (p == 1) ? g : 2 * g * z;
        if (weight_data != nullptr) {
          h *= weight_data[target_idx];
        }
        grad_input_target -= h;
        grad_input_row_data[d] = h;
      } else {
        grad_input_row_data[d] = 0;
      }
    }
    grad_input_row_data[target_idx] = grad_input_target;

    input_data += dim;
    grad_input_row_data += dim;
  }

  if (reduction != Reduction::None || grad_output.dim() == 0) {
    assert(
        reduction != Reduction::None || grad_output.dim() > 0 ||
        nframe == 1); // check 1d scalar fallback-case
    const auto d = *grad_output.data_ptr<scalar_t>();
    for (int64_t t = 0; t < nframe * dim; t++) {
      grad_input_data[t] *= d;
    }
  } else {
    auto grad_output_acc = grad_output.accessor<scalar_t, 1>();
    for (int64_t t = 0; t < nframe; t++) {
      for (int64_t d = 0; d < dim; d++) {
        grad_input_data[t * dim + d] *= grad_output_acc[t];
      }
    }
  }
}

void multi_margin_loss_backward_out_cpu_template(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& target,
    int p,
    Scalar margin,
    const Tensor& weight,
    int64_t reduction) {
  const auto ndims = input.dim();
  TORCH_CHECK(
      input.numel() > 0 && ndims <= 2,
      "non-empty vector or matrix expected, got size: ",
      input.sizes());

  TORCH_CHECK(p == 1 || p == 2, "only p == 1 and p == 2 supported");

  int64_t nframe, dim;
  if (ndims <= 1) {
    nframe = 1;
    dim = ndims == 0 ? 1 : input.size(0);
  } else {
    nframe = input.size(0);
    dim = input.size(1);
  }

  TORCH_CHECK(
      target.numel() > 0 && target.dim() <= 1 && target.numel() == nframe,
      "inconsistent target size, got: ",
      target.sizes());

  grad_input.resize_as_(input);
  TORCH_CHECK(grad_input.is_contiguous(), "grad_input must be contiguous");

  auto input_contiguous = input.contiguous();
  auto target_contiguous = target.contiguous();
  auto weight_contiguous = weight.contiguous();
  AT_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "multi_margin_loss_backward_cpu_kernel", [&] {
        auto grad_input_data = grad_input.data_ptr<scalar_t>();
        auto input_data = input_contiguous.data_ptr<scalar_t>();
        auto target_data = target_contiguous.data_ptr<int64_t>();
        auto weight_data = weight_contiguous.defined()
            ? weight_contiguous.data_ptr<scalar_t>()
            : nullptr;
        scalar_t g = reduction == Reduction::Mean
            ? static_cast<scalar_t>(1. / (nframe * dim))
            : static_cast<scalar_t>(1. / dim);
        multi_margin_loss_backward_cpu_kernel<scalar_t>(
            grad_input_data,
            grad_output,
            input_data,
            target_data,
            p,
            margin.to<scalar_t>(),
            g,
            weight_data,
            nframe,
            dim,
            reduction);
      });
}

} // namespace

Tensor multi_margin_loss_cpu(
    const Tensor& input,
    const Tensor& target,
    Scalar p,
    Scalar margin,
    const Tensor& weight,
    int64_t reduction) {
  auto output = at::empty({0}, input.options());
  multi_margin_loss_out_cpu_template(
      output, input, target, p.toInt(), margin, weight, reduction);
  return output;
}

Tensor& multi_margin_loss_cpu_out(
    Tensor& output,
    const Tensor& input,
    const Tensor& target,
    Scalar p,
    Scalar margin,
    const Tensor& weight,
    int64_t reduction) {
  multi_margin_loss_out_cpu_template(
      output, input, target, p.toInt(), margin, weight, reduction);
  return output;
}

Tensor multi_margin_loss_cpu_backward(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& target,
    Scalar p,
    Scalar margin,
    const Tensor& weight,
    int64_t reduction) {
  auto grad_input = at::empty({0}, input.options());
  multi_margin_loss_backward_out_cpu_template(
      grad_input,
      grad_output,
      input,
      target,
      p.toInt(),
      margin,
      weight,
      reduction);
  return grad_input;
}

Tensor& multi_margin_loss_cpu_backward_out(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& target,
    Scalar p,
    Scalar margin,
    const Tensor& weight,
    int64_t reduction) {
  multi_margin_loss_backward_out_cpu_template(
      grad_input,
      grad_output,
      input,
      target,
      p.toInt(),
      margin,
      weight,
      reduction);
  return grad_input;
}

} // namespace native
} // namespace at
