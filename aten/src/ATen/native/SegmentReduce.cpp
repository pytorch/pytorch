#include <ATen/native/SegmentReduce.h>

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/NumericUtils.h>

namespace at {
namespace native {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(_segment_reduce_stub);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(_segment_reduce_backward_stub);

namespace {

SegmentReductionType get_reduction_enum(const c10::string_view& reduce) {
  if (reduce == "max") {
    return SegmentReductionType::MAX;
  } else if (reduce == "mean") {
    return SegmentReductionType::MEAN;
  } else if (reduce == "min") {
    return SegmentReductionType::MIN;
  } else if (reduce == "sum") {
    return SegmentReductionType::SUM;
  } else {
    TORCH_CHECK(false, "unsopported reduction given! ", reduce);
  }
}

template <typename T>
void _segment_reduce_cpu_kernel1(
    SegmentReductionType reduction,
    const Tensor& data,
    const T* lengths_data,
    int64_t axis,
    const c10::optional<Scalar>& initial,
    Tensor& output,
    int64_t segment_count) {
  int64_t stride_count = data.numel() / data.size(axis);
  AT_DISPATCH_FLOATING_TYPES_AND2(
      kBFloat16, kHalf, data.scalar_type(), "_segment_reduce_cpu", [&]() {
        auto* output_data = output.data_ptr<scalar_t>();
        const auto* values_data = data.data_ptr<scalar_t>();
        int64_t lengths_cum_sum = 0;
        for (int64_t i = 0; i < segment_count; ++i) {
          for (int64_t l = 0; l < stride_count; ++l) {
            // ===== step1: initialize starting value
            scalar_t initial_value;
            if (initial.has_value()) {
              initial_value = initial.value().to<scalar_t>();
            } else if (reduction == SegmentReductionType::MAX) {
              initial_value = -std::numeric_limits<scalar_t>::infinity();
            } else if (
                reduction == SegmentReductionType::MEAN ||
                reduction == SegmentReductionType::SUM) {
              initial_value = 0;
            } else if (reduction == SegmentReductionType::MIN) {
              initial_value = std::numeric_limits<scalar_t>::infinity();
            }

            // ===== step2: apply reduction
            for (int64_t j = 0; j < lengths_data[i]; ++j) {
              int64_t starting_index =
                  ((lengths_cum_sum + j) * stride_count) + l;
              const auto data = values_data[starting_index];
              // TODO: There is no need to branch with every element
              if (reduction == SegmentReductionType::MAX) {
                initial_value = at::_isnan(data)
                    ? data
                    : std::max<scalar_t>(initial_value, data);
              } else if (
                  reduction == SegmentReductionType::MEAN ||
                  reduction == SegmentReductionType::SUM) {
                initial_value = initial_value + data;
              } else if (reduction == SegmentReductionType::MIN) {
                initial_value = at::_isnan(data)
                    ? data
                    : std::min<scalar_t>(initial_value, data);
              }
            }

            // ===== step3: finalize reduction
            TORCH_CHECK(lengths_data[i] >= 0);

            if (lengths_data[i] == 0 && !initial.has_value() &&
                reduction == SegmentReductionType::MEAN) {
              initial_value = static_cast<scalar_t>(NAN);
            } else if (
                reduction == SegmentReductionType::MEAN &&
                lengths_data[i] > 0 && !at::_isnan(initial_value)) {
              initial_value = initial_value / lengths_data[i];
            }
            int64_t output_index = (i * stride_count) + l;
            output_data[output_index] = initial_value;
          }
          lengths_cum_sum += lengths_data[i];
        }
      });
}
Tensor _segment_reduce_cpu_kernel(
    SegmentReductionType reduction,
    const Tensor& data,
    const Tensor& lengths,
    int64_t axis,
    const c10::optional<Scalar>& initial) {
  int64_t segment_count = lengths.numel();
  auto output_shape = data.sizes().vec();
  output_shape[axis] = segment_count;
  auto output = at::empty(output_shape, data.options());

  AT_DISPATCH_INDEX_TYPES(lengths.type(), "_segment_reduce_cpu_kernel1", [&]() {
    const auto* lengths_data = lengths.data_ptr<index_t>();
    _segment_reduce_cpu_kernel1(
        reduction, data, lengths_data, axis, initial, output, segment_count);
  });

  return output;
}

template <typename T>
void _segment_reduce_cpu_backward_kernel1(
    const Tensor& grad_contig,
    const Tensor& output_contig,
    const Tensor& data_contig,
    SegmentReductionType reduction,
    const T* lengths_data,
    int64_t axis,
    Tensor& grad_input,
    int64_t segment_count) {
  int64_t stride_count = data_contig.numel() / data_contig.size(axis);
  // TODO: Swtich to TensorIterator for better maintainablility and
  // readability
  AT_DISPATCH_FLOATING_TYPES_AND2(
      kBFloat16,
      kHalf,
      data_contig.scalar_type(),
      "_segment_reduce_cpu",
      [&]() {
        auto* output_data = output_contig.data_ptr<scalar_t>();
        auto* grad_data = grad_contig.data_ptr<scalar_t>();
        auto* grad_input_data = grad_input.data_ptr<scalar_t>();
        const auto* values_data = data_contig.data_ptr<scalar_t>();

        int64_t lengths_cum_sum = 0;
        for (int64_t i = 0; i < segment_count; ++i) {
          if (lengths_data[i] == 0) {
            continue;
          }

          for (int64_t l = 0; l < stride_count; ++l) {
            int64_t output_index = (i * stride_count) + l;

            if (reduction == SegmentReductionType::MAX ||
                reduction == SegmentReductionType::MIN) {
              int64_t counter = 0;
              for (int64_t j = 0; j < lengths_data[i]; ++j) {
                int64_t starting_index =
                    ((lengths_cum_sum + j) * stride_count) + l;
                if (at::_isnan(values_data[starting_index]) ||
                    values_data[starting_index] == output_data[output_index]) {
                  grad_input_data[starting_index] = grad_data[output_index];
                  counter++;
                }
              }
              // Average gradient based on number of maximum elements in
              // the segment
              if (counter < 2) {
                continue;
              }
              for (int64_t j = 0; j < lengths_data[i]; ++j) {
                int64_t starting_index =
                    ((lengths_cum_sum + j) * stride_count) + l;
                if (grad_input_data[starting_index] > 0) {
                  grad_input_data[starting_index] =
                      grad_input_data[starting_index] / counter;
                }
              }
            } else if (reduction == SegmentReductionType::MEAN) {
              auto grad_val = grad_data[output_index] / lengths_data[i];
              for (int64_t j = 0; j < lengths_data[i]; ++j) {
                int64_t starting_index =
                    ((lengths_cum_sum + j) * stride_count) + l;
                grad_input_data[starting_index] = grad_val;
              }
            } else if (reduction == SegmentReductionType::SUM) {
              const auto& grad_val = grad_data[output_index];
              for (int64_t j = 0; j < lengths_data[i]; ++j) {
                int64_t starting_index =
                    ((lengths_cum_sum + j) * stride_count) + l;
                grad_input_data[starting_index] = grad_val;
              }
            }
          }

          lengths_cum_sum += lengths_data[i];
        }
      });
}

Tensor _segment_reduce_cpu_backward_kernel(
    const Tensor& grad_contig,
    const Tensor& output_contig,
    const Tensor& data_contig,
    SegmentReductionType reduction,
    const Tensor& lengths_contig,
    int64_t axis) {
  int64_t segment_count = lengths_contig.numel();
  auto output_shape = data_contig.sizes().vec();
  output_shape[axis] = segment_count;
  auto grad_input = at::zeros({data_contig.sizes()}, grad_contig.options());

  AT_DISPATCH_INDEX_TYPES(
      lengths_contig.type(), "_segment_reduce_cpu_backward_kernel1", [&]() {
        const auto* lengths_data = lengths_contig.data_ptr<index_t>();
        _segment_reduce_cpu_backward_kernel1(
            grad_contig,
            output_contig,
            data_contig,
            reduction,
            lengths_data,
            axis,
            grad_input,
            segment_count);
      });

  return grad_input;
}

} // namespace

Tensor segment_reduce_kernel(
    const Tensor& data,
    c10::string_view reduce,
    const c10::optional<Tensor>& lengths,
    const c10::optional<Tensor>& indices,
    int64_t axis,
    bool unsafe,
    const c10::optional<Scalar>& initial) {
  axis = maybe_wrap_dim(axis, data.ndimension());
  TORCH_CHECK(axis == 0, "Currently only dim=0 is supported! ", axis);
  TORCH_CHECK(data.numel() > 0);

  // length related checks
  TORCH_CHECK(
      lengths.has_value() && !indices.has_value(),
      "Currently only lengths based reduction is supported!")
  const auto& lengths_value = lengths.value();
  TORCH_CHECK(lengths_value.dim() == 1);
  TORCH_CHECK(data.get_device() == lengths_value.get_device());
  TORCH_CHECK(data.dim() >= lengths_value.dim());

  if (!unsafe) {
    auto min_length = lengths_value.min().item<int64_t>();
    TORCH_CHECK((min_length >= 0), "lengths contains negative value!");
    TORCH_CHECK(lengths_value.sum().item<int64_t>() == data.size(axis));
  }

  auto reduction = get_reduction_enum(reduce);
  const auto data_contig = data.contiguous();
  const auto lengths_contig = lengths_value.contiguous();

  return _segment_reduce_stub(
      data_contig.device().type(),
      reduction,
      data_contig,
      lengths_contig,
      axis,
      initial);
}

REGISTER_ARCH_DISPATCH(
    _segment_reduce_stub,
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
    DEFAULT,
    &_segment_reduce_cpu_kernel);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_AVX_DISPATCH(_segment_reduce_stub, &_segment_reduce_cpu_kernel);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_AVX2_DISPATCH(_segment_reduce_stub, &_segment_reduce_cpu_kernel);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_VSX_DISPATCH(_segment_reduce_stub, &_segment_reduce_cpu_kernel);

// Currently some computation is being duplicated across forward and backward.
// TODO: Cache indices in forward pass to re-use in backward
Tensor _segment_reduce_backward_kernel(
    const Tensor& grad,
    const Tensor& output,
    const Tensor& data,
    c10::string_view reduce,
    const c10::optional<Tensor>& lengths,
    int64_t axis) {
  axis = maybe_wrap_dim(axis, data.ndimension());
  TORCH_CHECK(axis == 0, "Currently only dim=0 is supported! ", axis);
  TORCH_CHECK(
      lengths.has_value(),
      "Currently only lengths based reduction is supported!")
  const auto& lengths_value = lengths.value();

  const auto grad_contig = grad.contiguous();
  const auto output_contig = output.contiguous();
  const auto data_contig = data.contiguous();
  const auto lengths_contig = lengths_value.contiguous();

  auto reduction = get_reduction_enum(reduce);
  return _segment_reduce_backward_stub(
      grad_contig.device().type(),
      grad_contig,
      output_contig,
      data_contig,
      reduction,
      lengths_contig,
      axis);
}

REGISTER_ARCH_DISPATCH(
    _segment_reduce_backward_stub,
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
    DEFAULT,
    &_segment_reduce_cpu_backward_kernel);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_AVX_DISPATCH(
    _segment_reduce_backward_stub,
    &_segment_reduce_cpu_backward_kernel);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
REGISTER_AVX2_DISPATCH(
    _segment_reduce_backward_stub,
    &_segment_reduce_cpu_backward_kernel);
REGISTER_VSX_DISPATCH(
    _segment_reduce_backward_stub,
    &_segment_reduce_cpu_backward_kernel);

} // namespace native
} // namespace at
