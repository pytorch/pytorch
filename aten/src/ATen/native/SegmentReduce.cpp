#include <ATen/native/SegmentReduce.h>

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/NumericUtils.h>

namespace at {
namespace native {

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
DEFINE_DISPATCH(_segment_reduce_stub);
DEFINE_DISPATCH(_segment_reduce_backward_stub);

namespace {

SegmentReductionType get_reduction_enum(const c10::string_view& reduce) {
  if (reduce == "max") {
    return SegmentReductionType::MAX;
  } else if (reduce == "mean") {
    return SegmentReductionType::MEAN;
  } else {
    TORCH_CHECK(false, "unsopported reduction given! ", reduce);
  }
}

Tensor _segment_reduce_cpu_kernel(
    SegmentReductionType reduction,
    const Tensor& data,
    const Tensor& lengths,
    int64_t axis,
    const c10::optional<Scalar>& initial) {
  int64_t batch_size = lengths.numel();
  auto output = at::empty({batch_size}, data.options());

  const auto* lengths_data = lengths.data_ptr<int64_t>();

  AT_DISPATCH_ALL_TYPES_AND2(
      kBFloat16, kHalf, data.scalar_type(), "_segment_reduce_cpu", ([&]() {
        auto* output_data = output.data_ptr<scalar_t>();
        const auto* values_data = data.data_ptr<scalar_t>();
        int64_t k = 0;
        for (int64_t i = 0; i < batch_size; ++i) {
          // ===== step1: initialize starting value
          scalar_t initial_value;
          if (initial.has_value()) {
            initial_value = initial.value().to<scalar_t>();
          } else if (reduction == SegmentReductionType::MAX) {
            initial_value = std::numeric_limits<scalar_t>::lowest();
          } else if (reduction == SegmentReductionType::MEAN) {
            initial_value = 0;
          }

          // ===== step2: apply reduction
          for (int64_t j = 0; j < lengths_data[i]; ++j) {
            const auto data = values_data[k];
            // TODO: There is no need to branch with every element
            if (reduction == SegmentReductionType::MAX) {
              initial_value = at::_isnan(data)
                  ? data
                  : std::max<scalar_t>(initial_value, data);
            } else if (reduction == SegmentReductionType::MEAN) {
              initial_value = at::_isnan(data) ? data : (initial_value + data);
            }
            k++;
          }

          // ===== step3: finalize reduction
          TORCH_CHECK(lengths_data[i] >= 0);
          if (lengths_data[i] == 0 && !initial.has_value()) {
            output_data[i] = static_cast<scalar_t>(NAN);
            continue;
          }
          output_data[i] = initial_value;
          if (reduction == SegmentReductionType::MEAN && lengths_data[i] > 0 &&
              !at::_isnan(output_data[i])) {
            output_data[i] = output_data[i] / lengths_data[i];
          }
        }
      }));

  return output;
}

Tensor _segment_reduce_cpu_backward_kernel(
    const Tensor& grad_contig,
    const Tensor& output_contig,
    const Tensor& data_contig,
    SegmentReductionType reduction,
    const Tensor& lengths_contig) {
  auto grad_input = at::zeros({data_contig.sizes()}, grad_contig.options());

  int64_t batch_size = lengths_contig.numel();
  const auto* lengths_data = lengths_contig.data_ptr<int64_t>();

  AT_DISPATCH_ALL_TYPES_AND2(
      kBFloat16,
      kHalf,
      data_contig.scalar_type(),
      "_segment_reduce_cpu",
      ([&]() {
        auto* output_data = output_contig.data_ptr<scalar_t>();
        auto* grad_data = grad_contig.data_ptr<scalar_t>();
        auto* grad_input_data = grad_input.data_ptr<scalar_t>();
        const auto* values_data = data_contig.data_ptr<scalar_t>();
        int64_t k = 0;
        for (int64_t i = 0; i < batch_size; ++i) {
          if (lengths_data[i] == 0) {
            continue;
          }
          if (reduction == SegmentReductionType::MAX) {
            int64_t counter = 0;
            for (int64_t j = 0; j < lengths_data[i]; ++j) {
              if (at::_isnan(values_data[k]) ||
                  values_data[k] == output_data[i]) {
                grad_input_data[k] = grad_data[i];
                counter++;
              }
              k++;
            }
            // Average gradient based on number of maximum elements in the
            // segment
            if (counter < 2) {
              continue;
            }
            for (int64_t j = 0; j < lengths_data[i]; ++j) {
              int64_t index = k - j - 1;
              if (grad_input_data[index] > 0) {
                grad_input_data[index] = grad_input_data[index] / counter;
              }
            }
          } else if (reduction == SegmentReductionType::MEAN) {
            auto grad_val = grad_data[i] / lengths_data[i];
            for (int64_t j = 0; j < lengths_data[i]; ++j) {
              grad_input_data[k] = grad_val;
              k++;
            }
          }
        }
      }));

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
  TORCH_CHECK(axis == 0, "Currently only dim=0 is supported!");
  TORCH_CHECK(data.dim() == 1);
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
    TORCH_CHECK(min_length != 0 || initial.has_value());
    TORCH_CHECK(lengths_value.sum().item<int64_t>() == data.numel());
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
REGISTER_VSX_DISPATCH(_segment_reduce_stub, &_segment_reduce_cpu_kernel);

// Currently some computation is beind duplicated across forward and backward.
// TODO: Cache indices in forward pass to re-use in backward
Tensor _segment_reduce_backward_kernel(
    const Tensor& grad,
    const Tensor& output,
    const Tensor& data,
    c10::string_view reduce,
    const c10::optional<Tensor>& lengths) {
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
      lengths_contig);
}

REGISTER_ARCH_DISPATCH(
    _segment_reduce_backward_stub,
    DEFAULT,
    &_segment_reduce_cpu_backward_kernel);
REGISTER_AVX_DISPATCH(
    _segment_reduce_backward_stub,
    &_segment_reduce_cpu_backward_kernel);
REGISTER_AVX2_DISPATCH(
    _segment_reduce_backward_stub,
    &_segment_reduce_cpu_backward_kernel);
REGISTER_VSX_DISPATCH(
    _segment_reduce_backward_stub,
    &_segment_reduce_cpu_backward_kernel);

} // namespace native
} // namespace at
