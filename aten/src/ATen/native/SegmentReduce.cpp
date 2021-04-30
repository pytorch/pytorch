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

Tensor _segment_reduce_cpu_kernel(
    const Tensor& data,
    const Tensor& lengths,
    int64_t axis,
    bool unsafe) {
  const auto lengths_contig = lengths.contiguous();
  const auto data_contig = data.contiguous();

  int64_t batch_size = lengths_contig.numel();
  auto output = at::empty({batch_size}, data.options());

  const auto* lengths_data = lengths_contig.data_ptr<int64_t>();
  if (!unsafe) {
    int64_t sum = 0;
    for (int64_t i = 0; i < batch_size; ++i) {
      TORCH_CHECK(
          (lengths_data[i] > 0), "lengths contains non positive value!");
      sum += lengths_data[i];
    }
    TORCH_CHECK(sum == data.numel());
  }

  AT_DISPATCH_ALL_TYPES_AND2(
      kBFloat16,
      kHalf,
      data_contig.scalar_type(),
      "_segment_reduce_cpu",
      ([&]() {
        auto* output_data = output.data_ptr<scalar_t>();
        const auto* values_data = data_contig.data_ptr<scalar_t>();
        int64_t k = 0;
        for (int64_t i = 0; i < batch_size; ++i) {
          scalar_t reduction = std::numeric_limits<scalar_t>::lowest();
          for (int64_t j = 0; j < lengths_data[i]; ++j) {
            const auto data = values_data[k];
            reduction =
                at::_isnan(data) ? data : std::max<scalar_t>(reduction, data);
            k++;
          }
          // If unsafe is false, check on lengths or indices should cover cases
          // where lengths for a particular segment is non-positive. If unsafe
          // is true, simply set to numerical limits for particular reduction
          output_data[i] = reduction;
        }
      }));

  return output;
}

Tensor _segment_reduce_cpu_backward_kernel(
    const Tensor& grad_contig,
    const Tensor& output_contig,
    const Tensor& data_contig,
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
          int64_t counter = 0;
          for (int64_t j = 0; j < lengths_data[i]; ++j) {
            if (at::_isnan(values_data[k]) ||
                values_data[k] == output_data[i]) {
              grad_input_data[k] = grad_data[i];
              counter++;
            }
            k++;
          }
          // Average gradient output based on number of maximum in the segment
          TORCH_INTERNAL_ASSERT(counter > 0);
          if (counter < 2) {
            continue;
          }
          for (int64_t j = 0; j < lengths_data[i]; ++j) {
            int64_t index = k - j - 1;
            if (grad_input_data[index] > 0) {
              grad_input_data[index] = grad_input_data[index] / counter;
            }
          }
        }
      }));

  return grad_input;
}

} // namespace

enum SegmentReductionType { MAX };
static const std::map<std::string, SegmentReductionType> segmentReduce2REDUCE =
    {
        {"max", MAX},
};

Tensor segment_reduce_kernel(
    const Tensor& data,
    std::string reduce,
    const c10::optional<Tensor>& lengths,
    const c10::optional<Tensor>& indices,
    int64_t axis,
    bool unsafe) {
  axis = maybe_wrap_dim(axis, data.ndimension());
  TORCH_CHECK(axis == 0, "Currently only dim=0 is supported!");
  TORCH_CHECK(data.dim() == 1);
  TORCH_CHECK(data.numel() > 0);
  TORCH_CHECK(
      at::native::segmentReduce2REDUCE.at(reduce) == MAX,
      "Currently only 'max' reduction is supported!");

  // length related checks
  TORCH_CHECK(
      lengths.has_value() && !indices.has_value(),
      "Currently only lengths based reduction is supported!")
  const auto& lengths_value = lengths.value();
  TORCH_CHECK(lengths_value.dim() == 1);
  TORCH_CHECK(data.get_device() == lengths_value.get_device());
  TORCH_CHECK(data.dim() >= lengths_value.dim());

  return _segment_reduce_stub(
      data.device().type(), data, lengths_value, axis, unsafe);
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
Tensor segment_reduce_backward_kernel(
    const Tensor& grad,
    const Tensor& output,
    const Tensor& data,
    const c10::optional<Tensor>& lengths) {
  TORCH_CHECK(
      lengths.has_value(),
      "Currently only lengths based reduction is supported!")
  const auto& lengths_value = lengths.value();

  const auto grad_contig = grad.contiguous();
  const auto output_contig = output.contiguous();
  const auto data_contig = data.contiguous();
  const auto lengths_contig = lengths_value.contiguous();

  return _segment_reduce_backward_stub(
      grad_contig.device().type(),
      grad_contig,
      output_contig,
      data_contig,
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
