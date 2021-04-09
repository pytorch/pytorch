#include <ATen/native/SegmentReduce.h>

#include <ATen/ATen.h>
#include <ATen/NumericUtils.h>

namespace at {
namespace native {

DEFINE_DISPATCH(segment_reduce_stub);

enum ReductionType { MAX };
const std::map<std::string, ReductionType> reduce2REDUCE = {
    {"max", MAX},
};

Tensor _segment_reduce_cpu(
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
      reduce2REDUCE.at(reduce) == MAX,
      "Currently only 'max' reduction is supported!");

  // length related checks
  TORCH_CHECK(
      lengths.has_value() && !indices.has_value(),
      "Currently only lengths based reduction is supported!")
  const auto& lengths_value = lengths.value();
  TORCH_CHECK(lengths_value.dim() == 1);
  TORCH_CHECK(data.get_device() == lengths_value.get_device());
  TORCH_CHECK(data.dim() >= lengths_value.dim());

  const auto lengths_contig = lengths_value.contiguous();
  const auto data_contig = data.contiguous();

  int64_t batch_size = lengths_contig.numel();
  auto output = at::empty({batch_size}, data.options());

  const auto* lengths_data = lengths_contig.data_ptr<int64_t>();
  if (!unsafe) {
    int64_t sum = 0;
    for (int64_t i = 0; i < batch_size; ++i) {
      TORCH_CHECK(lengths_data[i] > 0);
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

} // namespace native
} // namespace at
