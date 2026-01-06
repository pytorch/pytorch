#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/cuda/cub.cuh>
#include <ATen/cuda/CUDAConfig.h>

namespace at::cuda::cub {

namespace {
template <typename scalar_t>
struct SumOp {
  __device__ scalar_t operator () (scalar_t a, scalar_t b) const {
    return a + b;
  }
};
}

template <typename input_t, typename output_t>
void inclusive_sum_truncating(const input_t *input, output_t *output, int64_t num_items) {
  inclusive_scan(input, output, NO_ROCM(::cuda)::std::plus<>{}, num_items);
}

template void inclusive_sum_truncating(const int32_t *input, int32_t *output, int64_t num_items);
template void inclusive_sum_truncating(const int64_t *input, int64_t *output, int64_t num_items);
template void inclusive_sum_truncating(const int32_t *input, int64_t *output, int64_t num_items);

template <typename input_t, typename output_t>
void exclusive_sum_in_common_type(const input_t *input, output_t *output, int64_t num_items) {
  using scalar_t = std::common_type_t<input_t, output_t>;
  exclusive_scan(input, output, SumOp<scalar_t>{}, scalar_t(0), num_items);
}

template void exclusive_sum_in_common_type(const int32_t *input, int32_t *output, int64_t num_items);
template void exclusive_sum_in_common_type(const int64_t *input, int64_t *output, int64_t num_items);

namespace {
struct CountMaskOp {
  __device__ int64_t operator() (const uint8_t &x) const {
    return x != 0;
  }
};
}

void mask_exclusive_sum(const uint8_t *mask, int64_t *output_idx, int64_t n) {
  CountMaskOp op{};
  auto iter = ATEN_CUB_TRANSFORM_ITERATOR(bool, decltype(op), decltype(mask))(mask, op);
  exclusive_scan(iter, output_idx, SumOp<int64_t>{}, int64_t{0}, n);
}

}  // namespace at::cuda::cub
