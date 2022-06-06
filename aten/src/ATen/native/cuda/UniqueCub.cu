#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/cuda/UniqueCub.cuh>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/cub.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/arange.h>
#include <ATen/ops/empty.h>
#endif

namespace at {
namespace native {
namespace internal {

namespace {

template <typename scalar_t>
__global__ void adjacent_difference_kernel(
    int64_t n,
    const scalar_t* input,
    int* output) {
  CUDA_KERNEL_LOOP(i, n) {
    output[i] = i > 0 ? input[i] != input[i - 1] : 0;
  }
}

__global__ void scatter_kernel(
    int64_t n,
    const int64_t* input,
    const int64_t* indices,
    int64_t* output) {
  CUDA_KERNEL_LOOP(i, n) {
    output[indices[i]] = input[i];
  }
}

// A variation of compute_unique (defined in Unique.cu) that doesn't allow
// customizing equal and not_equal (CUB doesn't allow them).
template <typename scalar_t>
std::tuple<Tensor, Tensor, Tensor, int64_t> compute_unique(
    const Tensor& sorted,
    const Tensor& sorted_indices,
    const bool return_inverse,
    const bool return_counts,
    const bool consecutive) {
  int64_t num_inp = sorted.numel();
  TORCH_CHECK(
      num_inp <= INT_MAX, "num_inp ", num_inp, " is too big to for CUB");
  auto options = sorted.options().dtype(kLong);
  const scalar_t* data = sorted.data_ptr<scalar_t>();
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // inverse indices
  Tensor inverse_indices;
  if (!return_inverse || num_inp == 0) {
    inverse_indices = at::empty({0}, options);
  } else {
    inverse_indices = at::empty({num_inp}, options);
    Tensor inv_loc = consecutive ? at::empty({num_inp}, options.dtype(kInt))
                                 : inverse_indices;
    int* inv_loc_ptr = static_cast<int*>(inv_loc.data_ptr());
    const dim3 block =
        dim3(std::min(static_cast<int64_t>(cuda::getApplyBlock().x), num_inp));
    dim3 grid;
    int curDevice = -1;
    cudaGetDevice(&curDevice);
    cuda::getApplyGrid(num_inp, grid, curDevice);
    adjacent_difference_kernel<scalar_t>
        <<<grid, block, 0, stream>>>(num_inp, data, inv_loc_ptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    Tensor inv_loc_out =
        consecutive ? inverse_indices : at::empty({num_inp}, options);
    at::cuda::cub::inclusive_sum_truncating(
        inv_loc_ptr,
        inv_loc_out.data_ptr<int64_t>(),
        num_inp);

    if (!consecutive) {
      TORCH_INTERNAL_ASSERT(
          sorted_indices.defined(),
          "return_inverse is set to true, but sorted_indices is undefined. Send a bug report!");
      scatter_kernel<<<grid, block, 0, stream>>>(
          num_inp,
          inv_loc_out.data_ptr<int64_t>(),
          sorted_indices.data_ptr<int64_t>(),
          inverse_indices.data_ptr<int64_t>());
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
  }

  // unique and count
  Tensor data_out = at::empty({num_inp}, sorted.options());
  Tensor counts = at::empty({0}, options);
  Tensor length = at::empty({1}, options);
  int64_t num_out;
  if (!return_counts) {
    cuda::cub::unique(data, data_out.data_ptr<scalar_t>(), length.data_ptr<int64_t>(), num_inp);
    num_out = length.item<int64_t>();
  } else {
    counts.resize_(num_inp);
    at::cuda::cub::run_length_encode(
        data,
        data_out.data_ptr<scalar_t>(),
        counts.data_ptr<int64_t>(),
        length.data_ptr<int64_t>(),
        num_inp);
    num_out = length.item<int64_t>();
    counts.resize_(num_out);
  }

  return std::tuple<Tensor, Tensor, Tensor, int64_t>(
      data_out, inverse_indices, counts, num_out);
}

} // namespace

// This function (and compute_unique above) are defined in a separate file from
// Unique.cu because for now ATen/cuda/cub.cuh can't be used together with
// thrust in the same compilation unit.
template <typename scalar_t>
std::tuple<Tensor, Tensor, Tensor> unique_cuda_template(
    const Tensor& self,
    const bool consecutive,
    const bool return_inverse,
    const bool return_counts) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  auto options = self.options().dtype(kLong);
  int64_t num_inp = self.numel();
  Tensor sorted;
  Tensor self_c = self.contiguous();
  if (consecutive) {
    sorted = self_c;
  } else {
    sorted = at::empty({num_inp}, self.options());
  }
  scalar_t* sorted_data = sorted.data_ptr<scalar_t>();

  Tensor sorted_indices;
  if (!return_inverse) {
    if (!consecutive) {
      cuda::cub::radix_sort_keys(self_c.data_ptr<scalar_t>(), sorted_data, num_inp);
    }
  } else {
    if (!consecutive) {
      Tensor range = at::arange(0, num_inp, options);
      sorted_indices = at::empty({num_inp}, options);
      cuda::cub::radix_sort_pairs(
          self_c.data_ptr<scalar_t>(),
          sorted_data,
          range.data_ptr<int64_t>(),
          sorted_indices.data_ptr<int64_t>(),
          num_inp);
    }
  }

  Tensor output, inverse_indices, counts;
  int64_t num_out;
  std::tie(output, inverse_indices, counts, num_out) = compute_unique<scalar_t>(
      sorted, sorted_indices, return_inverse, return_counts, consecutive);
  output.resize_(num_out);

  if (return_inverse) {
    inverse_indices.resize_(self.sizes());
  }

  return std::tuple<Tensor, Tensor, Tensor>(output, inverse_indices, counts);
}

#define INSTANTIATE_UNIQUE_CUDA_TEMPLATE(TYPE)                            \
  template std::tuple<Tensor, Tensor, Tensor> unique_cuda_template<TYPE>( \
      const Tensor& self,                                                 \
      const bool consecutive,                                             \
      const bool return_inverse,                                          \
      const bool return_counts)

INSTANTIATE_UNIQUE_CUDA_TEMPLATE(uint8_t);
INSTANTIATE_UNIQUE_CUDA_TEMPLATE(int8_t);
INSTANTIATE_UNIQUE_CUDA_TEMPLATE(double);
INSTANTIATE_UNIQUE_CUDA_TEMPLATE(float);
INSTANTIATE_UNIQUE_CUDA_TEMPLATE(int32_t);
INSTANTIATE_UNIQUE_CUDA_TEMPLATE(int64_t);
INSTANTIATE_UNIQUE_CUDA_TEMPLATE(int16_t);
INSTANTIATE_UNIQUE_CUDA_TEMPLATE(bool);
INSTANTIATE_UNIQUE_CUDA_TEMPLATE(at::Half);

#undef INSTANTIATE

} // namespace internal
} // namespace native
} // namespace at
