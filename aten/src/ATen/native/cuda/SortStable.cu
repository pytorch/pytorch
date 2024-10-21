
#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/cuda/SortStable.h>

#include <ATen/Dispatch.h>
#include <ATen/core/Array.h>
#include <ATen/core/TensorBase.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/cuda/cub.cuh>
#include <ATen/cuda/detail/OffsetCalculator.cuh>
#include <ATen/native/cuda/SortUtils.cuh>
#include <ATen/native/cuda/SortingCommon.cuh>

#include <c10/core/DeviceArray.h>
#include <limits>

namespace at::native {

namespace {

struct offset_t {
  int stride;
  int begin;
  __device__ int operator[](int i) {
    return stride * (begin + i);
  }
};
// Segmented sort by full sort algorithm:.
// Say we are sorting a (2, 3) tensor. We have in flattened form:
// values       0.4 1.2 5.3 6.2 1.3 2.3
// indices        0   1   2   0   1   2
// segment_id     0   0   0   1   1   1

// First we sort by values, globally:
// values       6.2 5.3 2.3 1.2 1.3 0.4
// indices        0   2   2   1   1   0
// segment_id     1   0   1   0   1   0

// Then we stable sort by segment id:
// values       5.3 1.2 0.4 6.2 2.3 1.3
// indices        2   1   0   0   2   1
// segment_id     0   0   0   1   1   1

// This method can only work if the slice we are sorting (`dim`) is
// innermost, and both values and indices are contiguous. We do this
// by re-arranging the input into this form as needed, which will
// unfortunately allocate memory if the request is not in this form.
// Vectorized sort is slower than iterated sort if the number of
// slices is small (since we're sorting twice, instead of invoking a
// smaller sort `numSlices` times), but the cub sort
// implementation here is a catch-all, so we're not looking for
// efficiency, but instead correctness.

template <typename scalar_t>
__global__ void sort_postprocess_kernel(
    const scalar_t* in,
    scalar_t* out,
    int64_t* index,
    const int2* i_s_ptr,
    int nsegments,
    int nsort) {
  CUDA_KERNEL_LOOP(i, nsegments * nsort) {
    int segment = i / nsort;
    int j = i % nsort;

    int offset = segment * nsort;
    const scalar_t* in_ = in + offset;
    scalar_t* out_ = out + offset;
    int64_t* index_ = index + offset;
    const int2* i_s_ptr_ = i_s_ptr + offset;

    int idx = i_s_ptr_[j].y;
    index_[j] = idx;
    out_[j] = in_[idx];
  }
}

C10_LAUNCH_BOUNDS_1(at::cuda::detail::CUDA_NUM_THREADS)
__global__ void fill_index_and_segment_kernel(
    int2* data,
    int numel,
    at::cuda::detail::IntDivider<uint32_t> nsort_divider) {
  CUDA_KERNEL_LOOP(idx, numel) {
    auto div_mod = nsort_divider.divmod(idx);
    auto segment = static_cast<int>(div_mod.div);
    auto sort = static_cast<int>(div_mod.mod);
    data[idx] = int2{segment, sort};
  }
}

C10_LAUNCH_BOUNDS_1(at::cuda::detail::CUDA_NUM_THREADS)
__global__ void fill_reverse_indices_kernel(
    int64_t* data,
    int numel,
    at::cuda::detail::IntDivider<uint32_t> nsort_divider) {
  CUDA_KERNEL_LOOP(idx, numel) {
    data[idx] = nsort_divider.mod(idx);
  }
}

template <typename scalar_t>
inline void segmented_sort_large_segments(
    const int64_t nsegments,
    const int64_t nsort,
    const int64_t n,
    const bool descending,
    const scalar_t* self_ptr,
    scalar_t* values_ptr,
    int64_t* indices_ptr) {
  using namespace at::cuda::detail;
  auto allocator = at::cuda::getCUDADeviceAllocator();
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 block = CUDA_NUM_THREADS;
  dim3 grid = GET_BLOCKS(nsort);
  c10::DeviceArray<int64_t> indices(*allocator, nsort);
  at::cuda::detail::IntDivider<uint32_t> nsort_divider(nsort);
  fill_reverse_indices_kernel<<<grid, block, 0, stream>>>(
      indices.get(), nsort, nsort_divider);
  const int64_t* initial_indices = indices.get();

  for (auto i : c10::irange(nsegments)) {
    at::cuda::cub::radix_sort_pairs<scalar_t, int64_t>(
        self_ptr, values_ptr, initial_indices, indices_ptr, nsort, descending);
    indices_ptr += nsort;
    self_ptr += nsort;
    values_ptr += nsort;
  }
}

template <typename scalar_t>
inline void segmented_sort_pairs_by_full_sort(
    const int64_t nsegments,
    const int64_t nsort,
    const int64_t n,
    const bool descending,
    const scalar_t* const self_ptr,
    scalar_t* const values_ptr,
    int64_t* const indices_ptr) {
  int64_t segment_bits = std::max<int64_t>(
      1L, static_cast<int64_t>(std::ceil(::log2(nsegments))));

  const auto numel = nsort * nsegments;
  auto cuda_allocator = at::cuda::getCUDADeviceAllocator();
  auto indices_and_segment = cuda_allocator->allocate(numel * sizeof(int2));
  auto i_s_ptr = static_cast<int2*>(indices_and_segment.get());

  using namespace at::cuda::detail;
  dim3 block = CUDA_NUM_THREADS;
  dim3 grid = GET_BLOCKS(numel);
  auto stream = c10::cuda::getCurrentCUDAStream();
  at::cuda::detail::IntDivider<uint32_t> nsort_divider(nsort);
  fill_index_and_segment_kernel<<<grid, block, 0, stream>>>(
      i_s_ptr, numel, nsort_divider);

  auto indices_and_segment2 =
      cuda_allocator->allocate(nsegments * nsort * sizeof(int2));
  auto i_s_ptr2 = static_cast<int2*>(indices_and_segment2.get());

  at::cuda::cub::radix_sort_pairs<scalar_t, int2>(
      self_ptr, nullptr, i_s_ptr, i_s_ptr2, n, descending);

  TORCH_INTERNAL_ASSERT(segment_bits <= 32);

  // sort on lower 32bits, i.e. segment index
  at::cuda::cub::radix_sort_keys<int64_t>(
      reinterpret_cast<int64_t*>(i_s_ptr2),
      reinterpret_cast<int64_t*>(i_s_ptr),
      n,
      false,
      0,
      segment_bits);

  sort_postprocess_kernel<<<
      (n + 511) / 512,
      512,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
      self_ptr, values_ptr, indices_ptr, i_s_ptr, nsegments, nsort);
}

template <typename scalar_t>
void segmented_sort_pairs(
    int64_t nsegments,
    int64_t nsort,
    int64_t n,
    bool descending,
    const scalar_t* self_ptr,
    scalar_t* values_ptr,
    int64_t* indices_ptr) {
  const auto numel = nsort * nsegments;
  auto cuda_allocator = at::cuda::getCUDADeviceAllocator();
  auto reverse_indices = cuda_allocator->allocate(numel * sizeof(int64_t));
  int64_t* reverse_indices_ptr = static_cast<int64_t*>(reverse_indices.get());

  using namespace at::cuda::detail;
  dim3 block = CUDA_NUM_THREADS;
  dim3 grid = GET_BLOCKS(numel);
  auto stream = c10::cuda::getCurrentCUDAStream();
  at::cuda::detail::IntDivider<uint32_t> nsort_divider(nsort);
  fill_reverse_indices_kernel<<<grid, block, 0, stream>>>(
      reverse_indices_ptr, numel, nsort_divider);

  at::cuda::cub::segmented_sort_pairs(
      self_ptr,
      values_ptr,
      reverse_indices_ptr,
      indices_ptr,
      n,
      nsegments,
      offset_t{(int)nsort, 0},
      offset_t{(int)nsort, 1},
      descending);
}

} // namespace

void launch_stable_sort_kernel(
    const TensorBase& self,
    int64_t dim,
    bool descending,
    const TensorBase& values,
    const TensorBase& indices) {
  const auto numel = self.numel();
  if (numel == 0) {
    return;
  }

  int64_t numel_or_intmax =
      std::min(numel, static_cast<int64_t>(std::numeric_limits<int>::max()));
  int64_t nsort = self.size(dim);
  int64_t nbatch = (numel_or_intmax / nsort) * nsort;
  TORCH_CHECK(nbatch > 0, "Cannot sort dimension of length ", nsort);
  int64_t* indices_ptr = indices.mutable_data_ptr<int64_t>();

  AT_DISPATCH_ALL_TYPES_AND3(
      kBool, kHalf, kBFloat16, self.scalar_type(), "sort", [&] {
        const scalar_t* self_ptr = self.const_data_ptr<scalar_t>();
        scalar_t* values_ptr = values.mutable_data_ptr<scalar_t>();
        int64_t remaining = numel;
        while (remaining > 0) {
          int64_t n = std::min(remaining, nbatch);
          int64_t nsegments = n / nsort;

          if (nsegments == 1 ||
              nsort >= 1000000) { // rough heuristics where even a single
                                  // sort occupies GPU
            segmented_sort_large_segments(
                nsegments,
                nsort,
                n,
                descending,
                self_ptr,
                values_ptr,
                indices_ptr);
          } else if (nsegments < 128) {
            segmented_sort_pairs_by_full_sort(
                nsegments,
                nsort,
                n,
                descending,
                self_ptr,
                values_ptr,
                indices_ptr);
          } else {
            segmented_sort_pairs(
                nsegments,
                nsort,
                n,
                descending,
                self_ptr,
                values_ptr,
                indices_ptr);
          }

          remaining -= n;
          self_ptr += n;
          values_ptr += n;
          indices_ptr += n;
        }
      });
}

} // namespace at::native
