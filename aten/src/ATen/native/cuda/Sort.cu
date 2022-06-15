#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/cuda/Sort.h>
#include <ATen/core/TensorBase.h>
#include <ATen/core/Array.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/cub.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/cuda/detail/OffsetCalculator.cuh>
#include <ATen/native/cuda/SortUtils.cuh>
#include <ATen/native/cuda/SortingCommon.cuh>

#include <limits>
#include <c10/core/DeviceArray.h>

namespace at { namespace native {

// In alignment with default sort on a c++ map, this function
// will permute key and value tensors identically, and
// in such a way that the 'key' tensor is ordered numerically
void sortKeyValueInplace(const TensorBase& key,
                         const TensorBase& value,
                         int dim, bool dir) {
  TORCH_CHECK(key.sizes() == value.sizes(),
              "Key tensor must have same size as value tensor");
  int dims = value.dim();
  TORCH_CHECK(dims <= MAX_DIMS, "value tensor has too many dimensions");
  // if key and value tensors have the same size, we do not need to check both

  ptrdiff_t inElements = key.numel();

  if (inElements == 0) {
    return;
  }

  int64_t keySliceSize = key.size(dim);
  ptrdiff_t keySlices = inElements / keySliceSize;

  // The amount of shared memory and block size is based on
  // 2^ceil(lg(n)); we choose that sorting implementation for a given
  // size.
  int64_t ceilPowerOf2 = nextHighestPowerOf2(keySliceSize);

  // FIXME: We'd have to find some other trick with Thrust to perform a
  // vectorized (key, value) sort by slice segment
  TORCH_INTERNAL_ASSERT(ceilPowerOf2 <= 2048, "sortKeyValueInplace only works for sizes <= 2048 at present");

  const auto stream = c10::cuda::getCurrentCUDAStream();

#define HANDLE_CASE(TYPE, A, SIZE, BATCH)                               \
  do {                                                                  \
    constexpr int items_per_thread = 2;                                 \
    static_assert(SIZE % items_per_thread == 0, "");                    \
    constexpr int block_x = SIZE / items_per_thread;                    \
    constexpr int block_y = BATCH;                                      \
    dim3 block(block_x, block_y);                                       \
                                                                        \
    dim3 grid;                                                          \
    const int grid_count = (keySlices + BATCH - 1) / BATCH;             \
    TORCH_INTERNAL_ASSERT(getGridFromTiles(grid_count, grid),           \
                          "Too many slices to sort");                   \
                                                                        \
    if (dir) {                                                          \
      bitonicSortKVInPlace<A, -1, block_x, block_y>                     \
        <<<grid, block, 0, stream>>>(                                   \
          keyInfo,                                                      \
          (TYPE) keySlices,                                             \
          (TYPE) keySliceSize,                                          \
          (TYPE) keyInfo.strides[collapseKeyDim],                       \
          valueInfo,                                                    \
          (TYPE) valueInfo.strides[collapseValueDim],                   \
          GTOp<scalar_t, true>());                                      \
      C10_CUDA_KERNEL_LAUNCH_CHECK();                                   \
    } else {                                                            \
      bitonicSortKVInPlace<A, -1, block_x, block_y>                     \
        <<<grid, block, 0, stream>>>(                                   \
          keyInfo,                                                      \
          (TYPE) keySlices,                                             \
          (TYPE) keySliceSize,                                          \
          (TYPE) keyInfo.strides[collapseKeyDim],                       \
          valueInfo,                                                    \
          (TYPE) valueInfo.strides[collapseValueDim],                   \
          LTOp<scalar_t, true>());                                      \
      C10_CUDA_KERNEL_LAUNCH_CHECK();                                   \
    }                                                                   \
  } while (0)

#define HANDLE_SORT_CASE(TYPE, A)                       \
  {                                                     \
    switch (ceilPowerOf2) {                             \
      case 2048:                                        \
        HANDLE_CASE(TYPE, A, 2048, 1);                  \
        break;                                          \
      case 1024:                                        \
      case 512:                                         \
      case 256:                                         \
        HANDLE_CASE(TYPE, A, 1024, 1);                  \
        break;                                          \
      case 128:                                         \
      case 64:                                          \
        HANDLE_CASE(TYPE, A, 128, 4);                   \
        break;                                          \
      case 32:                                          \
      case 16:                                          \
      case 8:                                           \
      case 4:                                           \
      case 2:                                           \
        HANDLE_CASE(TYPE, A, 32, 16);                   \
        break;                                          \
      case 1:                                           \
        /* Nothing to do, data already sorted */        \
        break;                                          \
      default:                                          \
        TORCH_INTERNAL_ASSERT(false);                   \
    }                                                   \
  }

  // The constructed key/value tensor info is used to select the slice
  // we are sorting on a per-block basis
  // The constructed key/value tensor info is used to select the slice
  // we are sorting on a per-block basis
  AT_DISPATCH_ALL_TYPES_AND3(at::ScalarType::Half, at::ScalarType::BFloat16, at::ScalarType::Bool, key.scalar_type(), "sortKeyValueInplace", [&]  {
    if (at::cuda::detail::canUse32BitIndexMath(key)) {
      at::cuda::detail::TensorInfo<scalar_t, unsigned int> keyInfo =
        at::cuda::detail::getTensorInfo<scalar_t, unsigned int>(key);
      at::cuda::detail::TensorInfo<int64_t, unsigned int> valueInfo =
        at::cuda::detail::getTensorInfo<int64_t, unsigned int>(value);

      auto strideKey = keyInfo.strides[dim];
      keyInfo.sizes[dim] = 1;
      int collapseKeyDim = keyInfo.collapseDims(dim);
      keyInfo.strides[collapseKeyDim] = strideKey;
      auto strideValue = valueInfo.strides[dim];
      valueInfo.sizes[dim]=1;
      int collapseValueDim = valueInfo.collapseDims(dim);
      valueInfo.strides[collapseValueDim] = strideValue;

      if (keyInfo.isContiguous()) {
        HANDLE_SORT_CASE(unsigned int, -2);
      } else {
        switch (keyInfo.dims) {
          case 2:
            HANDLE_SORT_CASE(unsigned int, 2);
            break;
          default:
            HANDLE_SORT_CASE(unsigned int, -1);
            break;
        }
      }

    } else {
      at::cuda::detail::TensorInfo<scalar_t, uint64_t> keyInfo =
        at::cuda::detail::getTensorInfo<scalar_t, uint64_t>(key);
      at::cuda::detail::TensorInfo<int64_t, uint64_t> valueInfo =
        at::cuda::detail::getTensorInfo<int64_t, uint64_t>(value);

      auto strideKey = keyInfo.strides[dim];
      keyInfo.sizes[dim] = 1;
      int collapseKeyDim = keyInfo.collapseDims(dim);
      keyInfo.strides[collapseKeyDim] = strideKey;
      auto strideValue = valueInfo.strides[dim];
      valueInfo.sizes[dim]=1;
      int collapseValueDim = valueInfo.collapseDims(dim);
      valueInfo.strides[collapseValueDim] = strideValue;

      // int64_t case is rare, just instantiate the generic version
      HANDLE_SORT_CASE(uint64_t, -1);
    }
  });
#undef HANDLE_CASE
#undef HANDLE_SORT_CASE
#undef HANDLE_A_CASE
}

namespace {

struct offset_t {
  int stride;
  int begin;
  __device__ int operator[](int i) {
    return stride * (begin + i);
  }
};

}

namespace {

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

template<typename scalar_t>
__global__ void sort_postprocess_kernel(const scalar_t *in, scalar_t *out, int64_t *index, const int2 *i_s_ptr, int nsegments, int nsort) {
  CUDA_KERNEL_LOOP(i, nsegments * nsort) {
    int segment = i / nsort;
    int j = i % nsort;

    int offset = segment * nsort;
    const scalar_t *in_ = in + offset;
    scalar_t *out_ = out + offset;
    int64_t *index_ = index + offset;
    const int2 *i_s_ptr_ = i_s_ptr + offset;

    int idx = i_s_ptr_[j].y;
    index_[j] = idx;
    out_[j] = in_[idx];
  }
}


C10_LAUNCH_BOUNDS_1(at::cuda::detail::CUDA_NUM_THREADS)
__global__ void fill_index_and_segment_kernel(
    int2 *data, int numel, at::cuda::detail::IntDivider<uint32_t> nsort_divider) {
  CUDA_KERNEL_LOOP(idx, numel) {
    auto div_mod = nsort_divider.divmod(idx);
    auto segment = static_cast<int>(div_mod.div);
    auto sort = static_cast<int>(div_mod.mod);
    data[idx] = int2{segment, sort};
  }
}

C10_LAUNCH_BOUNDS_1(at::cuda::detail::CUDA_NUM_THREADS)
__global__ void fill_reverse_indices_kernel(
    int64_t *data, int numel, at::cuda::detail::IntDivider<uint32_t> nsort_divider) {
  CUDA_KERNEL_LOOP(idx, numel) {
    data[idx] = nsort_divider.mod(idx);
  }
}

template<typename scalar_t>
inline void segmented_sort_large_segments(
    const int64_t nsegments, const int64_t nsort, const int64_t n, const bool descending,
    const scalar_t * self_ptr, scalar_t * values_ptr, int64_t * indices_ptr
  ) {
  using namespace at::cuda::detail;
  auto allocator = at::cuda::getCUDADeviceAllocator();
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 block = CUDA_NUM_THREADS;
  dim3 grid = GET_BLOCKS(nsort);
  c10::DeviceArray<int64_t> indices(*allocator, nsort);
  at::cuda::detail::IntDivider<uint32_t> nsort_divider(nsort);
  fill_reverse_indices_kernel<<<grid, block, 0, stream>>>(
      indices.get(), nsort, nsort_divider);
  const int64_t *initial_indices = indices.get();

  for (auto i: c10::irange(nsegments)){
    at::cuda::cub::radix_sort_pairs<scalar_t, int64_t>(
        self_ptr, values_ptr, initial_indices, indices_ptr,
        nsort, descending);
    indices_ptr += nsort;
    self_ptr += nsort;
    values_ptr += nsort;
  }
}

template<typename scalar_t>
inline void segmented_sort_pairs_by_full_sort(
  const int64_t nsegments, const int64_t nsort, const int64_t n, const bool descending,
  const scalar_t *const self_ptr, scalar_t *const values_ptr, int64_t *const indices_ptr
) {
  int64_t segment_bits = std::max<int64_t>(1L, static_cast<int64_t>(std::ceil(std::log2(nsegments))));

  const auto numel = nsort * nsegments;
  auto cuda_allocator = at::cuda::getCUDADeviceAllocator();
  auto indices_and_segment = cuda_allocator->allocate(numel * sizeof(int2));
  auto i_s_ptr = static_cast<int2 *>(indices_and_segment.get());

  using namespace at::cuda::detail;
  dim3 block = CUDA_NUM_THREADS;
  dim3 grid = GET_BLOCKS(numel);
  auto stream = c10::cuda::getCurrentCUDAStream();
  at::cuda::detail::IntDivider<uint32_t> nsort_divider(nsort);
  fill_index_and_segment_kernel<<<grid, block, 0, stream>>>(
      i_s_ptr, numel, nsort_divider);

  auto indices_and_segment2 = cuda_allocator->allocate(nsegments * nsort * sizeof(int2));
  auto i_s_ptr2 = static_cast<int2 *>(indices_and_segment2.get());

  at::cuda::cub::radix_sort_pairs<scalar_t, int2>(
    self_ptr, nullptr, i_s_ptr, i_s_ptr2,
    n, descending);

  TORCH_INTERNAL_ASSERT(segment_bits <= 32);

  // sort on lower 32bits, i.e. segment index
  at::cuda::cub::radix_sort_keys<int64_t>(
    reinterpret_cast<int64_t *>(i_s_ptr2), reinterpret_cast<int64_t *>(i_s_ptr),
    n, false, 0, segment_bits);

  sort_postprocess_kernel<<<(n + 511) / 512, 512, 0, at::cuda::getCurrentCUDAStream()>>>(
    self_ptr, values_ptr, indices_ptr, i_s_ptr, nsegments, nsort);
}

template<typename scalar_t>
void segmented_sort_pairs(
    int64_t nsegments, int64_t nsort, int64_t n, bool descending,
    const scalar_t *self_ptr, scalar_t *values_ptr, int64_t *indices_ptr) {
  const auto numel = nsort * nsegments;
  auto cuda_allocator = at::cuda::getCUDADeviceAllocator();
  auto reverse_indices = cuda_allocator->allocate(numel * sizeof(int64_t));
  int64_t *reverse_indices_ptr = static_cast<int64_t *>(reverse_indices.get());

  using namespace at::cuda::detail;
  dim3 block = CUDA_NUM_THREADS;
  dim3 grid = GET_BLOCKS(numel);
  auto stream = c10::cuda::getCurrentCUDAStream();
  at::cuda::detail::IntDivider<uint32_t> nsort_divider(nsort);
  fill_reverse_indices_kernel<<<grid, block, 0, stream>>>(
      reverse_indices_ptr, numel, nsort_divider);

  at::cuda::cub::segmented_sort_pairs(self_ptr, values_ptr,
                                      reverse_indices_ptr, indices_ptr, n, nsegments,
                                      offset_t{(int)nsort, 0}, offset_t{(int)nsort, 1}, descending);
}

}  // namespace

void launch_stable_sort_kernel(
    const TensorBase &self, int64_t dim, bool descending,
    const TensorBase &values, const TensorBase &indices) {
  const auto numel = self.numel();
  if (numel == 0) {
    return;
  }

  int64_t numel_or_intmax = std::min(numel, static_cast<int64_t>(std::numeric_limits<int>::max()));
  int64_t nsort = self.size(dim);
  int64_t nbatch = (numel_or_intmax / nsort) * nsort;
  TORCH_CHECK(nbatch > 0, "Cannot sort dimension of length ", nsort);
  int64_t *indices_ptr = indices.data_ptr<int64_t>();

#if (defined(USE_ROCM) && ROCM_VERSION < 40500)
  constexpr bool is_rocm_bf16_sort_unsupported = true;
#else
  constexpr bool is_rocm_bf16_sort_unsupported = false;
#endif

  AT_DISPATCH_ALL_TYPES_AND3(kBool, kHalf, kBFloat16, self.scalar_type(), "sort", [&]{
    c10::guts::if_constexpr<!(is_rocm_bf16_sort_unsupported && std::is_same<scalar_t, c10::BFloat16>::value)>([&](auto _){
      const scalar_t *self_ptr = self.data_ptr<scalar_t>();
      scalar_t *values_ptr = values.data_ptr<scalar_t>();
      int64_t remaining = _(numel);
      while (remaining > 0) {
        int64_t n = std::min(remaining, nbatch);
        int64_t nsegments = n / nsort;

        if (nsegments == 1 || nsort >= 1000000) { //rough heuristics where even a single sort occupies GPU
          segmented_sort_large_segments(
              nsegments, nsort, n, descending,
              self_ptr, values_ptr, indices_ptr);
        } else if (nsegments < 128) {
          segmented_sort_pairs_by_full_sort(nsegments, nsort, n, descending,
            self_ptr, values_ptr, indices_ptr);
        } else {
          segmented_sort_pairs(nsegments, nsort, n, descending,
                               self_ptr, values_ptr, indices_ptr);
        }

        remaining -= n;
        self_ptr += n;
        values_ptr += n;
        indices_ptr += n;
      }
    }, [&](auto _){ TORCH_CHECK(_(false), "BFloat16 is not supported on ROCm < 4.5"); });
  });
}

}}  // namespace at::native
