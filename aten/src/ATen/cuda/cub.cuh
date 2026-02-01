#pragma once
#include <ATen/cuda/cub.h>

#include <cstddef>
#include <type_traits>
#include <iterator>
#include <limits>

#ifndef USE_ROCM
#include <cuda/std/functional>
#endif

#include <ATen/cuda/cub_definitions.cuh>
#include <ATen/cuda/CUDAContextLight.h>

#if USE_GLOBAL_CUB_WRAPPED_NAMESPACE()

#include <cub/cub.cuh>

#else

// include cub in a safe manner, see:
// https://github.com/pytorch/pytorch/pull/55292
#undef CUB_NS_POSTFIX //undef to avoid redefinition warnings
#undef CUB_NS_PREFIX
#undef CUB_NS_QUALIFIER
#define CUB_NS_PREFIX namespace at_cuda_detail {
#define CUB_NS_POSTFIX }
#define CUB_NS_QUALIFIER ::at_cuda_detail::cub
#include <cub/cub.cuh>
#undef CUB_NS_POSTFIX
#undef CUB_NS_PREFIX
#undef CUB_NS_QUALIFIER

#endif

#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAStream.h>

// handle the temporary storage and 'twice' calls for cub API
#define CUB_WRAPPER(func, ...) do {                                       \
  size_t temp_storage_bytes = 0;                                          \
  AT_CUDA_CHECK(func(nullptr, temp_storage_bytes, __VA_ARGS__));          \
  auto& caching_allocator = *::c10::cuda::CUDACachingAllocator::get();    \
  auto temp_storage = caching_allocator.allocate(temp_storage_bytes);     \
  AT_CUDA_CHECK(func(temp_storage.get(), temp_storage_bytes, __VA_ARGS__));\
} while (false)

#ifdef USE_ROCM
#define NO_ROCM(x)
#define ROCM_HIPCUB(x) ::hipcub
#else
#define NO_ROCM(x) x
#define ROCM_HIPCUB(x) x
#endif

#if CUB_V3_PLUS()
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#define ATEN_CUB_TRANSFORM_ITERATOR(ValueType, ...) ::thrust::transform_iterator<__VA_ARGS__>
#define ATEN_CUB_COUNTING_ITERATOR(...) ::thrust::counting_iterator<__VA_ARGS__>
#define ATEN_CUB_CONSTANT_ITERATOR(...) ::thrust::constant_iterator<__VA_ARGS__>
#define ATEN_CUB_MAXIMUM() ::cuda::maximum<>()
#else
#define ATEN_CUB_TRANSFORM_ITERATOR(...) NO_ROCM(at_cuda_detail)ROCM_HIPCUB(::cub)::TransformInputIterator<__VA_ARGS__>
#define ATEN_CUB_COUNTING_ITERATOR(...) NO_ROCM(at_cuda_detail)ROCM_HIPCUB(::cub)::CountingInputIterator<__VA_ARGS__>
#define ATEN_CUB_CONSTANT_ITERATOR(...) NO_ROCM(at_cuda_detail)ROCM_HIPCUB(::cub)::ConstantInputIterator<__VA_ARGS__>
#define ATEN_CUB_MAXIMUM() NO_ROCM(at_cuda_detail)ROCM_HIPCUB(::cub)::Max()
#endif

#if defined(USE_ROCM)

// backport https://github.com/NVIDIA/cub/pull/306 for c10::BFloat16

template <>
struct ROCM_HIPCUB(cub)::FpLimits<c10::BFloat16>
{
    static __host__ __device__ __forceinline__ c10::BFloat16 Max() {
        unsigned short max_word = 0x7F7F;
        return reinterpret_cast<c10::BFloat16&>(max_word);
    }

    static __host__ __device__ __forceinline__ c10::BFloat16 Lowest() {
        unsigned short lowest_word = 0xFF7F;
        return reinterpret_cast<c10::BFloat16&>(lowest_word);
    }
};

template <>
struct ROCM_HIPCUB(cub)::NumericTraits<c10::BFloat16>:
       ROCM_HIPCUB(cub)::BaseTraits<ROCM_HIPCUB(cub)::FLOATING_POINT, true, false, unsigned short, c10::BFloat16> {};

#endif

#if !defined(USE_ROCM)
namespace at::native {
namespace cub = ::at_cuda_detail::cub;
} // namespace at::native
#endif

namespace at::cuda::cub {

namespace detail {

template<typename T>
struct cuda_type {
  using type = T;
};
template<>
struct cuda_type<c10::Half> {
  using type = __half;
};

#if !defined(USE_ROCM)

template<>
struct cuda_type<c10::BFloat16> {
  using type = __nv_bfloat16;
};

#elif defined(USE_ROCM)

template<>
struct cuda_type<c10::BFloat16> {
  using type = hip_bfloat16;
};

#endif

}  // namespace detail

template<typename key_t, typename value_t, typename OffsetIteratorT>
inline void segmented_sort_pairs(
    const key_t *keys_in, key_t *keys_out,
    const value_t *values_in, value_t *values_out,
    int64_t num_elements, int64_t num_segments,
    OffsetIteratorT begin_offsets, OffsetIteratorT end_offsets,
    bool descending=false, int64_t begin_bit=0, int64_t end_bit=sizeof(key_t)*8
) {
  TORCH_CHECK(num_elements <= std::numeric_limits<int>::max(),
    "cub sort does not support sorting more than INT_MAX elements");
  TORCH_CHECK(num_segments <= std::numeric_limits<int>::max(),
    "cub sort does not support sorting more than INT_MAX elements");
  using key_t_ = typename detail::cuda_type<key_t>::type;

  auto allocator = c10::cuda::CUDACachingAllocator::get();
  c10::DataPtr keys_out_owner;

  if (keys_out == nullptr) {
    keys_out_owner = allocator->allocate(num_elements * sizeof(key_t));
    keys_out = reinterpret_cast<key_t *>(keys_out_owner.get());
  }

  const key_t_ *keys_in_ = reinterpret_cast<const key_t_*>(keys_in);
  key_t_ *keys_out_ = reinterpret_cast<key_t_*>(keys_out);

  if (descending) {
    CUB_WRAPPER(NO_ROCM(at_cuda_detail)::cub::DeviceSegmentedRadixSort::SortPairsDescending,
      keys_in_, keys_out_, values_in, values_out,
      num_elements, num_segments, begin_offsets, end_offsets,
      begin_bit, end_bit, c10::cuda::getCurrentCUDAStream());
  } else {
    CUB_WRAPPER(NO_ROCM(at_cuda_detail)::cub::DeviceSegmentedRadixSort::SortPairs,
      keys_in_, keys_out_, values_in, values_out,
      num_elements, num_segments, begin_offsets, end_offsets,
      begin_bit, end_bit, c10::cuda::getCurrentCUDAStream());
  }
}

template <typename KeysInputIteratorT, typename ValuesInputIteratorT, typename ValuesOutputIteratorT, typename NumSelectedIteratorT>
inline void unique_by_key(
  KeysInputIteratorT keys_in, ValuesInputIteratorT values_in,
  ValuesOutputIteratorT values_out,
  NumSelectedIteratorT num_selected, int64_t num_input_items)
{
  // TODO: use thrust::discard_iterator to handle null keys_out when https://github.com/NVIDIA/cub/issues/406 is fixed.
  using KeyT = typename std::iterator_traits<KeysInputIteratorT>::value_type;
  auto allocator = c10::cuda::CUDACachingAllocator::get();
  c10::DataPtr keys_out_owner;
  keys_out_owner = allocator->allocate(num_input_items * sizeof(KeyT));
  auto keys_out_ = static_cast<KeyT *>(keys_out_owner.get());
  CUB_WRAPPER(NO_ROCM(at_cuda_detail)::cub::DeviceSelect::UniqueByKey,
    keys_in, values_in, keys_out_, values_out, num_selected, num_input_items, c10::cuda::getCurrentCUDAStream());
}

namespace impl {

template<typename InputIteratorT1, typename InputIteratorT2, typename OutputIteratorT, class ScanOpT>
C10_LAUNCH_BOUNDS_1(1)
__global__ void transform_vals(InputIteratorT1 a, InputIteratorT2 b, OutputIteratorT out, ScanOpT scan_op){
  // NOTE: out here not the final scan output, but an intermediate of the accumulation type.
  using acc_t = typename std::iterator_traits<OutputIteratorT>::value_type;
  *out = scan_op(static_cast<acc_t>(*a), static_cast<acc_t>(*b));
}

// even though cub is supposed to support tensors with int_max elements, in reality it doesn't,
// so split at int_max/2
constexpr int max_cub_size = std::numeric_limits<int>::max() / 2 + 1; // 2**30
}

// non synchronizing cub call
// even though cub is supposed to support tensors with int_max elements, in reality it doesn't,
// so split at int_max/2
template<typename InputIteratorT, typename OutputIteratorT, typename ScanOpT, int max_cub_size=impl::max_cub_size>
inline void inclusive_scan(InputIteratorT input, OutputIteratorT output, ScanOpT scan_op, int64_t num_items) {
#if defined(USE_ROCM)
  //For ROCm, use hipCUB chained iterators
  CUB_WRAPPER(NO_ROCM(detail)::hipcub::DeviceScan::InclusiveScan,
      input,
      output,
      scan_op,
      num_items,
      at::cuda::getCurrentCUDAStream());
  C10_HIP_KERNEL_LAUNCH_CHECK();
#else
  // non synchronizing cub call
  // even though cub is supposed to support tensors with int_max elements, in reality it doesn't,
  // so split at int_max/2
  int size_cub = std::min<int64_t>(num_items, max_cub_size);
  CUB_WRAPPER(NO_ROCM(at_cuda_detail)::cub::DeviceScan::InclusiveScan,
      input,
      output,
      scan_op,
      size_cub,
      at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  using input_t = typename std::iterator_traits<InputIteratorT>::value_type;
  for (int64_t i = max_cub_size; i < num_items; i += max_cub_size) {
    auto allocator = c10::cuda::CUDACachingAllocator::get();
    c10::DataPtr first_elem = allocator->allocate(sizeof(input_t));
    auto first_elem_ptr = reinterpret_cast<input_t *>(first_elem.get());

    size_cub = std::min<int64_t>(num_items - i, max_cub_size);
    impl::transform_vals<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(
        output + i - 1,
        input + i,
        first_elem_ptr,
        scan_op);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    CUB_WRAPPER(NO_ROCM(at_cuda_detail)::cub::DeviceScan::ExclusiveScan,
        input + i + 1,
        output + i,
        scan_op,
        ::at_cuda_detail::cub::FutureValue<input_t>(first_elem_ptr),
        size_cub,
        at::cuda::getCurrentCUDAStream());
  }
#endif
}

template<typename T>
struct BlockPrefixCallbackOp
{
    public:
    T running_total;

    __host__ __device__ BlockPrefixCallbackOp(T running_total) : running_total(running_total) {}

    // Callback operator to be entered by the first warp of threads in the block.
    // Thread-0 is responsible for returning a value for seeding the block-wide scan.
    __host__ __device__ T operator()(T block_aggregate)
    {
        T old_prefix = running_total;
        running_total += block_aggregate;
        return old_prefix;
    }
};

template<int BLOCK_THREADS, int ITEMS_PER_THREAD, typename T>
__global__ void final_scan_kernel(const T* d_in, T* d_out, T* agg, int64_t nelem, int iters_per_cta) {
  int64_t offset = BLOCK_THREADS * ITEMS_PER_THREAD * iters_per_cta * (int64_t)blockIdx.x;
  int64_t remaining =  nelem - offset;
  if (remaining <= 0) {
    return;
  }

  d_in += offset;
  d_out += offset;

  using BlockLoadT = ROCM_HIPCUB(at_cuda_detail::cub)::BlockLoad<T, BLOCK_THREADS, ITEMS_PER_THREAD, ROCM_HIPCUB(at_cuda_detail::cub)::BLOCK_LOAD_WARP_TRANSPOSE>;

  // Specialize BlockStore type for our thread block (uses warp-striped loads for coalescing, then transposes in shared
  // memory to a blocked arrangement)
  using BlockStoreT = ROCM_HIPCUB(at_cuda_detail::cub)::BlockStore<T, BLOCK_THREADS, ITEMS_PER_THREAD, ROCM_HIPCUB(at_cuda_detail::cub)::BLOCK_STORE_WARP_TRANSPOSE>;

  // Specialize BlockScan type for our thread block
  using BlockScanT = ROCM_HIPCUB(at_cuda_detail::cub)::BlockScan<T, BLOCK_THREADS, ROCM_HIPCUB(at_cuda_detail::cub)::BLOCK_SCAN_WARP_SCANS>;
  using BlockReduceT = ROCM_HIPCUB(at_cuda_detail::cub)::BlockReduce<T, BLOCK_THREADS>;


  // Shared memory
  __shared__ union TempStorage
  {
    typename BlockLoadT::TempStorage load;
    typename BlockStoreT::TempStorage store;
    typename BlockScanT::TempStorage scan;
    typename BlockReduceT::TempStorage reduce;
  } temp_storage;

  // load agg and reduce my starting value
  T agg_data;
  agg_data = threadIdx.x >= blockIdx.x ? T(0) : agg[threadIdx.x];
  // In case there are fewer threads than previous block aggregates to be read, add more aggregates (should be at most 2-3 aggregates per thread)
  for (unsigned int i=threadIdx.x + blockDim.x; i<blockIdx.x; i+=blockDim.x) {
    agg_data += agg[i];
  }

  T aggregate = BlockReduceT(temp_storage.reduce).Sum(agg_data);
  __syncthreads();
  BlockPrefixCallbackOp prefix_op(aggregate);


  // Per-thread tile data
  T data[ITEMS_PER_THREAD];

  for (int i=0; i<iters_per_cta; i++){
  // Load items into a blocked arrangement
    if (remaining >= BLOCK_THREADS * ITEMS_PER_THREAD) {
      BlockLoadT(temp_storage.load).Load(d_in, data);
    } else {
       #pragma unroll
       for (int j=0; j<ITEMS_PER_THREAD; j++) {
         data[j] = 0;
       }
       BlockLoadT(temp_storage.load).Load(d_in, data, remaining);
    }

    // Barrier for smem reuse
    __syncthreads();

    // Compute inclusive prefix sum
    BlockScanT(temp_storage.scan).InclusiveSum(data, data, prefix_op);

    // Barrier for smem reuse
    __syncthreads();

    // Store items from a blocked arrangement
    if (remaining >= BLOCK_THREADS * ITEMS_PER_THREAD) {
      BlockStoreT(temp_storage.store).Store(d_out, data);
    } else {
      BlockStoreT(temp_storage.store).Store(d_out, data, remaining);
    }
    d_in += BLOCK_THREADS * ITEMS_PER_THREAD;
    d_out += BLOCK_THREADS * ITEMS_PER_THREAD;
    remaining -= BLOCK_THREADS * ITEMS_PER_THREAD;
    if (remaining <= 0) return;
    __syncthreads();
  }

}

template <typename T, typename aggT, bool nonzero>
struct TransformFunctor {
  __device__ aggT operator()(T value) const {
    if constexpr (!nonzero) {
      return value;
    } else {
      return (value != T(0)) ? 1 : 0;
    }
  }
};

template<int BLOCK_THREADS, int ITEMS_PER_THREAD, bool nonzero, typename T, typename aggT>
__global__ void calc_block_sums(const T * d_in, aggT * agg, int64_t nelem, int iters_per_cta){
    int64_t offset = BLOCK_THREADS * ITEMS_PER_THREAD * iters_per_cta * (int64_t)blockIdx.x;
    int64_t remaining = nelem - offset;
    if (remaining <= 0) {
      return;
    }
    d_in += offset;

    using BlockLoadT = ROCM_HIPCUB(at_cuda_detail::cub)::BlockLoad<aggT, BLOCK_THREADS, ITEMS_PER_THREAD, ROCM_HIPCUB(at_cuda_detail::cub)::BLOCK_LOAD_STRIPED>;
    using BlockReduceT = ROCM_HIPCUB(at_cuda_detail::cub)::BlockReduce<aggT, BLOCK_THREADS>;
    // Shared memory
    __shared__ union TempStorage
    {
      typename BlockLoadT::TempStorage load;
      typename BlockReduceT::TempStorage reduce;
    } temp_storage;
    aggT data[ITEMS_PER_THREAD];
    aggT agg_val = 0;
    TransformFunctor<T, aggT, nonzero> transform_functor;
    auto iter_in = ATEN_CUB_TRANSFORM_ITERATOR(aggT, TransformFunctor<T, aggT, nonzero>, const T*)(d_in, transform_functor);
    for (int i=0; i<iters_per_cta; i++){
      if (remaining >= BLOCK_THREADS * ITEMS_PER_THREAD) {
        BlockLoadT(temp_storage.load).Load(iter_in, data);
        __syncthreads();
        agg_val += BlockReduceT(temp_storage.reduce).Sum(data);

      } else {
        BlockLoadT(temp_storage.load).Load(iter_in, data, remaining, aggT(0));
        __syncthreads();
        agg_val += BlockReduceT(temp_storage.reduce).Sum(data);
      }
      iter_in += BLOCK_THREADS * ITEMS_PER_THREAD;
      remaining -= BLOCK_THREADS * ITEMS_PER_THREAD;
      if (remaining <= 0) {
        // for nonzeros we need to write out last blocks
        // accumulated value to be able to compute
        // total number of nonzeros
        if (nonzero && threadIdx.x == 0) {
          agg[blockIdx.x] = agg_val;
        }
        return;
      }
      __syncthreads();

    }
    if (threadIdx.x == 0) {
      agg[blockIdx.x] = agg_val;
    }

}

template <typename T>
struct NonZeroOp {
  __host__ __device__ __forceinline__ int operator()(const T& a) const {
    return (a != T(0));
  }
};

template<int size>
constexpr int block_threads(){
  if constexpr (size >=16) {
    return 128;
  } else if constexpr (size >=8) {
    return 256;
  } else {
    return 512;
  }
}

template<typename scalar_t, typename ScanOpT>
inline void inclusive_deterministic_scan(const scalar_t *  input, scalar_t * output, ScanOpT scan_op, int64_t num_items) {
  static_assert(std::is_same_v<ScanOpT, std::plus<scalar_t>>, "");
  constexpr int BLOCK_THREADS = block_threads<sizeof(scalar_t)>();
  constexpr int ITEMS_PER_THREAD = 16;
  auto grid_size = (num_items + BLOCK_THREADS * ITEMS_PER_THREAD - 1) / (BLOCK_THREADS * ITEMS_PER_THREAD);
  const int64_t num_sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

  const int iters_per_cta = (grid_size + num_sms - 1)/num_sms;
  grid_size = std::min(num_sms, grid_size);
  auto& allocator = *c10::cuda::CUDACachingAllocator::get();
  auto agg = allocator.allocate(grid_size * sizeof(scalar_t));
  calc_block_sums<BLOCK_THREADS, ITEMS_PER_THREAD, false>
  <<<grid_size, BLOCK_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
    input, (scalar_t*)agg.get(), num_items, iters_per_cta);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  final_scan_kernel<BLOCK_THREADS, ITEMS_PER_THREAD>
  <<<grid_size, BLOCK_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
  input, output, (scalar_t*)agg.get(), num_items, iters_per_cta);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}


template<typename InputIteratorT, typename OutputIteratorT, typename ScanOpT, typename InitValueT, int max_cub_size=impl::max_cub_size>
inline void exclusive_scan(InputIteratorT input, OutputIteratorT output, ScanOpT scan_op, InitValueT init_value, int64_t num_items) {
#if defined(USE_ROCM)
  //For ROCm, use hipCUB chained iterators
  CUB_WRAPPER(NO_ROCM(detail)::hipcub::DeviceScan::ExclusiveScan,
      input,
      output,
      scan_op,
      init_value,
      num_items,
      at::cuda::getCurrentCUDAStream());
  C10_HIP_KERNEL_LAUNCH_CHECK();
#else
  // non synchronizing cub call
  // even though cub is supposed to support tensors with int_max elements, in reality it doesn't,
  // so split at int_max/2
  int size_cub = std::min<int64_t>(num_items, max_cub_size);
  CUB_WRAPPER(NO_ROCM(at_cuda_detail)::cub::DeviceScan::ExclusiveScan,
      input,
      output,
      scan_op,
      init_value,
      size_cub,
      at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  for (int64_t i = max_cub_size; i < num_items; i += max_cub_size) {
    auto allocator = c10::cuda::CUDACachingAllocator::get();
    c10::DataPtr first_elem = allocator->allocate(sizeof(InitValueT));
    auto first_elem_ptr = reinterpret_cast<InitValueT *>(first_elem.get());

    size_cub = std::min<int64_t>(num_items - i, max_cub_size);
    impl::transform_vals<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(
        output + i - 1,
        input + i - 1,
        first_elem_ptr,
        scan_op);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    CUB_WRAPPER(NO_ROCM(at_cuda_detail)::cub::DeviceScan::ExclusiveScan,
        input + i,
        output + i,
        scan_op,
        ::at_cuda_detail::cub::FutureValue<InitValueT>(first_elem_ptr),
        size_cub,
        at::cuda::getCurrentCUDAStream());
  }
#endif
}


template <typename KeysInputIteratorT, typename ValuesInputIteratorT, typename ValuesOutputIteratorT>
inline void inclusive_sum_by_key(KeysInputIteratorT keys, ValuesInputIteratorT input, ValuesOutputIteratorT output, int64_t num_items) {
  TORCH_CHECK(num_items <= std::numeric_limits<int>::max(),
    "cub InclusiveSumByKey does not support more than INT_MAX elements");
#if !defined(USE_ROCM)
  CUB_WRAPPER(at_cuda_detail::cub::DeviceScan::InclusiveSumByKey,
      keys, input, output, num_items, NO_ROCM(::cuda)::std::equal_to<>(), at::cuda::getCurrentCUDAStream());
#else
  CUB_WRAPPER(cub::DeviceScan::InclusiveSumByKey,
      keys, input, output, num_items, hipcub::Equality(), at::cuda::getCurrentCUDAStream());
#endif
}

template <typename KeysInputIteratorT, typename ValuesInputIteratorT, typename ValuesOutputIteratorT, typename ScanOpT>
inline void inclusive_scan_by_key(KeysInputIteratorT keys, ValuesInputIteratorT input, ValuesOutputIteratorT output, ScanOpT scan_op, int64_t num_items) {
  TORCH_CHECK(num_items <= std::numeric_limits<int>::max(),
    "cub InclusiveSumByKey does not support more than INT_MAX elements");
#if !defined(USE_ROCM)
  CUB_WRAPPER(at_cuda_detail::cub::DeviceScan::InclusiveScanByKey,
      keys, input, output, scan_op, num_items, NO_ROCM(::cuda)::std::equal_to<>(), at::cuda::getCurrentCUDAStream());
#else
  CUB_WRAPPER(cub::DeviceScan::InclusiveScanByKey,
      keys, input, output, scan_op, num_items, hipcub::Equality(), at::cuda::getCurrentCUDAStream());
#endif
}


template <typename InputIteratorT, typename OutputIteratorT, typename NumSelectedIteratorT>
void unique(InputIteratorT input, OutputIteratorT output,
            NumSelectedIteratorT num_selected_out, int64_t num_items) {
  TORCH_CHECK(num_items <= std::numeric_limits<int>::max(),
              "cub unique does not support more than INT_MAX elements");
  CUB_WRAPPER(NO_ROCM(at_cuda_detail)::cub::DeviceSelect::Unique,
              input, output, num_selected_out, num_items, at::cuda::getCurrentCUDAStream());
}

template <typename InputIteratorT, typename OutputIteratorT, typename CountsOutputIteratorT,
          typename LengthOutputIteratorT>
void run_length_encode(InputIteratorT input, OutputIteratorT output, CountsOutputIteratorT counts_out,
                       LengthOutputIteratorT length_out, int64_t num_items) {
  TORCH_CHECK(num_items <= std::numeric_limits<int>::max(),
              "cub run_length_encode does not support more than INT_MAX elements");
  CUB_WRAPPER(
      NO_ROCM(at_cuda_detail)::cub::DeviceRunLengthEncode::Encode,
      input, output, counts_out, length_out, num_items,
      at::cuda::getCurrentCUDAStream());
}

template <typename InputIteratorT, typename OutputIteratorT, typename ReductionOpT, typename T>
void reduce(InputIteratorT input, OutputIteratorT output, int64_t num_items, ReductionOpT op, T init) {
  TORCH_CHECK(num_items <= std::numeric_limits<int>::max(),
              "cub reduce does not support more than INT_MAX elements");
  CUB_WRAPPER(
      NO_ROCM(at_cuda_detail)::cub::DeviceReduce::Reduce,
      input, output, num_items, op, init,
      at::cuda::getCurrentCUDAStream());

}

}  // namespace at::cuda::cub
