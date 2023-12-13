#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

constexpr int64_t BYTES_PER_THREAD = 16;
constexpr int64_t MAX_NUM_PARAMS = 1024;
constexpr int64_t MAX_NUM_THREADS = 1024;
constexpr int64_t WARP_SIZE = 32;

template <typename T>
__device__ inline void streamLoad128(uint4& val, const T* addr) {
#if defined(USE_ROCM) || (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800))
  CUDA_KERNEL_ASSERT(false);
#else
  unsigned long long int low, high;
  asm("ld.global.nc.v2.u64 {%0, %1}, [%2];"
      : "=l"(low), "=l"(high)
      : "l"(addr));
  reinterpret_cast<unsigned long long int*>(&val)[0] = low;
  reinterpret_cast<unsigned long long int*>(&val)[1] = high;
#endif
}

template <typename T>
__device__ inline void streamStore128(T* addr, const uint4& val) {
#if defined(USE_ROCM) || (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800))
  CUDA_KERNEL_ASSERT(false);
#else
  unsigned long long int low, high;
  low = reinterpret_cast<const unsigned long long int*>(&val)[0];
  high = reinterpret_cast<const unsigned long long int*>(&val)[1];
  asm("st.global.cs.v2.u64 [%0], {%1, %2};" : : "l"(addr), "l"(low), "l"(high));
#endif
}

template <typename T>
__device__ inline void streamLoad64(uint2& val, const T* addr) {
  unsigned long long int* valPtr =
      reinterpret_cast<unsigned long long int*>(&val);
  asm("ld.global.nc.u64 %0, [%1];" : "=l"(*valPtr) : "l"(addr));
}

template <typename T>
__device__ inline void streamStore64(T* addr, const uint2& val) {
  unsigned long long int data;
  data = reinterpret_cast<const unsigned long long int*>(&val)[0];
  asm("st.global.cs.u64 [%0], %1;" : : "l"(addr), "l"(data));
}

static inline int64_t divUp(int64_t a, int64_t b) {
  return (a + b - 1) / b;
}

///////////////////////////////////////////////////////////////////////////////
// Requires shards to be 128-bit aligned with no gaps in-between shards.
///////////////////////////////////////////////////////////////////////////////
template <typename T>
static __global__ void fsdpAllGatherCopyOutKernel(
    T** paramPtrs,
    T* allGatherResPtr,
    int64_t numel,
    int64_t* shardDimCumSums,
    int64_t numParams,
    int64_t worldSize,
    int64_t shardDimSum) {
  const auto numelPerThread = BYTES_PER_THREAD / sizeof(T);
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  const auto srcOff = tid * numelPerThread;
  const auto rankOff = srcOff % shardDimSum; // Offset within the rank
  const auto rank = srcOff / shardDimSum;

  __shared__ int64_t dimCumSums[MAX_NUM_PARAMS + 1];
  if (threadIdx.x < numParams + 1) {
    dimCumSums[threadIdx.x] = shardDimCumSums[threadIdx.x];
  }
  __syncthreads();

  int paramIdx = 0;
  for (size_t i = 1; i < numParams; ++i) {
    // Threads in a warp are likely to take the same branch.
    // So branching is beneficial here especially with large numParams.
    if (rankOff < dimCumSums[i]) {
      break;
    }
    paramIdx += 1;
  }
  const auto shardBegin = dimCumSums[paramIdx];
  const auto shardEnd = dimCumSums[paramIdx + 1];
  const auto shardLen = shardEnd - shardBegin;
  const auto paramOff = shardLen * rank + rankOff - shardBegin;

  if (srcOff < numel) {
    uint4 val;
    streamLoad128(val, allGatherResPtr + srcOff);
    streamStore128(&paramPtrs[paramIdx][paramOff], val);
  }
}

void fsdpAllGatherCopyOut(
    std::vector<at::Tensor> params,
    at::Tensor allGatherRes,
    int64_t worldSize) {
  const auto numelPerThread = BYTES_PER_THREAD / allGatherRes.element_size();
  const auto device = allGatherRes.device();
  const auto scalarType = allGatherRes.scalar_type();

  TORCH_CHECK(allGatherRes.is_cuda());
  TORCH_CHECK(allGatherRes.is_non_overlapping_and_dense());
  TORCH_CHECK(allGatherRes.numel() % worldSize == 0);
  TORCH_CHECK(params.size() <= MAX_NUM_PARAMS);

  std::vector<void*> paramPtrs;
  std::vector<int64_t> shardDimCumSums{0};
  for (size_t i = 0; i < params.size(); ++i) {
    const auto& param = params[i];
    TORCH_CHECK(param.is_non_overlapping_and_dense());
    TORCH_CHECK(param.device() == device);
    TORCH_CHECK(param.scalar_type() == scalarType);
    TORCH_CHECK(
        param.numel() % (worldSize * numelPerThread) == 0,
        "Shard must be 128-bit aligned");
    paramPtrs.push_back(param.data_ptr());
    shardDimCumSums.push_back(shardDimCumSums[i] + param.numel() / worldSize);
  }

  TORCH_CHECK(
      shardDimCumSums.back() * worldSize == allGatherRes.numel(),
      "allGatherRes and params must contain the same number of elements.");

  auto packed = at::empty(
      {static_cast<int64_t>(paramPtrs.size() + shardDimCumSums.size())},
      at::TensorOptions().dtype(at::kLong).pinned_memory(true));
  memcpy(
      packed.data_ptr(), paramPtrs.data(), sizeof(int64_t) * paramPtrs.size());
  memcpy(
      packed.data_ptr<int64_t>() + paramPtrs.size(),
      shardDimCumSums.data(),
      sizeof(int64_t) * shardDimCumSums.size());
  packed = packed.to(device, /*non_blocking=*/true);
  auto paramPtrsDev = packed.data_ptr();
  auto shardDimCumSumsDev = packed.data_ptr<int64_t>() + paramPtrs.size();

  dim3 blocks(0, 1, 1);
  dim3 threads(0, 1, 1);

  auto numThreadsRequired = allGatherRes.numel() / numelPerThread;
  if (numThreadsRequired <= MAX_NUM_THREADS) {
    blocks.x = 1;
    threads.x = divUp(numThreadsRequired, WARP_SIZE) * WARP_SIZE;
  } else {
    blocks.x = divUp(numThreadsRequired, MAX_NUM_THREADS);
    threads.x = MAX_NUM_THREADS;
  }

  TORCH_WARN_ONCE("blocks: ", blocks.x, ", threads: ", threads.x);

  AT_DISPATCH_ALL_TYPES_AND(
      at::ScalarType::BFloat16, scalarType, "fsdp_all_gather_copy_out", [&] {
        fsdpAllGatherCopyOutKernel<scalar_t>
            <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                reinterpret_cast<scalar_t**>(paramPtrsDev),
                allGatherRes.data_ptr<scalar_t>(),
                allGatherRes.numel(),
                shardDimCumSumsDev,
                params.size(),
                worldSize,
                shardDimCumSums.back());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
}

///////////////////////////////////////////////////////////////////////////////
// No alignment requirements on shards.
// Requires all_gather input to be prepared with fsdp_all_gather_copy_in
// which aligns shards at 128-bit. There will be gaps in-between shards.
///////////////////////////////////////////////////////////////////////////////
template <typename T>
static __global__ void fsdpAllGatherCopyOutKernel_no_align(
    T** paramPtrs,
    T* allGatherResPtr,
    int64_t numel,
    int64_t* alignedDimCumSums,
    int64_t* unalignedDims,
    int64_t numParams,
    int64_t worldSize,
    int64_t shardDimSum) {
  const auto numelPerThread = BYTES_PER_THREAD / sizeof(T);
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  const auto srcOff = tid * numelPerThread;
  const auto rankOff = srcOff % shardDimSum; // Offset within the rank
  const auto rank = srcOff / shardDimSum;

  if (srcOff >= numel) {
    return;
  }

  __shared__ int64_t dimCumSums[MAX_NUM_PARAMS + 1];
  __shared__ int64_t unalignedDims_[MAX_NUM_PARAMS];
  if (threadIdx.x < numParams + 1) {
    dimCumSums[threadIdx.x] = alignedDimCumSums[threadIdx.x];
    unalignedDims_[threadIdx.x] = unalignedDims[threadIdx.x];
  }
  __syncthreads();

  int paramIdx = 0;
  for (size_t i = 1; i < numParams; ++i) {
    // Threads in a warp are likely to take the same branch.
    // So branching is beneficial here especially with large numParams.
    if (rankOff < dimCumSums[i]) {
      break;
    }
    paramIdx += 1;
  }

  const auto alignedShardBegin = dimCumSums[paramIdx];
  const auto unalignedShardLen = unalignedDims_[paramIdx];
  const auto paramOff = unalignedShardLen * rank + rankOff - alignedShardBegin;
  const auto paramShardEnd = unalignedShardLen * (rank + 1);

  for (size_t i = 0; i < numelPerThread; ++i) {
    if (paramOff + i >= paramShardEnd) {
      break;
    }
    paramPtrs[paramIdx][paramOff + i] = allGatherResPtr[srcOff + i];
  }
}

void fsdpAllGatherCopyOut_no_align(
    std::vector<at::Tensor> params,
    at::Tensor allGatherRes,
    int64_t worldSize) {
  const auto numelPerThread = BYTES_PER_THREAD / allGatherRes.element_size();
  const auto device = allGatherRes.device();
  const auto scalarType = allGatherRes.scalar_type();

  TORCH_CHECK(allGatherRes.is_cuda());
  TORCH_CHECK(allGatherRes.is_non_overlapping_and_dense());
  TORCH_CHECK(allGatherRes.numel() % worldSize == 0);
  TORCH_CHECK(params.size() <= MAX_NUM_PARAMS);

  std::vector<void*> paramPtrs;
  std::vector<int64_t> alignedDimCumSums{0};
  std::vector<int64_t> unalignedDims;
  for (size_t i = 0; i < params.size(); ++i) {
    const auto& param = params[i];
    TORCH_CHECK(param.is_non_overlapping_and_dense());
    TORCH_CHECK(param.device() == device);
    TORCH_CHECK(param.scalar_type() == scalarType);
    // All params are expected to be aligned at worldSize.
    // But not neccessarily worldSize * numelPerThread.
    TORCH_CHECK(param.numel() % worldSize == 0);
    paramPtrs.push_back(param.data_ptr());
    const auto alignedDim =
        divUp(param.numel(), worldSize * numelPerThread) * numelPerThread;
    alignedDimCumSums.push_back(alignedDimCumSums[i] + alignedDim);
    unalignedDims.push_back(param.numel() / worldSize);
  }

  TORCH_CHECK(
      alignedDimCumSums.back() * worldSize == allGatherRes.numel(),
      "allGatherRes and params must contain the same number of elements.");

  auto packed = at::empty(
      {static_cast<int64_t>(
          paramPtrs.size() + alignedDimCumSums.size() + unalignedDims.size())},
      at::TensorOptions().dtype(at::kLong).pinned_memory(true));
  memcpy(
      packed.data_ptr(), paramPtrs.data(), sizeof(int64_t) * paramPtrs.size());
  memcpy(
      packed.data_ptr<int64_t>() + paramPtrs.size(),
      alignedDimCumSums.data(),
      sizeof(int64_t) * alignedDimCumSums.size());
  memcpy(
      packed.data_ptr<int64_t>() + paramPtrs.size() + alignedDimCumSums.size(),
      unalignedDims.data(),
      sizeof(int64_t) * unalignedDims.size());
  packed = packed.to(device, /*non_blocking=*/true);
  auto paramPtrsDev = packed.data_ptr();
  auto shardDimCumSumsDev = packed.data_ptr<int64_t>() + paramPtrs.size();
  auto unalignedDimsDev =
      packed.data_ptr<int64_t>() + paramPtrs.size() + alignedDimCumSums.size();

  dim3 blocks(0, 1, 1);
  dim3 threads(0, 1, 1);

  auto numThreadsRequired = allGatherRes.numel() / numelPerThread;
  if (numThreadsRequired <= MAX_NUM_THREADS) {
    blocks.x = 1;
    threads.x = divUp(numThreadsRequired, WARP_SIZE) * WARP_SIZE;
  } else {
    blocks.x = divUp(numThreadsRequired, MAX_NUM_THREADS);
    threads.x = MAX_NUM_THREADS;
  }

  TORCH_WARN_ONCE("blocks: ", blocks.x, ", threads: ", threads.x);

  AT_DISPATCH_ALL_TYPES_AND(
      at::ScalarType::BFloat16, scalarType, "fsdp_all_gather_copy_out", [&] {
        fsdpAllGatherCopyOutKernel_no_align<scalar_t>
            <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                reinterpret_cast<scalar_t**>(paramPtrsDev),
                allGatherRes.data_ptr<scalar_t>(),
                allGatherRes.numel(),
                shardDimCumSumsDev,
                unalignedDimsDev,
                params.size(),
                worldSize,
                alignedDimCumSums.back());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
}

///////////////////////////////////////////////////////////////////////////////
// No requirement on alignment at any stage.
///////////////////////////////////////////////////////////////////////////////
__device__ inline bool isAligned(const void* ptr, size_t alignment) {
  uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
  return addr % alignment == 0;
}

template <typename T>
static __global__ void fsdpAllGatherCopyOutKernel_no_align_2(
    T** paramPtrs,
    T* allGatherResPtr,
    int64_t numel,
    int64_t* shardDimCumSums,
    int64_t numParams,
    int64_t worldSize,
    int64_t shardDimSum,
    int64_t groupSize) {
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  const auto warpId = tid / groupSize;
  if (warpId >= numParams * worldSize) {
    return;
  }
  // These values are the same across the group which is warp-aligned,
  // so we can be
  const auto memberId = tid % groupSize;
  const auto rank = warpId / numParams;
  const auto paramIdx = warpId % numParams;
  const auto shardBegin = shardDimCumSums[paramIdx];
  const auto shardEnd = shardDimCumSums[paramIdx + 1];
  const auto shardLen = shardEnd - shardBegin;
  const auto srcOff = rank * shardDimSum + shardBegin;
  const auto dstOff = rank * shardLen;
  const auto srcPtr = allGatherResPtr + srcOff;
  auto dstPtr = &paramPtrs[paramIdx][dstOff];

  using Holder128 = uint4;
  using Holder64 = uint2;

#define TRY_VECTORIZE(alignment)                                          \
  do {                                                                    \
    /* Skip vectorization attempt */                                      \
    if (sizeof(T) > alignment) {                                          \
      break;                                                              \
    }                                                                     \
    constexpr size_t numelPerThread = alignment / 8 / sizeof(T);          \
    const size_t stride = groupSize * numelPerThread;                     \
    const size_t begin = memberId * numelPerThread;                       \
    /* Check if vectorized store is possible */                           \
    if (isAligned(dstPtr, alignment) && shardLen % numelPerThread == 0) { \
      for (size_t i = begin; i < shardLen; i += stride) {                 \
        Holder##alignment val;                                            \
        /* Check if vectorized load is possible */                        \
        if (isAligned(srcPtr, alignment)) {                               \
          streamLoad##alignment(val, srcPtr + i);                         \
        } else {                                                          \
          for (size_t j = 0; j < numelPerThread; ++j) {                   \
            reinterpret_cast<T*>(&val)[j] = srcPtr[i + j];                \
          }                                                               \
        }                                                                 \
        streamStore##alignment(&dstPtr[i], val);                          \
      }                                                                   \
      return;                                                             \
    }                                                                     \
  } while (0)

  TRY_VECTORIZE(128);
  TRY_VECTORIZE(64);

  for (size_t i = memberId; i < shardLen; i += groupSize) {
    paramPtrs[paramIdx][dstOff + i] = allGatherResPtr[srcOff + i];
  }
}

void fsdpAllGatherCopyOut_no_align_2(
    std::vector<at::Tensor> params,
    at::Tensor allGatherRes,
    int64_t worldSize,
    int64_t warpsPerShard) {
  const auto device = allGatherRes.device();
  const auto scalarType = allGatherRes.scalar_type();

  TORCH_CHECK(warpsPerShard >= 1);
  TORCH_CHECK(allGatherRes.is_cuda());
  TORCH_CHECK(allGatherRes.is_non_overlapping_and_dense());
  TORCH_CHECK(allGatherRes.numel() % worldSize == 0);
  TORCH_CHECK(params.size() <= MAX_NUM_PARAMS);

  std::vector<void*> paramPtrs;
  std::vector<int64_t> dimCumSums{0};
  for (size_t i = 0; i < params.size(); ++i) {
    const auto& param = params[i];
    TORCH_CHECK(param.is_non_overlapping_and_dense());
    TORCH_CHECK(param.device() == device);
    TORCH_CHECK(param.scalar_type() == scalarType);
    // All params are expected to be aligned at worldSize.
    // But not neccessarily worldSize * numelPerThread.
    TORCH_CHECK(param.numel() % worldSize == 0);
    paramPtrs.push_back(param.data_ptr());
    dimCumSums.push_back(dimCumSums[i] + param.numel() / worldSize);
  }

  TORCH_CHECK(
      dimCumSums.back() * worldSize == allGatherRes.numel(),
      "allGatherRes and params must contain the same number of elements.");

  auto packed = at::empty(
      {static_cast<int64_t>(paramPtrs.size() + dimCumSums.size())},
      at::TensorOptions().dtype(at::kLong).pinned_memory(true));
  memcpy(
      packed.data_ptr(), paramPtrs.data(), sizeof(int64_t) * paramPtrs.size());
  memcpy(
      packed.data_ptr<int64_t>() + paramPtrs.size(),
      dimCumSums.data(),
      sizeof(int64_t) * dimCumSums.size());
  packed = packed.to(device, /*non_blocking=*/true);
  auto paramPtrsDev = packed.data_ptr();
  auto shardDimCumSumsDev = packed.data_ptr<int64_t>() + paramPtrs.size();

  dim3 blocks(0, 1, 1);
  dim3 threads(0, 1, 1);

  // Each group is responsible for copying one shard
  const auto groupSize = warpsPerShard * WARP_SIZE;
  const auto numWarpsRequired = params.size() * worldSize;
  const auto numThreadsRequired = numWarpsRequired * groupSize;
  if (allGatherRes.numel() <= MAX_NUM_THREADS) {
    blocks.x = 1;
    threads.x = numThreadsRequired;
  } else {
    blocks.x = divUp(numThreadsRequired, MAX_NUM_THREADS);
    threads.x = MAX_NUM_THREADS;
  }

  TORCH_WARN_ONCE("blocks: ", blocks.x, ", threads: ", threads.x);

  AT_DISPATCH_ALL_TYPES_AND(
      at::ScalarType::BFloat16, scalarType, "fsdp_all_gather_copy_out", [&] {
        fsdpAllGatherCopyOutKernel_no_align_2<scalar_t>
            <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                reinterpret_cast<scalar_t**>(paramPtrsDev),
                allGatherRes.data_ptr<scalar_t>(),
                allGatherRes.numel(),
                shardDimCumSumsDev,
                params.size(),
                worldSize,
                dimCumSums.back(),
                groupSize);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
}
