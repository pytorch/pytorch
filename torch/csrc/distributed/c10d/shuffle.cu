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

static inline int64_t divUp(int64_t a, int64_t b) {
  return (a + b - 1) / b;
}

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
