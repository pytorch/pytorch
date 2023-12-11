#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

constexpr int64_t BYTES_PER_THREAD = 16;
constexpr int64_t MAX_NUM_FEATURES = 512;
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

static __global__ void fsdpCopyOutKernel(
    at::BFloat16** outputPtrs,
    at::BFloat16* inputPtr,
    int64_t numel,
    int64_t* shardDimCumSums,
    int64_t numParams,
    int64_t worldSize,
    int64_t shardDimSum) {
  // TODO: support all types
  const auto numelPerThread = BYTES_PER_THREAD / 2;
  auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  auto srcOff = tid * numelPerThread;
  // Offset within the rank
  auto rankOff = srcOff % shardDimSum;
  auto rank = srcOff / shardDimSum;

  __shared__ int64_t dimCumSums[MAX_NUM_FEATURES + 1];
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
  auto shardBegin = dimCumSums[paramIdx];
  auto shardEnd = dimCumSums[paramIdx + 1];
  auto shardLen = shardEnd - shardBegin;
  auto paramOff = shardLen * rank + rankOff - shardBegin;

  if (srcOff < numel) {
    uint4 val;
    streamLoad128(val, inputPtr + srcOff);
    streamStore128(&outputPtrs[paramIdx][paramOff], val);
  }
}

static inline std::vector<int64_t> makeCumSums(
    std::vector<int64_t> seq,
    int worldSize) {
  std::vector<int64_t> cumSums = {0};
  int64_t acc = 0;
  for (const auto& n : seq) {
    acc += n;
    cumSums.push_back(acc);
  }
  return cumSums;
}

static inline int64_t divUp(int64_t a, int64_t b) {
  return (a + b - 1) / b;
}

template <typename T>
at::Tensor toTensor(const std::vector<T>& vec) {
  static_assert(sizeof(T) == sizeof(int64_t));
  auto tensor = at::empty({static_cast<int64_t>(vec.size())}, at::kLong);
  std::memcpy(tensor.data_ptr(), vec.data(), sizeof(T) * vec.size());
  return tensor;
}

void fsdpCopyOut(
    std::vector<at::Tensor> outputs,
    at::Tensor input,
    int64_t worldSize) {
  const auto numelPerThread = BYTES_PER_THREAD / input.element_size();

  TORCH_CHECK(input.is_cuda());
  TORCH_CHECK(input.is_non_overlapping_and_dense());

  int64_t outputNumel = 0;
  std::vector<void*> outputPtrs;
  std::vector<int64_t> shardDims;
  for (auto& output : outputs) {
    TORCH_CHECK(output.is_cuda());
    TORCH_CHECK(output.is_non_overlapping_and_dense());
    TORCH_CHECK(output.device() == input.device());
    TORCH_CHECK(
        output.numel() % (worldSize * numelPerThread) == 0,
        "Shard must be 128-bit aligned");
    outputNumel += output.numel();
    outputPtrs.push_back(output.data_ptr());
    shardDims.push_back(output.numel() / worldSize);
  }

  TORCH_CHECK(
      outputNumel == input.numel(),
      "Input and output must contain the same number of elements.");
  TORCH_CHECK(input.numel() % worldSize == 0);
  TORCH_CHECK(outputs.size() <= MAX_NUM_FEATURES);

  const auto shardDimSum = std::accumulate(shardDims.begin(), shardDims.end(), 0);

  // Instead of using cudaMalloc, put these in GPU tensors and leverage the
  // caching allocator to manage their lifetime.
  auto outputPtrsTensor = toTensor(outputPtrs).cuda();
  auto shardDimCumSums = toTensor(makeCumSums(shardDims, worldSize)).cuda();

  dim3 blocks(0, 1, 1);
  dim3 threads(0, 1, 1);

  auto numThreadsRequired = input.numel() / numelPerThread;
  if (numThreadsRequired <= MAX_NUM_THREADS) {
    blocks.x = 1;
    threads.x = divUp(numThreadsRequired, WARP_SIZE) * WARP_SIZE;
  } else {
    blocks.x = divUp(numThreadsRequired, MAX_NUM_THREADS);
    threads.x = MAX_NUM_THREADS;
  }

  TORCH_WARN_ONCE("blocks: ", blocks.x, ", threads: ", threads.x);

  fsdpCopyOutKernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<at::BFloat16**>(outputPtrsTensor.data_ptr()),
      input.data_ptr<at::BFloat16>(),
      input.numel(),
      shardDimCumSums.data_ptr<int64_t>(),
      outputs.size(),
      worldSize,
      shardDimSum);
}
