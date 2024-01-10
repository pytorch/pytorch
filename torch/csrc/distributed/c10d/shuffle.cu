#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

constexpr int64_t BLOCK_SIZE = 128;
constexpr int64_t BYTES_PER_THREAD = 16;
constexpr int64_t WARP_SIZE = 32;

template <typename T>
__device__ inline void streamLoad128(uint4& val, const T* addr) {
#if defined(USE_ROCM) || (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800))
  val.x = reinterpret_cast<uint4*>(addr)->x;
  val.y = reinterpret_cast<uint4*>(addr)->y;
  val.z = reinterpret_cast<uint4*>(addr)->z;
  val.w = reinterpret_cast<uint4*>(addr)->w;
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
  reinterpret_cast<uint4*>(addr)->x = val.x;
  reinterpret_cast<uint4*>(addr)->y = val.y;
  reinterpret_cast<uint4*>(addr)->z = val.z;
  reinterpret_cast<uint4*>(addr)->w = val.w;
#else
  unsigned long long int low, high;
  low = reinterpret_cast<const unsigned long long int*>(&val)[0];
  high = reinterpret_cast<const unsigned long long int*>(&val)[1];
  asm("st.global.cs.v2.u64 [%0], {%1, %2};" : : "l"(addr), "l"(low), "l"(high));
#endif
}

static __host__ __device__ inline int64_t divUp(int64_t a, int64_t b) {
  return (a + b - 1) / b;
}

static __device__ inline bool isAligned(const void* addr, size_t alignment) {
  return reinterpret_cast<uintptr_t>(addr) % alignment == 0;
}

static __device__ inline void read128(uint4& val, const char* addr) {
  if (isAligned(addr, BYTES_PER_THREAD)) {
    streamLoad128(val, addr);
  } else if (isAligned(addr, sizeof(uint64_t))) {
    for (size_t j = 0; j < BYTES_PER_THREAD / sizeof(uint64_t); ++j) {
      reinterpret_cast<uint64_t*>(&val)[j] =
          reinterpret_cast<const uint64_t*>(addr)[j];
    }
  } else if (isAligned(addr, sizeof(uint32_t))) {
    for (size_t j = 0; j < BYTES_PER_THREAD / sizeof(uint32_t); ++j) {
      reinterpret_cast<uint32_t*>(&val)[j] =
          reinterpret_cast<const uint32_t*>(addr)[j];
    }
  } else if (isAligned(addr, sizeof(uint64_t))) {
    for (size_t j = 0; j < BYTES_PER_THREAD / sizeof(uint64_t); ++j) {
      reinterpret_cast<uint16_t*>(&val)[j] =
          reinterpret_cast<const uint16_t*>(addr)[j];
    }
  } else {
    for (size_t j = 0; j < BYTES_PER_THREAD; ++j) {
      reinterpret_cast<char*>(&val)[j] = (addr)[j];
    }
  }
}

static __global__ void fsdpAllGatherCopyOutKernel(
    char** params,
    char* allGatherRes,
    int64_t totalSize,
    int64_t* blockOffsetToParamIdx,
    int64_t* blockCumSums,
    int64_t* shardSizeCumSums,
    int64_t blockDimSum,
    int64_t shardSizeSum,
    int64_t ranksPerBlock,
    int64_t worldSize) {
  const int64_t blockOffset = blockIdx.x % blockDimSum;
  const int64_t paramIdx = blockOffsetToParamIdx[blockOffset];
  const int64_t shardBlockCount =
      blockCumSums[paramIdx + 1] - blockCumSums[paramIdx];
  const int64_t shardBlockId = blockOffset - blockCumSums[paramIdx];
  const int64_t groupSize = shardBlockCount * blockDim.x;
  const int64_t groupOff = shardBlockId * blockDim.x + threadIdx.x;
  const int64_t shardBegin = shardSizeCumSums[paramIdx];
  const int64_t shardEnd = shardSizeCumSums[paramIdx + 1];
  const int64_t shardSize = shardEnd - shardBegin;

  for (int64_t rank = blockIdx.x / blockDimSum; rank < worldSize;
       rank += worldSize / ranksPerBlock) {
    const int64_t srcOff = rank * shardSizeSum + shardBegin;
    const int64_t dstOff = rank * shardSize;

    const char* src = allGatherRes + srcOff;
    char* dst = params[paramIdx] + dstOff;

    if (shardSize < blockDim.x) {
      if (groupOff < shardSize) {
        dst[groupOff] = src[groupOff];
      }
      continue;
    }

    const int64_t vecBegin =
        divUp(dstOff, BYTES_PER_THREAD) * BYTES_PER_THREAD - dstOff;
    const int64_t vecEnd =
        vecBegin + (shardSize - vecBegin) / BYTES_PER_THREAD * BYTES_PER_THREAD;
    const int64_t stride = groupSize * BYTES_PER_THREAD;

    for (size_t i = vecBegin + groupOff * BYTES_PER_THREAD; i < vecEnd;
         i += stride) {
      uint4 val;
      read128(val, &src[i]);
      streamStore128(&dst[i], val);
    }
    if (groupOff < vecBegin && groupOff < shardSize) {
      dst[groupOff] = src[groupOff];
    }
    if (vecEnd + groupOff < shardSize) {
      dst[vecEnd + groupOff] = src[vecEnd + groupOff];
    }
  }
}

/**
 * Pack multiple std::vector<int64_t> into a single cuda tensor.
 */
std::pair<at::Tensor, std::vector<int64_t*>> packArgs(
    std::vector<std::vector<int64_t>> vecs,
    const at::Device& device) {
  int64_t numel = 0;
  for (const auto& vec : vecs) {
    numel += vec.size();
  }

  auto packed = at::empty(
      {numel}, at::TensorOptions().dtype(at::kLong).pinned_memory(true));
  size_t offset = 0;
  for (const auto& vec : vecs) {
    memcpy(
        packed.data_ptr<int64_t>() + offset,
        vec.data(),
        sizeof(int64_t) * vec.size());
    offset += vec.size();
  }
  packed = packed.to(device, /*non_blocking=*/true);

  std::vector<int64_t*> ptrs;
  offset = 0;
  for (const auto& vec : vecs) {
    ptrs.push_back(packed.data_ptr<int64_t>() + offset);
    offset += vec.size();
  }
  return std::make_pair(packed, ptrs);
}

void fsdpAllGatherCopyOut(
    std::vector<at::Tensor> params,
    at::Tensor allGatherRes,
    int64_t worldSize) {
  const auto device = allGatherRes.device();
  const auto totalSize = allGatherRes.numel() * allGatherRes.element_size();

  TORCH_CHECK(allGatherRes.is_cuda());
  TORCH_CHECK(allGatherRes.is_non_overlapping_and_dense());

  std::vector<int64_t> paramPtrs; // Param pointers stored as int64_t
  std::vector<int64_t> shardSizes; // In bytes
  std::vector<int64_t> shardSizeCumSums{0}; // In bytes
  for (size_t i = 0; i < params.size(); ++i) {
    const auto& param = params[i];
    TORCH_CHECK(param.device() == device);
    TORCH_CHECK(param.is_non_overlapping_and_dense());
    TORCH_CHECK(param.numel() > 0);
    TORCH_CHECK(param.numel() % worldSize == 0);
    // Deduce the shard size from the param size
    const auto shardSize = param.numel() * param.element_size() / worldSize;
    paramPtrs.push_back(reinterpret_cast<int64_t>(param.data_ptr()));
    shardSizes.push_back(shardSize);
    shardSizeCumSums.push_back(shardSizeCumSums[i] + shardSize);
  }

  TORCH_CHECK(
      shardSizeCumSums.back() * worldSize == totalSize,
      "The total byte size must be identical between params and allGatherRes");

  static int64_t numSMs = -1;
  static int64_t maxThreadsPerSM = -1;
  if (numSMs == -1) {
    cudaDeviceProp deviceProp;
    AT_CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, device.index()));
    numSMs = deviceProp.multiProcessorCount;
    maxThreadsPerSM = deviceProp.maxThreadsPerMultiProcessor;
  }

  // Calculate the amount of blocks needed if each thread only processes
  // NUM_BYTES_PER_THREAD.
  int64_t nBlks = 0;
  for (const auto& shardSize : shardSizes) {
    nBlks += divUp(shardSize, BLOCK_SIZE * BYTES_PER_THREAD);
  }

  // The kernel uses no shared memory and little registers.
  const int64_t maxBlks = numSMs * maxThreadsPerSM * 4.0;
  int64_t iterFactor = divUp(BLOCK_SIZE * nBlks * worldSize, maxBlks);
  int64_t ranksPerBlock = std::ceil(std::sqrt(iterFactor));
  ranksPerBlock = std::min(static_cast<int64_t>(worldSize), ranksPerBlock);
  iterFactor = divUp(iterFactor, ranksPerBlock);

  std::vector<int64_t> blockOffsetToParamIdx;
  std::vector<int64_t> blockCumSums{0};
  for (int64_t paramIdx = 0; paramIdx < static_cast<int64_t>(params.size());
       ++paramIdx) {
    int64_t numBlocks =
        divUp(shardSizes[paramIdx], BLOCK_SIZE * BYTES_PER_THREAD * iterFactor);
    blockOffsetToParamIdx.insert(
        blockOffsetToParamIdx.end(), numBlocks, paramIdx);
    blockCumSums.push_back(blockCumSums.back() + numBlocks);
  }
  const auto numBlocks = blockCumSums.back();

  auto packedArgs = packArgs(
      {paramPtrs, blockOffsetToParamIdx, blockCumSums, shardSizeCumSums},
      device);

  dim3 blocks(numBlocks * (worldSize / ranksPerBlock), 1, 1);
  dim3 threads(BLOCK_SIZE, 1, 1);

  LOG(INFO) << "iterFactor: " << iterFactor
            << ", ranksPerBlock: " << ranksPerBlock << ", blocks: " << blocks.x
            << ", threads: " << threads.x;

  fsdpAllGatherCopyOutKernel<<<
      blocks,
      threads,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
      /*paramPtrs=*/reinterpret_cast<char**>(packedArgs.second[0]),
      reinterpret_cast<char*>(allGatherRes.data_ptr()),
      totalSize,
      /*blockOffsetToParamIdx=*/packedArgs.second[1],
      /*blockCumSums=*/packedArgs.second[2],
      /*shardSizeCumSums=*/packedArgs.second[3],
      blockCumSums.back(),
      shardSizeCumSums.back(),
      ranksPerBlock,
      worldSize);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
