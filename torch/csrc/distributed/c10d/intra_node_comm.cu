#include <torch/csrc/distributed/c10d/intra_node_comm.hpp>

#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace c10d {

static constexpr size_t kBytesPerThread = 16;
static constexpr size_t kMaxAllReduceBlocks = 24;
static constexpr size_t kThreadsPerBlock = 1024;
static constexpr size_t kWarpSize = 32;
static constexpr uint32_t kNumelPerThread =
    kBytesPerThread / sizeof(at::BFloat16);
;
static constexpr uint32_t kNumelPerWarp = kNumelPerThread * kWarpSize;

#define DEVICE_INLINE __device__ inline __attribute__((always_inline))

struct __align__(16) bf16x8 {
  __nv_bfloat162 vals[4];
};

DEVICE_INLINE __nv_bfloat162
bf16hadd2(const __nv_bfloat162 x, const __nv_bfloat162 y) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  float fxl, fxh, fyl, fyh;
  fxl = __low2float(x);
  fxh = __high2float(x);
  fyl = __low2float(y);
  fyh = __high2float(y);
  return __floats2bfloat162_rn(fxl + fyl, fxh + fyh);
#else
  return __hadd2(x, y);
#endif
}

DEVICE_INLINE bf16x8 add_bf16x8(bf16x8 a, bf16x8 b) {
  bf16x8 c;
  c.vals[0] = bf16hadd2(a.vals[0], b.vals[0]);
  c.vals[1] = bf16hadd2(a.vals[1], b.vals[1]);
  c.vals[2] = bf16hadd2(a.vals[2], b.vals[2]);
  c.vals[3] = bf16hadd2(a.vals[3], b.vals[3]);
  return c;
}

// releaseSignal and acquireSignal also enforces memory ordering
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-synchronization-domains
DEVICE_INLINE void releaseSignal(uint32_t* addr) {
  atomicInc_system(addr, 1);
}

DEVICE_INLINE void acquireSignal(uint32_t* addr) {
  volatile uint32_t* signal = addr;
  uint32_t val;
  do {
    val = *signal;
  } while (val == 0 || atomicCAS_system(addr, val, val - 1) != val);
}

#define LOAD_16(a, b) \
  *reinterpret_cast<uint4*>(a) = reinterpret_cast<const uint4*>(b)[0]

static inline size_t divUp(uint32_t a, uint32_t b) {
  return (a + b - 1) / b;
}

static inline size_t alignUp(uint32_t a, uint32_t b) {
  return divUp(a, b) * b;
}

////////////////////////////////////////////////////////////////////////////////
// Fully Connected Algos
////////////////////////////////////////////////////////////////////////////////

struct FcP2pState {
  uint32_t signals[kMaxAllReduceBlocks][kMaxDevices];
  uint32_t signals1[kMaxAllReduceBlocks][kMaxDevices];
  uint32_t flags[kMaxAllReduceBlocks];
};

void* initFcP2pState() {
  void* state = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&state, sizeof(FcP2pState)));
  C10_CUDA_CHECK(cudaMemset(state, 0, sizeof(FcP2pState)));
  return state;
}

template <uint32_t kWorldSize>
static __global__ void oneShotAllReduceKernel(
    at::BFloat16* input,
    size_t N,
    size_t N_aligned,
    std::array<FcP2pState*, kMaxDevices> p2pStates,
    std::array<at::BFloat16*, kMaxDevices> buffers,
    size_t rank) {
  const size_t offset =
      (blockDim.x * blockIdx.x + threadIdx.x) * kNumelPerThread;
  const size_t stride = blockDim.x * gridDim.x * kNumelPerThread;

  // Wait for all other ranks to enter the kernel
  if (threadIdx.x < kWorldSize) {
    auto targetRank = threadIdx.x;
    releaseSignal(&p2pStates[targetRank]->signals[blockIdx.x][rank]);
    acquireSignal(&p2pStates[rank]->signals[blockIdx.x][targetRank]);
  }
  __syncthreads();

  // The source pointers. Distributed round-robin for the different warps
  const at::BFloat16* srcs[kWorldSize];
#pragma unroll kWorldSize
  for (int ii = 0; ii < kWorldSize; ++ii) {
    int srcRank = (rank + ii) % kWorldSize;
    srcs[ii] = buffers[srcRank];
  }

  for (size_t i = offset; i < N_aligned; i += stride) {
    bf16x8 vals[kWorldSize];
#pragma unroll kWorldSize
    for (size_t ii = 0; ii < kWorldSize; ++ii) {
      LOAD_16(&vals[ii], &srcs[ii][i]);
    }
    bf16x8 sums;
    memset(reinterpret_cast<void*>(&sums), 0, sizeof(sums));

#pragma unroll kWorldSize
    for (size_t ii = 0; ii < kWorldSize; ++ii) {
      sums = add_bf16x8(sums, vals[ii]);
    }
    for (size_t ii = 0; ii < kNumelPerThread; ++ii) {
      if (i + ii < N) {
        input[i + ii] = reinterpret_cast<at::BFloat16*>(&sums)[ii];
      }
    }
  }
}

template <uint32_t kWorldSize>
static __launch_bounds__(1024) __global__ void twoShotAllReduceKernel(
    at::BFloat16* input,
    size_t N_aligned,
    std::array<FcP2pState*, kMaxDevices> p2pStates,
    std::array<at::BFloat16*, kMaxDevices> buffers,
    size_t rank) {
  const size_t offset =
      (blockDim.x * blockIdx.x + threadIdx.x) * kNumelPerThread;
  const size_t stride = blockDim.x * gridDim.x * kNumelPerThread;
  const size_t N_per_rank = N_aligned / kWorldSize;
  const size_t N_start = N_per_rank * rank;

  // Wait for all other ranks to enter the kernel
  if (threadIdx.x < kWorldSize) {
    auto targetRank = threadIdx.x;
    releaseSignal(&p2pStates[targetRank]->signals[blockIdx.x][rank]);
    acquireSignal(&p2pStates[rank]->signals[blockIdx.x][targetRank]);
  }
  __syncthreads();

  at::BFloat16* localRelay = buffers[rank] + kMaxIntraNodeSize / 2;

  // The source pointers. Distributed round-robin for the different warps
  at::BFloat16* srcs[kWorldSize];
  size_t distRank[kWorldSize];
#pragma unroll kWorldSize
  for (int ii = 0; ii < kWorldSize; ++ii) {
    int srcRank = (rank + ii) % kWorldSize;
    srcs[ii] = buffers[srcRank];
    distRank[ii] = srcRank;
  }

  for (size_t i = offset; i < N_per_rank; i += stride) {
    bf16x8 vals[kWorldSize];
#pragma unroll kWorldSize
    for (size_t ii = 0; ii < kWorldSize; ++ii) {
      LOAD_16(&vals[ii], &srcs[ii][N_start + i]);
    }
    bf16x8 sums;
    memset(reinterpret_cast<void*>(&sums), 0, sizeof(sums));

#pragma unroll kWorldSize
    for (size_t ii = 0; ii < kWorldSize; ++ii) {
      sums = add_bf16x8(sums, vals[ii]);
    }
    auto relayBuf = srcs[0] + kMaxIntraNodeSize / 2;
    LOAD_16(&relayBuf[N_start + i], &sums);
  }
  __syncthreads();

  if (threadIdx.x < kWorldSize) {
    auto targetRank = threadIdx.x;
    releaseSignal(&p2pStates[targetRank]->signals1[blockIdx.x][rank]);
    acquireSignal(&p2pStates[rank]->signals1[blockIdx.x][targetRank]);
  }
  __syncthreads();

  for (size_t i = offset; i < N_per_rank; i += stride) {
#pragma unroll kWorldSize
    for (size_t ii = 0; ii < kWorldSize; ++ii) {
      size_t i_r = N_start + i + (distRank[ii] - rank) * N_per_rank;
      auto relayBuf = srcs[ii] + kMaxIntraNodeSize / 2;
      LOAD_16(&input[i_r], &relayBuf[i_r]);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// Hybrid Cube Mesh Algos
////////////////////////////////////////////////////////////////////////////////

struct HcmP2pState {
  uint32_t signals[kMaxAllReduceBlocks][kMaxDevices];
  size_t neighborRanks[3];
  size_t relayRank;
};

using HybridCubeMesh = std::array<std::array<int, 4>, kMaxDevices>;

HybridCubeMesh getHybridCubeMesh(NvlMesh nvlMesh) {
  std::array<std::unordered_set<size_t>, kMaxDevices> neighbors = {};
  std::array<size_t, kMaxDevices> neighborMasks = {};
  for (size_t i = 0; i < kMaxDevices; ++i) {
    for (size_t j = 0; j < kMaxDevices; ++j) {
      if (nvlMesh[i][j] > 0) {
        neighbors[i].insert(j);
        neighborMasks[i] |= (1ul << j);
      }
    }
  }
  HybridCubeMesh hcm = {};
  for (auto& row : hcm) {
    row.fill(-1);
  }
  for (size_t i = 0; i < kMaxDevices; ++i) {
    TORCH_CHECK(neighbors[i].size() == 4);
    for (size_t j = 0; j < kMaxDevices; ++j) {
      if ((neighborMasks[i] & neighborMasks[j]) == 0) {
        neighbors[i].erase(j);
        hcm[i][3] = j;
      }
    }
  }

  for (size_t i = 0; i < kMaxDevices; ++i) {
    for (size_t k = 0; k < 3; ++k) {
      // We can only fill hcm[i][k] with j hcm[j][k] is not filled
      for (size_t j : neighbors[i]) {
        if (hcm[j][k] == -1) {
          hcm[i][k] = j;
          hcm[j][k] = i;
          break;
        }
      }
      TORCH_CHECK(hcm[i][k] != -1);
      neighbors[i].erase(hcm[i][k]);
    }
  }
  return hcm;
}

void* initHcmP2pState(NvlMesh nvlMesh, size_t rank) {
  HcmP2pState state;
  memset(&state, 0, sizeof(state));

  auto hcm = getHybridCubeMesh(nvlMesh);
  std::copy(hcm[rank].begin(), hcm[rank].begin() + 3, state.neighborRanks);
  state.relayRank = hcm[rank][3];

  void* stateDev = nullptr;
  C10_CUDA_CHECK(cudaMalloc(&stateDev, sizeof(state)));
  AT_CUDA_CHECK(
      cudaMemcpy(stateDev, &state, sizeof(state), cudaMemcpyHostToDevice));
  return stateDev;
}

static __global__ void hybridCubeMeshAllReduceKernel(
    at::BFloat16* input,
    size_t M,
    size_t N,
    std::array<HcmP2pState*, kMaxDevices> p2pStates,
    std::array<at::BFloat16*, kMaxDevices> buffers,
    size_t rank) {
  const size_t offset =
      (blockDim.x * blockIdx.x + threadIdx.x) * kNumelPerThread;
  const size_t stride = blockDim.x * gridDim.x * kNumelPerThread;

  // Wait for HCM neigbors to enter the kernel
  if (threadIdx.x < 3) {
    auto targetRank = p2pStates[rank]->neighborRanks[threadIdx.x];
    releaseSignal(&p2pStates[targetRank]->signals[blockIdx.x][rank]);
    acquireSignal(&p2pStates[rank]->signals[blockIdx.x][targetRank]);
  }
  __syncthreads();

  const auto neighborRanks = p2pStates[rank]->neighborRanks;
  const auto relayRank = p2pStates[rank]->relayRank;
  const at::BFloat16* srcs[4] = {
      buffers[rank],
      buffers[neighborRanks[0]],
      buffers[neighborRanks[1]],
      buffers[neighborRanks[2]],
  };
  at::BFloat16* localRelay = buffers[rank] + kMaxIntraNodeSize / 2;
  at::BFloat16* remoteRelay = buffers[relayRank] + kMaxIntraNodeSize / 2;

  // During the first stage, every rank loads data from non-relay HCM
  // neighbors, sums up with local data, and store in local relay buffer
  for (size_t i = offset; i < N; i += stride) {
    bf16x8 vals[4];

#pragma unroll 4
    for (size_t ii = 0; ii < 4; ++ii) {
      LOAD_16(&vals[ii], &srcs[ii][i]);
    }
    bf16x8 sums;
    memset(reinterpret_cast<void*>(&sums), 0, sizeof(sums));

#pragma unroll 4
    for (size_t ii = 0; ii < 4; ++ii) {
      sums = add_bf16x8(sums, vals[ii]);
    }
    LOAD_16(&localRelay[i], &sums);
  }
  __syncthreads();

  // Each block syncs with the same block on the relay rank
  if (threadIdx.x == 0) {
    releaseSignal(&p2pStates[relayRank]->signals[blockIdx.x][rank]);
    acquireSignal(&p2pStates[rank]->signals[blockIdx.x][relayRank]);
  }
  __syncthreads();

  for (size_t i = offset; i < N; i += stride) {
    bf16x8 localSum, remoteSum;
    LOAD_16(&localSum, &localRelay[i]);
    LOAD_16(&remoteSum, &remoteRelay[i]);
    localSum = add_bf16x8(localSum, remoteSum);
    for (size_t ii = 0; ii < kNumelPerThread; ++ii) {
      if (i + ii < M) {
        input[i + ii] = reinterpret_cast<at::BFloat16*>(&localSum)[ii];
      }
    }
  }
}

static void checkInput(const at::Tensor& input, size_t rank) {
  TORCH_CHECK(
      input.dtype() == at::kBFloat16,
      "oneShotAllReduce only supports bf16 for now");
  TORCH_CHECK(input.is_non_overlapping_and_dense());
  TORCH_CHECK(input.device().is_cuda());
  TORCH_CHECK(static_cast<size_t>(input.get_device()) == rank);
  TORCH_CHECK(
      static_cast<size_t>(input.numel() * input.element_size()) <
      kMaxIntraNodeSize);
}

static void getLaunchConfig(size_t N, dim3& blocks, dim3& threads) {
  blocks = dim3(0, 1, 1);
  threads = dim3(0, 1, 1);
  if (N < kNumelPerThread * kThreadsPerBlock) {
    threads.x = divUp(N, kNumelPerWarp) * kWarpSize;
    blocks.x = 1;
  } else {
    auto warpsRequired = divUp(N, kNumelPerWarp);
    auto threadsRequired = divUp(N, kNumelPerThread);
    blocks.x =
        std::min(divUp(threadsRequired, kThreadsPerBlock), kMaxAllReduceBlocks);
    auto warpsPerBlock = divUp(warpsRequired, blocks.x);
    threads.x = std::min(kThreadsPerBlock, warpsPerBlock * kWarpSize);
  }
}

template <typename T>
auto castArr(std::array<void*, kMaxDevices> arr) {
  std::array<T, kMaxDevices> arr_;
  for (size_t i = 0; i < kMaxDevices; ++i) {
    arr_[i] = reinterpret_cast<T>(arr[i]);
  }
  return arr_;
}

at::Tensor oneShotAllReduce(
    const at::Tensor& input,
    std::array<void*, kMaxDevices> p2pStates,
    std::array<void*, kMaxDevices> buffers,
    size_t rank,
    size_t worldSize) {
  checkInput(input, rank);

  size_t N_aligned = alignUp(input.numel(), kNumelPerWarp);
  TORCH_CHECK(N_aligned % kNumelPerWarp == 0);
  TORCH_CHECK(N_aligned <= kMaxIntraNodeSize / sizeof(at::BFloat16));

  dim3 blocks, threads;
  getLaunchConfig(N_aligned, blocks, threads);

  TORCH_WARN_ONCE("blocks: ", blocks.x, " threads: ", threads.x);

  at::cuda::OptionalCUDAGuard guard(input.get_device());
  AT_CUDA_CHECK(cudaMemcpyAsync(
      buffers[rank],
      input.data_ptr(),
      input.numel() * input.element_size(),
      cudaMemcpyDeviceToDevice,
      at::cuda::getCurrentCUDAStream()));

#define X(kWorldSize)                                               \
  if (worldSize == kWorldSize) {                                    \
    oneShotAllReduceKernel<kWorldSize>                              \
        <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>( \
            input.data_ptr<at::BFloat16>(),                         \
            input.numel(),                                          \
            N_aligned,                                              \
            castArr<FcP2pState*>(p2pStates),                        \
            castArr<at::BFloat16*>(buffers),                        \
            rank);                                                  \
    C10_CUDA_KERNEL_LAUNCH_CHECK();                                 \
  }
  X(2);
  X(3);
  X(4);
  X(5);
  X(6);
  X(7);
  X(8);
#undef X
  return input;
}

at::Tensor twoShotAllReduce(
    const at::Tensor& input,
    std::array<void*, kMaxDevices> p2pStates,
    std::array<void*, kMaxDevices> buffers,
    size_t rank,
    size_t worldSize) {
  checkInput(input, rank);

  size_t N_aligned = alignUp(input.numel(), worldSize * kNumelPerWarp);
  size_t N_per_rank = N_aligned / worldSize;
  TORCH_CHECK(N_per_rank % kNumelPerWarp == 0);
  TORCH_CHECK(N_aligned <= kMaxIntraNodeSize / sizeof(at::BFloat16));

  dim3 blocks, threads;
  getLaunchConfig(N_per_rank, blocks, threads);

  TORCH_WARN_ONCE("two shot: blocks: ", blocks.x, " threads: ", threads.x);

  auto output = N_aligned == static_cast<size_t>(input.numel())
      ? input
      : input.new_empty(N_aligned);

  at::cuda::OptionalCUDAGuard guard(input.get_device());
  AT_CUDA_CHECK(cudaMemcpyAsync(
      buffers[rank],
      input.data_ptr(),
      input.numel() * input.element_size(),
      cudaMemcpyDeviceToDevice,
      at::cuda::getCurrentCUDAStream()));

#define X(kWorldSize)                                               \
  if (worldSize == kWorldSize) {                                    \
    twoShotAllReduceKernel<kWorldSize>                              \
        <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>( \
            output.data_ptr<at::BFloat16>(),                        \
            N_aligned,                                              \
            castArr<FcP2pState*>(p2pStates),                        \
            castArr<at::BFloat16*>(buffers),                        \
            rank);                                                  \
    C10_CUDA_KERNEL_LAUNCH_CHECK();                                 \
  }
  X(2);
  X(3);
  X(4);
  X(5);
  X(6);
  X(7);
  X(8);
#undef X

  if (output.data_ptr() != input.data_ptr()) {
    AT_CUDA_CHECK(cudaMemcpyAsync(
        input.data_ptr(),
        output.data_ptr(),
        input.numel() * input.element_size(),
        cudaMemcpyDeviceToDevice,
        at::cuda::getCurrentCUDAStream()));
  }
  return input;
}

at::Tensor hybridCubeMeshAllReduce(
    const at::Tensor& input,
    std::array<void*, kMaxDevices> p2pStates,
    std::array<void*, kMaxDevices> buffers,
    size_t rank,
    size_t worldSize) {
  checkInput(input, rank);

  size_t N = alignUp(input.numel(), kNumelPerWarp);
  TORCH_CHECK(N % kNumelPerWarp == 0);
  TORCH_CHECK(N <= kMaxIntraNodeSize / sizeof(at::BFloat16));

  dim3 blocks, threads;
  getLaunchConfig(N, blocks, threads);

  TORCH_WARN_ONCE("blocks: ", blocks.x, " threads: ", threads.x);

  at::cuda::OptionalCUDAGuard guard(input.get_device());
  AT_CUDA_CHECK(cudaMemcpyAsync(
      buffers[rank],
      input.data_ptr(),
      input.numel() * input.element_size(),
      cudaMemcpyDeviceToDevice,
      at::cuda::getCurrentCUDAStream()));

  hybridCubeMeshAllReduceKernel<<<
      blocks,
      threads,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
      input.data_ptr<at::BFloat16>(),
      input.numel(),
      N,
      castArr<HcmP2pState*>(p2pStates),
      castArr<at::BFloat16*>(buffers),
      rank);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return input;
}

AllReduceAlgo selectAllReduceAlgo(
    const at::Tensor& input,
    Topology topology,
    size_t worldSize) {
  // Only supports bf16 for now
  if (input.dtype() != at::kBFloat16) {
    return AllReduceAlgo::NONE;
  }
  const size_t numel = input.numel();
  const size_t elem_sz = input.element_size();
  if (topology == Topology::HYBRID_CUBE_MESH) {
    TORCH_CHECK(
        worldSize == 8, "hyperCubeAllReduce only supports exactly 8 GPUs");
    if (alignUp(numel, kNumelPerWarp) * elem_sz <= 256 * 1024) {
      return AllReduceAlgo::HCM;
    }
  }
  if (topology == Topology::FULLY_CONNECTED) {
    if (alignUp(numel, kNumelPerWarp) * elem_sz <= 256 * 1024) {
      return AllReduceAlgo::ONE_SHOT;
    }
    if (alignUp(numel, kNumelPerWarp * worldSize) * elem_sz <=
        10 * 1024 * 1024) {
      return AllReduceAlgo::TWO_SHOT;
    }
  }
  return AllReduceAlgo::NONE;
}

} // namespace c10d
