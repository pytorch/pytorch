#include <torch/csrc/distributed/c10d/intra_node_comm.hpp>

// TODO
#include <iostream>
#include <sstream>

#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace c10d {

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

#define LOAD_16(a, b) \
  *reinterpret_cast<uint4*>(a) = reinterpret_cast<const uint4*>(b)[0]

static constexpr size_t kMaxAllReduceBlocks = 24;
static constexpr size_t kThreadsPerBlock = 1024;
static constexpr size_t kWarpSize = 32;

template <uint32_t kWorldSize>
static __global__ void oneShotAllReduceKernel(
    at::BFloat16* output,
    size_t N,
    at::BFloat16* input,
    size_t M,
    std::array<at::BFloat16*, kMaxDevices> buffers,
    std::array<uint32_t*, kMaxDevices> barriers,
    uint32_t barrierFlag,
    size_t rank) {
  for (size_t i = blockDim.x * blockIdx.x + threadIdx.x; i < M;
       i += blockDim.x * gridDim.x) {
    buffers[rank][i] = input[i];
  }

  // Synchronize the ranks.
  volatile uint32_t* barrier = barriers[rank];
  if (threadIdx.x < kWorldSize) {
    // The 1st block notifies the other ranks.
    if (blockIdx.x == 0) {
      assert(barriers[threadIdx.x][rank] < barrierFlag);
      barriers[threadIdx.x][rank] = barrierFlag;
    }

    // Busy-wait until all ranks are ready.
    while (barrier[threadIdx.x] < barrierFlag) {
    }
  }

  // Make sure we can move on...
  __syncthreads();

  // The source pointers. Distributed round-robin for the different warps.
  const at::BFloat16* src_d[kWorldSize];
#pragma unroll kWorldSize
  for (int ii = 0; ii < kWorldSize; ++ii) {
    int srcRank = (rank + ii) % kWorldSize;
    src_d[ii] = buffers[srcRank];
  }

  // Load 8 fp16s
  constexpr size_t numelPerThread = 8;
  const size_t offset =
      (blockDim.x * blockIdx.x + threadIdx.x) * numelPerThread;
  const size_t stride = blockDim.x * gridDim.x * numelPerThread;

  // Each block accumulates the values from the different GPUs on the same
  // node.
  for (size_t i = offset; i < N; i += stride) {
    // Iterate over the different ranks/devices on the node to load the
    // values.
    bf16x8 vals[kWorldSize];
#pragma unroll kWorldSize
    for (size_t ii = 0; ii < kWorldSize; ++ii) {
      LOAD_16(&vals[ii], &src_d[ii][i]);
    }

    // Sum the values from the different ranks.
    bf16x8 sums;
    memset(reinterpret_cast<void*>(&sums), 0, sizeof(sums));

#pragma unroll kWorldSize
    for (size_t ii = 0; ii < kWorldSize; ++ii) {
      sums = add_bf16x8(sums, vals[ii]);
    }

    // Store to the destination buffer.
    LOAD_16(&output[i], &sums);
  }
}

static inline size_t divUp(uint32_t a, uint32_t b) {
  return (a + b - 1) / b;
}

at::Tensor oneShotAllReduce(
    const at::Tensor& input,
    std::array<void*, kMaxDevices> buffers,
    std::array<uint32_t*, kMaxDevices> barriers,
    uint32_t barrierFlag,
    size_t rank,
    size_t worldSize) {
  constexpr uint32_t numelPerThread = 8;
  constexpr uint32_t numelPerWarp = numelPerThread * kWarpSize;
  TORCH_CHECK(
      input.dtype() == at::kBFloat16,
      "oneShotAllReduce only supports bf16 for now");

  TORCH_CHECK(worldSize == 2 || worldSize == 4 || worldSize == 8);
  TORCH_CHECK(input.is_non_overlapping_and_dense());
  TORCH_CHECK(input.device().is_cuda());
  TORCH_CHECK(static_cast<size_t>(input.numel()) < kMaxIntraNodeSize);

  // Potentially over allocate the output buffer to align with warp size.
  size_t M = input.numel();
  size_t N = divUp(M, numelPerWarp) * numelPerWarp;
  TORCH_CHECK(N % numelPerWarp == 0);
  TORCH_CHECK(N <= kMaxIntraNodeSize / 2);
  auto output = input.new_zeros(N);

  dim3 threads(0, 1, 1);
  dim3 blocks(0, 1, 1);

  if (N < numelPerThread * kThreadsPerBlock) {
    threads.x = divUp(N, numelPerWarp) * kWarpSize;
    blocks.x = 1;
  } else {
    auto warpsRequired = divUp(N, numelPerWarp);
    auto threadsRequired = divUp(N, numelPerThread);
    blocks.x =
        std::min(divUp(threadsRequired, kThreadsPerBlock), kMaxAllReduceBlocks);
    auto warpsPerBlock = divUp(warpsRequired, blocks.x);
    threads.x = std::min(kThreadsPerBlock, warpsPerBlock * kWarpSize);
  }

  std::array<at::BFloat16*, kMaxDevices> bf16Buffers;
  for (size_t i = 0; i < kMaxDevices; ++i) {
    bf16Buffers[i] = static_cast<at::BFloat16*>(buffers[i]);
  }

  TORCH_CHECK(input.get_device() == rank);
  at::cuda::OptionalCUDAGuard guard(input.get_device());

// TODO: maybe not specialize if unrolling doesn't provide much perf gain
#define X(kWorldSize)                                               \
  if (worldSize == kWorldSize) {                                    \
    oneShotAllReduceKernel<kWorldSize>                              \
        <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>( \
            output.data_ptr<at::BFloat16>(),                        \
            N,                                                      \
            input.data_ptr<at::BFloat16>(),                         \
            M,                                                      \
            bf16Buffers,                                            \
            barriers,                                               \
            barrierFlag,                                            \
            rank);                                                  \
  }
  X(2);
  X(4);
  X(8);
#undef X
  return output.as_strided(input.sizes(), input.strides());
}

static __global__ void hybridCubeAllReduceKernel(
    at::BFloat16* output,
    size_t N,
    at::BFloat16* input,
    size_t M,
    std::array<at::BFloat16*, kMaxDevices> buffers,
    std::array<uint32_t*, kMaxDevices> barriers,
    uint32_t barrierFlag,
    size_t neighbors[5],
    size_t rank) {
  for (size_t i = blockDim.x * blockIdx.x + threadIdx.x; i < M;
       i += blockDim.x * gridDim.x) {
    buffers[rank][i] = input[i];
  }

  // Synchronize the ranks.
  volatile uint32_t* barrier = barriers[rank];
  if (threadIdx.x < kMaxDevices) {
    // The 1st block notifies the other ranks.
    if (blockIdx.x == 0) {
      assert(barriers[threadIdx.x][rank] != barrierFlag);
      barriers[threadIdx.x][rank] = barrierFlag;
    }

    // Busy-wait until all ranks are ready.
    while (barrier[threadIdx.x] < barrierFlag) {
    }
  }

  // Make sure we can move on...
  __syncthreads();

  const at::BFloat16* src_d[4];
#pragma unroll 4
  for (int ii = 0; ii < 4; ++ii) {
    src_d[ii] = buffers[neighbors[ii]];
  }

  // Load 8 fp16s
  constexpr size_t numelPerThread = 8;
  constexpr size_t relayOffset = kMaxIntraNodeSize / 2;
  const size_t offset =
      (blockDim.x * blockIdx.x + threadIdx.x) * numelPerThread;
  const size_t stride = blockDim.x * gridDim.x * numelPerThread;

  // Each block accumulates the values from the different GPUs on the same
  // node.
  for (size_t i = offset; i < N; i += stride) {
    // Iterate over the different ranks/devices on the node to load the
    // values.
    bf16x8 vals[4];
#pragma unroll 4
    for (size_t ii = 0; ii < 4; ++ii) {
      LOAD_16(&vals[ii], &src_d[ii][i]);
    }

    // Sum the values from the different ranks.
    bf16x8 sums;
    memset(reinterpret_cast<void*>(&sums), 0, sizeof(sums));

#pragma unroll 4
    // Sum up local and non-opposite vertex neighbors buffers.
    // Write to both output buffer and relay buffer.
    for (size_t ii = 0; ii < 4; ++ii) {
      sums = add_bf16x8(sums, vals[ii]);
    }
    at::BFloat16* relay = buffers[rank] + relayOffset;
    LOAD_16(&relay[i], &sums);
    LOAD_16(&output[i], &sums);
  }

  barrierFlag += 1;
  if (threadIdx.x < kMaxDevices) {
    // The 1st block notifies the other ranks.
    if (blockIdx.x == 0) {
      barriers[threadIdx.x][rank] = barrierFlag;
    }

    // Busy-wait until all ranks are ready.
    while (barrier[threadIdx.x] < barrierFlag) {
    }
  }

  __syncthreads();

  // Sum up output buffer and the opposite vertex neighbors's relay buffer.
  at::BFloat16* relay = buffers[neighbors[4]] + relayOffset;
  for (size_t i = offset; i < N; i += stride) {
    bf16x8 a, b;
    LOAD_16(&a, &output[i]);
    LOAD_16(&b, &relay[i]);
    a = add_bf16x8(a, b);
    LOAD_16(&output[i], &a);
  }
}

std::array<size_t, 5> initHybridOneShotAllReduceConfig(
    NvlMesh nvlMesh,
    size_t rank) {
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
  std::array<size_t, kMaxDevices> opposite = {};
  for (size_t i = 0; i < kMaxDevices; ++i) {
    TORCH_CHECK(neighbors[i].size() == 4);
    for (size_t j = 0; j < kMaxDevices; ++j) {
      if ((neighborMasks[i] & neighborMasks[j]) == 0) {
        neighbors[i].erase(j);
        opposite[i] = j;
      }
    }
    TORCH_CHECK(neighbors[i].size() == 3);
  }
  // The first 3 values are non-opposite vertex neighbor devices.
  // The 4th values is the local device.
  // The 5th values is the opposite vertex neighbor device.
  std::array<size_t, 5> conf = {};
  std::copy(neighbors[rank].begin(), neighbors[rank].end(), conf.begin());
  conf[3] = rank;
  conf[4] = opposite[rank];
  return conf;
}

at::Tensor hybridCubeOneShotAllReduce(
    const at::Tensor& input,
    std::array<void*, kMaxDevices> buffers,
    std::array<uint32_t*, kMaxDevices> barriers,
    uint32_t barrierFlag,
    size_t rank,
    size_t worldSize,
    NvlMesh nvlMesh_) {
  TORCH_CHECK(
      worldSize == 8, "hyperCubeAllReduce only supports exactly 8 GPUs");

  static size_t* conf = nullptr;
  if (conf == nullptr) {
    auto confHost = initHybridOneShotAllReduceConfig(nvlMesh_, rank);
    C10_CUDA_CHECK(cudaMalloc(&conf, sizeof(confHost)));
    AT_CUDA_CHECK(cudaMemcpyAsync(
        conf,
        confHost.data(),
        sizeof(confHost),
        cudaMemcpyHostToDevice,
        at::cuda::getCurrentCUDAStream()));
  }

  constexpr uint32_t numelPerThread = 8;
  constexpr uint32_t numelPerWarp = numelPerThread * kWarpSize;
  TORCH_CHECK(
      input.dtype() == at::kBFloat16,
      "oneShotAllReduce only supports bf16 for now");

  TORCH_CHECK(worldSize == 2 || worldSize == 4 || worldSize == 8);
  TORCH_CHECK(input.is_non_overlapping_and_dense());
  TORCH_CHECK(input.device().is_cuda());
  TORCH_CHECK(static_cast<size_t>(input.numel()) < kMaxIntraNodeSize);

  // Potentially over allocate the output buffer to align with warp size.
  size_t M = input.numel();
  size_t N = divUp(M, numelPerWarp) * numelPerWarp;
  TORCH_CHECK(N % numelPerWarp == 0);
  TORCH_CHECK(N <= kMaxIntraNodeSize / 2);
  auto output = input.new_zeros(N);

  dim3 threads(0, 1, 1);
  dim3 blocks(0, 1, 1);

  if (N < numelPerThread * kThreadsPerBlock) {
    threads.x = divUp(N, numelPerWarp) * kWarpSize;
    blocks.x = 1;
  } else {
    auto warpsRequired = divUp(N, numelPerWarp);
    auto threadsRequired = divUp(N, numelPerThread);
    blocks.x =
        std::min(divUp(threadsRequired, kThreadsPerBlock), kMaxAllReduceBlocks);
    auto warpsPerBlock = divUp(warpsRequired, blocks.x);
    threads.x = std::min(kThreadsPerBlock, warpsPerBlock * kWarpSize);
  }

  std::array<at::BFloat16*, kMaxDevices> bf16Buffers;
  for (size_t i = 0; i < kMaxDevices; ++i) {
    bf16Buffers[i] = static_cast<at::BFloat16*>(buffers[i]);
  }

  TORCH_CHECK(input.get_device() == rank);
  at::cuda::OptionalCUDAGuard guard(input.get_device());

  hybridCubeAllReduceKernel<<<
      blocks,
      threads,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
      output.data_ptr<at::BFloat16>(),
      N,
      input.data_ptr<at::BFloat16>(),
      M,
      bf16Buffers,
      barriers,
      barrierFlag,
      conf,
      rank);
  return output.as_strided(input.sizes(), input.strides());
}

} // namespace c10d
