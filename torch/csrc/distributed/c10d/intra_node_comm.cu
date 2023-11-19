#include <torch/csrc/distributed/c10d/intra_node_comm.hpp>

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

static constexpr size_t kMaxAllReduceBlocks = 24;
static constexpr size_t kThreadsPerBlock = 1024;
static constexpr size_t kWarpSize = 32;

template <uint32_t kWorldSize>
static __global__ void oneShotAllReduceKernel(
    at::BFloat16* output,
    size_t N,
    at::BFloat16* input,
    size_t M,
    std::array<void*, kMaxDevices> buffers,
    std::array<uint32_t*, kMaxDevices> barriers,
    uint32_t barrierFlag,
    size_t rank) {

  for (size_t i = blockDim.x * blockIdx.x + threadIdx.x; i < M;
       i += blockDim.x * gridDim.x) {
    static_cast<at::BFloat16*>(buffers[rank])[i] = input[i];
  }

  // Synchronize the ranks.
  volatile uint32_t* barrier_d = barriers[rank];
  if (threadIdx.x < kWorldSize) {
    // The 1st block notifies the other ranks.
    if (blockIdx.x == 0) {
      assert(barriers[threadIdx.x][rank] != barrierFlag);
      barriers[threadIdx.x][rank] = barrierFlag;
    }

    // Busy-wait until all ranks are ready.
    while (barrier_d[threadIdx.x] != barrierFlag) {
    }
  }

  // Make sure we can move on...
  __syncthreads();

  // The source pointers. Distributed round-robin for the different warps.
  const at::BFloat16* src_d[kWorldSize];
#pragma unroll kWorldSize
  for (int ii = 0; ii < kWorldSize; ++ii) {
    int srcRank = (rank + ii) % kWorldSize;
    src_d[ii] = static_cast<at::BFloat16*>(buffers[srcRank]);
  }

  // Load 8 fp16s
  constexpr size_t numelPerThread = 8;
  const size_t offset = (blockDim.x * blockIdx.x + threadIdx.x) * numelPerThread;
  const size_t stride = blockDim.x * gridDim.x * numelPerThread;

  // Each block accumulates the values from the different GPUs on the same
  // node.
  for (size_t i = offset; i < N; i += stride) {
    // Iterate over the different ranks/devices on the node to load the
    // values.
    bf16x8 vals[kWorldSize];
#pragma unroll kWorldSize
    for (size_t ii = 0; ii < kWorldSize; ++ii) {
      *reinterpret_cast<uint4*>(&vals[ii]) =
          reinterpret_cast<const uint4*>(&src_d[ii][i])[0];
    }

    // Sum the values from the different ranks.
    bf16x8 sums;
    memset(reinterpret_cast<void*>(&sums), 0, sizeof(sums));

#pragma unroll kWorldSize
    for (size_t ii = 0; ii < kWorldSize; ++ii) {
      sums = add_bf16x8(sums, vals[ii]);
    }

    // Store to the destination buffer.
    *reinterpret_cast<uint4*>(&output[i]) =
        *reinterpret_cast<const uint4*>(&sums);
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
  // TODO: support other dtypes
  TORCH_CHECK(
      input.dtype() == at::kBFloat16,
      "oneShotAllReduce only supports bf16 for now");

  TORCH_CHECK(worldSize == 2 || worldSize == 4 || worldSize == 8);
  TORCH_CHECK(input.is_non_overlapping_and_dense());
  TORCH_CHECK(input.device().is_cuda());
  TORCH_CHECK(static_cast<size_t>(input.numel()) < kMaxIntraNodeSize);

  // Align output size by warp size. The output buffer will be
  // narrowed later.
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
    blocks.x = std::min(
        divUp(threadsRequired, kThreadsPerBlock), kMaxAllReduceBlocks);
    auto warpsPerBlock = divUp(warpsRequired, blocks.x);
    threads.x = std::min(kThreadsPerBlock, warpsPerBlock * kWarpSize);
  }

  // TODO: check input device == buffer device
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
            buffers,                                                \
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

} // namespace c10d
