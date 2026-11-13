#include <gtest/gtest.h>

#include <torch/headeronly/cuda/KernelUtils.h>
#include <torch/headeronly/util/BFloat16.h>
#include <torch/headeronly/util/Exception.h>
#include <torch/headeronly/util/Half.h>

#include <cuda_runtime.h>

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace {

#define SKIP_IF_NO_CUDA()                                  \
  do {                                                     \
    int n = 0;                                             \
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) { \
      GTEST_SKIP() << "No CUDA device available";          \
    }                                                      \
  } while (0)

#define CUDA_ASSERT_OK(expr)                                        \
  do {                                                              \
    cudaError_t err_ = (expr);                                      \
    STD_TORCH_CHECK(                                                \
        err_ == cudaSuccess,                                        \
        #expr,                                                      \
        " failed: ",                                                \
        cudaGetErrorString(err_));                                  \
  } while (0)

template <typename scalar_t, typename index_t>
__global__ void scatter_add_kernel(
    scalar_t* out,
    const index_t* indices,
    int64_t n,
    index_t numel,
    bool fast_atomics) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    torch::headeronly::fastAtomicAdd(
        out, indices[i], numel, static_cast<scalar_t>(1), fast_atomics);
  }
}

template <typename scalar_t, typename index_t>
__global__ void specialized_add_kernel(scalar_t* out, index_t numel) {
  torch::headeronly::fastSpecializedAtomicAdd(
      out, index_t{0}, numel, static_cast<scalar_t>(1));
}

// Scatters +1 into each of `indices` of a `numel`-element tensor starting
// `offset` elements into `buf`. Returns the whole buffer, so writes landing
// outside the tensor are visible.
template <typename scalar_t, typename index_t = int64_t>
std::vector<scalar_t> scatterAdd(
    std::vector<scalar_t> buf,
    const std::vector<index_t>& indices,
    index_t numel,
    bool fast_atomics,
    int64_t offset = 0) {
  const int64_t n = static_cast<int64_t>(indices.size());
  const size_t bytes = buf.size() * sizeof(scalar_t);
  scalar_t* d_buf = nullptr;
  index_t* d_idx = nullptr;
  CUDA_ASSERT_OK(cudaMalloc(&d_buf, bytes));
  CUDA_ASSERT_OK(cudaMalloc(&d_idx, n * sizeof(index_t)));
  CUDA_ASSERT_OK(cudaMemcpy(d_buf, buf.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_ASSERT_OK(cudaMemcpy(
      d_idx, indices.data(), n * sizeof(index_t), cudaMemcpyHostToDevice));

  constexpr int threads = 256;
  scatter_add_kernel<scalar_t, index_t>
      <<<static_cast<int>((n + threads - 1) / threads), threads>>>(
          d_buf + offset, d_idx, n, numel, fast_atomics);
  CUDA_ASSERT_OK(cudaGetLastError());
  CUDA_ASSERT_OK(cudaDeviceSynchronize());

  CUDA_ASSERT_OK(cudaMemcpy(buf.data(), d_buf, bytes, cudaMemcpyDeviceToHost));
  cudaFree(d_buf);
  cudaFree(d_idx);
  return buf;
}

template <typename scalar_t>
void testScatterAdd(bool fast_atomics) {
  constexpr int64_t n = 2048;
  constexpr int64_t numel = 33;
  std::vector<int64_t> indices(n);
  std::vector<int> expected(numel, 0);
  for (int64_t i = 0; i < n; ++i) {
    indices[i] = i % numel;
    expected[i % numel]++;
  }
  const auto out = scatterAdd<scalar_t>(
      std::vector<scalar_t>(numel), indices, numel, fast_atomics);
  for (int64_t j = 0; j < numel; ++j) {
    EXPECT_EQ(static_cast<float>(out[j]), static_cast<float>(expected[j]))
        << "slot " << j << " fast_atomics=" << fast_atomics;
  }
}

// Everything fastAtomicAdd does beyond gpuAtomicAdd is Half/BFloat16 only: it
// issues one (fast) 32-bit __half2/__nv_bfloat162 atomic covering the target
// *and* an adjacent element, padding the neighbour's lane with -0.0. `numel`
// exists so it can decline to do that when the neighbour is outside the tensor.
//
// Adding -0.0 leaves every non-NaN value unchanged, but quiets a signaling NaN.
// So we seed memory with a signaling NaN to make paired writes detectable.
constexpr uint16_t kNegZero = 0x8000; // -0.0 in both fp16 and bf16

template <typename scalar_t>
constexpr uint16_t kSignalingNaN = std::numeric_limits<scalar_t>::signaling_NaN().x;

// The tensor is buf[offset, offset + numel); targets start at -0.0 and every
// other slot holds a signaling NaN.
template <typename scalar_t>
std::vector<scalar_t> seededBuffer(
    int64_t buf_numel,
    int64_t offset,
    const std::vector<int64_t>& targets) {
  std::vector<scalar_t> buf(buf_numel);
  for (auto& v : buf) {
    v.x = kSignalingNaN<scalar_t>;
  }
  for (auto i : targets) {
    buf[offset + i].x = kNegZero;
  }
  return buf;
}

template <typename scalar_t>
bool wasWritten(scalar_t v) {
  return v.x != kSignalingNaN<scalar_t>;
}

template <typename scalar_t>
void testVectorizedBounds() {
  constexpr int64_t buf_numel = 34;
  constexpr int64_t numel = 32;
  constexpr int64_t offset = 1; // odd 16-bit alignment, e.g., in a view
  const std::vector<int64_t> targets = {0, 5, 16, numel - 1};

  // The tensor is buf[1..32]. At this offset odd indices pair right and even
  // ones pair left, so index 0 could only pair left and index numel-1 only
  // right -- both neighbours are outside the tensor, so both must fall back to
  // a scalar atomic. Indices 5 and 16 have legal neighbours and must pair, one
  // in each direction; they are the positive control proving a pair write is
  // detectable here at all.
  const auto out = scatterAdd<scalar_t>(
      seededBuffer<scalar_t>(buf_numel, offset, targets),
      targets,
      numel,
      /*fast_atomics=*/true,
      offset);

  for (auto i : targets) {
    EXPECT_EQ(static_cast<float>(out[offset + i]), 1.0f) << "index " << i;
  }
  EXPECT_TRUE(wasWritten(out[offset + 6])) << "index 5 should pair right";
  EXPECT_TRUE(wasWritten(out[offset + 15])) << "index 16 should pair left";
  EXPECT_FALSE(wasWritten(out[0])) << "wrote below the start of the tensor";
  EXPECT_FALSE(wasWritten(out[buf_numel - 1])) << "wrote past the end";
}

// A paired neighbour must keep its value, including the sign of -0.0.
template <typename scalar_t>
void testPairedNeighbourUnchanged() {
  constexpr int64_t numel = 32;
  constexpr int64_t offset = 1;
  std::vector<scalar_t> buf(numel + 2);
  for (auto& v : buf) {
    v.x = kNegZero;
  }
  const auto out = scatterAdd<scalar_t>(
      std::move(buf), {5, 16}, numel, /*fast_atomics=*/true, offset);
  EXPECT_EQ(static_cast<float>(out[offset + 5]), 1.0f);
  EXPECT_EQ(static_cast<float>(out[offset + 16]), 1.0f);
  EXPECT_EQ(out[offset + 6].x, kNegZero) << "index 5's neighbour changed";
  EXPECT_EQ(out[offset + 15].x, kNegZero) << "index 16's neighbour changed";
}

} // namespace

TEST(TestFastAtomicAdd, ScatterAdd) {
  SKIP_IF_NO_CUDA();
  for (bool fast : {true, false}) {
    testScatterAdd<float>(fast);
    testScatterAdd<double>(fast);
    testScatterAdd<torch::headeronly::Half>(fast);
    testScatterAdd<torch::headeronly::BFloat16>(fast);
    testScatterAdd<int32_t>(fast);
    testScatterAdd<int64_t>(fast);
  }
}

// Packed fp16 atomics need sm_70 and packed bf16 atomics sm_80; below that
// fastAtomicAdd never pairs, so the pairing tests do not apply.
#define SKIP_IF_NO_PACKED_ATOMIC(min_major)                            \
  do {                                                                 \
    int device_ = 0;                                                   \
    cudaDeviceProp prop_{};                                            \
    CUDA_ASSERT_OK(cudaGetDevice(&device_));                           \
    CUDA_ASSERT_OK(cudaGetDeviceProperties(&prop_, device_));          \
    if (prop_.major < (min_major)) {                                   \
      GTEST_SKIP() << "packed atomics need sm_" << (min_major) << "0"; \
    }                                                                  \
  } while (0)

TEST(TestFastAtomicAdd, VectorizedBoundsHalf) {
  SKIP_IF_NO_CUDA();
  SKIP_IF_NO_PACKED_ATOMIC(7);
  testVectorizedBounds<torch::headeronly::Half>();
}

TEST(TestFastAtomicAdd, VectorizedBoundsBFloat16) {
  SKIP_IF_NO_CUDA();
  SKIP_IF_NO_PACKED_ATOMIC(8);
  testVectorizedBounds<torch::headeronly::BFloat16>();
}

TEST(TestFastAtomicAdd, PairedNeighbourUnchangedHalf) {
  SKIP_IF_NO_CUDA();
  SKIP_IF_NO_PACKED_ATOMIC(7);
  testPairedNeighbourUnchanged<torch::headeronly::Half>();
}

TEST(TestFastAtomicAdd, PairedNeighbourUnchangedBFloat16) {
  SKIP_IF_NO_CUDA();
  SKIP_IF_NO_PACKED_ATOMIC(8);
  testPairedNeighbourUnchanged<torch::headeronly::BFloat16>();
}

TEST(TestFastAtomicAdd, FastSpecializedAtomicAdd) {
  SKIP_IF_NO_CUDA();
  float* d = nullptr;
  CUDA_ASSERT_OK(cudaMalloc(&d, sizeof(float)));
  CUDA_ASSERT_OK(cudaMemset(d, 0, sizeof(float)));
  specialized_add_kernel<float, int64_t><<<1, 1>>>(d, 1);
  CUDA_ASSERT_OK(cudaGetLastError());
  CUDA_ASSERT_OK(cudaDeviceSynchronize());
  float host = -1.0f;
  CUDA_ASSERT_OK(cudaMemcpy(&host, d, sizeof(float), cudaMemcpyDeviceToHost));
  EXPECT_EQ(host, 1.0f);
  cudaFree(d);
}
