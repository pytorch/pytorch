#include <gtest/gtest.h>

#include <torch/headeronly/cuda/KernelUtils.h>
#include <torch/headeronly/util/BFloat16.h>
#include <torch/headeronly/util/Half.h>

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
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

#define CUDA_ASSERT_OK(expr)                                            \
  do {                                                                  \
    cudaError_t err_ = (expr);                                          \
    if (err_ != cudaSuccess) {                                          \
      throw std::runtime_error(                                         \
          std::string(#expr) + " failed: " + cudaGetErrorString(err_)); \
    }                                                                   \
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
// *and* an adjacent element, padding the neighbour's lane with +0. `numel`
// exists so it can decline to do that when the neighbour is outside the tensor.
//
// A +0 write is invisible when doing a value check BUT one observable trace
// is the sign bit: IEEE 754 gives (-0.0) + (+0.0) == +0.0. So we seed memory
// with -0.0 to make out of bound writes detectable.
constexpr uint16_t kNegZero = 0x8000; // -0.0 in both fp16 and bf16

// A slot the vectorized path wrote to comes back as +0.0; one it skipped still
// holds the -0.0 it was seeded with. Both read as 0.0 as a float, so the raw
// bits are the only way to tell them apart.
template <typename scalar_t>
bool wasWritten(scalar_t v) {
  return v.x != kNegZero;
}

template <typename scalar_t>
void testVectorizedBounds() {
  constexpr int64_t buf_numel = 34;
  constexpr int64_t numel = 32;
  constexpr int64_t offset = 1; // odd 16-bit alignment, e.g., in a view
  std::vector<scalar_t> buf(buf_numel);
  for (auto& v : buf) {
    v.x = kNegZero;
  }

  // The tensor is buf[1..32]. At this offset odd indices pair right and even
  // ones pair left, so index 0 could only pair left and index numel-1 only
  // right -- both neighbours are outside the tensor, so both must fall back to
  // a scalar atomic. Indices 5 and 16 have legal neighbours and must pair, one
  // in each direction; they are the positive control proving a pair write is
  // detectable here at all.
  const auto out = scatterAdd<scalar_t>(
      std::move(buf), {0, 5, 16, numel - 1}, numel, /*fast_atomics=*/true, offset);

  EXPECT_EQ(static_cast<float>(out[offset]), 1.0f);
  EXPECT_EQ(static_cast<float>(out[offset + 5]), 1.0f);
  EXPECT_EQ(static_cast<float>(out[offset + 16]), 1.0f);
  EXPECT_EQ(static_cast<float>(out[offset + numel - 1]), 1.0f);

  EXPECT_TRUE(wasWritten(out[offset + 6])) << "index 5 should pair right";
  EXPECT_TRUE(wasWritten(out[offset + 15])) << "index 16 should pair left";
  EXPECT_FALSE(wasWritten(out[0])) << "wrote below the start of the tensor";
  EXPECT_FALSE(wasWritten(out[buf_numel - 1])) << "wrote past the end";
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

TEST(TestFastAtomicAdd, VectorizedBounds) {
  SKIP_IF_NO_CUDA();
  testVectorizedBounds<torch::headeronly::Half>();
  testVectorizedBounds<torch::headeronly::BFloat16>();
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
