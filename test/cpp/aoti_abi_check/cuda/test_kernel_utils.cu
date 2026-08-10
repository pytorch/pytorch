#include <gtest/gtest.h>

#include <torch/headeronly/cuda/KernelUtils.h>
#include <torch/headeronly/util/BFloat16.h>
#include <torch/headeronly/util/Half.h>

#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace {

#define SKIP_IF_NO_CUDA()                                       \
  do {                                                          \
    int n = 0;                                                  \
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {      \
      GTEST_SKIP() << "No CUDA device available";               \
    }                                                           \
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
__global__ void specialized_add_kernel(
    scalar_t* out,
    index_t index,
    index_t numel) {
  torch::headeronly::fastSpecializedAtomicAdd(
      out, index, numel, static_cast<scalar_t>(1));
}

// Runs `indices` through fastAtomicAdd against a buffer of `buf_numel`, with
// the tensor base offset `out_offset` elements into the allocation, and returns
// the whole buffer. The offset lets us place the tensor on an odd 16-bit
// alignment, which is the case the Half/BFloat16 vectorized path handles.
template <typename scalar_t, typename index_t = int64_t>
std::vector<scalar_t> scatterAdd(
    const std::vector<index_t>& indices,
    int64_t buf_numel,
    int64_t out_offset,
    index_t out_numel,
    bool fast_atomics) {
  const int64_t n = static_cast<int64_t>(indices.size());
  scalar_t* d_buf = nullptr;
  index_t* d_idx = nullptr;
  EXPECT_EQ(cudaMalloc(&d_buf, buf_numel * sizeof(scalar_t)), cudaSuccess);
  EXPECT_EQ(cudaMalloc(&d_idx, n * sizeof(index_t)), cudaSuccess);
  EXPECT_EQ(cudaMemset(d_buf, 0, buf_numel * sizeof(scalar_t)), cudaSuccess);
  EXPECT_EQ(
      cudaMemcpy(
          d_idx, indices.data(), n * sizeof(index_t), cudaMemcpyHostToDevice),
      cudaSuccess);

  constexpr int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  scatter_add_kernel<scalar_t, index_t><<<blocks, threads>>>(
      d_buf + out_offset, d_idx, n, out_numel, fast_atomics);
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);

  std::vector<scalar_t> host(buf_numel);
  EXPECT_EQ(
      cudaMemcpy(
          host.data(),
          d_buf,
          buf_numel * sizeof(scalar_t),
          cudaMemcpyDeviceToHost),
      cudaSuccess);
  cudaFree(d_buf);
  cudaFree(d_idx);
  return host;
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
  const auto out = scatterAdd<scalar_t>(indices, numel, 0, numel, fast_atomics);
  for (int64_t j = 0; j < numel; ++j) {
    EXPECT_EQ(static_cast<float>(out[j]), static_cast<float>(expected[j]))
        << "slot " << j << " fast_atomics=" << fast_atomics;
  }
}

// The `numel` argument exists so the vectorized Half/BFloat16 path can fall
// back to a scalar atomic rather than pairing with an out-of-bounds neighbour.
template <typename scalar_t>
void testStaysInBounds() {
  constexpr int64_t buf_numel = 34;
  constexpr int64_t numel = 32;
  const auto buf = scatterAdd<scalar_t>(
      {0, numel - 1}, buf_numel, /*out_offset=*/1, numel, /*fast_atomics=*/true);
  for (int64_t j = 0; j < buf_numel; ++j) {
    const float want = (j == 1 || j == numel) ? 1.0f : 0.0f;
    EXPECT_EQ(static_cast<float>(buf[j]), want) << "slot " << j;
  }
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

TEST(TestFastAtomicAdd, StaysInBounds) {
  SKIP_IF_NO_CUDA();
  testStaysInBounds<torch::headeronly::Half>();
  testStaysInBounds<torch::headeronly::BFloat16>();
}

TEST(TestFastAtomicAdd, FastSpecializedAtomicAdd) {
  SKIP_IF_NO_CUDA();
  float* d = nullptr;
  ASSERT_EQ(cudaMalloc(&d, 2 * sizeof(float)), cudaSuccess);
  ASSERT_EQ(cudaMemset(d, 0, 2 * sizeof(float)), cudaSuccess);
  specialized_add_kernel<float, int64_t><<<1, 1>>>(d, 0, 2);
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);
  float host[2] = {-1.0f, -1.0f};
  ASSERT_EQ(
      cudaMemcpy(host, d, 2 * sizeof(float), cudaMemcpyDeviceToHost),
      cudaSuccess);
  EXPECT_EQ(host[0], 1.0f);
  EXPECT_EQ(host[1], 0.0f);
  cudaFree(d);
}
