#include <gtest/gtest.h>

#include <torch/headeronly/cuda/AtomicAdd.h>
#include <torch/headeronly/util/BFloat16.h>
#include <torch/headeronly/util/Half.h>

#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace {

template <typename scalar_t>
__global__ void scatter_add_kernel(
    scalar_t* out,
    const int64_t* indices,
    const scalar_t* values,
    int64_t n,
    int64_t out_numel,
    bool fast_atomics) {
  for (int64_t i = (blockIdx.x * blockDim.x) + threadIdx.x; i < n;
       i += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    torch::headeronly::fastAtomicAdd(
        out, indices[i], out_numel, values[i], fast_atomics);
  }
}

bool hasCudaDevice() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

// Accumulates ones into a buffer of buf_numel elements through
// fastAtomicAdd on the sub-buffer starting at out_offset with out_numel
// elements, then returns the whole buffer so callers can also check the
// elements outside the sub-buffer.
template <typename scalar_t>
std::vector<scalar_t> launchScatterAdd(
    const std::vector<int64_t>& indices,
    int64_t buf_numel,
    int64_t out_offset,
    int64_t out_numel,
    bool fast_atomics) {
  const int64_t n = static_cast<int64_t>(indices.size());
  const std::vector<scalar_t> values(n, scalar_t(1.0f));
  std::vector<scalar_t> buf(buf_numel, scalar_t(0.0f));

  scalar_t* d_buf = nullptr;
  int64_t* d_indices = nullptr;
  scalar_t* d_values = nullptr;
  EXPECT_EQ(cudaMalloc(&d_buf, buf_numel * sizeof(scalar_t)), cudaSuccess);
  EXPECT_EQ(cudaMalloc(&d_indices, n * sizeof(int64_t)), cudaSuccess);
  EXPECT_EQ(cudaMalloc(&d_values, n * sizeof(scalar_t)), cudaSuccess);
  EXPECT_EQ(
      cudaMemcpy(
          d_buf, buf.data(), buf_numel * sizeof(scalar_t), cudaMemcpyDefault),
      cudaSuccess);
  EXPECT_EQ(
      cudaMemcpy(
          d_indices, indices.data(), n * sizeof(int64_t), cudaMemcpyDefault),
      cudaSuccess);
  EXPECT_EQ(
      cudaMemcpy(
          d_values, values.data(), n * sizeof(scalar_t), cudaMemcpyDefault),
      cudaSuccess);

  constexpr int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  scatter_add_kernel<scalar_t><<<blocks, threads>>>(
      d_buf + out_offset, d_indices, d_values, n, out_numel, fast_atomics);
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_EQ(
      cudaMemcpy(
          buf.data(), d_buf, buf_numel * sizeof(scalar_t), cudaMemcpyDefault),
      cudaSuccess);
  EXPECT_EQ(cudaFree(d_buf), cudaSuccess);
  EXPECT_EQ(cudaFree(d_indices), cudaSuccess);
  EXPECT_EQ(cudaFree(d_values), cudaSuccess);
  return buf;
}

// indices arange(2048) % 33 hit both __half2 alignments and the boundary
// fallbacks; ones sum to small integer counts, exact in every dtype.
template <typename scalar_t>
void testScatterAdd(bool fast_atomics) {
  if (!hasCudaDevice()) {
    GTEST_SKIP() << "No CUDA device available";
  }
  constexpr int64_t n = 2048;
  constexpr int64_t out_numel = 33;
  std::vector<int64_t> indices(n);
  std::vector<int> expected(out_numel, 0);
  for (int64_t i = 0; i < n; i++) {
    indices[i] = i % out_numel;
    expected[i % out_numel]++;
  }
  const auto out =
      launchScatterAdd<scalar_t>(indices, out_numel, 0, out_numel, fast_atomics);
  for (int64_t j = 0; j < out_numel; j++) {
    EXPECT_EQ(static_cast<float>(out[j]), static_cast<float>(expected[j]))
        << "slot " << j << " fast_atomics=" << fast_atomics;
  }
}

// numel exists so the 16-bit fast path never pairs past the tensor. The odd
// 2-byte offset puts index 0 on the pair-with-predecessor branch, which must
// fall back to a scalar atomic rather than touch buf[0]; index numel - 1
// likewise must not pair with buf[buf_numel - 1].
template <typename scalar_t>
void testStaysInBounds() {
  if (!hasCudaDevice()) {
    GTEST_SKIP() << "No CUDA device available";
  }
  constexpr int64_t buf_numel = 34;
  constexpr int64_t out_numel = 32;
  const std::vector<int64_t> indices = {0, out_numel - 1};
  const auto buf = launchScatterAdd<scalar_t>(
      indices, buf_numel, /*out_offset=*/1, out_numel, /*fast_atomics=*/true);
  for (int64_t j = 0; j < buf_numel; j++) {
    const float expected = (j == 1 || j == out_numel) ? 1.0f : 0.0f;
    EXPECT_EQ(static_cast<float>(buf[j]), expected) << "slot " << j;
  }
}

} // namespace

TEST(TestAtomicAdd, FastAtomicAddFloat) {
  testScatterAdd<float>(true);
  testScatterAdd<float>(false);
}

TEST(TestAtomicAdd, FastAtomicAddDouble) {
  testScatterAdd<double>(true);
  testScatterAdd<double>(false);
}

TEST(TestAtomicAdd, FastAtomicAddHalf) {
  testScatterAdd<torch::headeronly::Half>(true);
  testScatterAdd<torch::headeronly::Half>(false);
}

TEST(TestAtomicAdd, FastAtomicAddBFloat16) {
  testScatterAdd<torch::headeronly::BFloat16>(true);
  testScatterAdd<torch::headeronly::BFloat16>(false);
}

TEST(TestAtomicAdd, FastAtomicAddHalfStaysInBounds) {
  testStaysInBounds<torch::headeronly::Half>();
}

TEST(TestAtomicAdd, FastAtomicAddBFloat16StaysInBounds) {
  testStaysInBounds<torch::headeronly::BFloat16>();
}
