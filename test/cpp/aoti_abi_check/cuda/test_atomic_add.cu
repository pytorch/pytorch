#include <test/cpp/aoti_abi_check/cuda/utils.cuh>

#include <torch/headeronly/cuda/AtomicAdd.h>
#include <torch/headeronly/util/BFloat16.h>
#include <torch/headeronly/util/Half.h>

#include <cstdint>
#include <vector>

namespace {

using torch::test::DeviceBuffer;

template <typename scalar_t, typename index_t>
__global__ void scatter_add_kernel(
    scalar_t* out,
    const index_t* indices,
    const scalar_t* values,
    int64_t n,
    index_t out_numel,
    bool fast_atomics) {
  for (int64_t i = (blockIdx.x * blockDim.x) + threadIdx.x; i < n;
       i += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    torch::headeronly::fastAtomicAdd(
        out, indices[i], out_numel, values[i], fast_atomics);
  }
}

template <typename scalar_t, typename index_t = int64_t>
std::vector<scalar_t> launchScatterAdd(
    const std::vector<index_t>& indices,
    int64_t buf_numel,
    int64_t out_offset,
    int64_t out_numel,
    bool fast_atomics) {
  const int64_t n = static_cast<int64_t>(indices.size());
  DeviceBuffer<scalar_t> d_buf(std::vector<scalar_t>(buf_numel, scalar_t(0.0f)));
  DeviceBuffer<index_t> d_indices(indices);
  DeviceBuffer<scalar_t> d_values(std::vector<scalar_t>(n, scalar_t(1.0f)));

  constexpr int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  scatter_add_kernel<scalar_t, index_t><<<blocks, threads>>>(
      d_buf.get() + out_offset,
      d_indices.get(),
      d_values.get(),
      n,
      static_cast<index_t>(out_numel),
      fast_atomics);
  CUDA_EXPECT_OK(cudaGetLastError());
  CUDA_EXPECT_OK(cudaDeviceSynchronize());
  return d_buf.to_host();
}

template <typename scalar_t, typename index_t = int64_t>
void testScatterAdd(bool fast_atomics) {
  SKIP_IF_NO_CUDA_DEVICE();
  constexpr int64_t n = 2048;
  constexpr int64_t out_numel = 33;
  std::vector<index_t> indices(n);
  std::vector<int> expected(out_numel, 0);
  for (int64_t i = 0; i < n; i++) {
    indices[i] = static_cast<index_t>(i % out_numel);
    expected[i % out_numel]++;
  }
  const auto out = launchScatterAdd<scalar_t, index_t>(
      indices, out_numel, 0, out_numel, fast_atomics);
  for (int64_t j = 0; j < out_numel; j++) {
    EXPECT_EQ(static_cast<float>(out[j]), static_cast<float>(expected[j]))
        << "slot " << j << " fast_atomics=" << fast_atomics;
  }
}

template <typename scalar_t>
void testStaysInBounds() {
  SKIP_IF_NO_CUDA_DEVICE();
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

TEST(TestAtomicAdd, FastAtomicAddInt16) {
  testScatterAdd<int16_t>(true);
  testScatterAdd<int16_t>(false);
}

TEST(TestAtomicAdd, FastAtomicAddInt8) {
  testScatterAdd<int8_t>(true);
  testScatterAdd<int8_t>(false);
}

TEST(TestAtomicAdd, FastAtomicAddIntIndex) {
  testScatterAdd<float, int32_t>(true);
  testScatterAdd<float, int32_t>(false);
}

TEST(TestAtomicAdd, FastAtomicAddHalfStaysInBounds) {
  testStaysInBounds<torch::headeronly::Half>();
}

TEST(TestAtomicAdd, FastAtomicAddBFloat16StaysInBounds) {
  testStaysInBounds<torch::headeronly::BFloat16>();
}
