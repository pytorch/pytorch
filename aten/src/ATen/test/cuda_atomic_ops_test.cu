#include <gtest/gtest.h>
#include <THC/THCAtomics.cuh>
#include <c10/test/util/Macros.h>

#include <cmath>

constexpr int blocksize = 256;
constexpr int factor = 4;
constexpr int arraysize = blocksize / factor;

template <typename T>
__global__ void addition_test_kernel(T * a, T * sum) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int idx = (tid) % arraysize;

  gpuAtomicAdd(&sum[idx], a[idx]);
}

template <typename T>
__global__ void mul_test_kernel(T * a, T * sum) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int idx = (tid) % arraysize;

  gpuAtomicMul(&sum[idx], a[idx]);
}

template <typename T>
void test_atomic_add() {
  dim3 dimBlock(blocksize, 1);
  dim3 dimGrid(1, 1);

  T *ad, *sumd;

  std::vector<T> a(arraysize);
  std::vector<T> sum(arraysize);
  std::vector<T> answer(arraysize);

  for (int i = 0; i < arraysize; ++i) {
    a[i] = 1;
    sum[i] = 0;
    answer[i] = factor;
  }

  cudaMalloc((void**)&ad, arraysize * sizeof(T));
  cudaMalloc((void**)&sumd, arraysize * sizeof(T));

  cudaMemcpy(ad, a.data(), arraysize * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy(sumd, sum.data(), arraysize * sizeof(T), cudaMemcpyHostToDevice);

  addition_test_kernel<<<dimGrid, dimBlock>>>(ad, sumd);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  cudaMemcpy(sum.data(), sumd, arraysize * sizeof(T), cudaMemcpyDeviceToHost);

  for (int i = 0; i < arraysize; ++i) {
    ASSERT_EQ(sum[i], answer[i]) << typeid(T).name();
  }

  cudaFree(ad);
  cudaFree(sumd);
}

template <typename T>
void test_atomic_mul() {
  dim3 dimBlock(blocksize, 1);
  dim3 dimGrid(1, 1);

  T *ad, *sumd;

  std::vector<T> a(arraysize);
  std::vector<T> sum(arraysize);
  std::vector<T> answer(arraysize);

  for (int i = 0; i < arraysize; ++i) {
    a[i] = 2;
    sum[i] = 2;
    answer[i] = pow(sum[i], static_cast<T>(factor));
  }

  cudaMalloc((void**)&ad, arraysize * sizeof(T));
  cudaMalloc((void**)&sumd, arraysize * sizeof(T));

  cudaMemcpy(ad, a.data(), arraysize * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy(sumd, sum.data(), arraysize * sizeof(T), cudaMemcpyHostToDevice);

  mul_test_kernel<<<dimGrid, dimBlock>>>(ad, sumd);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  cudaMemcpy(sum.data(), sumd, arraysize * sizeof(T), cudaMemcpyDeviceToHost);

  for (int i = 0; i < arraysize; ++i) {
    ASSERT_EQ(sum[i], answer[i]) << typeid(T).name();
  }

  cudaFree(ad);
  cudaFree(sumd);
}

TEST(TestAtomicOps, TestAtomicAdd) {
  test_atomic_add<uint8_t>();
  test_atomic_add<int8_t>();
  test_atomic_add<int16_t>();
  test_atomic_add<int32_t>();
  test_atomic_add<int64_t>();

  test_atomic_add<at::BFloat16>();
  test_atomic_add<at::Half>();
  test_atomic_add<float>();
  test_atomic_add<double>();
  test_atomic_add<c10::complex<float> >();
  test_atomic_add<c10::complex<double> >();
}

TEST(TestAtomicOps, DISABLED_ON_WINDOWS(TestAtomicMul)) {
  test_atomic_mul<at::BFloat16>();
  test_atomic_mul<at::Half>();
  test_atomic_mul<float>();
  test_atomic_mul<double>();
}
