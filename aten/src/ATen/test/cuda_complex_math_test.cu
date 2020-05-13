#include <gtest/gtest.h>

int safeDeviceCount() {
  int count;
  cudaError_t err = cudaGetDeviceCount(&count);
  if (err == cudaErrorInsufficientDriver || err == cudaErrorNoDevice) {
    return 0;
  }
  return count;
}

#define SKIP_IF_NO_GPU()                    \
  do {                                      \
    if (safeDeviceCount() == 0) {           \
      return;                               \
    }                                       \
  } while(0)

#define C10_ASSERT_NEAR(a, b, tol) assert(abs(a - b) < tol)
#define C10_DEFINE_TEST(a, b)                       \
__global__ void CUDA##a##b();                       \
TEST(a##Device, b) {                                \
  SKIP_IF_NO_GPU();                                 \
  cudaDeviceSynchronize();                          \
  CUDA##a##b<<<1, 1>>>();                           \
  cudaDeviceSynchronize();                          \
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);       \
}                                                   \
__global__ void CUDA##a##b()
#include <c10/test/util/complex_math_test_common.h>


#undef C10_DEFINE_TEST
#undef C10_ASSERT_NEAR
#define C10_DEFINE_TEST(a, b) TEST(a##Host, b)
#define C10_ASSERT_NEAR(a, b, tol) ASSERT_NEAR(a, b, tol)
#include <c10/test/util/complex_math_test_common.h>

