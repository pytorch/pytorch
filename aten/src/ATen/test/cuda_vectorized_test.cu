#include <gtest/gtest.h>
#include <ATen/native/cuda/MemoryAccess.cuh>

using namespace at::native::memory;
__managed__ double4 buffer1[1024];
__managed__ double4 buffer2[1024];

void reset_buffers() {
  for (int i = 0; i < 1024; i++) {
    buffer1[i].x = i;
    buffer1[i].y = i + 0.1;
    buffer1[i].z = i + 0.2;
    buffer1[i].t = i + 0.3;

    buffer2[2].x = -i;
    buffer2[2].y = -(i + 0.1);
    buffer2[2].z = -(i + 0.2);
    buffer2[2].t = -(i + 0.3);
  }
}

TEST(TestVectorizedMemoryAccess, CanVectorizeUpTo) {
  char *ptr = reinterpret_cast<char *>(buffer1);

  ASSERT_EQ(can_vectorize_up_to<bool>(ptr), 1);
  ASSERT_EQ(can_vectorize_up_to<int8_t>(ptr), 4);
  ASSERT_EQ(can_vectorize_up_to<int16_t>(ptr), 4);
  ASSERT_EQ(can_vectorize_up_to<int>(ptr), 4);
  ASSERT_EQ(can_vectorize_up_to<int64_t>(ptr), 4);

  ASSERT_EQ(can_vectorize_up_to<bool>(ptr + 1), 1);
  ASSERT_EQ(can_vectorize_up_to<int8_t>(ptr + 1), 1);

  ASSERT_EQ(can_vectorize_up_to<bool>(ptr + 2), 1);
  ASSERT_EQ(can_vectorize_up_to<int8_t>(ptr + 2), 2);
  ASSERT_EQ(can_vectorize_up_to<int16_t>(ptr + 2), 1);

  ASSERT_EQ(can_vectorize_up_to<bool>(ptr + 4), 1);
  ASSERT_EQ(can_vectorize_up_to<int8_t>(ptr + 4), 4);
  ASSERT_EQ(can_vectorize_up_to<int16_t>(ptr + 4), 2);
  ASSERT_EQ(can_vectorize_up_to<int>(ptr + 4), 1);

  ASSERT_EQ(can_vectorize_up_to<bool>(ptr + 8), 1);
  ASSERT_EQ(can_vectorize_up_to<int8_t>(ptr + 8), 4);
  ASSERT_EQ(can_vectorize_up_to<int16_t>(ptr + 8), 4);
  ASSERT_EQ(can_vectorize_up_to<int>(ptr + 8), 2);
  ASSERT_EQ(can_vectorize_up_to<int64_t>(ptr + 8), 1);
}

template <typename scalar_t, int vec_size>
__global__ void vectorized_copy(scalar_t *dst, scalar_t *src) {
  using vectorized = vectorized<scalar_t, 64, 256, vec_size>;
  scalar_t buf[vectorized::thread_work_size];
  vectorized::load(buf, src + 256 * blockIdx.x);
  vectorized::store(dst + 256 * blockIdx.x, buf);
}

TEST(TestVectorizedMemoryAccess, CopyKernel) {
  double *b1 = reinterpret_cast<double *>(buffer1);
  double *b2 = reinterpret_cast<double *>(buffer2);

  // vec4 copy
  reset_buffers();
  cudaDeviceSynchronize();
  vectorized_copy<double, 4><<<16, 64>>>(b2, b1);
  cudaDeviceSynchronize();
  for (int i = 0; i < 1024; i++) {
    ASSERT_EQ(buffer1[i], buffer2[i]);
  }

  // vec2 copy
  reset_buffers();
  cudaDeviceSynchronize();
  vectorized_copy<double, 2><<<16, 64>>>(b2, b1);
  cudaDeviceSynchronize();
  for (int i = 0; i < 1024; i++) {
    ASSERT_EQ(buffer1[i], buffer2[i]);
  }

  // vec1 copy
  reset_buffers();
  cudaDeviceSynchronize();
  vectorized_copy<double, 1><<<16, 64>>>(b2, b1);
  cudaDeviceSynchronize();
  for (int i = 0; i < 1024; i++) {
    ASSERT_EQ(buffer1[i], buffer2[i]);
  }

  // unaligned
  for (int i = 0; i <= 16; i++) {
    for (int j = 0; j <= 16; j++) {
      b1 = reinterpret_cast<double *>(reinterpret_cast<char *>(buffer1) + i);
      b2_ = reinterpret_cast<double *>(reinterpret_cast<char *>(buffer2) + j);
      auto f = []() {
        cudaDeviceSynchronize();
        vectorized_copy<double, 4><<<1, 64>>>(b2, b1);
        cudaDeviceSynchronize();
      };
      if (i % 16 == 0 && j % 16 == 0) {
        f();
      } else {
        ASSERT_THROW(f(), std::runtime_error);
      }
    }
  }
}
