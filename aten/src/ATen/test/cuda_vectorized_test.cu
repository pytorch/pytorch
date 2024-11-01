#include <gtest/gtest.h>
#include <ATen/ATen.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/MemoryAccess.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/core/Array.h>

using namespace at::native;
using namespace at::native::memory;

constexpr int buffer_size = 1024;

__managed__ double4 buffer1[buffer_size];
__managed__ double4 buffer2[buffer_size];

void reset_buffers() {
  for (int i = 0; i < buffer_size; i++) {
    buffer1[i].x = i;
    buffer1[i].y = i + 0.1;
    buffer1[i].z = i + 0.2;
    buffer1[i].w = i + 0.3;

    buffer2[2].x = -i;
    buffer2[2].y = -(i + 0.1);
    buffer2[2].z = -(i + 0.2);
    buffer2[2].w = -(i + 0.3);
  }
}

#if defined(USE_ROCM)
TEST(TestLoops, HasSameArgTypes) {
  // This is a compile-time unit test. If this file compiles without error,
  // then the test passes and during runtime, we just need to return.
  using namespace at::native::modern::detail;
  using func1_t = int (*)(float, float);
  using func2_t = int (*)(bool, float, float);
  using func3_t = int (*)(float);
  using func4_t = int (*)();
  static_assert(has_same_arg_types<func1_t>::value, "func1_t has the same argument types");
  static_assert(!has_same_arg_types<func2_t>::value, "func2_t does not have the same argument types");
  static_assert(has_same_arg_types<func3_t>::value, "func3_t has the same argument types");
  static_assert(has_same_arg_types<func4_t>::value, "func4_t has the same argument types");
  return;
}
#endif

TEST(TestVectorizedMemoryAccess, CanVectorizeUpTo) {
  char *ptr = reinterpret_cast<char *>(buffer1);

  ASSERT_EQ(memory::can_vectorize_up_to<bool>(ptr), 4);
  ASSERT_EQ(memory::can_vectorize_up_to<int8_t>(ptr), 4);
  ASSERT_EQ(memory::can_vectorize_up_to<int16_t>(ptr), 4);
  ASSERT_EQ(memory::can_vectorize_up_to<int>(ptr), 4);
  ASSERT_EQ(memory::can_vectorize_up_to<int64_t>(ptr), 4);

  ASSERT_EQ(memory::can_vectorize_up_to<bool>(ptr + 1), 1);
  ASSERT_EQ(memory::can_vectorize_up_to<int8_t>(ptr + 1), 1);

  ASSERT_EQ(memory::can_vectorize_up_to<bool>(ptr + 2), 2);
  ASSERT_EQ(memory::can_vectorize_up_to<int8_t>(ptr + 2), 2);
  ASSERT_EQ(memory::can_vectorize_up_to<int16_t>(ptr + 2), 1);

  ASSERT_EQ(memory::can_vectorize_up_to<bool>(ptr + 4), 4);
  ASSERT_EQ(memory::can_vectorize_up_to<int8_t>(ptr + 4), 4);
  ASSERT_EQ(memory::can_vectorize_up_to<int16_t>(ptr + 4), 2);
  ASSERT_EQ(memory::can_vectorize_up_to<int>(ptr + 4), 1);

  ASSERT_EQ(memory::can_vectorize_up_to<bool>(ptr + 8), 4);
  ASSERT_EQ(memory::can_vectorize_up_to<int8_t>(ptr + 8), 4);
  ASSERT_EQ(memory::can_vectorize_up_to<int16_t>(ptr + 8), 4);
  ASSERT_EQ(memory::can_vectorize_up_to<int>(ptr + 8), 2);
  ASSERT_EQ(memory::can_vectorize_up_to<int64_t>(ptr + 8), 1);
}

// The following kernel copy values by using vectorized policies
// defined in `ATen/native/cuda/MemoryAccess.cuh`
template <typename scalar_t, int vec_size>
__global__ void vectorized_copy(scalar_t *dst, scalar_t *src) {
  static_assert(vec_size <= thread_work_size() && thread_work_size() % vec_size == 0, "Invalid vec_size");
  using array_t = at::detail::Array<char*, 2>;
  array_t data;
  data[0] = reinterpret_cast<char *>(dst);
  data[1] = reinterpret_cast<char *>(src);
  int idx = blockIdx.x;
  using vectorized = policies::vectorized<vec_size, array_t, thread_work_size()>;
  auto policy = vectorized(data);
  scalar_t buf[thread_work_size()];
#if !defined(USE_ROCM)
  // This fails only on CUDA 10.x, remove this after CUDA 10.x support is dropped
  scalar_t *buf_ = &buf[0];
  auto accessor = [&](int index) -> scalar_t & { return buf_[index]; };
#else
  auto accessor = [&](int index) -> scalar_t & { return buf[index]; };
#endif
  policy.load_single_arg(accessor, src + block_work_size() * blockIdx.x);
  policy.store(buf, idx);
}

TEST(TestVectorizedMemoryAccess, CopyKernel) {
  if (!at::cuda::is_available()) {
    return;
  }

  double *b1 = reinterpret_cast<double *>(buffer1);
  double *b2 = reinterpret_cast<double *>(buffer2);

  // vec4 copy
  reset_buffers();
  cudaDeviceSynchronize();
  constexpr int total_work_size = buffer_size * 4;
  vectorized_copy<double, 4><<<total_work_size / block_work_size() , num_threads()>>>(b2, b1);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
  for (int i = 0; i < 1024; i++) {
    ASSERT_EQ(buffer1[i].x, buffer2[i].x);
    ASSERT_EQ(buffer1[i].y, buffer2[i].y);
    ASSERT_EQ(buffer1[i].z, buffer2[i].z);
    ASSERT_EQ(buffer1[i].w, buffer2[i].w);
  }

  // vec2 copy
  reset_buffers();
  cudaDeviceSynchronize();
  vectorized_copy<double, 2><<<total_work_size / block_work_size() , num_threads()>>>(b2, b1);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
  for (int i = 0; i < 1024; i++) {
    ASSERT_EQ(buffer1[i].x, buffer2[i].x);
    ASSERT_EQ(buffer1[i].y, buffer2[i].y);
    ASSERT_EQ(buffer1[i].z, buffer2[i].z);
    ASSERT_EQ(buffer1[i].w, buffer2[i].w);
  }

  // vec1 copy
  reset_buffers();
  cudaDeviceSynchronize();
  vectorized_copy<double, 1><<<total_work_size / block_work_size() , num_threads()>>>(b2, b1);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());
  for (int i = 0; i < 1024; i++) {
    ASSERT_EQ(buffer1[i].x, buffer2[i].x);
    ASSERT_EQ(buffer1[i].y, buffer2[i].y);
    ASSERT_EQ(buffer1[i].z, buffer2[i].z);
    ASSERT_EQ(buffer1[i].w, buffer2[i].w);
  }

// Skipping this part until https://github.com/pytorch/pytorch/issues/51863 is resolved
#if 0
  // unaligned
  for (int i = 0; i < 16; i++) {
    for (int j = 0; j < 16; j++) {
      b1 = reinterpret_cast<double *>(reinterpret_cast<char *>(buffer1) + i);
      b2 = reinterpret_cast<double *>(reinterpret_cast<char *>(buffer2) + j);
      (void)cudaGetLastError();
      cudaDeviceSynchronize();
      vectorized_copy<double, 4><<<1, num_threads()>>>(b2, b1);
      cudaDeviceSynchronize();
      auto err = cudaGetLastError();
      if (i % 16 == 0 && j % 16 == 0) {
        ASSERT_EQ(err, cudaSuccess);
      } else {
        ASSERT_EQ(err, cudaErrorMisalignedAddress);
      }
    }
  }
#endif
}
