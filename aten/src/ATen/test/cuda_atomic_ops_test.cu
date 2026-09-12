#include <gtest/gtest.h>
#include <ATen/cuda/Atomic.cuh>
#include <c10/test/util/Macros.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/KernelUtils.cuh>
#include <c10/cuda/CUDAException.h>

#include <ATen/ATen.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

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
__global__ void max_test_kernel(T * a, T * max) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int a_idx = (tid) % (arraysize * factor);
  int idx = a_idx / factor;

  gpuAtomicMax(&max[idx], a[a_idx]);
}

template <typename T>
__global__ void min_test_kernel(T * a, T * min) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int a_idx = (tid) % (arraysize * factor);
  int idx = a_idx / factor;

  gpuAtomicMin(&min[idx], a[a_idx]);
}

template <typename T>
__global__ void return_value_test_kernel(T * dst, T * ret) {
  *ret = gpuAtomicMax(dst, static_cast<T>(4));
}

// gpuAtomic* return the value in memory before the update. Max has no native
// fp16/bf16 instruction, so this exercises the AtomicFPOp Compare-and-Swap (CAS)
// loop on all archs.
template <typename T>
void test_atomic_return_value() {
  T *dstd, *retd;
  T dst = 2, ret = 0;

  cudaMalloc((void**)&dstd, sizeof(T));
  cudaMalloc((void**)&retd, sizeof(T));
  cudaMemcpy(dstd, &dst, sizeof(T), cudaMemcpyHostToDevice);

  return_value_test_kernel<<<1, 1>>>(dstd, retd);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  cudaMemcpy(&dst, dstd, sizeof(T), cudaMemcpyDeviceToHost);
  cudaMemcpy(&ret, retd, sizeof(T), cudaMemcpyDeviceToHost);

  ASSERT_EQ(ret, static_cast<T>(2)) << typeid(T).name();
  ASSERT_EQ(dst, static_cast<T>(4)) << typeid(T).name();

  cudaFree(dstd);
  cudaFree(retd);
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
    answer[i] = pow(sum[i], static_cast<T>(factor + 1));
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

template <typename T>
void test_atomic_max() {
  dim3 dimBlock(blocksize, 1);
  dim3 dimGrid(1, 1);

  T *ad, *sumd;

  std::vector<T> a(arraysize * factor);
  std::vector<T> sum(arraysize);
  std::vector<T> answer(arraysize);

  int j;
  for (int i = 0; i < arraysize * factor; ++i) {
    a[i] = i;
    if (i % factor == 0) {
      j = i / factor;
      sum[j] = std::numeric_limits<T>::lowest();
      answer[j] = (j + 1) * factor - 1;
    }
  }

  cudaMalloc((void**)&ad, arraysize * factor * sizeof(T));
  cudaMalloc((void**)&sumd, arraysize * sizeof(T));

  cudaMemcpy(ad, a.data(), arraysize * factor * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy(sumd, sum.data(), arraysize * sizeof(T), cudaMemcpyHostToDevice);

  max_test_kernel<<<dimGrid, dimBlock>>>(ad, sumd);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  cudaMemcpy(sum.data(), sumd, arraysize * sizeof(T), cudaMemcpyDeviceToHost);

  for (int i = 0; i < arraysize; ++i) {
    ASSERT_EQ(sum[i], answer[i]) << typeid(T).name();
  }

  cudaFree(ad);
  cudaFree(sumd);
}

template <typename T>
void test_atomic_min() {
  dim3 dimBlock(blocksize, 1);
  dim3 dimGrid(1, 1);

  T *ad, *sumd;

  std::vector<T> a(arraysize * factor);
  std::vector<T> sum(arraysize);
  std::vector<T> answer(arraysize);

  int j;
  for (int i = 0; i < arraysize * factor; ++i) {
    a[i] = i;
    if (i % factor == 0) {
      j = i / factor;
      sum[j] = std::numeric_limits<T>::max();
      answer[j] = j * factor;
    }
  }

  cudaMalloc((void**)&ad, arraysize * factor * sizeof(T));
  cudaMalloc((void**)&sumd, arraysize * sizeof(T));

  cudaMemcpy(ad, a.data(), arraysize * factor * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy(sumd, sum.data(), arraysize * sizeof(T), cudaMemcpyHostToDevice);

  min_test_kernel<<<dimGrid, dimBlock>>>(ad, sumd);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  cudaMemcpy(sum.data(), sumd, arraysize * sizeof(T), cudaMemcpyDeviceToHost);

  for (int i = 0; i < arraysize; ++i) {
    ASSERT_EQ(sum[i], answer[i]) << typeid(T).name();
  }

  cudaFree(ad);
  cudaFree(sumd);
}

TEST(TestAtomicOps, TestAtomicAdd) {
  if (!at::cuda::is_available()) return;
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
  if (!at::cuda::is_available()) return;
  test_atomic_mul<uint8_t>();
  test_atomic_mul<int8_t>();
  test_atomic_mul<int16_t>();
  test_atomic_mul<int32_t>();
  test_atomic_mul<int64_t>();
  test_atomic_mul<at::BFloat16>();
  test_atomic_mul<at::Half>();
  test_atomic_mul<float>();
  test_atomic_mul<double>();
}

TEST(TestAtomicOps, DISABLED_ON_WINDOWS(TestAtomicMax)) {
  if (!at::cuda::is_available()) return;
  test_atomic_max<uint8_t>();
  test_atomic_max<int8_t>();
  test_atomic_max<int16_t>();
  test_atomic_max<int32_t>();
  test_atomic_max<int64_t>();
  test_atomic_max<at::BFloat16>();
  test_atomic_max<at::Half>();
  test_atomic_max<float>();
  test_atomic_max<double>();
}

TEST(TestAtomicOps, DISABLED_ON_WINDOWS(TestAtomicMin)) {
  if (!at::cuda::is_available()) return;
  test_atomic_min<uint8_t>();
  test_atomic_min<int8_t>();
  test_atomic_min<int16_t>();
  test_atomic_min<int32_t>();
  test_atomic_min<int64_t>();
  test_atomic_min<at::BFloat16>();
  test_atomic_min<at::Half>();
  test_atomic_min<float>();
  test_atomic_min<double>();
}

TEST(TestAtomicOps, DISABLED_ON_WINDOWS(TestAtomicReturnValue)) {
  if (!at::cuda::is_available()) return;
  test_atomic_return_value<at::BFloat16>();
  test_atomic_return_value<at::Half>();
  test_atomic_return_value<float>();
  test_atomic_return_value<double>();
}

namespace {

// What the accessor overload has to get right is the bound it derives: the
// distance to the last addressable element, which stops matching the element
// count as soon as the accessor is strided.
//
// A pairing writes +0 to the neighbour, which no value check can see. The sign
// bit gives it away, since (-0.0) + (+0.0) == +0.0, so a buffer seeded with
// -0.0 answers "was this slot touched". Same trick as
// TestFastAtomicAdd.VectorizedBounds in
// test/cpp/aoti_abi_check/cuda/test_kernel_utils.cu, for the pointer form.
constexpr uint16_t kNegZeroBits = 0x8000; // -0.0 as both fp16 and bf16

template <typename scalar_t>
__global__ void accessor_scatter_kernel(
    at::PackedTensorAccessor64<scalar_t, 2> acc,
    int64_t i0,
    int64_t i1) {
  at::native::fastAtomicAdd(acc, static_cast<scalar_t>(1), i0, i1);
}

// A 2x3 view into a 1-D tensor, `offset` elements in with the given strides --
// the strided accessor is the object under test, so it is built the way one
// actually reaches a kernel. Scatters +1 at (i0, i1) and returns the whole
// buffer, so a write outside the view's region shows up.
template <typename scalar_t>
std::vector<scalar_t> accessorScatter(
    std::vector<scalar_t> buf,
    int64_t offset,
    int64_t row_stride,
    int64_t col_stride,
    int64_t i0,
    int64_t i1) {
  const auto opts =
      at::TensorOptions().dtype(c10::CppTypeToScalarType<scalar_t>::value);
  const auto base =
      at::from_blob(buf.data(), {static_cast<int64_t>(buf.size())}, opts)
          .to(at::kCUDA);
  const auto view =
      at::as_strided(base, {2, 3}, {row_stride, col_stride}, offset);

  accessor_scatter_kernel<scalar_t>
      <<<1, 1>>>(view.packed_accessor64<scalar_t, 2>(), i0, i1);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const auto out = base.cpu();
  std::memcpy(buf.data(), out.const_data_ptr(), buf.size() * sizeof(scalar_t));
  return buf;
}

template <typename scalar_t>
bool wasWritten(scalar_t v) {
  return v.x != kNegZeroBits;
}

template <typename scalar_t>
void test_accessor_bounds() {
  constexpr int64_t buf_numel = 16;
  constexpr int64_t offset = 1; // odd 16-bit alignment, as a view would give

  // Contiguous 2x3, strides 3/1: memory_span is 6. At this alignment even
  // offsets pair left and odd ones pair right, so (0,1) at offset 1 pairs and
  // is the positive control, while (1,2) at offset 5 can only look right, past
  // the last element, and must fall back to a scalar atomic.
  {
    std::vector<scalar_t> buf(buf_numel);
    for (auto& v : buf) {
      v.x = kNegZeroBits;
    }
    const auto out =
        accessorScatter<scalar_t>(std::move(buf), offset, 3, 1, 0, 1);
    EXPECT_EQ(static_cast<float>(out[offset + 1]), 1.0f);
    EXPECT_TRUE(wasWritten(out[offset + 2])) << "offset 1 should pair right";
    EXPECT_FALSE(wasWritten(out[0])) << "wrote below the start";
  }
  {
    std::vector<scalar_t> buf(buf_numel);
    for (auto& v : buf) {
      v.x = kNegZeroBits;
    }
    const auto out =
        accessorScatter<scalar_t>(std::move(buf), offset, 3, 1, 1, 2);
    EXPECT_EQ(static_cast<float>(out[offset + 5]), 1.0f);
    EXPECT_FALSE(wasWritten(out[offset + 6])) << "paired past the last element";
    EXPECT_FALSE(wasWritten(out[buf_numel - 1])) << "wrote past the end";
  }

  // Non-contiguous 2x3, strides 6/2: element count is still 6, but the last
  // addressable offset is 10, so memory_span is 11. Writing (1, 2) lands there
  // and pairs left into offset 9, a gap the view steps over -- legal, and what
  // a numel-derived bound of 6 would wrongly have refused.
  {
    std::vector<scalar_t> buf(buf_numel);
    for (auto& v : buf) {
      v.x = kNegZeroBits;
    }
    const auto out =
        accessorScatter<scalar_t>(std::move(buf), offset, 6, 2, 1, 2);
    EXPECT_EQ(static_cast<float>(out[offset + 10]), 1.0f);
    EXPECT_TRUE(wasWritten(out[offset + 9])) << "declined to pair inside the region";
    EXPECT_FALSE(wasWritten(out[offset + 11])) << "paired past the region";
    EXPECT_FALSE(wasWritten(out[0])) << "wrote below the start";
  }
}
} // namespace

TEST(TestAtomicOps, DISABLED_ON_WINDOWS(TestFastAtomicAddAccessorBounds)) {
  if (!at::cuda::is_available()) return;
  test_accessor_bounds<at::Half>();
  // fastSpecializedAtomicAdd has no packed bf16 path before sm_80, so the
  // pairing these cases assert on does not happen there.
  if (at::cuda::getCurrentDeviceProperties()->major >= 8) {
    test_accessor_bounds<at::BFloat16>();
  }
}
