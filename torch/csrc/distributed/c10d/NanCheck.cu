#ifdef USE_C10D_NCCL

#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/torch.h>
#include <algorithm>
#include <torch/csrc/distributed/c10d/NanCheck.hpp>
#include <stdint.h>

namespace c10d {

// CUDA kernel to check if data has NAN, device side assert
// is raised if NAN is found

// Using ulong2 as a "byte pack", with 16 bytes, for efficient data load
union BytePack16 {
  ulong2 ul2;
  uint64_t ul[2];
};

typedef union BytePack16 BytePack;

// AMD HIP doesn't define `__trap()`, using `assert` instead
#ifdef USE_ROCM
#define __trap() assert(0)
#endif

//// Start of templated functions for checking NaNs inside a BytePack

// (i) General implementation (aka fallback)
// We use a for loop to iterate over the elements in a BytePack.
// EltPerPack would be greater than 8 if falling in this case.

template <typename T, int EltPerPack>
struct CheckBytePack {
  static __device__ __forceinline__ void check(BytePack* tmp) {
    T* data = (T*)tmp;
    #pragma unroll 8
    for (int i = 0; i < EltPerPack; i++) {
      if (isnan(data[i])) __trap();
    }
  }
};

// (ii) Template Specialization for 8-byte data types, e.g. double
// EltPerPack = 16 / 8 = 2

template <typename T>
struct CheckBytePack<T, /*EltPerPack*/2> {
  static __device__ __forceinline__ void check(BytePack* tmp) {
    T* data = (T*)tmp;
    if (isnan(data[0]) || isnan(data[1])) __trap();
  }
};

// (iii) Template specialization for 4-byte data types, e.g. float32
// EltPerPack = 16 / 4 = 4

template <typename T>
struct CheckBytePack<T, /*EltPerPack*/4> {
  static __device__ __forceinline__ void check(BytePack* tmp) {
    T* data = (T*)tmp;
    if (isnan(data[0]) || isnan(data[1]) || isnan(data[2]) || isnan(data[3])) __trap();
  }
};

// (iv) Template specialization for 2-byte data types, e.g. float16, bfloat16, half.
// EltPerPack = 16 / 2 = 8

template <typename T>
struct CheckBytePack<T, /*EltPerPack*/8> {
  static __device__ __forceinline__ void check(BytePack* tmp) {
    T* data = (T*)tmp;
    if (isnan(data[0]) || isnan(data[1]) || isnan(data[2]) || isnan(data[3]) ||
        isnan(data[4]) || isnan(data[5]) || isnan(data[6]) || isnan(data[7])) {
          __trap();
    }
  }
};

// (v) Template specialization for Float8 types.
// EltPerPack = 16 / 1 = 16

// We want to check 8 x FP8 simultaneously, hence this template definition.
template<typename T>
struct HasNanFP8x8 {
  static __device__ __forceinline__ bool check(uint64_t fp8x8) = delete;
  /*
  {
    // `static_assert` in template definition requires c++23 onwards.
    // But the error message still applies if you find yourself here.
    static_assert(
      false,
      "You should never call this template definition because it is empty. You "
      "can follow the example of Float8_e4m3fn below to implement the check for "
      "your new datatype."
    );
  }
  */
};

// isnan condition for Float8_e4m3fn:
// (x & 0b01111111) == 0b01111111
// i.e.
// (x & 0x7f) == 0x7f

// The algorithm is as follows:
// (1) Mask out the most significant bit with mask 0x7f.
// (2) If the result is 0x7f (is nan), the following arithmetic would cause the
//     8th bit to be 1: x[i] = x[i] + 0x01
// (3) Only leave the 8th bit by masking with 0x80.
// (4) If any x[i] is nan, then the whole x != 0.

template<>
struct HasNanFP8x8<c10::Float8_e4m3fn> {
  static __device__ __forceinline__ bool check(uint64_t fp8x8) {
    auto t = fp8x8 & 0x7F7F7F7F7F7F7F7FULL;
    auto incremented = t + 0x0101010101010101ULL;
    auto overflow = incremented & 0x8080808080808080ULL;
    return overflow != 0;
  }
};

// isnan condition for Float8_e5m2:
// (x & 0x7f) > 0x7c
// This case does not overflow: 0x7c + 0x03 == 0x7f but adding 0x03 to anything
// greater than 0x7c will overflow.

template<>
struct HasNanFP8x8<c10::Float8_e5m2> {
  static __device__ __forceinline__ bool check(uint64_t fp8x8) {
    auto t = fp8x8 & 0x7F7F7F7F7F7F7F7FULL;
    auto incremented = t + 0x0303030303030303ULL;
    auto overflow = incremented & 0x8080808080808080ULL;
    return overflow != 0;
  }
};

template<typename T>
struct CheckBytePack<T, /*EltPerPack*/16> {
  static __device__ __forceinline__ void check(BytePack* tmp) {
    if (HasNanFP8x8<T>::check(tmp->ul[0]) || HasNanFP8x8<T>::check(tmp->ul[1]))
        __trap();
  }
};

//// End of templated functions for checking NaNs inside a BytePack


// Fast-path check routine:
// each thread will load and check 8 BytePacks in this routine

// Create a tmp buffer of size 8, also unroll for loop by 8
#define UNROLL 8

template <typename T>
__device__ __forceinline__ void checkChunk(BytePack* ptr) {
  BytePack tmp[UNROLL];
  int nWorkers = blockDim.x * gridDim.x;
  // First load values from global memory into tmp buffer
  #pragma unroll 8
  for (int j = 0; j < UNROLL; j++) {
    tmp[j] = ptr[nWorkers * j];
  }
  // Then check each BytePack in the tmp buffer
  #pragma unroll 8
  for (int j = 0; j < UNROLL; j++) {
    CheckBytePack<T, sizeof(BytePack)/sizeof(T)>::check(tmp + j);
  }
  // Note: we separate the check from the load for efficient loading
}

// Align address of `ptr` up, to the alignment of `T`
#define ALIGN_UP(ptr, T) (((uintptr_t)ptr + sizeof(T) - 1) / sizeof(T) * sizeof(T))

// This is the host-facing kernel

template <typename T>
__global__ void checkForNaN(T* data, size_t size) {
  constexpr int EltPerPack = sizeof(BytePack) / sizeof(T);
  // Offset of current thread
  size_t offset = blockIdx.x * blockDim.x + threadIdx.x;

  // Align input address up to BytePack in case it is not
  T* ptrAlign = (T*)ALIGN_UP(data, BytePack);
  // Pre-process the data before alignment
  size_t preProcElts = min(ptrAlign - data, size);
  // Read memory by T (slow). One iter is enough bc the number of threads would
  // be bigger than `preProcElts`
  if (offset < preProcElts) {
    if (isnan(data[offset])) __trap();
  }
  // We have processes this amount of data
  size -= preProcElts;

  // Start BytePack processing
  BytePack* ptr = (BytePack*)ptrAlign;
  // Size of input data in unit of BytePack
  size_t sizeInBP = size * sizeof(T) / sizeof(BytePack);
  // Number of BytePacks processed in one fast-path iteration
  size_t loopSize = blockDim.x * gridDim.x * UNROLL;

  // Fast path
  // The condition below makes sure there is enough data to process (`loopSize`)
  for (; offset + loopSize <= sizeInBP; offset += loopSize) {
    checkChunk<T>(ptr + offset);
  }

  // The rest data goes on slow path
  // We just do regular load and check
  for (; offset < sizeInBP; offset += blockDim.x * gridDim.x) {
    BytePack tmp = ptr[offset];
    CheckBytePack<T, EltPerPack>::check(&tmp);
  }

  // We can still have a tail smaller than 1 BytePack
  // TODO: merge this tail check with head check to make them concurrent
  if (threadIdx.x < size % EltPerPack) {
    T* tailPtr = (T*)(ptr + sizeInBP);
    if (isnan(tailPtr[threadIdx.x])) __trap();
  }
}

// CHECK if a Tensor contains NAN in any of its element
void checkForNan(const at::Tensor& tensor, at::cuda::CUDAStream& stream) {
  // skip check for non float types
  if (!torch::is_floating_point(tensor)) {
    return;
  }
  const size_t maxNumThreadsPerBlock = 512;
  const size_t maxNumBlocks = 24;
  const size_t numThreadsPerBlock =
      std::min<size_t>(maxNumThreadsPerBlock, tensor.numel());

  const size_t numBlocks = std::min<size_t>(
      maxNumBlocks,
      (tensor.numel() + numThreadsPerBlock - 1) / numThreadsPerBlock);

  AT_DISPATCH_FLOATING_TYPES_AND4(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      at::ScalarType::Float8_e4m3fn,
      at::ScalarType::Float8_e5m2,
      tensor.scalar_type(),
      "checkForNaN",
      [&] {
        checkForNaN<scalar_t><<<numBlocks, numThreadsPerBlock, 0, stream>>>(
            tensor.data_ptr<scalar_t>(), tensor.numel());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
}

} // namespace c10d

#endif // USE_C10D_NCCL
