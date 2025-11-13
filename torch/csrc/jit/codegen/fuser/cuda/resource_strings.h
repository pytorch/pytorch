#pragma once

#include <ATen/code_template.h>
#include <torch/csrc/Export.h>

namespace torch::jit::fuser::cuda {

/*with type_as not checking type of its input, a fusion group can have non-fp32
tensor as input. Correct code for this case is generated, however, nvrtc does
not know how to handle int*_t integer types, so typedefs help it handle those
cases*/

static constexpr auto bfloat16_type_string = "__nv_bfloat16";

#if defined(USE_ROCM) && ROCM_VERSION < 70000
static auto type_declarations_template = at::jit::CodeTemplate(R"(
${HalfHeader}
${BFloat16Header}
${RandHeader}

#define NAN __int_as_float(0x7fffffff)
#define POS_INFINITY __int_as_float(0x7f800000)
#define NEG_INFINITY __int_as_float(0xff800000)

typedef ${IndexType} IndexType;
template<typename T, size_t N>
struct TensorInfo {
  T* data;
  IndexType sizes[N];
  IndexType strides[N];
};
template<typename T>
struct TensorInfo<T, 0> {
  T * data;
};
)");
#else
static auto type_declarations_template = at::jit::CodeTemplate(R"(
typedef unsigned char uint8_t;
typedef signed char int8_t;
typedef short int  int16_t;
typedef long long int int64_t;
typedef unsigned long long int uint64_t;
${HalfHeader}
${BFloat16Header}
${RandHeader}

#define NAN __int_as_float(0x7fffffff)
#define POS_INFINITY __int_as_float(0x7f800000)
#define NEG_INFINITY __int_as_float(0xff800000)

typedef ${IndexType} IndexType;
template<typename T, size_t N>
struct TensorInfo {
  T* data;
  IndexType sizes[N];
  IndexType strides[N];
};
template<typename T>
struct TensorInfo<T, 0> {
  T * data;
};
)");
#endif

// We rewrite the code for philox RNG from curand as nvrtc couldn't resolve the
// curand header correctly.
constexpr auto rand_support_literal = R"(

  class Philox {
  public:
    __device__ inline Philox(unsigned long long seed,
                             unsigned long long subsequence,
                             unsigned long long offset) {
      key.x = (unsigned int)seed;
      key.y = (unsigned int)(seed >> 32);
      counter = make_uint4(0, 0, 0, 0);
      counter.z = (unsigned int)(subsequence);
      counter.w = (unsigned int)(subsequence >> 32);
      STATE = 0;
      incr_n(offset / 4);
    }

    __device__ inline unsigned long operator()() {
      if(STATE == 0) {
        uint4 counter_ = counter;
        uint2 key_ = key;
        for(int i = 0; i < 9; i++) {
          counter_ = single_round(counter_, key_);
          key_.x += (kPhilox10A); key_.y += (kPhilox10B);
        }
        output = single_round(counter_, key_);
        incr();
      }
      unsigned long ret;
      switch(STATE) {
        case 0: ret = output.x; break;
        case 1: ret = output.y; break;
        case 2: ret = output.z; break;
        case 3: ret = output.w; break;
      }
      STATE = (STATE + 1) % 4;
      return ret;
    }

  private:
    uint4 counter;
    uint4 output;
    uint2 key;
    unsigned int STATE;
    __device__ inline void incr_n(unsigned long long n) {
      unsigned int nlo = (unsigned int)(n);
      unsigned int nhi = (unsigned int)(n >> 32);
      counter.x += nlo;
      if (counter.x < nlo)
        nhi++;
      counter.y += nhi;
      if (nhi <= counter.y)
        return;
      if (++counter.z)
        return;
      ++counter.w;
    }
    __device__ inline void incr() {
      if (++counter.x)
        return;
      if (++counter.y)
        return;
      if (++counter.z)
        return;
      ++counter.w;
    }
    __device__ unsigned int mulhilo32(unsigned int a, unsigned int b,
                                      unsigned int *result_high) {
      *result_high = __umulhi(a, b);
      return a*b;
    }

    __device__ inline uint4 single_round(uint4 ctr, uint2 key) {
      unsigned int hi0;
      unsigned int hi1;
      unsigned int lo0 = mulhilo32(kPhiloxSA, ctr.x, &hi0);
      unsigned int lo1 = mulhilo32(kPhiloxSB, ctr.z, &hi1);

      uint4 ret = {hi1 ^ ctr.y ^ key.x, lo1, hi0 ^ ctr.w ^ key.y, lo0};
      return ret;
    }

    static const unsigned long kPhilox10A = 0x9E3779B9;
    static const unsigned long kPhilox10B = 0xBB67AE85;
    static const unsigned long kPhiloxSA = 0xD2511F53;
    static const unsigned long kPhiloxSB = 0xCD9E8D57;
  };

  // Inverse of 2^32.
  #define M_RAN_INVM32 2.3283064e-10f
  __device__  __inline__ float uniform(unsigned int x) {
    return x * M_RAN_INVM32;
  }
)";

constexpr auto rand_param =
    ",unsigned long long seed, unsigned long long offset";

constexpr auto rand_init = R"(
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  Philox rnd(seed, idx, offset);
)";

static auto cuda_compilation_unit_template = at::jit::CodeTemplate(R"(
${type_declarations}

extern "C" __global__
void ${kernelName}(IndexType totalElements, ${formals} ${RandParam}) {
  ${RandInit}
  // check whether do vectorized load/store and allocate buffer
  bool flag_vec4 = true;
  ${tensorChecks}
  if (flag_vec4) {
    for (IndexType linearIndex = 4 * (blockIdx.x * blockDim.x + threadIdx.x);
         linearIndex < totalElements;
         linearIndex += 4 * gridDim.x * blockDim.x) {
      // Convert `linearIndex` into an offset of tensor as it is:
      ${tensorOffsets}
      // load 4 at a time
      ${kernelLoad}
      #pragma unroll 4
      for (int i=0; i<4; i++) {
        // calculate the results
        ${kernelBody_vec4}
      }
      // store 4 at a time
      ${kernelStore}
    }
  } else {
    for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
         linearIndex < totalElements;
         linearIndex += gridDim.x * blockDim.x) {
      // Convert `linearIndex` into an offset of tensor:
      ${tensorOffsets}
      // calculate the results
      ${kernelBody}
    }
  }
}
)");

// This snippet enables half support in the jit. Following the pattern for
// reductions, fp16 input data is immediately upconverted to float
// with __half2float(). All mathematical operations are done on float
// values, and if needed the intermediate float representation is
// converted to half with __float2half() when writing to a half tensor.
#if defined(USE_ROCM)
constexpr auto half_support_literal =
    R"(
typedef __half half;
)";
#else
constexpr auto half_support_literal =
    R"(
#define __HALF_TO_US(var) *(reinterpret_cast<unsigned short *>(&(var)))
#define __HALF_TO_CUS(var) *(reinterpret_cast<const unsigned short *>(&(var)))
#if defined(__cplusplus)
  struct __align__(2) __half {
    __host__ __device__ __half() { }

  protected:
    unsigned short __x;
  };

  /* All intrinsic functions are only available to nvcc compilers */
  #if defined(__CUDACC__)
    /* Definitions of intrinsics */
    __device__ __half __float2half(const float f) {
      __half val;
      asm("{  cvt.rn.f16.f32 %0, %1;}\n" : "=h"(__HALF_TO_US(val)) : "f"(f));
      return val;
    }

    __device__ float __half2float(const __half h) {
      float val;
      asm("{  cvt.f32.f16 %0, %1;}\n" : "=f"(val) : "h"(__HALF_TO_CUS(h)));
      return val;
    }
)"
    // MSVC's preprocessor (but not the standard compiler) has a bug
    // where it incorrectly tokenizes raw string literals, ending when it sees a
    // " this causes the #endif in this string literal to be treated as a
    // preprocessor token which, in turn, cause sccache on windows CI to fail.
    // See https://godbolt.org/z/eVTIJq as an example.
    // This workaround uses string-pasting to separate the " and the #endif into
    // different strings
    R"(
  #endif /* defined(__CUDACC__) */
#endif /* defined(__cplusplus) */
#undef __HALF_TO_US
#undef __HALF_TO_CUS

typedef __half half;
)";
#endif

#if defined(USE_ROCM)

#if ROCM_VERSION >= 70000
#define BF16_UINT32_DEF "typedef unsigned int uint32_t;\n"
#else
#define BF16_UINT32_DEF ""
#endif

constexpr auto bfloat16_support_literal =
    R"(
#ifndef __align__
#define __align__(x) __attribute__((aligned(x)))
#endif
)" BF16_UINT32_DEF R"(
typedef struct __align__(2) {
  unsigned short x;
}
__nv_bfloat16_raw;

#if defined(__cplusplus)
struct __align__(2) __nv_bfloat16 {
  __host__ __device__ __nv_bfloat16() {}

  __host__ __device__ __nv_bfloat16& operator=(const __nv_bfloat16_raw& hr) {
    __x = hr.x;
    return *this;
  }

  unsigned short __x;
};

__device__ unsigned short __internal_float2bfloat16(
    const float f,
    unsigned int& sign,
    unsigned int& remainder) {
  unsigned int x;

  x = __float_as_uint(f);

  if ((x & 0x7fffffffU) > 0x7f800000U) {
    sign = 0U;
    remainder = 0U;
    return static_cast<unsigned short>(0x7fffU);
  }
  sign = x >> 31;
  remainder = x << 16;
  return static_cast<unsigned short>(x >> 16);
}

/* Definitions of intrinsics */
__device__ __nv_bfloat16 __float2bfloat16(const float a) {
  __nv_bfloat16 val;
  __nv_bfloat16_raw r;
  unsigned int sign;
  unsigned int remainder;
  r.x = __internal_float2bfloat16(a, sign, remainder);
  if ((remainder > 0x80000000U) ||
      ((remainder == 0x80000000U) && ((r.x & 0x1U) != 0U))) {
    r.x++;
  }
  val = r;
  return val;
}

__device__ float __bfloat162float(const __nv_bfloat16 a) {
  union
  {
      uint32_t int32;
      float    fp32;
  } u = {uint32_t(a.__x) << 16};
  return u.fp32;
}
#endif /* defined(__cplusplus) */
)";
#else
constexpr auto bfloat16_support_literal =
    R"(
#define __BFLOAT16_TO_US(var) *(reinterpret_cast<unsigned short*>(&(var)))
#define __BFLOAT16_TO_CUS(var) \
  *(reinterpret_cast<const unsigned short*>(&(var)))

typedef struct __align__(2) {
  unsigned short x;
}
__nv_bfloat16_raw;

#if defined(__cplusplus)
struct __align__(2) __nv_bfloat16 {
  __host__ __device__ __nv_bfloat16() {}

  __host__ __device__ __nv_bfloat16& operator=(const __nv_bfloat16_raw& hr) {
    __x = hr.x;
    return *this;
  }

 protected:
  unsigned short __x;
};

#if defined(__CUDACC__)
__device__ unsigned short __internal_float2bfloat16(
    const float f,
    unsigned int& sign,
    unsigned int& remainder) {
  unsigned int x;

  x = __float_as_uint(f);

  if ((x & 0x7fffffffU) > 0x7f800000U) {
    sign = 0U;
    remainder = 0U;
    return static_cast<unsigned short>(0x7fffU);
  }
  sign = x >> 31;
  remainder = x << 16;
  return static_cast<unsigned short>(x >> 16);
}

/* Definitions of intrinsics */
__device__ __nv_bfloat16 __float2bfloat16(const float a) {
  __nv_bfloat16 val;
#if __CUDA_ARCH__ >= 800
  asm("{  cvt.rn.bf16.f32 %0, %1;}\n" : "=h"(__BFLOAT16_TO_US(val)) : "f"(a));
#else
  __nv_bfloat16_raw r;
  unsigned int sign;
  unsigned int remainder;
  r.x = __internal_float2bfloat16(a, sign, remainder);
  if ((remainder > 0x80000000U) ||
      ((remainder == 0x80000000U) && ((r.x & 0x1U) != 0U))) {
    r.x++;
  }
  val = r;
#endif
  return val;
}

__device__ float __bfloat162float(const __nv_bfloat16 a) {
  float val;
  asm("{ mov.b32 %0, {0,%1};}\n" : "=f"(val) : "h"(__BFLOAT16_TO_CUS(a)));
  return val;
}
#endif /* defined(__CUDACC__) */
#endif /* defined(__cplusplus) */
#undef __BFLOAT16_TO_US
#undef __BFLOAT16_TO_CUS
)";
#endif

} // namespace torch::jit::fuser::cuda
