// Pytorch also has an implementation of Philox RNG: https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/codegen/cuda/runtime/random_numbers.cu
#pragma once
// Philox CUDA.

#include <ATen/cuda/CUDAContext.h>

namespace {

class Philox {
public:
  __device__ inline Philox(unsigned long long seed,
                           unsigned long long subsequence,
                           unsigned long long offset)
      : STATE(0)
      , key(reinterpret_cast<const uint2&>(seed)) {
    //key.x = (unsigned int)seed;
    //key.y = (unsigned int)(seed >> 32);
    //counter = make_uint4(0, 0, 0, 0);
    //counter.z = (unsigned int)(subsequence);
    //counter.w = (unsigned int)(subsequence >> 32);
    //STATE = 0;
    //incr_n(offset / 4);

    // key = reinterpret_cast<const uint2&>(seed);
    ull2 * tmp = reinterpret_cast<ull2*>(&counter);
    tmp->x = offset / 4;
    tmp->y = subsequence;
    // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0)) {
    //     printf("Philox counter: %d, %d, %d, %d\n", counter.x, counter.y, counter.z, counter.w);
    // }
  }
  __device__ inline uint4 operator()() {
    // if (STATE == 0) {
      uint4 counter_ = counter;
      uint2 key_ = key;
      // 7-round philox
      #pragma unroll
      for (int i = 0; i < 6; i++) {
        counter_ = single_round(counter_, key_);
        key_.x += (kPhilox10A);
        key_.y += (kPhilox10B);
      }
      // output = single_round(counter_, key_);
      uint4 output = single_round(counter_, key_);
      // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0)) {
      //     printf("Philox counter: %u, %u, %u, %u\n", counter.x, counter.y, counter.z, counter.w);
      //     printf("Philox output: %u, %u, %u, %u\n", output.x, output.y, output.z, output.w);
      // }
      incr();
    // }
    // return a float4 directly
    // unsigned long ret;
    // switch(STATE) {
    //  case 0: ret = output.x; break;
    //  case 1: ret = output.y; break;
    //  case 2: ret = output.z; break;
    //  case 3: ret = output.w; break;
    //}
    // STATE = (STATE + 1) % 4;
    return output;
  }

private:
  struct ull2 {
      uint64_t x;
      uint64_t y;
  };
  uint4 counter;
  // uint4 output;
  const uint2 key;
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

  __device__ uint4 incr128 (uint4 ctr)
  {
    uint4 res;
    asm ("add.cc.u32      %0, %4, %8;\n\t"
         "addc.cc.u32     %1, %5, %9;\n\t"
         "addc.cc.u32     %2, %6, %10;\n\t"
         "addc.u32        %3, %7, %11;\n\t"
         : "=r"(res.x), "=r"(res.y), "=r"(res.z), "=r"(res.w)
         : "r"(ctr.x), "r"(ctr.y), "r"(ctr.z), "r"(ctr.w),
           "n"(1), "n"(0), "n"(0), "n"(0));
    return res;
  }

  __device__ inline void incr() {
    // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0)) {
    //     printf("Counter before: %u, %u, %u, %u\n", counter.x, counter.y, counter.z, counter.w);
    // }
    counter = incr128(counter);
    // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0)) {
    //     printf("Counter after: %u, %u, %u, %u\n", counter.x, counter.y, counter.z, counter.w);
    // }
  }
  __device__ unsigned int mulhilo32(unsigned int a, unsigned int b,
                                    unsigned int *result_high) {
    *result_high = __umulhi(a, b);
    return a * b;
  }
  __device__ uint2 mulhilo32_v2 (const unsigned int a, const unsigned int b)
  {
    uint2 *res;
    unsigned long long tmp;
    asm ("mul.wide.u32      %0, %1, %2;\n\t"
         : "=l"(tmp)
         : "r"(a), "r"(b));
    res = (uint2*)(&tmp);
    return *res;
  }
  __device__ inline uint4 single_round(const uint4 ctr, const uint2 key) {
    //unsigned int hi0;
    //unsigned int hi1;
    //unsigned int lo0 = mulhilo32(kPhiloxSA, ctr.x, &hi0);
    //unsigned int lo1 = mulhilo32(kPhiloxSB, ctr.z, &hi1);
    //uint4 ret = {hi1 ^ ctr.y ^ key.x, lo1, hi0 ^ ctr.w ^ key.y, lo0};
    uint2 res0 = mulhilo32_v2(kPhiloxSA, ctr.x);
    uint2 res1 = mulhilo32_v2(kPhiloxSB, ctr.z);
    uint4 ret = {res1.y ^ ctr.y ^ key.x, res1.x, res0.y ^ ctr.w ^ key.y, res0.x};
    return ret;
  }
  static const unsigned long kPhilox10A = 0x9E3779B9;
  static const unsigned long kPhilox10B = 0xBB67AE85;
  static const unsigned long kPhiloxSA = 0xD2511F53;
  static const unsigned long kPhiloxSB = 0xCD9E8D57;
};
// Inverse of 2^32.
constexpr float M_RAN_INVM32 = 2.3283064e-10f;
__device__ __inline__ float4 uniform4(const uint4 x) {
  return make_float4(x.x * M_RAN_INVM32, x.y * M_RAN_INVM32, x.z * M_RAN_INVM32,
                     x.w * M_RAN_INVM32);
}

} // namespace
