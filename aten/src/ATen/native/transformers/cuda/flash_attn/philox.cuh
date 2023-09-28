// Pytorch also has an implementation of Philox RNG: https://github.com/pytorch/pytorch/blob/8ca3c881db3e3510fcb7725389f6a0633c9b992c/torch/csrc/jit/tensorexpr/cuda_random.h
#pragma once
// Philox CUDA.

#include <ATen/cuda/CUDAContext.h>

namespace pytorch_flash{

struct ull2 {
    unsigned long long x;
    unsigned long long y;
};

inline __device__ uint2 mulhilo32(const unsigned int a, const unsigned int b) {
    uint2 *res;
    unsigned long long tmp;
    asm ("mul.wide.u32 %0, %1, %2;\n\t"
          : "=l"(tmp)
          : "r"(a), "r"(b));
    res = (uint2*)(&tmp);
    return *res;
}

inline __device__ uint4 philox_single_round(const uint4 ctr, const uint2 key) {
    constexpr unsigned long kPhiloxSA = 0xD2511F53;
    constexpr unsigned long kPhiloxSB = 0xCD9E8D57;
    uint2 res0 = mulhilo32(kPhiloxSA, ctr.x);
    uint2 res1 = mulhilo32(kPhiloxSB, ctr.z);
    uint4 ret = {res1.y ^ ctr.y ^ key.x, res1.x, res0.y ^ ctr.w ^ key.y, res0.x};
    return ret;
}

inline __device__ uint4 philox(unsigned long long seed,
                               unsigned long long subsequence,
                               unsigned long long offset) {
    constexpr unsigned long kPhilox10A = 0x9E3779B9;
    constexpr unsigned long kPhilox10B = 0xBB67AE85;
    uint2 key = reinterpret_cast<uint2&>(seed);
    uint4 counter;
    ull2 *tmp = reinterpret_cast<ull2*>(&counter);
    tmp->x = offset;
    tmp->y = subsequence;
    #pragma unroll
    for (int i = 0; i < 6; i++) {
        counter = philox_single_round(counter, key);
        key.x += (kPhilox10A);
        key.y += (kPhilox10B);
    }
    uint4 output = philox_single_round(counter, key);
    return output;
}

} // namespace flash

namespace {

class Philox {
public:
  __device__ inline Philox(unsigned long long seed,
                           unsigned long long subsequence,
                           unsigned long long offset)
      : STATE(0)
      , seed_(seed)
      , offset_(offset)
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
    // // if (STATE == 0) {
    //   uint4 counter_ = counter;
    //   uint2 key_ = key;
    //   // 7-round philox
    //   #pragma unroll
    //   for (int i = 0; i < 6; i++) {
    //       counter_ = pytorch_flash::philox_single_round(counter_, key_);
    //     key_.x += (kPhilox10A);
    //     key_.y += (kPhilox10B);
    //   }
    //   // output = philox_single_round(counter_, key_);
    //   uint4 output = pytorch_flash::philox_single_round(counter_, key_);
    //   // if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0)) {
    //   //     printf("Philox counter: %u, %u, %u, %u\n", counter.x, counter.y, counter.z, counter.w);
    //   //     printf("Philox output: %u, %u, %u, %u\n", output.x, output.y, output.z, output.w);
    //   // }
    //   incr();
    // // }
    // // return a float4 directly
    // // unsigned long ret;
    // // switch(STATE) {
    // //  case 0: ret = output.x; break;
    // //  case 1: ret = output.y; break;
    // //  case 2: ret = output.z; break;
    // //  case 3: ret = output.w; break;
    // //}
    // // STATE = (STATE + 1) % 4;
    // return output;
      return pytorch_flash::philox(seed_, offset_, offset_);
  }

private:
  unsigned long long offset_, seed_;
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

  static const unsigned long kPhilox10A = 0x9E3779B9;
  static const unsigned long kPhilox10B = 0xBB67AE85;
  // static const unsigned long kPhiloxSA = 0xD2511F53;
  // static const unsigned long kPhiloxSB = 0xCD9E8D57;
};

} // namespace pytorch_flash
